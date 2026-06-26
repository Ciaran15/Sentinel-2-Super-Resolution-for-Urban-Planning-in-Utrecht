#!/usr/bin/env python3
"""Fine-tune a pretrained SwinIR model on paired LR/GT imagery.

This script is designed for supervised fine-tuning workflows in which
low-resolution (LR) input images and high-resolution ground-truth (GT) images
are stored in separate folders and matched by filename stem. It is intended for
paired remote-sensing experiments such as Sentinel-2 to aerial-image
super-resolution, but it can also be used for other RGB paired-image datasets.

Key features
------------
- Loads a pretrained SwinIR checkpoint for initialization.
- Matches LR and GT images by filename stem, with optional fallback suffix
  handling for generated SR filenames.
- Supports standard train/validation folder splits.
- Uses random cropped patches for training and full-image validation.
- Computes validation PSNR on RGB images.
- Saves periodic and best-model checkpoints for later reuse.
- Includes professional docstrings and clear command-line arguments for public
  repository use.

Expected directory structure
----------------------------
A typical dataset layout is:

    dataset/
      train/
        inputs/
        ground_truth/
      val/
        inputs/
        ground_truth/

Example usage
-------------
python train_swinir_finetune.py \
    --train_lq /path/to/dataset/train/inputs \
    --train_gt /path/to/dataset/train/ground_truth \
    --val_lq /path/to/dataset/val/inputs \
    --val_gt /path/to/dataset/val/ground_truth \
    --checkpoint_init model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth \
    --save_dir experiments/swinir_remote_finetune \
    --scale 4 \
    --large_model

Notes
-----
1. This script imports the SwinIR network definition from the official
   repository path ``models/network_swinir.py``. Run it from the SwinIR project
   root, or ensure that the repository root is on ``PYTHONPATH``.
2. The script assumes 3-channel RGB imagery.
3. For a first domain-adaptation baseline, the default loss is L1
   reconstruction loss, which is typically more stable than GAN-based training
   for remote-sensing use cases.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.network_swinir import SwinIR

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class PairedSample:
    """Container describing one matched LR/GT image pair.

    Attributes
    ----------
    lq_path:
        Path to the low-resolution input image.
    gt_path:
        Path to the ground-truth target image.
    stem:
        Common sample identifier derived from the matched filename stem.
    """

    lq_path: Path
    gt_path: Path
    stem: str


def list_images(folder: Path) -> List[Path]:
    """Return all supported image files in a folder in sorted order.

    Parameters
    ----------
    folder:
        Directory containing image files.

    Returns
    -------
    list[Path]
        Sorted image paths whose suffixes are included in ``VALID_EXTS``.
    """
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def build_stem_index(folder: Path) -> Dict[str, List[Path]]:
    """Build a lookup from filename stem to candidate image paths.

    Parameters
    ----------
    folder:
        Directory containing images.

    Returns
    -------
    dict[str, list[Path]]
        Mapping from image stem to one or more matching file paths.
    """
    index: Dict[str, List[Path]] = {}
    for path in list_images(folder):
        index.setdefault(path.stem, []).append(path)
    return index


def resolve_pair_path(lq_path: Path, gt_index: Dict[str, List[Path]]) -> Optional[Path]:
    """Resolve the most likely GT path for a given LR image.

    The function first tries an exact stem match. If that fails, it removes a
    number of common super-resolution suffixes and tries again. This makes the
    loader more tolerant of filenames such as ``tile001_x4.png`` or
    ``tile001_SwinIR.png``.

    Parameters
    ----------
    lq_path:
        Input LR image path.
    gt_index:
        Ground-truth index created by :func:`build_stem_index`.

    Returns
    -------
    Path or None
        The matched GT path if found; otherwise ``None``.
    """
    stem = lq_path.stem
    if stem in gt_index:
        return gt_index[stem][0]

    suffixes = ["_SwinIR", "_swinir", "_x4", "_x2", "_x3", "_x8", "_out", "_result"]
    for suffix in suffixes:
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            if base in gt_index:
                return gt_index[base][0]

    return None


def build_pairs(folder_lq: Path, folder_gt: Path) -> List[PairedSample]:
    """Match LR and GT images by filename stem.

    Parameters
    ----------
    folder_lq:
        Directory containing LR input images.
    folder_gt:
        Directory containing GT reference images.

    Returns
    -------
    list[PairedSample]
        Matched LR/GT pairs in deterministic order.

    Raises
    ------
    FileNotFoundError
        If no LR images are found.
    RuntimeError
        If no valid LR/GT pairs can be matched.
    """
    lq_paths = list_images(folder_lq)
    if not lq_paths:
        raise FileNotFoundError(f"No LR images found in: {folder_lq}")

    gt_index = build_stem_index(folder_gt)
    pairs: List[PairedSample] = []
    missing: List[str] = []

    for lq_path in lq_paths:
        gt_path = resolve_pair_path(lq_path, gt_index)
        if gt_path is None:
            missing.append(lq_path.name)
            continue
        pairs.append(PairedSample(lq_path=lq_path, gt_path=gt_path, stem=gt_path.stem))

    if not pairs:
        raise RuntimeError(
            f"No matched LR/GT pairs found between {folder_lq} and {folder_gt}."
        )

    if missing:
        print(
            f"[Warning] {len(missing)} LR files did not find a matching GT file. "
            f"Example: {missing[:5]}"
        )

    return pairs


def read_image_rgb(path: Path) -> np.ndarray:
    """Read an image as normalized RGB ``float32`` data.

    Parameters
    ----------
    path:
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        RGB image with shape ``(H, W, 3)`` and values in ``[0, 1]``.

    Raises
    ------
    FileNotFoundError
        If the image cannot be loaded.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert an ``HWC`` RGB image array to a ``CHW`` torch tensor.

    Parameters
    ----------
    img:
        RGB image array with values in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(3, H, W)`` and dtype ``float32``.
    """
    return torch.from_numpy(np.transpose(img, (2, 0, 1))).float()


def ensure_min_size(img: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
    """Pad an image by reflection if it is smaller than a required size.

    Parameters
    ----------
    img:
        Input RGB image.
    min_h:
        Minimum required height.
    min_w:
        Minimum required width.

    Returns
    -------
    numpy.ndarray
        Image that is at least ``(min_h, min_w)``.
    """
    h, w = img.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return img
    return cv2.copyMakeBorder(
        img,
        top=0,
        bottom=pad_h,
        left=0,
        right=pad_w,
        borderType=cv2.BORDER_REFLECT_101,
    )


def random_crop_pair(lq: np.ndarray, gt: np.ndarray, lq_patch_size: int, scale: int) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly crop a spatially corresponding LR/GT patch pair.

    Parameters
    ----------
    lq:
        LR image in RGB ``float32`` format.
    gt:
        GT image in RGB ``float32`` format.
    lq_patch_size:
        Patch size sampled on the LR image.
    scale:
        Super-resolution scale factor relating GT size to LR size.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Cropped LR patch and the corresponding GT patch.

    Notes
    -----
    The GT crop is assumed to be aligned to the LR crop by a factor of
    ``scale``. The function will raise an error if the images are too small or
    inconsistent for the requested crop.
    """
    lq = ensure_min_size(lq, lq_patch_size, lq_patch_size)
    gt = ensure_min_size(gt, lq_patch_size * scale, lq_patch_size * scale)

    h_lq, w_lq = lq.shape[:2]
    h_gt, w_gt = gt.shape[:2]

    if h_gt < lq_patch_size * scale or w_gt < lq_patch_size * scale:
        raise ValueError("GT image is smaller than the required scaled patch size.")

    max_top = h_lq - lq_patch_size
    max_left = w_lq - lq_patch_size
    top = random.randint(0, max_top) if max_top > 0 else 0
    left = random.randint(0, max_left) if max_left > 0 else 0

    top_gt = top * scale
    left_gt = left * scale

    lq_patch = lq[top : top + lq_patch_size, left : left + lq_patch_size]
    gt_patch = gt[
        top_gt : top_gt + lq_patch_size * scale,
        left_gt : left_gt + lq_patch_size * scale,
    ]

    return lq_patch, gt_patch


def center_crop_to_scale_pair(lq: np.ndarray, gt: np.ndarray, scale: int) -> Tuple[np.ndarray, np.ndarray]:
    """Center-crop LR and GT images so their shapes are mutually consistent.

    Parameters
    ----------
    lq:
        LR image.
    gt:
        GT image.
    scale:
        Super-resolution scale factor.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Center-cropped LR and GT arrays satisfying the expected ``scale``
        relationship.
    """
    h_lq, w_lq = lq.shape[:2]
    h_gt, w_gt = gt.shape[:2]

    target_h_lq = min(h_lq, h_gt // scale)
    target_w_lq = min(w_lq, w_gt // scale)
    target_h_gt = target_h_lq * scale
    target_w_gt = target_w_lq * scale

    def crop_center(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        top = max((img.shape[0] - target_h) // 2, 0)
        left = max((img.shape[1] - target_w) // 2, 0)
        return img[top : top + target_h, left : left + target_w]

    lq = crop_center(lq, target_h_lq, target_w_lq)
    gt = crop_center(gt, target_h_gt, target_w_gt)
    return lq, gt


class PairedTrainDataset(Dataset):
    """Training dataset for paired LR/GT image super-resolution.

    Each sample is matched by filename stem and converted into a random
    corresponding LR/GT crop pair. Optional horizontal and vertical flips plus
    90-degree rotation augmentation can be enabled.
    """

    def __init__(
        self,
        folder_lq: Path,
        folder_gt: Path,
        scale: int,
        lq_patch_size: int,
        augment: bool = True,
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        folder_lq:
            Directory containing LR training inputs.
        folder_gt:
            Directory containing GT training targets.
        scale:
            Super-resolution scale factor.
        lq_patch_size:
            Random training crop size on the LR images.
        augment:
            Whether to apply simple geometric augmentations.
        """
        self.pairs = build_pairs(folder_lq, folder_gt)
        self.scale = scale
        self.lq_patch_size = lq_patch_size
        self.augment = augment

    def __len__(self) -> int:
        """Return the number of matched training pairs."""
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load one paired training example.

        Parameters
        ----------
        index:
            Sample index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            LR tensor and GT tensor with shapes ``(3, H, W)`` and
            ``(3, scale*H, scale*W)``.
        """
        pair = self.pairs[index]
        lq = read_image_rgb(pair.lq_path)
        gt = read_image_rgb(pair.gt_path)
        lq, gt = center_crop_to_scale_pair(lq, gt, self.scale)
        lq, gt = random_crop_pair(lq, gt, self.lq_patch_size, self.scale)

        if self.augment:
            if random.random() < 0.5:
                lq = np.flip(lq, axis=1).copy()
                gt = np.flip(gt, axis=1).copy()
            if random.random() < 0.5:
                lq = np.flip(lq, axis=0).copy()
                gt = np.flip(gt, axis=0).copy()
            if random.random() < 0.5:
                lq = np.transpose(lq, (1, 0, 2)).copy()
                gt = np.transpose(gt, (1, 0, 2)).copy()

        return numpy_to_tensor(lq), numpy_to_tensor(gt)


class PairedValDataset(Dataset):
    """Validation dataset for full-image paired LR/GT evaluation."""

    def __init__(self, folder_lq: Path, folder_gt: Path, scale: int) -> None:
        """Initialize the validation dataset.

        Parameters
        ----------
        folder_lq:
            Directory containing LR validation inputs.
        folder_gt:
            Directory containing GT validation targets.
        scale:
            Super-resolution scale factor.
        """
        self.pairs = build_pairs(folder_lq, folder_gt)
        self.scale = scale

    def __len__(self) -> int:
        """Return the number of matched validation pairs."""
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Load one validation example.

        Parameters
        ----------
        index:
            Sample index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, str]
            LR tensor, GT tensor, and sample stem identifier.
        """
        pair = self.pairs[index]
        lq = read_image_rgb(pair.lq_path)
        gt = read_image_rgb(pair.gt_path)
        lq, gt = center_crop_to_scale_pair(lq, gt, self.scale)
        return numpy_to_tensor(lq), numpy_to_tensor(gt), pair.stem


class L1Loss(nn.Module):
    """Simple wrapper around mean absolute error loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 reconstruction loss.

        Parameters
        ----------
        pred:
            Predicted SR tensor.
        target:
            Ground-truth tensor.

        Returns
        -------
        torch.Tensor
            Scalar L1 loss value.
        """
        return F.l1_loss(pred, target)


def set_random_seed(seed: int) -> None:
    """Set deterministic random seeds for reproducible experiments.

    Parameters
    ----------
    seed:
        Base seed used for Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(scale: int, large_model: bool) -> SwinIR:
    """Construct a SwinIR network matching the selected pretrained variant.

    Parameters
    ----------
    scale:
        Super-resolution scale factor.
    large_model:
        Whether to build the large real-SR SwinIR-L configuration.

    Returns
    -------
    SwinIR
        Initialized SwinIR network.

    Notes
    -----
    The hyperparameters here follow the standard real-SR configurations used in
    the public SwinIR release. The large model uses a larger embedding
    dimension, more residual Swin Transformer blocks, and a larger window size.
    """
    if large_model:
        return SwinIR(
            upscale=scale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=240,
            num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
            mlp_ratio=2,
            upsampler="nearest+conv",
            resi_connection="3conv",
        )

    return SwinIR(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection="1conv",
    )


def load_checkpoint_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """Load pretrained weights into the SwinIR model.

    Parameters
    ----------
    model:
        Target model instance.
    checkpoint_path:
        Path to the pretrained or previously fine-tuned checkpoint.
    device:
        Active PyTorch device.

    Raises
    ------
    FileNotFoundError
        If the checkpoint path does not exist.
    RuntimeError
        If no compatible state dictionary can be extracted.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = None

    if isinstance(checkpoint, dict):
        for key in ["params_ema", "params", "state_dict", "model", "net"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
        if state_dict is None and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint

    if state_dict is None:
        raise RuntimeError(f"Could not extract a model state dictionary from: {checkpoint_path}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if missing:
        print(f"[Info] Missing keys during load: {len(missing)}")
    if unexpected:
        print(f"[Info] Unexpected keys during load: {len(unexpected)}")


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor batch item to a clipped RGB image array.

    Parameters
    ----------
    tensor:
        Tensor with shape ``(1, 3, H, W)`` or ``(3, H, W)``.

    Returns
    -------
    numpy.ndarray
        RGB image array with dtype ``float32`` and values in ``[0, 1]``.
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().clamp_(0.0, 1.0)
    return np.transpose(tensor.numpy(), (1, 2, 0)).astype(np.float32)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute RGB PSNR between two normalized images.

    Parameters
    ----------
    img1:
        First image in ``[0, 1]``.
    img2:
        Second image in ``[0, 1]``.

    Returns
    -------
    float
        Peak signal-to-noise ratio in decibels.
    """
    mse = float(np.mean((img1 - img2) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


@torch.no_grad()
def run_model_tiled(model: nn.Module, lq: torch.Tensor, scale: int, tile: int, tile_overlap: int) -> torch.Tensor:
    """Run tiled inference for memory-efficient validation.

    Parameters
    ----------
    model:
        SwinIR model in evaluation mode.
    lq:
        LR tensor with shape ``(1, 3, H, W)``.
    scale:
        Super-resolution scale factor.
    tile:
        Tile size on the LR image.
    tile_overlap:
        Overlap size between adjacent LR tiles.

    Returns
    -------
    torch.Tensor
        Reconstructed SR tensor.
    """
    _, _, h, w = lq.size()
    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [max(h - tile, 0)]
    w_idx_list = list(range(0, w - tile, stride)) + [max(w - tile, 0)]

    output = torch.zeros(1, 3, h * scale, w * scale, device=lq.device)
    weight = torch.zeros_like(output)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            patch = lq[:, :, h_idx : h_idx + tile, w_idx : w_idx + tile]
            sr_patch = model(patch)
            out_h = h_idx * scale
            out_w = w_idx * scale
            output[:, :, out_h : out_h + sr_patch.size(2), out_w : out_w + sr_patch.size(3)] += sr_patch
            weight[:, :, out_h : out_h + sr_patch.size(2), out_w : out_w + sr_patch.size(3)] += 1

    return output / weight.clamp_min(1e-8)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scale: int,
    tile: int,
    tile_overlap: int,
) -> Tuple[float, List[Tuple[str, float]]]:
    """Evaluate the model on the validation split using RGB PSNR.

    Parameters
    ----------
    model:
        SwinIR model to evaluate.
    loader:
        Validation data loader.
    device:
        Active PyTorch device.
    scale:
        Super-resolution scale factor.
    tile:
        Tile size for validation. Use ``0`` to disable tiled inference.
    tile_overlap:
        Overlap between validation tiles.

    Returns
    -------
    tuple[float, list[tuple[str, float]]]
        Mean PSNR and the per-image PSNR values.
    """
    model.eval()
    psnr_rows: List[Tuple[str, float]] = []

    for lq, gt, stem in loader:
        lq = lq.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        if tile > 0:
            pred = run_model_tiled(model, lq, scale=scale, tile=tile, tile_overlap=tile_overlap)
        else:
            pred = model(lq)

        pred_img = tensor_to_image(pred)
        gt_img = tensor_to_image(gt)
        psnr_rows.append((stem[0], calculate_psnr(pred_img, gt_img)))

    mean_psnr = float(np.mean([row[1] for row in psnr_rows])) if psnr_rows else float("nan")
    return mean_psnr, psnr_rows


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_psnr: float,
    args: argparse.Namespace,
) -> None:
    """Serialize model and optimizer state to a checkpoint file.

    Parameters
    ----------
    path:
        Output checkpoint path.
    model:
        Model whose weights will be saved.
    optimizer:
        Optimizer whose state will be saved.
    epoch:
        Current epoch number.
    global_step:
        Current global training step.
    best_psnr:
        Best validation PSNR observed so far.
    args:
        Parsed command-line arguments for reproducibility.
    """
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "best_psnr": best_psnr,
        "params": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, path)


def append_metrics_csv(csv_path: Path, epoch: int, global_step: int, val_psnr: float) -> None:
    """Append one validation summary row to a CSV log.

    Parameters
    ----------
    csv_path:
        CSV file path.
    epoch:
        Epoch number.
    global_step:
        Training step number.
    val_psnr:
        Validation PSNR for the current checkpoint.
    """
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["epoch", "global_step", "val_psnr_rgb"])
        writer.writerow([epoch, global_step, f"{val_psnr:.6f}"])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for fine-tuning.

    Returns
    -------
    argparse.Namespace
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained SwinIR model on paired LR/GT imagery."
    )
    parser.add_argument("--train_lq", type=Path, required=True, help="Path to the LR training folder.")
    parser.add_argument("--train_gt", type=Path, required=True, help="Path to the GT training folder.")
    parser.add_argument("--val_lq", type=Path, required=True, help="Path to the LR validation folder.")
    parser.add_argument("--val_gt", type=Path, required=True, help="Path to the GT validation folder.")
    parser.add_argument("--checkpoint_init", type=Path, required=True, help="Pretrained checkpoint used for initialization.")
    parser.add_argument("--save_dir", type=Path, required=True, help="Directory for logs and checkpoints.")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution scale factor.")
    parser.add_argument("--large_model", action="store_true", help="Use the SwinIR-L real-SR architecture.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader worker processes.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of fine-tuning epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--lq_patch_size", type=int, default=64, help="Training crop size on the LR image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--val_every", type=int, default=1, help="Validate every N epochs.")
    parser.add_argument("--save_every", type=int, default=5, help="Save a regular checkpoint every N epochs.")
    parser.add_argument("--tile", type=int, default=0, help="Tile size for validation inference; use 0 to disable.")
    parser.add_argument("--tile_overlap", type=int, default=32, help="Overlap between validation tiles.")
    return parser.parse_args()


def main() -> None:
    """Run supervised fine-tuning and periodic validation.

    The training procedure performs the following steps:
    1. Parse command-line arguments and create output directories.
    2. Build paired train/validation datasets from LR and GT folders.
    3. Construct a SwinIR model matching the chosen pretrained variant.
    4. Load pretrained initialization weights.
    5. Fine-tune the model with L1 reconstruction loss.
    6. Periodically validate on the validation split and save checkpoints.
    """
    args = parse_args()
    set_random_seed(args.seed)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.save_dir / "val_metrics.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = PairedTrainDataset(
        folder_lq=args.train_lq,
        folder_gt=args.train_gt,
        scale=args.scale,
        lq_patch_size=args.lq_patch_size,
        augment=True,
    )
    val_dataset = PairedValDataset(
        folder_lq=args.val_lq,
        folder_gt=args.val_gt,
        scale=args.scale,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, min(args.num_workers, 2)),
        pin_memory=True,
    )

    model = build_model(scale=args.scale, large_model=args.large_model).to(device)
    load_checkpoint_weights(model, args.checkpoint_init, device=device)

    criterion = L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_psnr = -float("inf")
    global_step = 0

    print(f"Training pairs   : {len(train_dataset)}")
    print(f"Validation pairs : {len(val_dataset)}")
    print(f"Save directory   : {args.save_dir}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for lq, gt in train_loader:
            lq = lq.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(lq)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            global_step += 1

        scheduler.step()
        mean_train_loss = running_loss / max(len(train_loader), 1)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_l1={mean_train_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        if epoch % args.val_every == 0:
            val_psnr, _ = validate(
                model=model,
                loader=val_loader,
                device=device,
                scale=args.scale,
                tile=args.tile,
                tile_overlap=args.tile_overlap,
            )
            append_metrics_csv(metrics_csv, epoch=epoch, global_step=global_step, val_psnr=val_psnr)
            print(f"Validation PSNR (RGB): {val_psnr:.4f} dB")

            latest_path = ckpt_dir / "latest.pth"
            save_checkpoint(
                path=latest_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_psnr=max(best_psnr, val_psnr),
                args=args,
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_path = ckpt_dir / "best.pth"
                save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_psnr=best_psnr,
                    args=args,
                )
                print(f"[Best] Updated best checkpoint: {best_path}")

        if epoch % args.save_every == 0:
            epoch_path = ckpt_dir / f"epoch_{epoch:03d}.pth"
            save_checkpoint(
                path=epoch_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_psnr=best_psnr,
                args=args,
            )
            print(f"[Checkpoint] Saved: {epoch_path}")

    print("Fine-tuning completed.")
    print(f"Best validation PSNR: {best_psnr:.4f} dB")
    print(f"Checkpoint directory : {ckpt_dir}")


if __name__ == "__main__":
    main()
