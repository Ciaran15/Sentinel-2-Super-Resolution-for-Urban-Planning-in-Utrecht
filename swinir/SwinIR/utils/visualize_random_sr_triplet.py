#!/usr/bin/env python3
"""Visualize a random LR/SR/GT image triplet for qualitative inspection.

This utility is intended for paired super-resolution workflows in which:
- ``folder_sr`` contains model output images,
- ``folder_lq`` contains the low-resolution input images, and
- ``folder_gt`` contains the ground-truth reference images.

The script selects one random super-resolved image, resolves the matching LR and
GT files by filename stem, and displays all three images side by side using
Matplotlib. It is designed for research demos, reporting, and public GitHub
repositories where a quick visual comparison is useful.

Key features
------------
- Randomly selects one SR image from a results folder.
- Matches LR and GT images by filename stem.
- Includes fallback handling for common SR suffixes such as ``_SwinIR`` and
  ``_x4``.
- Supports different file extensions across SR, LR, and GT folders.
- Can optionally save the comparison figure to disk.
- Can be imported as a reusable function in a notebook or executed as a script.

Example usage
-------------
Run as a script:

    python visualize_random_sr_triplet.py \
        --folder_sr results/swinir_real_sr_x4_large \
        --folder_lq /home/datalab/zDatalab2/dataset/test/inputs \
        --folder_gt /home/datalab/zDatalab2/dataset/test/ground_truth \
        --seed 42 \
        --save_path results/swinir_real_sr_x4_large/random_triplet.png

Use from a notebook:

    from visualize_random_sr_triplet import show_random_triplet

    show_random_triplet(
        folder_sr='results/swinir_real_sr_x4_large',
        folder_lq='/home/datalab/zDatalab2/dataset/test/inputs',
        folder_gt='/home/datalab/zDatalab2/dataset/test/ground_truth',
        seed=42,
    )
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
COMMON_SR_SUFFIXES = [
    "_SwinIR",
    "_swinir",
    "_SR",
    "_sr",
    "_x2",
    "_x3",
    "_x4",
    "_x8",
    "_out",
    "_result",
]


def list_images(folder: Path) -> List[Path]:
    """Return all supported image files in a folder in sorted order.

    Parameters
    ----------
    folder:
        Directory containing image files.

    Returns
    -------
    list[Path]
        Sorted list of image file paths.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS
    )


def build_stem_index(folder: Path) -> Dict[str, List[Path]]:
    """Build a lookup table from filename stem to image paths.

    Parameters
    ----------
    folder:
        Directory containing images.

    Returns
    -------
    dict[str, list[Path]]
        Mapping from filename stem to one or more image paths.
    """
    index: Dict[str, List[Path]] = {}
    for path in list_images(folder):
        index.setdefault(path.stem, []).append(path)
    return index


def candidate_stems(stem: str) -> List[str]:
    """Generate candidate filename stems for robust image matching.

    The function first tries the original stem and then removes a number of
    common super-resolution suffixes. This allows SR outputs such as
    ``tile001_SwinIR.png`` or ``tile001_x4.png`` to match LR/GT files named
    ``tile001.jpg``.

    Parameters
    ----------
    stem:
        Filename stem of the super-resolved output image.

    Returns
    -------
    list[str]
        Ordered list of candidate stems.
    """
    stems = [stem]
    for suffix in COMMON_SR_SUFFIXES:
        if stem.endswith(suffix):
            stems.append(stem[: -len(suffix)])
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(stems))


def resolve_match(reference_path: Path, index: Dict[str, List[Path]]) -> Path:
    """Resolve the most likely matching image for a reference file.

    Parameters
    ----------
    reference_path:
        Reference image path, typically the SR output.
    index:
        Lookup table created by :func:`build_stem_index`.

    Returns
    -------
    Path
        Matched image path.

    Raises
    ------
    FileNotFoundError
        If no corresponding image can be found.
    """
    for stem in candidate_stems(reference_path.stem):
        matches = index.get(stem, [])
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find a matching file for '{reference_path.name}' using stems {candidate_stems(reference_path.stem)}"
    )


def read_image_rgb(path: Path) -> np.ndarray:
    """Load an image as an RGB NumPy array.

    Parameters
    ----------
    path:
        Image file to load.

    Returns
    -------
    numpy.ndarray
        RGB image array with shape ``(H, W, 3)``.
    """
    return np.array(Image.open(path).convert("RGB"))


def show_random_triplet(
    folder_sr: str | Path,
    folder_lq: str | Path,
    folder_gt: str | Path,
    seed: Optional[int] = None,
    save_path: Optional[str | Path] = None,
    figsize: Tuple[float, float] = (18, 6),
) -> Dict[str, str]:
    """Display a random LR/SR/GT triplet and optionally save the figure.

    Parameters
    ----------
    folder_sr:
        Directory containing super-resolved output images.
    folder_lq:
        Directory containing low-resolution input images.
    folder_gt:
        Directory containing ground-truth reference images.
    seed:
        Optional random seed for reproducible sampling.
    save_path:
        Optional output path for saving the generated comparison figure.
    figsize:
        Matplotlib figure size as ``(width, height)``.

    Returns
    -------
    dict[str, str]
        Dictionary with the resolved ``sr_path``, ``lq_path``, and ``gt_path``.

    Raises
    ------
    FileNotFoundError
        If one of the input folders does not exist or no matching images are
        found.
    RuntimeError
        If the SR folder contains no supported images.
    """
    folder_sr = Path(folder_sr)
    folder_lq = Path(folder_lq)
    folder_gt = Path(folder_gt)

    sr_paths = list_images(folder_sr)
    if not sr_paths:
        raise RuntimeError(f"No SR images found in folder: {folder_sr}")

    lq_index = build_stem_index(folder_lq)
    gt_index = build_stem_index(folder_gt)

    rng = random.Random(seed)
    sr_path = rng.choice(sr_paths)
    lq_path = resolve_match(sr_path, lq_index)
    gt_path = resolve_match(sr_path, gt_index)

    lq_img = read_image_rgb(lq_path)
    sr_img = read_image_rgb(sr_path)
    gt_img = read_image_rgb(gt_path)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    triplets: Sequence[Tuple[str, np.ndarray, Path]] = (
        ("Low-Resolution Input", lq_img, lq_path),
        ("Super-Resolved Output", sr_img, sr_path),
        ("Ground Truth", gt_img, gt_path),
    )

    for axis, (title, image, path) in zip(axes, triplets):
        axis.imshow(image)
        axis.set_title(f"{title}\n{path.name}", fontsize=11)
        axis.axis("off")

    fig.suptitle("Random LR / SR / GT Comparison", fontsize=14)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved comparison figure to: {save_path}")

    plt.show()

    return {
        "sr_path": str(sr_path),
        "lq_path": str(lq_path),
        "gt_path": str(gt_path),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the visualization utility.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Display one random LR/SR/GT triplet for qualitative comparison."
    )
    parser.add_argument("--folder_sr", type=str, required=True, help="Folder containing SR result images.")
    parser.add_argument("--folder_lq", type=str, required=True, help="Folder containing LR input images.")
    parser.add_argument("--folder_gt", type=str, required=True, help="Folder containing GT reference images.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible selection.")
    parser.add_argument("--save_path", type=str, default="", help="Optional output path for saving the comparison figure.")
    return parser.parse_args()


def main() -> None:
    """Run the random triplet visualization utility from the command line."""
    args = parse_args()
    show_random_triplet(
        folder_sr=args.folder_sr,
        folder_lq=args.folder_lq,
        folder_gt=args.folder_gt,
        seed=args.seed,
        save_path=args.save_path or None,
    )


if __name__ == "__main__":
    main()
