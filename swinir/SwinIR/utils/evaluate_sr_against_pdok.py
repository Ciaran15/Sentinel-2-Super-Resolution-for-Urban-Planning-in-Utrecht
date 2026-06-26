#!/usr/bin/env python3
"""Batch-evaluate super-resolved images against ground-truth reference imagery.

This utility compares a folder of super-resolved (SR) output images against a
folder of ground-truth (GT) images and reports aggregate PSNR and SSIM scores.
It is designed for paired-image evaluation workflows in which SR outputs should
correspond to reference images by file stem.

Key features:
- Matches SR and GT images by filename stem.
- Includes fallback logic for common SR suffixes such as ``_x4`` or ``_SwinIR``.
- Optionally center-crops images to a shared spatial extent when dimensions do
  not match exactly.
- Supports border cropping before metric computation to reduce edge artefacts.
- Optionally exports per-image results to CSV for downstream analysis.

This script is intended as a lightweight evaluation baseline for image
super-resolution experiments, including remote sensing workflows where outputs
must be compared to higher-resolution reference imagery.
"""

import argparse
import csv
import math
import os
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

VALID_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def list_images(folder):
    """Return all supported image files in a folder in sorted order.

    Parameters
    ----------
    folder : str or Path
        Directory containing image files.

    Returns
    -------
    list[Path]
        Sorted list of image paths whose extensions are included in
        ``VALID_EXTS``.
    """
    paths = []
    for p in sorted(Path(folder).iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            paths.append(p)
    return paths


def build_gt_index(folder_gt):
    """Create a lookup table from ground-truth filename stems to file paths.

    The index enables fast matching between SR outputs and GT images based on
    their shared basename, regardless of file extension.

    Parameters
    ----------
    folder_gt : str or Path
        Directory containing ground-truth images.

    Returns
    -------
    dict[str, list[Path]]
        Mapping from filename stem to one or more candidate GT image paths.
    """
    index = {}
    for p in list_images(folder_gt):
        index.setdefault(p.stem, []).append(p)
    return index


def resolve_gt_path(sr_path, gt_index):
    """Resolve the most likely GT path for a given SR output image.

    The function first tries an exact filename-stem match. If that fails, it
    removes common SR-related suffixes (for example ``_x4`` or ``_SwinIR``) and
    retries the lookup.

    Parameters
    ----------
    sr_path : Path
        Path to the super-resolved image.
    gt_index : dict[str, list[Path]]
        Lookup table produced by :func:`build_gt_index`.

    Returns
    -------
    Path or None
        Matched GT path if found; otherwise ``None``.
    """
    stem = sr_path.stem
    candidates = gt_index.get(stem, [])
    if candidates:
        return candidates[0]

    suffixes = [
        '_SwinIR', '_swinir', '_SR', '_sr', '_x4', '_x2', '_x3', '_x8', '_out', '_result'
    ]
    for suffix in suffixes:
        if stem.endswith(suffix):
            base = stem[:-len(suffix)]
            candidates = gt_index.get(base, [])
            if candidates:
                return candidates[0]

    return None


def read_image(path):
    """Load an image as normalized RGB float data in the range ``[0, 1]``.

    Parameters
    ----------
    path : str or Path
        Image file to load.

    Returns
    -------
    numpy.ndarray
        RGB image as ``float32`` with shape ``(H, W, 3)`` and values in
        ``[0, 1]``.

    Raises
    ------
    FileNotFoundError
        If the image cannot be read by OpenCV.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Could not load image: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def center_crop_to_common(img1, img2):
    """Center-crop two images to their shared minimum height and width.

    This is useful when the SR and GT images are nearly aligned but do not have
    identical dimensions. The crop is applied symmetrically from the center of
    each image.

    Parameters
    ----------
    img1 : numpy.ndarray
        First image array.
    img2 : numpy.ndarray
        Second image array.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Center-cropped versions of ``img1`` and ``img2`` with matching spatial
        dimensions.
    """
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])

    def crop(img, h, w):
        top = max((img.shape[0] - h) // 2, 0)
        left = max((img.shape[1] - w) // 2, 0)
        return img[top:top + h, left:left + w]

    return crop(img1, h, w), crop(img2, h, w)


def crop_border(img, border):
    """Crop a fixed border width from all four sides of an image.

    Border cropping is often used in SR evaluation to reduce the influence of
    padding artefacts near image edges.

    Parameters
    ----------
    img : numpy.ndarray
        Image to crop.
    border : int
        Number of pixels to remove from each edge.

    Returns
    -------
    numpy.ndarray
        Cropped image.

    Raises
    ------
    ValueError
        If the requested crop would remove the full image content.
    """
    if border <= 0:
        return img
    h, w = img.shape[:2]
    if h <= 2 * border or w <= 2 * border:
        raise ValueError(
            f'crop_border={border} is too large for image shape {img.shape[:2]}'
        )
    return img[border:h - border, border:w - border]


def calculate_psnr(img1, img2):
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters
    ----------
    img1 : numpy.ndarray
        First normalized image.
    img2 : numpy.ndarray
        Second normalized image.

    Returns
    -------
    float
        PSNR value in decibels. Returns ``inf`` for identical images.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_ssim_rgb(img1, img2):
    """Compute mean RGB SSIM by averaging the per-channel SSIM scores.

    Parameters
    ----------
    img1 : numpy.ndarray
        First normalized RGB image.
    img2 : numpy.ndarray
        Second normalized RGB image.

    Returns
    -------
    float
        Mean SSIM across the three RGB channels.
    """
    scores = []
    for c in range(3):
        scores.append(ssim(img1[:, :, c], img2[:, :, c], data_range=1.0))
    return float(np.mean(scores))


def main():
    """Run batch image matching and compute aggregate evaluation metrics.

    The command-line interface loads all SR images, resolves their matching GT
    files, optionally harmonizes dimensions, computes per-image PSNR and SSIM,
    and reports dataset-level averages. A CSV export can be written for later
    inspection.
    """
    parser = argparse.ArgumentParser(
        description='Batch evaluation of super-resolved images against ground-truth images.'
    )
    parser.add_argument('--folder_sr', type=str, required=True, help='Folder containing super-resolved output images.')
    parser.add_argument('--folder_gt', type=str, required=True, help='Folder containing ground-truth reference images.')
    parser.add_argument('--crop_border', type=int, default=0, help='Number of pixels to crop from each image border before evaluation.')
    parser.add_argument('--csv_path', type=str, default='', help='Optional output path for a CSV file with per-image metrics.')
    parser.add_argument('--strict_size', action='store_true', help='Fail if SR and GT images do not have identical spatial dimensions.')
    args = parser.parse_args()

    sr_paths = list_images(args.folder_sr)
    if not sr_paths:
        raise FileNotFoundError(f'No images found in folder_sr: {args.folder_sr}')

    gt_index = build_gt_index(args.folder_gt)
    if not gt_index:
        raise FileNotFoundError(f'No images found in folder_gt: {args.folder_gt}')

    rows = []
    missing = []

    for sr_path in sr_paths:
        gt_path = resolve_gt_path(sr_path, gt_index)
        if gt_path is None:
            missing.append(sr_path.name)
            continue

        sr_img = read_image(sr_path)
        gt_img = read_image(gt_path)

        if args.strict_size and sr_img.shape != gt_img.shape:
            raise ValueError(
                f'Size mismatch for {sr_path.name}: SR={sr_img.shape}, GT={gt_img.shape}'
            )

        if sr_img.shape != gt_img.shape:
            sr_img, gt_img = center_crop_to_common(sr_img, gt_img)

        sr_img = crop_border(sr_img, args.crop_border)
        gt_img = crop_border(gt_img, args.crop_border)

        psnr_val = calculate_psnr(sr_img, gt_img)
        ssim_val = calculate_ssim_rgb(sr_img, gt_img)

        rows.append({
            'image': sr_path.name,
            'gt_image': gt_path.name,
            'height': sr_img.shape[0],
            'width': sr_img.shape[1],
            'psnr_rgb': psnr_val,
            'ssim_rgb': ssim_val,
        })

    if not rows:
        raise RuntimeError('No valid SR-GT image pairs were found for evaluation.')

    mean_psnr = float(np.mean([r['psnr_rgb'] for r in rows]))
    mean_ssim = float(np.mean([r['ssim_rgb'] for r in rows]))

    print('==== Batch evaluation completed ====')
    print(f'Number of evaluated pairs : {len(rows)}')
    print(f'Mean PSNR (RGB)           : {mean_psnr:.4f} dB')
    print(f'Mean SSIM (RGB)           : {mean_ssim:.6f}')

    if missing:
        print(f'Unmatched SR files: {len(missing)}')
        for name in missing[:20]:
            print(f'  - {name}')
        if len(missing) > 20:
            print('  ...')

    if args.csv_path:
        os.makedirs(os.path.dirname(args.csv_path) or '.', exist_ok=True)
        with open(args.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['image', 'gt_image', 'height', 'width', 'psnr_rgb', 'ssim_rgb']
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f'CSV saved to: {args.csv_path}')


if __name__ == '__main__':
    main()
