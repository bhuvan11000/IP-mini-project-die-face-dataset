"""Batch preprocessing for the die-face dataset (set1, Mini Project 1).

Pipeline (per image):
  1. Load full-resolution photo with cv2.imread (applies EXIF orientation).
  2. Downscale a working copy to width 1024 (INTER_AREA), keep scale factor.
  3. Gaussian blur the working copy, kernel (7, 7).
  4. Segment by colour distance from background in Lab space:
     background = per-channel median (die is tiny, so median ~= background);
     distance map = per-pixel Euclidean norm of (Lab - background),
     normalized to 0-255 uint8, then Otsu-thresholded. This typically lights
     up only the die outline + pips, not the die interior.
  5. Clean the mask: CLOSE (elliptical 9x9) to bridge outline gaps, fill all
     external contours, then OPEN (same kernel) to remove specks.
  6. Largest connected component (excluding background) = the die; take bbox.
  7. Square crop box centred on bbox centre, side = max(w, h) * (1 + 2*margin),
     clamped to working-image bounds, mapped back to full resolution, cropped
     from the FULL-RESOLUTION image (detect-before-resize so pips survive).
  8. Resize crop to NxN (INTER_AREA), grayscale, bilateral filter
     (d=7, sigmaColor=40, sigmaSpace=7), linear contrast stretch via
     cv2.normalize MINMAX. No hist-equalization / CLAHE.
  9. Save lossless single-channel 8-bit PNG + metadata row + contact sheet.

Classical image processing only (OpenCV + NumPy), no deep learning.
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# All paths resolve relative to the script location, so this works from any CWD.
SET_DIR = Path(__file__).resolve().parent.parent
SOURCE_LABEL = "set1 (bhuvan), phone camera"
WORK_WIDTH = 1024
BLUR_KERNEL = (7, 7)
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def detect_die(work_bgr):
    """Detect the die in the working-size BGR image.

    Returns (bbox, mask, debug_dict) where bbox = (x, y, w, h) in
    working-image coordinates (or None if nothing found), mask is the
    cleaned binary mask of the die (uint8 0/255), and debug_dict holds
    intermediate images for --debug visualization.
    """
    debug = {}
    # 3. Working blur to suppress sensor noise / surface texture.
    blurred = cv2.GaussianBlur(work_bgr, BLUR_KERNEL, 0)
    debug["blur"] = blurred
    # 4. Colour distance from background in Lab. Median ~= pure background
    #    because the die covers only a few percent of the frame.
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_colour = np.median(lab.reshape(-1, 3), axis=0)  # shape (3,)
    dist = np.linalg.norm(lab - bg_colour, axis=2)  # float32 HxW
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
    dist_u8 = dist_norm.astype(np.uint8)
    debug["distance"] = dist_u8
    # Otsu: die body is close to background colour, so this typically keeps
    # only the die outline + pips.
    _, otsu_mask = cv2.threshold(
        dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    debug["otsu"] = otsu_mask
    # 5a. CLOSE bridges gaps in the thin die outline.
    closed = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    # 5b. Fill every external contour -> enclosed die body becomes solid.
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    # 5c. OPEN removes isolated specks.
    cleaned = cv2.morphologyEx(filled, cv2.MORPH_OPEN, MORPH_KERNEL)
    debug["cleaned"] = cleaned
    # 6. Largest connected component (skip background label 0) = the die.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num <= 1:  # only background found
        return None, cleaned, debug
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    w = int(stats[best, cv2.CC_STAT_WIDTH])
    h = int(stats[best, cv2.CC_STAT_HEIGHT])
    mask = (labels == best).astype(np.uint8) * 255
    mask_area = int(stats[best, cv2.CC_STAT_AREA])
    return (x, y, w, h, mask_area), mask, debug


def crop_and_normalize(full_bgr, work_bbox, work_shape, scale, size, margin):
    """Map the working-coords bbox to a square full-res crop and normalize it.

    Returns (gray_224, fullres_bbox) where gray_224 is the final single-channel
    8-bit image and fullres_bbox = (x, y, w, h) of the square crop in
    full-resolution coordinates.
    """
    x, y, w, h = work_bbox
    work_h, work_w = work_shape[:2]
    # 7. Square box centred on bbox centre, padded by margin on every side.
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1 + 2 * margin)
    half = side / 2.0
    x1 = int(round(max(0, cx - half)))
    y1 = int(round(max(0, cy - half)))
    x2 = int(round(min(work_w, cx + half)))
    y2 = int(round(min(work_h, cy + half)))
    # Map back to full resolution by dividing by the scale factor.
    full_h, full_w = full_bgr.shape[:2]
    fx1 = int(round(max(0, x1 / scale)))
    fy1 = int(round(max(0, y1 / scale)))
    fx2 = int(round(min(full_w, x2 / scale)))
    fy2 = int(round(min(full_h, y2 / scale)))
    fullres_bbox = (fx1, fy1, fx2 - fx1, fy2 - fy1)
    crop = full_bgr[fy1:fy2, fx1:fx2]
    # 8. Resize first (INTER_AREA is best for downsampling), then gray,
    #    edge-preserving smoothing, linear contrast stretch.
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=7)
    out = cv2.normalize(smooth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return out, fullres_bbox


def process_image(path, output_dir, size, margin, debug_dir=None):
    """Process one image. Returns (metadata_dict, final_or_None, debug_fig_path)."""
    stem = path.stem
    category = stem.split("_")[0]  # face number before the underscore
    row = {
        "filename": path.name,
        "source": SOURCE_LABEL,
        "category": category,
        "original_width": "",
        "original_height": "",
        "original_format": path.suffix.lower(),
        "processed_width": size,
        "processed_height": size,
        "processed_format": ".png",
        "bbox_x": "",
        "bbox_y": "",
        "bbox_w": "",
        "bbox_h": "",
        "mask_area_fraction": "",
        "qc_flag": "",
    }
    # 1. Full-resolution load (cv2 applies EXIF orientation).
    full = cv2.imread(str(path))
    if full is None:
        row["qc_flag"] = "load_failed"
        print(f"[FAIL] {path.name}: could not load image")
        return row, None, None
    full_h, full_w = full.shape[:2]
    row["original_width"] = full_w
    row["original_height"] = full_h
    # 2. Working copy at width 1024, keep scale factor.
    scale = WORK_WIDTH / full_w
    work_w = WORK_WIDTH
    work_h = int(round(full_h * scale))
    work = cv2.resize(full, (work_w, work_h), interpolation=cv2.INTER_AREA)

    result = detect_die(work)
    bbox_work, mask, debug = result
    if bbox_work is None:
        row["qc_flag"] = "no_object_found"
        print(f"[FAIL] {path.name}: no object found")
        return row, None, None
    x, y, w, h, mask_area = bbox_work
    mask_area_fraction = mask_area / float(work_w * work_h)
    row["mask_area_fraction"] = round(mask_area_fraction, 6)

    final, fullres_bbox = crop_and_normalize(
        full, (x, y, w, h), work.shape, scale, size, margin
    )
    row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"] = fullres_bbox

    # QC checks (accumulate flags, don't crash).
    flags = []
    if not (0.001 <= mask_area_fraction <= 0.02):
        flags.append(f"mask_area_out_of_range({mask_area_fraction:.4f})")
    # Bbox touches the (working-image) border -> die may be cut off.
    if x <= 0 or y <= 0 or x + w >= work_w or y + h >= work_h:
        flags.append("bbox_touches_border")
    row["qc_flag"] = ";".join(flags) if flags else "ok"

    out_path = output_dir / f"{stem}.png"
    cv2.imwrite(str(out_path), final)  # single-channel 8-bit -> lossless gray PNG

    debug_saved = None
    if debug_dir is not None:
        debug_saved = save_debug_figure(
            debug_dir, stem, full, work, debug, (x, y, w, h), final
        )
    status = "ok" if row["qc_flag"] == "ok" else row["qc_flag"]
    print(f"[{status}] {path.name} -> {out_path.name} (frac={mask_area_fraction:.4f})")
    return row, final, debug_saved


def save_debug_figure(debug_dir, stem, full, work, debug, bbox, final):
    """Save a 2x4 step-by-step visualization for one image. Returns its path."""
    x, y, w, h = bbox
    overlay = work.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
    panels = [
        ("original (full-res)", cv2.cvtColor(full, cv2.COLOR_BGR2RGB)),
        ("working blur", cv2.cvtColor(debug["blur"], cv2.COLOR_BGR2RGB)),
        ("distance map", debug["distance"]),
        ("Otsu mask", debug["otsu"]),
        ("cleaned mask", debug["cleaned"]),
        ("detected box", cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)),
        ("final crop", final),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(f"debug: {stem}")
    for ax, (title, img) in zip(axes.flat, panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    axes.flat[len(panels)].axis("off")
    fig.tight_layout()
    out = debug_dir / f"{stem}_debug.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def make_contact_sheet(images_in_order, output_path, cols=10):
    """Save a labeled grid of all processed images. Returns output_path."""
    import math

    n = len(images_in_order)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < n:
            name, img = images_in_order[i]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(name, fontsize=7)
    fig.suptitle(f"contact sheet: {n} processed die images (ordered by face, index)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess die-face photos (set1).")
    p.add_argument("--size", type=int, default=224, help="output square size")
    p.add_argument("--margin", type=float, default=0.5, help="crop margin fraction")
    p.add_argument("--input", default="unprocessed", help="input dir name under set dir")
    p.add_argument("--output", default="processed", help="output dir name under set dir")
    p.add_argument(
        "--debug",
        action="store_true",
        help="also save per-image step figures to scripts/debug/",
    )
    return p.parse_args()


COLUMNS = [
    "filename",
    "source",
    "category",
    "original_width",
    "original_height",
    "original_format",
    "processed_width",
    "processed_height",
    "processed_format",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "mask_area_fraction",
    "qc_flag",
]


def main():
    args = parse_args()
    input_dir = SET_DIR / args.input
    output_dir = SET_DIR / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = None
    if args.debug:
        debug_dir = Path(__file__).resolve().parent / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Ordered by face then index: filenames are {face}_{index}.jpg.
    paths = sorted(
        [p for p in input_dir.glob("*.jpg")],
        key=lambda p: (
            int(p.stem.split("_")[0]),
            int(p.stem.split("_")[1]),
        ),
    )
    print(f"Found {len(paths)} images in {input_dir}")

    rows, sheet_items, n_flagged = [], [], 0
    for path in paths:
        row, final, _ = process_image(path, output_dir, args.size, args.margin, debug_dir)
        rows.append(row)
        if final is not None:
            sheet_items.append((path.stem, final))
        if row["qc_flag"] not in ("ok", ""):
            n_flagged += 1

    # Safe re-run: overwrite metadata.csv.
    with open(SET_DIR / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Contact sheet (leading underscore so it sorts apart from data images).
    sheet_path = output_dir / "_contact_sheet.png"
    if sheet_items:
        make_contact_sheet(sheet_items, sheet_path)

    n_done = len(sheet_items)
    print(f"\nSummary: processed {n_done}/{len(paths)} images, flagged {n_flagged}.")
    print(f"Outputs: {output_dir} (*.png), {SET_DIR / 'metadata.csv'}, {sheet_path}")


if __name__ == "__main__":
    main()
