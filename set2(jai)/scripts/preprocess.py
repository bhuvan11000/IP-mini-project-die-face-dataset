"""Batch preprocessing for the die-face dataset (set2, Mini Project 1).

Adapted from set1(bhuvan)/scripts/preprocess.py. SAME pipeline, SAME output
spec (224x224, single-channel 8-bit PNG, background level 200); only the
parameters that depend on the input images differ (see "Differences from set1"
in the README): inputs are 128x128 PNGs with a LARGE die (~30-45% of the
frame), not 4080x3072 JPG phone photos with a tiny die.

Pipeline (per image):
  1. Load the image with cv2.imread.
  2. Working copy = the image itself (scale 1.0; set1 downscaled to width 1024,
     here the inputs are already small so no resize is done).
  3. Gaussian blur the working copy, kernel (3, 3) (inputs are small).
  4. Segment by colour distance from background in Lab space:
     background = per-channel median of the OUTER BORDER STRIP (the die is large
     here, so the whole-image median is not safe);
     distance map = per-pixel Euclidean norm of (Lab - background),
     normalized to 0-255 uint8, then Otsu-thresholded. This typically lights
     up only the die outline + pips, not the die interior.
  5. Clean the mask: CLOSE (elliptical 9x9) to bridge outline gaps, fill all
     external contours, then OPEN (same kernel) to remove specks.
  6. Largest connected component (excluding background) = the die; take bbox.
  7. Square crop box centred on bbox centre, side = max(w, h) * (1 + 2*margin),
     margin = 0.15. The image is first padded with a constant background colour (median
     of the border strip) so the crop is always exactly square.
  8. Resize crop to NxN (INTER_CUBIC, since crops are smaller than 224), grayscale, bilateral filter
     (d=7, sigmaColor=40, sigmaSpace=7), then background-level
     normalization: map the border-strip median to 200 (piecewise-linear, no clipping).
     No hist-equalization / CLAHE.
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
SOURCE_LABEL = "set2 (jai), synthetic (rendered by generator script)"
BLUR_KERNEL = (3, 3)
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
MASK_FRAC_RANGE = (0.10, 0.60)  # die is large here (set1: 0.001-0.02)
BORDER_FRAC = 0.08  # border strip used to estimate background colour


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
    # 4. Colour distance from background in Lab. Background colour is taken
    #    from the border strip because the die covers a large part of the frame.
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    bh, bw = lab.shape[:2]
    bm = max(2, int(round(BORDER_FRAC * min(bh, bw))))
    strip = np.concatenate(
        [
            lab[:bm].reshape(-1, 3),
            lab[-bm:].reshape(-1, 3),
            lab[:, :bm].reshape(-1, 3),
            lab[:, -bm:].reshape(-1, 3),
        ]
    )
    bg_colour = np.median(strip, axis=0)  # shape (3,)
    dist = np.linalg.norm(lab - bg_colour, axis=2)  # float32 HxW
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
    dist_u8 = dist_norm.astype(np.uint8)
    debug["distance"] = dist_u8
    # Otsu (set1 behaviour): when the die body is close to the background colour
    # this keeps only outline + pips. In set2 the die body is often a strong
    # colour, so a single Otsu split isolates just the bright pips. Fix:
    # (a) first Otsu gives threshold t1 (pips vs rest); (b) a SECOND Otsu on the
    # pixels below t1 splits background vs die body (a 3-class "multi-Otsu").
    # If the 3-class mask has an implausible area, fall back to the set1 mask.
    t1, otsu_mask = cv2.threshold(
        dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    lower = dist_u8[dist_u8 <= t1].reshape(-1, 1)
    if lower.size > 0:
        t2, _ = cv2.threshold(lower, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        multi = ((dist_u8 > t2).astype(np.uint8)) * 255
        frac = multi.mean() / 255.0
        if 0.05 <= frac <= 0.70:
            otsu_mask = multi
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

    Returns (gray_out, fullres_bbox, bg_before, bg_after) where gray_out is
    the final single-channel 8-bit image, fullres_bbox = (x, y, w, h) of the
    square crop in full-resolution coordinates, and bg_before / bg_after are
    the border-strip median gray levels before / after normalization.
    """
    x, y, w, h = work_bbox
    # 7. Square box centred on bbox centre, padded by margin on every side.
    #    The image is padded (constant background colour) so the box is never clamped.
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1 + 2 * margin)
    half = side / 2.0
    pad = int(np.ceil(side))
    # Constant pad = median colour of the outer border strip (replicating edge
    # pixels would smear a die corner that touches the frame edge).
    m = max(2, int(round(BORDER_FRAC * min(full_bgr.shape[:2]))))
    strip = np.concatenate(
        [
            full_bgr[:m].reshape(-1, 3),
            full_bgr[-m:].reshape(-1, 3),
            full_bgr[:, :m].reshape(-1, 3),
            full_bgr[:, -m:].reshape(-1, 3),
        ]
    )
    pad_colour = tuple(int(v) for v in np.median(strip, axis=0))
    padded = cv2.copyMakeBorder(
        full_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=pad_colour
    )
    x1 = int(round(cx - half)) + pad
    y1 = int(round(cy - half)) + pad
    s_px = int(round(side))
    fullres_bbox = (x1 - pad, y1 - pad, s_px, s_px)  # may extend past the image
    crop = padded[y1 : y1 + s_px, x1 : x1 + s_px]
    # 8. Resize first (INTER_AREA is best for downsampling), then gray,
    #    edge-preserving smoothing, then background-level normalization:
    #    scale intensities so the border strip (outer 12% on all sides,
    #    i.e. background only) has median 200 in every output image.
    interp = cv2.INTER_CUBIC if crop.shape[0] < size else cv2.INTER_AREA
    resized = cv2.resize(crop, (size, size), interpolation=interp)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=7)
    b = int(round(0.12 * size))  # ~27 px at size 224
    border = np.zeros_like(smooth, dtype=bool)
    border[:b, :] = True
    border[-b:, :] = True
    border[:, :b] = True
    border[:, -b:] = True
    bg_before = float(np.median(smooth[border]))
    # Piecewise-linear level map with bg_before -> 200 (set1 used a single linear
    # scale, out = img * 200/bg, which CLIPS at 255 when bg is dark; set2
    # backgrounds range from very dark to very bright, so that destroyed pips on
    # bright dice). Below bg this is identical to set1's formula; above bg the
    # range [bg, 255] is compressed into [200, 255] instead of being clipped.
    sm = smooth.astype(np.float32)
    lo = sm * (200.0 / bg_before)
    hi = 200.0 + (sm - bg_before) * (55.0 / max(255.0 - bg_before, 1.0))
    out = np.clip(np.where(sm <= bg_before, lo, hi), 0, 255).astype(np.uint8)
    bg_after = float(np.median(out[border]))
    return out, fullres_bbox, round(bg_before, 2), round(bg_after, 2)


def process_image(path, output_dir, size, margin, debug_dir=None):
    """Process one image. Returns (metadata_dict, final_or_None, debug_fig_path)."""
    stem = path.stem
    category = stem.split("_")[0].replace("die", "")  # "die3_0042" -> "3"
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
        "bg_level_before": "",
        "bg_level_after": "",
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
    # 2. Working copy = the image itself (already small), scale factor 1.0.
    scale = 1.0
    work = full
    work_h, work_w = full_h, full_w

    result = detect_die(work)
    bbox_work, mask, debug = result
    if bbox_work is None:
        row["qc_flag"] = "no_object_found"
        print(f"[FAIL] {path.name}: no object found")
        return row, None, None
    x, y, w, h, mask_area = bbox_work
    mask_area_fraction = mask_area / float(work_w * work_h)
    row["mask_area_fraction"] = round(mask_area_fraction, 6)

    final, fullres_bbox, bg_before, bg_after = crop_and_normalize(
        full, (x, y, w, h), work.shape, scale, size, margin
    )
    row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"] = fullres_bbox
    row["bg_level_before"] = bg_before
    row["bg_level_after"] = bg_after

    # QC checks (accumulate flags, don't crash).
    flags = []
    if not (MASK_FRAC_RANGE[0] <= mask_area_fraction <= MASK_FRAC_RANGE[1]):
        flags.append(f"mask_area_out_of_range({mask_area_fraction:.4f})")
    # Bbox touches the image border -> die may be cut off.
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
    p = argparse.ArgumentParser(description="Preprocess die-face images (set2).")
    p.add_argument("--size", type=int, default=224, help="output square size")
    p.add_argument("--margin", type=float, default=0.15, help="crop margin fraction")
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
    "bg_level_before",
    "bg_level_after",
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

    # Ordered by face then index: filenames are die{face}_{index}.png.
    paths = sorted(
        [p for p in input_dir.glob("*.png")],
        key=lambda p: (
            int(p.stem.split("_")[0].replace("die", "")),
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
