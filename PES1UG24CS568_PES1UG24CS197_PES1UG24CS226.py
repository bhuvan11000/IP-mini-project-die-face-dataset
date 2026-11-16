#!/usr/bin/env python3
"""
Mini Project 2 - Die Face Classification (combined Part 1 + Part 2)
===================================================================

Part 1 - Classical features + SVM
    1a. Harris corner feature vector (response statistics, strongest corner
        coordinates and corner intensity histogram) -> RBF SVM
    1b. Rotation-invariant Harris descriptor (pip-like corner clusters and
        pairwise distance histograms) -> RBF SVM
    1c. SIFT descriptors -> K-Means visual vocabulary -> bag-of-visual-words
        histogram -> RBF SVM
    1d. Harris keypoints + SIFT descriptors + BoVW -> RBF SVM
    1e. Cosine-similarity nearest-neighbour baseline on the SIFT BoVW vectors
    Plus the Part 1 analysis: cross-source generalisation (set1 <-> set2) and
    a 5-fold cross-validation comparison.

Part 2 - Deep learning
    2a. ResNet (torchvision, ImageNet pre-trained) fine-tuned on the die faces
    2b. MViTv2 (timm, ImageNet pre-trained) fine-tuned on the die faces
    Use --freeze_backbone to train only the final classification layer instead
    of fine-tuning the whole network (the "train last layer" variant).

The dataset is pulled automatically from GitHub:
    https://github.com/bhuvan11000/IP-mini-project-die-face-dataset
By default the repository archive is downloaded once (about 62 MB) and only
the two `processed` folders and their metadata CSV files are extracted and
cached (about 6 MB).  `--download_mode api` fetches only those files one by one
instead, which uses less bandwidth but makes many more requests.  A shallow
git clone is the last fallback.  The labels come from each set's metadata.csv,
so they always match the preprocessing that produced the images.

Usage
-----
    python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py
    python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py --skip_dl
    python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py --skip_classical
    python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py --epochs 20
    python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py --models resnet --epochs 5
    python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py --seed 123
    python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py --data_root /path/to/local/clone
    python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py --refresh

Requirements
------------
    pip install numpy pandas opencv-python scikit-learn matplotlib
    pip install torch torchvision timm            # only for Part 2

Notes
-----
* All models use the same stratified train/validation/test split, so the
  holdout numbers are directly comparable.
* The test split is never used for training, vocabulary fitting or epoch
  selection.
* A pooled random split (both sources in train and test) is optimistic.  The
  cross-source section (train on set1, test on set2 and vice versa) is the
  honest domain-shift estimate.  Use --seed to check split stability.
* Figures and results.json are written to --out_dir (default: results).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # save figures to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 6
FACE_NAMES = [f"face {i}" for i in range(1, NUM_CLASSES + 1)]
IMG_SIZE = 224
SEED = 42

GITHUB_OWNER = "bhuvan11000"
GITHUB_REPO = "IP-mini-project-die-face-dataset"
GITHUB_BRANCH = "main"
REPO_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}.git"

DATASET_SPECS = {
    "set1": {
        "source": "bhuvan",
        "dir": "set1(bhuvan)",
        "processed": "set1(bhuvan)/processed",
        "metadata": "set1(bhuvan)/metadata.csv",
    },
    "set2": {
        "source": "jai",
        "dir": "set2(jai)",
        "processed": "set2(jai)/processed",
        "metadata": "set2(jai)/metadata.csv",
    },
}


# ===========================================================================
# 1. Dataset download / loading
# ===========================================================================
def _http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "die-face-mini-project"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_download(url: str, dest: Path, timeout: int = 60) -> None:
    """Download one file atomically (write .part, then rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    data = _http_get(url, timeout=timeout)
    if not data:
        raise RuntimeError(f"empty response for {url}")
    tmp.write_bytes(data)
    tmp.replace(dest)


def _download_one(raw_base: str, rel_path: str, cache_dir: Path,
                  retries: int = 3) -> bool:
    dest = cache_dir / rel_path
    if dest.exists() and dest.stat().st_size > 0:
        return False
    last_error = None
    for attempt in range(retries):
        try:
            _http_download(raw_base + urllib.parse.quote(rel_path), dest)
            return True
        except Exception as exc:  # network hiccup / transient 5xx
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"could not download {rel_path}: {last_error}")


def _wanted_paths(paths):
    """Keep only the PNG images and metadata.csv files we actually need."""
    wanted = []
    for path in paths:
        if path in {s["metadata"] for s in DATASET_SPECS.values()}:
            wanted.append(path)
            continue
        for spec in DATASET_SPECS.values():
            prefix = spec["processed"] + "/"
            if path.startswith(prefix) and path.lower().endswith(".png"):
                name = path.rsplit("/", 1)[-1]
                if not name.startswith("_"):
                    wanted.append(path)
                break
    return wanted


def _dataset_complete(root: Path) -> bool:
    """True when metadata and every processed image listed in it exist."""
    for spec in DATASET_SPECS.values():
        meta_path = root / spec["metadata"]
        pdir = root / spec["processed"]
        if not meta_path.exists() or not pdir.is_dir():
            return False
        try:
            meta = pd.read_csv(meta_path)
        except Exception:
            return False
        if "filename" not in meta.columns:
            return False
        for filename in meta["filename"]:
            if not (pdir / f"{Path(str(filename)).stem}.png").exists():
                return False
    return True


def _fetch_file_list(branch: str):
    api = (
        f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
        f"/git/trees/{branch}?recursive=1"
    )
    data = json.loads(_http_get(api))
    if data.get("truncated"):
        raise RuntimeError("GitHub tree listing was truncated")
    return [item["path"] for item in data.get("tree", []) if item.get("type") == "blob"]


def _download_files(cache_dir: Path, branch: str, workers: int = 16) -> int:
    """Fast path: list the repo via the API and fetch the wanted files in parallel."""
    paths = _wanted_paths(_fetch_file_list(branch))
    if not paths:
        raise RuntimeError("no processed images found in the repository tree")

    raw_base = (
        f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/"
        f"{branch}/"
    )
    downloaded = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, raw_base, path, cache_dir): path
            for path in paths
        }
        for i, future in enumerate(as_completed(futures), 1):
            try:
                if future.result():
                    downloaded += 1
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
            if i % 25 == 0 or i == len(paths):
                print(f"  checked {i}/{len(paths)} files", flush=True)
    print(f"  downloaded {downloaded} new files, {len(paths) - downloaded} cached")
    return len(paths)


def _download_zip(cache_dir: Path, branch: str) -> int:
    """Download the repository archive and extract only the wanted files."""
    url = (
        f"https://codeload.github.com/{GITHUB_OWNER}/{GITHUB_REPO}"
        f"/zip/refs/heads/{branch}"
    )
    print("  downloading repository archive (about 70 MB) ...", flush=True)
    archive = cache_dir / "_repo_archive.zip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "die-face-mini-project"}),
        timeout=300,
    ) as resp, open(archive, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)

    wanted = {}
    n = 0
    try:
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                path = member.split("/", 1)[-1]  # drop '<repo>-<branch>/'
                if path in {s["metadata"] for s in DATASET_SPECS.values()}:
                    wanted[path] = member
                    continue
                for spec in DATASET_SPECS.values():
                    prefix = spec["processed"] + "/"
                    if path.startswith(prefix) and path.lower().endswith(".png"):
                        name = path.rsplit("/", 1)[-1]
                        if not name.startswith("_"):
                            wanted[path] = member
                        break
            for path, member in wanted.items():
                dest = cache_dir / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(member))
                n += 1
    finally:
        archive.unlink(missing_ok=True)
    if n == 0:
        raise RuntimeError("archive did not contain the processed folders")
    print(f"  extracted {n} files from the archive")
    return n


def _git_clone(cache_dir: Path, branch: str) -> Path:
    """Last-resort fallback: shallow clone the public repository."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "_shallow_clone"
    if target.exists():
        shutil.rmtree(target)
    print(f"  cloning {REPO_URL} (shallow) ...", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, REPO_URL, str(target)],
        check=True,
    )
    return target


def resolve_data_root(data_root, cache_dir, branch=GITHUB_BRANCH,
                      refresh=False, download_mode="archive") -> Path:
    """
    Return a folder containing `set1(bhuvan)/processed` and
    `set2(jai)/processed`.  If --data_root is given it is used as-is (offline
    mode); otherwise the processed files and metadata are downloaded from
    GitHub into `cache_dir` and reused on later runs.
    """
    if data_root:
        root = Path(data_root).expanduser().resolve()
        if not root.is_dir():
            sys.exit(f"ERROR: data_root '{data_root}' is not a folder.")
        if not _dataset_complete(root):
            print(f"  warning: {root} does not look like a complete clone; "
                  f"continuing anyway")
        return root

    cache_dir = Path(cache_dir).expanduser()
    if not refresh and _dataset_complete(cache_dir):
        print(f"Using cached dataset: {cache_dir.resolve()}")
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Fetching dataset from github.com/{GITHUB_OWNER}/{GITHUB_REPO} "
        f"(branch '{branch}') -> {cache_dir}",
        flush=True,
    )
    if download_mode == "api":
        attempts = [
            ("file-by-file API", lambda: _download_files(cache_dir, branch)),
            ("repository archive", lambda: _download_zip(cache_dir, branch)),
        ]
    else:
        attempts = [
            ("repository archive", lambda: _download_zip(cache_dir, branch)),
            ("file-by-file API", lambda: _download_files(cache_dir, branch)),
        ]

    for label, download in attempts:
        try:
            download()
            break
        except Exception as exc:
            print(f"  {label} download failed ({exc}); trying the next option")

    if not _dataset_complete(cache_dir):
        print("  trying a shallow git clone")
        try:
            clone = _git_clone(cache_dir, branch)
            if _dataset_complete(clone):
                print(f"  dataset ready in {clone.resolve()}")
                return clone
            raise RuntimeError("clone does not contain the processed folders")
        except Exception as exc:
            sys.exit(
                "ERROR: could not download the dataset "
                f"({exc}).\nCheck your internet connection, or clone "
                "the repository manually and pass --data_root <clone folder>."
            )

    print(f"  dataset ready in {cache_dir.resolve()}")
    return cache_dir


def parse_label(filename: str) -> int:
    """Fallback label parser: '3_0.png' -> 3, 'die5_0042.png' -> 5."""
    match = re.match(r"^(?:die)?([1-6])_", filename)
    if match is None:
        raise ValueError(f"Cannot parse face label from '{filename}'")
    return int(match.group(1))


def load_dataset(root: Path):
    """
    Load the two processed sets.

    Returns
        images  : (N, 224, 224) uint8 grayscale
        labels  : (N,) int face values 1..6
        sources : (N,) str 'bhuvan' or 'jai'
        names   : list of '<source>/<file>' strings
    Labels are read from metadata.csv when it is available; filename parsing is
    only a fallback.  Files starting with '_' (contact sheets) are skipped.
    """
    images, labels, sources, names = [], [], [], []

    for spec in DATASET_SPECS.values():
        set_dir = root / spec["dir"]
        pdir = set_dir / "processed"
        meta_path = set_dir / "metadata.csv"

        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            rows = [
                (str(row.filename), int(row.category))
                for row in meta.itertuples(index=False)
            ]
        else:
            rows = [
                (f.name, parse_label(f.name))
                for f in sorted(pdir.glob("*.png"))
                if not f.name.startswith("_")
            ]

        count = 0
        for filename, label in rows:
            image_path = pdir / f"{Path(filename).stem}.png"
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"  warning: could not read {image_path}")
                continue
            if gray.shape != (IMG_SIZE, IMG_SIZE):
                gray = cv2.resize(
                    gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
                )
            images.append(gray)
            labels.append(label)
            sources.append(spec["source"])
            names.append(f"{spec['source']}/{image_path.name}")
            count += 1

        print(f"Loaded {count:3d} images from {pdir}")

    if not images:
        sys.exit("ERROR: no images were loaded.")

    images = np.stack(images)
    labels = np.array(labels, dtype=int)
    sources = np.array(sources)

    print(f"Total: {len(images)} images, each {images.shape[1]}x{images.shape[2]} "
          f"(uint8)")
    print("Images per class and source:")
    print(pd.crosstab(pd.Series(sources, name="set"),
                      pd.Series(labels, name="face")))
    if sorted(np.unique(labels)) != list(range(1, NUM_CLASSES + 1)):
        sys.exit(f"ERROR: expected classes 1..{NUM_CLASSES}, found "
                 f"{sorted(np.unique(labels))}")
    return images, labels, sources, names


def make_splits(labels, sources, test_size=0.25, val_size=0.15):
    """
    Stratified on (source, face), so every split has all six faces and both
    domains.  The classical models train on train+val; the deep models use
    train and pick the best epoch on val.  The test set is shared by all.
    """
    strat = np.array([f"{s}_{l}" for s, l in zip(sources, labels)])
    idx = np.arange(len(labels))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, random_state=SEED, stratify=strat
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size,
        random_state=SEED,
        stratify=strat[idx_trainval],
    )
    print(f"Split -> train {len(idx_train)}, val {len(idx_val)}, "
          f"test {len(idx_test)}")
    return idx_train, idx_val, idx_test


# ===========================================================================
# 2. Reporting helpers
# ===========================================================================
def report(name, y_true, y_pred, sources_test, out_dir: Path):
    """Print metrics, save a confusion matrix, and return a result dict."""
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 62)
    print(name)
    print("=" * 62)
    print(f"Test accuracy: {acc * 100:.2f}%  ({len(y_true)} test images)")
    if acc == 1.0 and len(y_true) < 100:
        print("  note: perfect score on a small pooled test set; check --seed "
              "and the cross-source results before generalising")

    per_set = {}
    for source in sorted(np.unique(sources_test)):
        mask = sources_test == source
        per_set[source] = accuracy_score(y_true[mask], y_pred[mask])
        print(f"  {source:7s} only: {per_set[source] * 100:6.2f}% "
              f"({int(mask.sum())} images)")

    print(classification_report(
        y_true, y_pred, labels=range(1, NUM_CLASSES + 1),
        target_names=FACE_NAMES, zero_division=0,
    ))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=range(1, NUM_CLASSES + 1),
        display_labels=FACE_NAMES, cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(f"{name}\naccuracy {acc * 100:.1f}%")
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    fname = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    fig.savefig(out_dir / f"cm_{fname}.png", dpi=130)
    plt.close(fig)
    return {
        "model": name,
        "test_accuracy": float(acc),
        "per_source_accuracy": {k: float(v) for k, v in per_set.items()},
    }


# ===========================================================================
# 3. Part 1 - Harris corner features
# ===========================================================================
def _to_gray_uint8(img_array) -> np.ndarray:
    if img_array.max() <= 1.0:
        return (img_array * 255).astype(np.uint8)
    return img_array.astype(np.uint8)


def extract_harris_layout_features(img_array, block_size=2, ksize=3, k=0.04,
                                   num_top_corners=20):
    """
    The Part 1 Harris feature vector (52-D).

    The image is resized to 64x64, then:
      4  number of corners, mean / std / max Harris response
      40 coordinates of the 20 strongest corners
      8  intensity histogram at corner pixels
    This encodes the absolute corner layout, so it is sensitive to rotation.
    """
    gray = _to_gray_uint8(img_array)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    response = cv2.cornerHarris(np.float32(gray), block_size, ksize, k)

    max_response = float(response.max()) if response.size else 0.0
    if max_response > 0:
        mask = response > 0.01 * max_response
    else:
        mask = np.zeros_like(response, dtype=bool)

    corner_responses = response[mask]
    num_corners = int(mask.sum())
    mean_response = float(corner_responses.mean()) if num_corners else 0.0
    std_response = float(corner_responses.std()) if num_corners else 0.0

    corner_y, corner_x = np.where(mask)
    if num_corners:
        order = np.argsort(response[corner_y, corner_x])[::-1][:num_top_corners]
        top_x = corner_x[order]
        top_y = corner_y[order]
    else:
        top_x = np.array([], dtype=int)
        top_y = np.array([], dtype=int)
    if len(top_x) < num_top_corners:
        pad = num_top_corners - len(top_x)
        top_x = np.pad(top_x, (0, pad), mode="constant")
        top_y = np.pad(top_y, (0, pad), mode="constant")

    if num_corners:
        hist, _ = np.histogram(gray[mask], bins=8, range=(0, 256))
        hist = hist / (hist.sum() + 1e-7)
    else:
        hist = np.zeros(8, dtype=float)

    return np.hstack([
        num_corners, mean_response, std_response, max_response,
        top_x, top_y, hist,
    ]).astype(np.float32)


def harris_corners(gray, sigma=2.5, block_size=9, ksize=3, k=0.04,
                   rel_thresh=0.05, nms_size=15, max_corners=150):
    """
    Harris detector with Gaussian smoothing and non-maximum suppression.

    Returns (xs, ys, responses, response_map), strongest corners first.
    """
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigma)
    response = cv2.cornerHarris(np.float32(smoothed), block_size, ksize, k)
    if response.size == 0 or response.max() <= 0:
        return np.array([], int), np.array([], int), np.array([]), response

    local_max = response == cv2.dilate(
        response, np.ones((nms_size, nms_size), np.uint8)
    )
    mask = local_max & (response > rel_thresh * response.max())
    ys, xs = np.nonzero(mask)
    values = response[ys, xs]
    order = np.argsort(-values)[:max_corners]
    return xs[order], ys[order], values[order], response


def harris_rotation_features(gray, cluster_size=20):
    """
    Rotation-invariant Harris descriptor (33-D), from Part 2.

    A round pip produces a cluster of Harris corners, while the die border and
    noise produce isolated corners.  Corners are merged into clusters and the
    vector describes only distances (never absolute positions):
      5  number of corners / clusters / clusters with >=2, >=3, >=4 corners
      12 histogram of pairwise distances between pip-like clusters
      8  histogram of nearest-neighbour distances
      2  mean / std of the nearest-neighbour distance
      6  histogram of the number of corners per cluster
    """
    xs, ys, _, _ = harris_corners(gray)
    n = len(xs)

    if n:
        mask = np.zeros(gray.shape, np.uint8)
        mask[ys, xs] = 255
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cluster_size, cluster_size)),
        )
        n_labels, labels, _, centroids = cv2.connectedComponentsWithStats(mask)
        counts = np.bincount(labels[ys, xs], minlength=n_labels)[1:]
        centroids = centroids[1:]
    else:
        counts = np.array([], dtype=int)
        centroids = np.zeros((0, 2))

    feats = [
        n,
        len(counts),
        int((counts >= 2).sum()),
        int((counts >= 3).sum()),
        int((counts >= 4).sum()),
    ]

    pips = centroids[counts >= 2]
    m = len(pips)
    if m >= 2:
        diff = pips[:, None, :] - pips[None, :, :]
        distances = np.hypot(diff[..., 0], diff[..., 1])
        upper = np.triu_indices(m, 1)
        feats += list(
            np.histogram(distances[upper], bins=12, range=(0, 160))[0]
            / len(upper[0])
        )
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(1)
        feats += list(
            np.histogram(nearest, bins=8, range=(0, 80))[0] / m
        )
        feats += [nearest.mean(), nearest.std()]
    else:
        feats += [0] * 22

    feats += list(np.histogram(counts, bins=[1, 2, 3, 4, 6, 10, 100])[0])
    return np.array(feats, dtype=np.float32)


def visualise_harris(images, labels, out_dir: Path):
    """Save the Part 1 Harris response demo."""
    demo_idx = int(np.where(labels == 5)[0][2])
    demo_gray = cv2.resize(images[demo_idx], (64, 64), interpolation=cv2.INTER_AREA)
    demo_response = cv2.cornerHarris(np.float32(demo_gray), 2, 3, 0.04)
    demo_response = demo_response / (np.abs(demo_response).max() + 1e-12)
    demo_mask = demo_response > 0.01 * demo_response.max()
    corner_y, corner_x = np.where(demo_mask)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(demo_gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Input resized to 64x64")
    axes[1].imshow(demo_response, cmap="inferno")
    axes[1].set_title("Harris response")
    axes[2].imshow(demo_gray, cmap="gray", vmin=0, vmax=255)
    axes[2].scatter(corner_x, corner_y, s=10, c="#e31a1c", linewidths=0)
    axes[2].set_title(f"{len(corner_x)} corners above threshold")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "harris_response.png", dpi=130)
    plt.close(fig)


def visualise_keypoints(images, labels, sources, sift, out_dir: Path):
    """Save a Harris/SIFT detector comparison for face 5 from each source."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    for row, source in enumerate(("bhuvan", "jai")):
        idx = int(np.where((sources == source) & (labels == 5))[0][0])
        img = images[idx]
        xs, ys, _, _ = harris_corners(img)

        axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"{source}: face 5")
        axes[row, 1].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].scatter(xs, ys, s=18, c="red")
        axes[row, 1].set_title(f"Harris corners ({len(xs)})")

        keypoints = sift.detect(img, None)
        drawn = cv2.drawKeypoints(
            img, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        axes[row, 2].imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        axes[row, 2].set_title(f"SIFT keypoints ({len(keypoints)})")
        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_dir / "harris_sift_keypoints.png", dpi=130)
    plt.close(fig)


# ===========================================================================
# 4. Part 1 - SIFT + bag of visual words
# ===========================================================================
def sift_descriptors(gray, sift):
    """Standard SIFT: DoG keypoints + 128-D descriptors."""
    _, desc = sift.detectAndCompute(gray, None)
    return desc


def harris_sift_descriptors(gray, sift, size=32.0):
    """Hybrid: Harris corners as keypoints, SIFT descriptors at those points."""
    xs, ys, _, _ = harris_corners(gray)
    if len(xs) == 0:
        return None
    keypoints = [
        cv2.KeyPoint(float(x), float(y), size) for x, y in zip(xs, ys)
    ]
    _, desc = sift.compute(gray, keypoints)
    return desc


def fit_vocab(desc_list, k, seed=None):
    """Fit a visual vocabulary on descriptors from the training split only."""
    seed = SEED if seed is None else seed
    valid = [
        d.astype(np.float32) for d in desc_list
        if d is not None and len(d) > 0
    ]
    if not valid:
        raise RuntimeError("no descriptors available to fit the vocabulary")
    stacked = np.vstack(valid)
    k = min(k, len(stacked))
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(stacked)
    return kmeans


def bovw_histogram(desc, kmeans):
    """Variable-length descriptors -> L2 + power-normalised histogram."""
    k = kmeans.n_clusters
    if desc is None or len(desc) == 0:
        return np.zeros(k, dtype=np.float32)
    words = kmeans.predict(desc.astype(np.float32))
    hist = np.bincount(words, minlength=k).astype(np.float32)
    hist /= (np.linalg.norm(hist) + 1e-12)
    return np.sqrt(hist)


def bovw_histograms(desc_list, kmeans):
    return np.stack([bovw_histogram(d, kmeans) for d in desc_list])


def fit_svm(X_train, y_train, tune=True):
    """Standardise + RBF SVM; optionally tune C/gamma by 5-fold CV."""
    pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    if not tune:
        pipeline.set_params(svc__C=10.0, svc__gamma="scale")
        pipeline.fit(X_train, y_train)
        return pipeline

    grid = {
        "svc__C": [1, 10, 100],
        "svc__gamma": ["scale", 0.1, 0.01, 0.001],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipeline, grid, cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"  best SVM params: {search.best_params_}  "
          f"(CV acc {search.best_score_ * 100:.1f}%)")
    return search.best_estimator_


def cosine_similarity_predict(X_train, y_train, X_test):
    """
    Simple 1-nearest-neighbour baseline using cosine similarity.
    Useful for checking whether the SVM is doing more than matching histograms.
    """
    train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-12)
    test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-12)
    similarity = test @ train.T
    return y_train[np.argmax(similarity, axis=1)]


def run_classical(images, labels, sources, idx_trainval, idx_test,
                  out_dir: Path, args):
    """Run all Part 1 models and return (results, precomputed features)."""
    results = []
    y_trainval = labels[idx_trainval]
    y_test = labels[idx_test]
    sources_test = sources[idx_test]
    tune = not args.no_grid_search

    # ------------------------------------------------------------------
    # 1a. Harris layout descriptor (Part 1 notebook)
    # ------------------------------------------------------------------
    print("\n[Part 1a] Harris layout features (52-D) + SVM")
    t0 = time.time()
    X_harris_layout = np.stack([
        extract_harris_layout_features(img) for img in images
    ])
    print(f"  feature matrix {X_harris_layout.shape} "
          f"({time.time() - t0:.1f}s)")
    svm = fit_svm(X_harris_layout[idx_trainval], y_trainval, tune=tune)
    pred = svm.predict(X_harris_layout[idx_test])
    results.append(report(
        "Harris layout (52-D) + SVM", y_test, pred, sources_test, out_dir
    ))

    # ------------------------------------------------------------------
    # 1b. Rotation-invariant Harris descriptor (Part 2)
    # ------------------------------------------------------------------
    print("\n[Part 1b] Harris rotation-invariant features (33-D) + SVM")
    t0 = time.time()
    X_harris_rotation = np.stack([
        harris_rotation_features(img) for img in images
    ])
    print(f"  feature matrix {X_harris_rotation.shape} "
          f"({time.time() - t0:.1f}s)")
    svm = fit_svm(X_harris_rotation[idx_trainval], y_trainval, tune=tune)
    pred = svm.predict(X_harris_rotation[idx_test])
    results.append(report(
        "Harris rotation-invariant (33-D) + SVM", y_test, pred,
        sources_test, out_dir,
    ))

    # ------------------------------------------------------------------
    # 1c. SIFT + BoVW
    # ------------------------------------------------------------------
    print("\n[Part 1c] SIFT + Bag of Visual Words + SVM")
    sift = cv2.SIFT_create()
    desc_sift = [sift_descriptors(img, sift) for img in images]
    counts = [0 if d is None else len(d) for d in desc_sift]
    print(f"  SIFT keypoints/image: mean {np.mean(counts):.1f}, "
          f"min {min(counts)}, max {max(counts)}")
    vocab = fit_vocab([desc_sift[i] for i in idx_trainval], args.vocab_size)
    X_sift = bovw_histograms(desc_sift, vocab)
    print(f"  BoVW feature matrix {X_sift.shape} "
          f"({vocab.n_clusters} visual words)")
    svm = fit_svm(X_sift[idx_trainval], y_trainval, tune=tune)
    pred = svm.predict(X_sift[idx_test])
    results.append(report(
        "SIFT + BoVW + SVM", y_test, pred, sources_test, out_dir
    ))

    # Cosine-similarity baseline on the same SIFT BoVW vectors.
    pred_cos = cosine_similarity_predict(
        X_sift[idx_trainval], y_trainval, X_sift[idx_test]
    )
    results.append(report(
        "SIFT + BoVW + cosine 1-NN", y_test, pred_cos,
        sources_test, out_dir,
    ))

    # ------------------------------------------------------------------
    # 1d. Harris keypoints + SIFT descriptors + BoVW
    # ------------------------------------------------------------------
    print("\n[Part 1d] Harris keypoints + SIFT descriptors + BoVW + SVM")
    desc_harris_sift = [
        harris_sift_descriptors(img, sift) for img in images
    ]
    vocab_hs = fit_vocab(
        [desc_harris_sift[i] for i in idx_trainval], args.vocab_size
    )
    X_harris_sift = bovw_histograms(desc_harris_sift, vocab_hs)
    print(f"  BoVW feature matrix {X_harris_sift.shape} "
          f"({vocab_hs.n_clusters} visual words)")
    svm = fit_svm(X_harris_sift[idx_trainval], y_trainval, tune=tune)
    pred = svm.predict(X_harris_sift[idx_test])
    results.append(report(
        "Harris-SIFT hybrid + BoVW + SVM", y_test, pred,
        sources_test, out_dir,
    ))

    # ------------------------------------------------------------------
    # Detector figures
    # ------------------------------------------------------------------
    visualise_harris(images, labels, out_dir)
    visualise_keypoints(images, labels, sources, sift, out_dir)

    features = {
        "X_harris_layout": X_harris_layout,
        "X_harris_rotation": X_harris_rotation,
        "X_sift": X_sift,
        "X_harris_sift": X_harris_sift,
        "desc_sift": desc_sift,
        "desc_harris_sift": desc_harris_sift,
    }
    return results, features


# ===========================================================================
# 5. Part 1 - cross-source generalisation and 5-fold CV
# ===========================================================================
def evaluate_cross_set(desc_sift, labels, sources, out_dir: Path,
                       vocab_size: int):
    """
    Train on one entire source and test on the other (SIFT + BoVW).
    This measures the domain gap that a pooled random split hides.
    """
    print("\n[Part 1e] Cross-source generalisation (SIFT + BoVW)")
    results = []
    for train_source, test_source in (("jai", "bhuvan"), ("bhuvan", "jai")):
        train_idx = np.where(sources == train_source)[0]
        test_idx = np.where(sources == test_source)[0]

        vocab = fit_vocab([desc_sift[i] for i in train_idx], vocab_size)
        X_train = bovw_histograms([desc_sift[i] for i in train_idx], vocab)
        X_test = bovw_histograms([desc_sift[i] for i in test_idx], vocab)

        scaler = StandardScaler().fit(X_train)
        svm = SVC(kernel="rbf", C=10.0, gamma="scale")
        svm.fit(scaler.transform(X_train), labels[train_idx])
        pred = svm.predict(scaler.transform(X_test))
        acc = accuracy_score(labels[test_idx], pred)
        print(f"  train on {train_source:6s} ({len(train_idx):3d}) -> "
              f"test on {test_source:6s} ({len(test_idx):3d}): "
              f"{acc * 100:6.2f}%")

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ConfusionMatrixDisplay.from_predictions(
            labels[test_idx], pred, labels=range(1, NUM_CLASSES + 1),
            display_labels=FACE_NAMES, cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(
            f"SIFT cross-set: {train_source} -> {test_source}\n"
            f"accuracy {acc * 100:.1f}%"
        )
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.tight_layout()
        fig.savefig(out_dir / f"cm_cross_set_{train_source}_to_{test_source}.png",
                    dpi=130)
        plt.close(fig)

        results.append({
            "train_source": train_source,
            "test_source": test_source,
            "train_images": int(len(train_idx)),
            "test_images": int(len(test_idx)),
            "accuracy": float(acc),
        })

    print(f"  chance level: {100 / NUM_CLASSES:.1f}%")
    return results


def run_cross_validation(images, labels, features, out_dir: Path,
                         vocab_size: int):
    """
    5-fold stratified cross-validation of the classical pipelines.
    A fixed RBF SVM (C=10, gamma='scale') is used so the comparison is cheap
    and the folds are directly comparable.  The SIFT/hybrid vocabularies are
    re-fitted inside every fold (no test descriptors leak into the codebook).
    """
    print("\n[Part 1f] 5-fold cross-validation of the classical pipelines")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {
        "Harris layout (52-D)": [],
        "Harris rotation-invariant (33-D)": [],
        "SIFT + BoVW": [],
        "Harris-SIFT hybrid + BoVW": [],
    }

    def fixed_svm(X_train, y_train, X_test):
        scaler = StandardScaler().fit(X_train)
        svm = SVC(kernel="rbf", C=10.0, gamma="scale")
        svm.fit(scaler.transform(X_train), y_train)
        return svm.predict(scaler.transform(X_test))

    for fold, (fold_train, fold_test) in enumerate(
            skf.split(images, labels), 1):
        y_train, y_test = labels[fold_train], labels[fold_test]

        scores["Harris layout (52-D)"].append(accuracy_score(
            y_test,
            fixed_svm(features["X_harris_layout"][fold_train], y_train,
                      features["X_harris_layout"][fold_test]),
        ))
        scores["Harris rotation-invariant (33-D)"].append(accuracy_score(
            y_test,
            fixed_svm(features["X_harris_rotation"][fold_train], y_train,
                      features["X_harris_rotation"][fold_test]),
        ))

        vocab = fit_vocab(
            [features["desc_sift"][i] for i in fold_train], vocab_size
        )
        X_train = bovw_histograms(
            [features["desc_sift"][i] for i in fold_train], vocab
        )
        X_test = bovw_histograms(
            [features["desc_sift"][i] for i in fold_test], vocab
        )
        scores["SIFT + BoVW"].append(accuracy_score(
            y_test, fixed_svm(X_train, y_train, X_test)
        ))

        vocab_hs = fit_vocab(
            [features["desc_harris_sift"][i] for i in fold_train], vocab_size
        )
        X_train = bovw_histograms(
            [features["desc_harris_sift"][i] for i in fold_train], vocab_hs
        )
        X_test = bovw_histograms(
            [features["desc_harris_sift"][i] for i in fold_test], vocab_hs
        )
        scores["Harris-SIFT hybrid + BoVW"].append(accuracy_score(
            y_test, fixed_svm(X_train, y_train, X_test)
        ))
        print(f"  fold {fold}/5 done", flush=True)

    summary = {}
    print(f"\n  {'Method':<34}{'mean accuracy':>15}{'std':>9}")
    print("  " + "-" * 56)
    for name, values in scores.items():
        mean, std = float(np.mean(values)), float(np.std(values))
        summary[name] = {"mean": mean, "std": std, "folds": values}
        print(f"  {name:<34}{mean * 100:>14.2f}%{std * 100:>8.2f}%")
    return summary


# ===========================================================================
# 6. Part 2 - deep learning (ResNet, MViT)
# ===========================================================================
def _classifier_head(model):
    if hasattr(model, "fc"):
        return model.fc
    if hasattr(model, "head"):
        return model.head
    if hasattr(model, "get_classifier"):
        return model.get_classifier()
    return None


def run_deep(images, labels, sources, idx_train, idx_val, idx_test,
             args, out_dir: Path):
    """Fine-tune ResNet and MViTv2 on the same split as the classical models."""
    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms as T
    except ImportError as exc:
        print(f"\nSkipping Part 2: {exc}")
        print("Install with:  pip install torch torchvision timm")
        return []

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDeep learning device: {device}")
    if device.type == "cpu":
        print("  warning: training deep models on CPU is slow; use a GPU/Colab "
              "or --models resnet --epochs 1 for a quick check")

    # ------------------------------------------------------------------
    # Data pipeline.  Grayscale is replicated to 3 channels so ImageNet
    # pre-trained weights can be used.  Rotations and flips do not change the
    # number of pips, so they are label-preserving augmentations.
    # ------------------------------------------------------------------
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.RandomRotation(180, fill=200),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    class DieDataset(Dataset):
        def __init__(self, indices, transform):
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, item):
            index = self.indices[item]
            image = Image.fromarray(images[index])  # 2-D uint8 -> mode 'L'
            return self.transform(image), int(labels[index]) - 1

    def make_loader(indices, transform, shuffle):
        return DataLoader(
            DieDataset(indices, transform),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            drop_last=shuffle and len(indices) >= args.batch_size,
        )

    train_loader = make_loader(idx_train, train_tf, True)
    val_loader = make_loader(idx_val, eval_tf, False)
    test_loader = make_loader(idx_test, eval_tf, False)

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------
    def build_resnet():
        import torchvision.models as tvm

        ctor = getattr(tvm, args.resnet, None)
        if ctor is None:
            raise RuntimeError(f"unknown torchvision model '{args.resnet}'")
        try:
            model = ctor(weights=None if args.no_pretrained else "DEFAULT")
        except Exception as exc:
            print(f"  warning: could not load pre-trained {args.resnet} "
                  f"({exc}); using random initialisation")
            model = ctor(weights=None)

        if args.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
        return model

    def build_mvit():
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("MViT needs timm:  pip install timm") from exc

        try:
            model = timm.create_model(
                args.mvit,
                pretrained=not args.no_pretrained,
                num_classes=NUM_CLASSES,
            )
        except Exception as exc:
            print(f"  warning: could not load pre-trained {args.mvit} "
                  f"({exc}); using random initialisation")
            model = timm.create_model(
                args.mvit, pretrained=False, num_classes=NUM_CLASSES
            )

        if args.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            if hasattr(model, "reset_classifier"):
                model.reset_classifier(NUM_CLASSES)
            else:
                head = _classifier_head(model)
                if head is None or not hasattr(head, "in_features"):
                    raise RuntimeError(
                        f"cannot find the classifier head of {args.mvit}"
                    )
                setattr(model, "head", nn.Linear(head.in_features, NUM_CLASSES))
        return model

    # ------------------------------------------------------------------
    # Train / evaluate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(model, loader):
        model.eval()
        predictions, loss_sum, n = [], 0.0, 0
        criterion = nn.CrossEntropyLoss()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * len(y)
            n += len(y)
            predictions.append(logits.argmax(1).cpu())
        if n == 0:
            raise RuntimeError("empty data loader")
        return torch.cat(predictions).numpy(), loss_sum / n

    def train_model(name, model, lr):
        model.to(device)

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(f"{name}: no trainable parameters")
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(
            trainable, lr=lr, weight_decay=0.05
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.15,
        )

        y_val = labels[idx_val] - 1
        best_acc, best_state, best_epoch = -1.0, None, 0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, args.epochs + 1):
            if args.freeze_backbone:
                model.eval()
                head = _classifier_head(model)
                if head is not None:
                    head.train()
            else:
                model.train()

            total, seen = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                total += loss.item() * len(y)
                seen += len(y)

            val_pred, val_loss = predict(model, val_loader)
            val_acc = accuracy_score(y_val, val_pred)
            history["train_loss"].append(total / max(seen, 1))
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_acc >= best_acc:
                best_acc, best_epoch = val_acc, epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                print(
                    f"  [{name}] epoch {epoch:3d}/{args.epochs}  "
                    f"train loss {total / max(seen, 1):.4f}  "
                    f"val loss {val_loss:.4f}  val acc {val_acc * 100:.1f}%",
                    flush=True,
                )

        if best_state is not None:
            model.load_state_dict(best_state)
        print(f"  [{name}] best val acc {best_acc * 100:.1f}% "
              f"at epoch {best_epoch}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].plot(history["train_loss"], label="train")
        axes[0].plot(history["val_loss"], label="val")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].legend()
        axes[1].plot(np.array(history["val_acc"]) * 100)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("val accuracy (%)")
        fig.suptitle(f"{name} training curves")
        plt.tight_layout()
        fig.savefig(out_dir / f"curves_{name.lower()}.png", dpi=130)
        plt.close(fig)
        return model

    # ------------------------------------------------------------------
    # Run the requested models
    # ------------------------------------------------------------------
    results = []
    y_test = labels[idx_test]
    sources_test = sources[idx_test]
    candidates = []
    if args.models in ("both", "resnet"):
        candidates.append(("ResNet", build_resnet, args.lr_resnet, args.resnet))
    if args.models in ("both", "mvit"):
        candidates.append(("MViT", build_mvit, args.lr_mvit, args.mvit))

    for name, builder, lr, model_id in candidates:
        print(f"\n[Part 2] {name} ({model_id}) - "
              f"{'linear probe' if args.freeze_backbone else 'fine-tuning'}")
        try:
            model = builder()
        except Exception as exc:
            print(f"  skipped: {exc}")
            continue

        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"  {total_params:.1f}M parameters "
              f"({trainable_params:,} trainable)")

        started = time.time()
        try:
            model = train_model(name, model, lr)
        except Exception as exc:
            print(f"  training failed: {exc}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        print(f"  training time {time.time() - started:.0f}s")

        test_pred, _ = predict(model, test_loader)
        results.append(report(
            f"{name} ({model_id})", y_test, test_pred + 1,
            sources_test, out_dir,
        ))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ===========================================================================
# 7. Main
# ===========================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Mini Project 2: die face classification "
                    "(classical features + deep learning)"
    )
    parser.add_argument(
        "--data_root", default=None,
        help="optional local clone of the dataset repository (offline mode); "
             "by default the data is downloaded from GitHub",
    )
    parser.add_argument(
        "--cache_dir", default="die_dataset_cache",
        help="where downloaded images are cached",
    )
    parser.add_argument(
        "--branch", default=GITHUB_BRANCH,
        help="GitHub branch to download",
    )
    parser.add_argument(
        "--download_mode", choices=["archive", "api"], default="archive",
        help="archive: one 62 MB download, extract only the processed folders; "
             "api: fetch only the needed files (slower, less bandwidth)",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="ignore the cache and download the dataset again",
    )
    parser.add_argument(
        "--out_dir", default="results",
        help="where figures and results.json are written",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="random seed for the split, CV folds and deep training",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=200,
        help="SIFT bag-of-visual-words codebook size",
    )
    parser.add_argument(
        "--no_grid_search", action="store_true",
        help="use fixed SVM C=10 / gamma=scale instead of 5-fold tuning",
    )
    parser.add_argument("--skip_classical", action="store_true")
    parser.add_argument("--skip_cross_set", action="store_true")
    parser.add_argument("--skip_cv", action="store_true")
    parser.add_argument("--skip_dl", action="store_true")

    parser.add_argument(
        "--models", choices=["both", "resnet", "mvit"], default="both",
        help="which deep models to train",
    )
    parser.add_argument(
        "--resnet", default="resnet18",
        help="torchvision ResNet: resnet18 / resnet34 / resnet50",
    )
    parser.add_argument(
        "--mvit", default="mvitv2_tiny",
        help="timm MViT: mvitv2_tiny / mvitv2_small / mvitv2_base",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_resnet", type=float, default=5e-4)
    parser.add_argument("--lr_mvit", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--freeze_backbone", action="store_true",
        help="train only the final classification layer (linear probe)",
    )
    parser.add_argument(
        "--no_pretrained", action="store_true",
        help="train from scratch instead of ImageNet initialisation",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if args.vocab_size < 1:
        parser.error("--vocab_size must be at least 1")
    return args


def main():
    global SEED

    args = parse_args()
    SEED = args.seed
    np.random.seed(SEED)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = resolve_data_root(
        args.data_root, args.cache_dir, args.branch,
        args.refresh, args.download_mode,
    )
    images, labels, sources, _ = load_dataset(root)
    idx_train, idx_val, idx_test = make_splits(labels, sources)
    idx_trainval = np.concatenate([idx_train, idx_val])

    main_results = []
    features = None
    cross_set_results = []
    cv_results = {}

    if not args.skip_classical:
        classical_results, features = run_classical(
            images, labels, sources, idx_trainval, idx_test,
            out_dir, args,
        )
        main_results += classical_results

        if not args.skip_cross_set:
            cross_set_results = evaluate_cross_set(
                features["desc_sift"], labels, sources, out_dir,
                args.vocab_size,
            )
        if not args.skip_cv:
            cv_results = run_cross_validation(
                images, labels, features, out_dir, args.vocab_size
            )

    if not args.skip_dl:
        main_results += run_deep(
            images, labels, sources, idx_train, idx_val, idx_test,
            args, out_dir,
        )

    # Report the pooled numbers with the caveat, and treat the cross-source /
    # leave-one-source-out numbers as the generalisation result.
    # Do not present 100% as the headline accuracy.
    print("\n" + "=" * 62)
    print("SUMMARY (same held-out test set for every model)")
    print("=" * 62)
    for result in main_results:
        print(f"  {result['model']:40s} "
              f"{result['test_accuracy'] * 100:6.2f}%")

    payload = {
        "dataset": {
            "repository": f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}",
            "branch": args.branch,
            "images": int(len(images)),
            "classes": list(range(1, NUM_CLASSES + 1)),
        },
        "config": {
            "seed": args.seed,
            "vocab_size": args.vocab_size,
            "svm_grid_search": not args.no_grid_search,
            "epochs": args.epochs,
            "deep_models": args.models,
            "freeze_backbone": args.freeze_backbone,
            "pretrained": not args.no_pretrained,
        },
        "split": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
            "test_indices": [int(i) for i in idx_test],
        },
        "results": main_results,
        "cross_set": cross_set_results,
        "cross_validation": cv_results,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nFigures and results.json saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
