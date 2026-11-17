# IP Mini Project: Die Face Dataset

Mini Project 1 for an image processing course. Images of a single six-sided
die, one face up per image (set 1: phone photos, set 2: synthetic renders).
The preprocessing pipeline below turns the raw photos into clean, uniform,
grayscale crops of the die, ready for a later stage that counts pips and
classifies faces 1-6. Mini Project 1 is preprocessing only, using classical
image processing (OpenCV + NumPy).

The repository also contains the **Mini Project 2 classifier**,
`PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py`, which uses these processed
images for classical (Harris / SIFT + SVM) and deep (ResNet / MViT) face
classification. See [Mini Project 2](#mini-project-2---die-face-classification)
below.

## Set 1

All work described here uses `set1(bhuvan)`.

### Original image structure

- Location: `set1(bhuvan)/unprocessed/` (read only, never modified)
- 60 images, all `.jpg`, all 4080x3072 (4:3), RGB, JPEG, taken with a phone
  camera (capture script: `capture.sh`, via adb)
- Filename format: `{face}_{index}.jpg`, face = 1..6, index = 0..9, so 10
  images per face (example: `3_0.jpg`)
- Scene: plain, slightly purple-gray, evenly lit surface with a soft lighting
  gradient. The die is small, about 7-8% of the frame width, and its position
  and rotation vary (some shots are tilted about 45 degrees)
- The die is white/translucent. Pips are blue on faces 2, 3, 5, 6 and red on
  faces 1 and 4. Grayscale conversion is deliberate so color cannot leak the
  label into any later model

### Repository structure

```
IP-mini-project-die-face-dataset/
|-- README.md
|-- LICENSE
|-- capture.sh                  # phone capture script used for set 1 acquisition
|-- PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py  # Mini Project 2 classifier
|-- set1(bhuvan)/
|   |-- unprocessed/            # 60 original photos (INPUT, read-only)
|   |-- processed/              # OUTPUT: 60 PNGs + _contact_sheet.png
|   |-- metadata.csv            # one row per image (see below)
|   |-- pipeline_walkthrough.png  # example figure used in this README
|   `-- scripts/
|       |-- preprocess.py       # main pipeline script
|       |-- requirements.txt    # opencv-python, numpy, pandas, matplotlib
|       `-- debug/              # step-by-step figures, only with --debug
`-- set2(jai)/
    |-- unprocessed/            # 119 synthetic 128x128 PNGs (INPUT, read-only)
    |-- processed/              # OUTPUT: 119 PNGs + _contact_sheet.png
    |-- metadata.csv            # one row per image (same columns as set 1)
    |-- pipeline_walkthrough.png  # example figure used in this README
    `-- scripts/
        |-- preprocess.py       # set 2 pipeline (adapted from set 1)
        |-- generate_synthetic.py  # renders the synthetic images
        |-- requirements.txt    # opencv-python, numpy, pandas, matplotlib
        `-- debug/              # step-by-step figures, only with --debug
```

Folder names contain parentheses, so quote paths in shell commands and use
`pathlib` in code.

### Environment setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r "set1(bhuvan)/scripts/requirements.txt"
```

### Preprocessing pipeline (per image)

The die is small relative to the frame, so it is detected and cropped BEFORE
resizing, otherwise the pips would shrink to a few pixels.

1. Load with `cv2.imread` (applies EXIF orientation). Keep the
   full-resolution image for the final crop.
2. Working copy downscaled to width 1024 (`cv2.INTER_AREA`), remembering the
   scale factor.
3. Gaussian blur on the working copy, kernel (7, 7).
4. Color distance segmentation from the background:
   - Convert blurred working copy to Lab (`cv2.COLOR_BGR2LAB`, float32).
   - Background color = per-channel MEDIAN of all pixels (the die is tiny,
     so the median is essentially pure background).
   - Distance map = Euclidean norm of (Lab - background color) per pixel,
     normalized to 0-255 uint8.
   - Otsu threshold (`THRESH_BINARY + THRESH_OTSU`). The die body is similar
     in color to the background, so Otsu typically lights up only the die
     outline and pips, not the interior. Step 5 handles this.
5. Mask cleanup:
   - Morphological CLOSE with an elliptical 9x9 kernel to bridge outline gaps.
   - `findContours` (RETR_EXTERNAL), fill all external contours
     (`drawContours` with `cv2.FILLED`) to fill the enclosed die body.
   - Morphological OPEN with the same kernel to remove specks.
6. Largest connected component (`connectedComponentsWithStats`, ignoring
   background label 0) is taken as the die. Read off its bounding box
   (x, y, w, h).
7. Square crop box centered on the bbox center with side
   `max(w, h) * (1 + 2 * margin)`, margin = 0.5. Clamp to working-image
   bounds, map back to full-resolution coordinates by dividing by the scale
   factor, and crop from the FULL-RESOLUTION image.
8. Resize crop to 224x224 (`cv2.INTER_AREA`), then:
   - Grayscale (`cv2.COLOR_BGR2GRAY`).
   - Bilateral filter (`d=7, sigmaColor=40, sigmaSpace=7`) to smooth die
     surface texture while keeping pip edges sharp.
   - Background-level normalization: `bg` = median gray value of the border
     strip of the 224x224 crop (outer 12% on all four sides, about 27 px),
     then `out = clip(img * (200.0 / bg), 0, 255)`. Every output image ends
     up with background level 200. No histogram equalization or CLAHE (those
     amplified surface texture and noise in testing).
9. Save as lossless PNG, single-channel 8-bit grayscale, to
   `set1(bhuvan)/processed/` with the input stem (`3_0.jpg` -> `3_0.png`).

Expected result: smooth grayscale image, die centered, clean light
background, clearly separated dark round pips.

Walkthrough of every stage on example `2_0.jpg` (face 2, tilted). Note how
the Otsu mask keeps only the die outline and pips, and the cleanup stages
turn that into a solid die mask:

![Pipeline walkthrough: original, blur, distance map, Otsu mask, cleaned mask, detected box, final crop](set1%28bhuvan%29/pipeline_walkthrough.png)

### Quality control

Per image, `qc_flag` in `metadata.csv` is set when:

- `mask_area_fraction` (die mask area / working-image area) is outside
  0.001-0.02,
- the detected bbox touches the image border (die may be cut off),
- no object is found or the image fails to load (logged, skipped, no crash).

A summary (processed count, flagged count) prints at the end of each run.

### Output structure

- `set1(bhuvan)/processed/*.png`: 60 processed images, each 224x224,
  single-channel 8-bit grayscale.
- `set1(bhuvan)/processed/_contact_sheet.png`: all 60 in a labeled grid
  ordered by face then index (leading underscore keeps it sorted apart from
  data images), for eyeballing everything at once.

![Contact sheet: all 60 processed images, ordered by face then index](set1%28bhuvan%29/processed/_contact_sheet.png)

- `set1(bhuvan)/metadata.csv`: one row per image with columns
  `filename, source, category, original_width, original_height,
  original_format, processed_width, processed_height, processed_format,
  bbox_x, bbox_y, bbox_w, bbox_h, mask_area_fraction, qc_flag,
  bg_level_before, bg_level_after`.
  - `source` = "set1 (bhuvan), phone camera".
  - `category` = face number parsed from the filename.
  - bbox values are full-resolution pixel coordinates.
  - `bg_level_before` / `bg_level_after` = border-strip median gray value
    before / after step 8 normalization.

### Usage

Paths resolve relative to the script location, so it runs from any directory.
Safe to re-run (outputs are overwritten, not duplicated).

```bash
.venv/bin/python "set1(bhuvan)/scripts/preprocess.py"               # defaults
.venv/bin/python "set1(bhuvan)/scripts/preprocess.py" --debug       # + per-image step figures in scripts/debug/
.venv/bin/python "set1(bhuvan)/scripts/preprocess.py" --help        # all options
```

Options: `--size` (default 224), `--margin` (default 0.5),
`--input` (default `unprocessed`), `--output` (default `processed`),
`--debug` (save original, blur, distance map, Otsu mask, cleaned mask, box
overlay, final crop per image to `scripts/debug/`).

### Results on this set

- 60/60 images processed, 0 flagged.
- Output check: 60 PNGs, each 224x224 single-channel 8-bit; CSV has 60 rows,
  10 per face 1-6.
- Background levels: before normalization min 138.00, max 150.00, std 2.32;
  after: exactly 200.00 on all 60 (std 0.00).
- Contact sheet review: every die centered with all pips visible and correct
  counts per face. Minor notes: `5_0` is rotated with one pip near the die
  edge (fully inside the crop), and face-1 shots show a bright specular ring
  around the single pip.

---

## Set 2

All work described here uses `set2(jai)`. It follows the same pipeline and the
same output specification as Set 1 (224x224, single-channel 8-bit grayscale PNG,
background level 200), so the two processed sets can be combined. The input
images are different, so a few pipeline parameters differ (see "Differences from
set 1").

### Original image structure

- Location: `set2(jai)/unprocessed/` (read only, never modified)
- 119 images, all `.png`, all 128x128 (1:1), RGB, 8-bit
- **Synthetic, not photographed.** Rendered with Pillow by
  `set2(jai)/scripts/generate_synthetic.py` (fixed seed 42). That script
  produces 3000 images; set 2 is a 119-image sample of them
- Filename format: `die{face}_{index}.png`, face = 1..6, index = the
  generator's running index (so indices have gaps; example: `die3_0048.png`).
  Images per face: 20, 19, 20, 20, 21, 19
- Scene: random flat background colour with grain noise, soft drop shadow, one
  rounded-square die per image in one of 7 colours (white, red, black, blue,
  green, yellow, purple). Pip colour is black or white, whichever contrasts with
  the die. The die takes up about 50-68% of the frame side, at a random rotation
  (0-360 degrees) and a small random shift. About 40% of images are blurred;
  brightness and noise vary
- Die colour and background colour are random and independent of the face, so
  colour does not carry the label. Grayscale conversion is kept so the output
  matches set 1

### Differences from set 1

The input images are small (128x128) with a LARGE die (about 26-52% of the frame
once the shadow is included), while set 1 is 4080x3072 with a die of 7-8% of the
frame width. Everything else in the pipeline and the output format is the same.

| Step | Set 1 | Set 2 | Why |
|---|---|---|---|
| Input | `*.jpg`, 4080x3072 | `*.png`, 128x128 | synthetic images |
| Working copy | downscale to width 1024 | none (scale 1.0) | already small; upscaling adds nothing |
| Gaussian blur | (7, 7) | (3, 3) | small image |
| Background colour | median of all pixels | median of the outer 8% border strip | die covers up to ~half the frame, so the global median is unsafe |
| Threshold | Otsu | Otsu, then a second Otsu on the lower group (3-class); falls back to set 1's mask if the area is implausible | die body is a strong colour here, so one Otsu split isolates only the pips |
| Morphology kernel | elliptical 9x9 | elliptical 5x5 | small image |
| Crop margin | 0.5 | 0.15 | die already fills much of the frame |
| Crop edges | clamped to image | image padded with its border-median colour so the crop is always square | die can sit near the frame edge |
| Resize | INTER_AREA | INTER_CUBIC (crops are smaller than 224) | upsampling |
| Level normalization | `out = img * 200/bg`, clipped | piecewise-linear, `bg -> 200`, identical below bg, `[bg, 255]` compressed into `[200, 255]` | background brightness varies from 50 to 227 here; the linear scale (up to 4x) clipped bright dice and pips to white |
| QC mask-area range | 0.001-0.02 | 0.10-0.60 | die is much larger |
| `source` column | `set1 (bhuvan), phone camera` | `set2 (jai), synthetic (rendered by generator script)` | provenance |

Two problems found while adapting the script, both fixed:

- With a single Otsu threshold, 8 of 119 images segmented only one pip as "the
  die" (mask area fraction about 0.01). The 3-class Otsu fixes this.
- With set 1's linear level normalization, 26 images had more than 10% of their
  pixels clipped to 255 and several dice lost their pips completely
  (for example `die4_0245`, `die5_0071`). The piecewise-linear map fixes this.

Walkthrough of every stage on example `die2_0017.png` (face 2):

![Pipeline walkthrough](set2%28jai%29/pipeline_walkthrough.png)

### Output structure

- `set2(jai)/processed/*.png`: 119 processed images, each 224x224,
  single-channel 8-bit grayscale, named after the input stem
- `set2(jai)/processed/_contact_sheet.png`: all 119 in a labeled grid ordered by
  face then index

![Contact sheet](set2%28jai%29/processed/_contact_sheet.png)

- `set2(jai)/metadata.csv`: one row per image, same columns as set 1.
  `category` = face number parsed from the filename (`die3_0048` -> 3);
  `bbox_*` are the square crop box in input-image pixel coordinates (it can
  extend past the image by the padded margin, so values can be negative).

### Usage

```bash
.venv/bin/pip install -r "set2(jai)/scripts/requirements.txt"
.venv/bin/python "set2(jai)/scripts/preprocess.py"               # defaults
.venv/bin/python "set2(jai)/scripts/preprocess.py" --debug       # + per-image step figures in scripts/debug/
.venv/bin/python "set2(jai)/scripts/preprocess.py" --help        # all options
```

Options are the same as set 1, except the default `--margin` is 0.15. To
regenerate the synthetic images: `python3 "set2(jai)/scripts/generate_synthetic.py"`
(needs `pillow` and `numpy`; writes 3000 images to `dice_dataset/` in the
current directory).

### Results on this set

- 119/119 images processed, 21 flagged.
- All 21 flags are `bbox_touches_border` (the die or its shadow reaches the
  frame edge). The mask-area check flags none. Reviewed on the flagged crops:
  the dice are fully inside the frame, with all pips visible.
- Output check: 119 PNGs, each 224x224 single-channel 8-bit; CSV has 119 rows
  (face 1-6: 20, 19, 20, 20, 21, 19).
- Mask area fraction: min 0.260, max 0.522.
- Background levels: before normalization min 50, max 227, std 37.6; after:
  199-200 on all 119 (std 0.16; the 199s are integer rounding).
- Note: on white dice many pixels sit at 255 after normalization (7 images
  have more than 10% saturated pixels). The pips are dark and clearly visible
  in those.

---

## Final Count

- Set 1 -> capture using phone camera -> 60 images (10 per face)
- Set 2 -> generated using a script -> 119 images
  (face 1-6: 20, 19, 20, 20, 21, 19)
- Total -> 179 images

---

## Mini Project 2 - Die Face Classification

Mini Project 2 uses the 179 processed crops from Mini Project 1 (60 real phone
photos + 119 synthetic renders, each 224x224 grayscale) to classify the visible
face value 1-6. Team SRs: `PES1UG24CS568`, `PES1UG24CS197`,
`PES1UG24CS226`. The complete submission is a single Python file:

```
PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py
```

The script does not assume the dataset is already present. It downloads the
two `processed` folders and their `metadata.csv` files from
`https://github.com/bhuvan11000/IP-mini-project-die-face-dataset`, caches them
in `die_dataset_cache/`, and reads the labels from the metadata. It runs from
any directory.

### What the script implements

| Course requirement | Implementation |
|---|---|
| SIFT + Harris feature extraction with SVM classification | Harris layout (52-D) + SVM, rotation-invariant Harris (33-D) + SVM, SIFT + BoVW + SVM, and Harris–SIFT hybrid + SVM |
| ResNet image classification | `torchvision` ResNet-18, ImageNet-pretrained and fine-tuned |
| MViT image classification | `timm` MViTv2-Tiny (Multiscale Vision Transformer), ImageNet-pretrained and fine-tuned |
| Single `.py` submission | `PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py` |
| SR-based filename | three SRs separated by underscores |

**Part 1 - classical features + SVM**

- Harris corner feature vector: corner response statistics, the 20 strongest
  corner coordinates, and an intensity histogram at corner pixels (52-D).
- Rotation-invariant Harris descriptor: pip-like corner clusters and pairwise
  distance histograms, using distances rather than absolute positions (33-D).
- SIFT descriptors clustered into 200 visual words with K-Means; each image is
  encoded as an L2/power-normalised bag-of-visual-words histogram.
- RBF SVM with `StandardScaler`; hyperparameters can be tuned by 5-fold CV.
- A cosine-similarity 1-NN baseline on the SIFT BoVW vectors.
- Harris keypoints + SIFT descriptors + BoVW as a hybrid pipeline.
- Cross-source evaluation: train on one entire source and test on the other.
- 5-fold cross-validation with the visual vocabulary re-fitted inside each
  fold, so test descriptors never leak into the codebook.

**Part 2 - deep learning**

- ResNet-18 fine-tuned from ImageNet weights.
- MViTv2-Tiny fine-tuned from ImageNet weights.
- `--freeze_backbone` trains only the final classification layer (linear
  probe) instead of fine-tuning the whole network.
- The same source-stratified train/validation/test split is used as in Part 1.
  The validation split selects the best epoch; the test split is never used
  for training, vocabulary fitting, or model selection.

### Split

The split is stratified on `(source, face)`:

| Split | Images | Use |
|---|---:|---|
| Train | 113 | classical fits; deep training |
| Validation | 21 | deep epoch selection |
| Test | 45 | final evaluation for every model |

Both sources and all six faces are represented in every split. The classical
models train on train + validation, while the deep models use validation for
best-epoch selection.

### Results

The numbers below are from the verification runs used for this project. The
script defaults to 30 deep-learning epochs; the deep numbers below are from
10-epoch verification runs, so they are indicative rather than a tuned
benchmark.

**Part 1 - classical holdout** (fixed RBF SVM, `C=10`, `gamma="scale"`):

| Method | Holdout accuracy | 5-fold CV |
|---|---:|---:|
| Harris layout (52-D) + SVM | 46.7% | 48.1% ± 6.6% |
| Harris rotation-invariant (33-D) + SVM | 84.4% | 84.4% ± 4.5% |
| SIFT + BoVW + SVM | 91.1% | 88.8% ± 3.2% |
| SIFT + BoVW + cosine 1-NN | 80.0% | — |
| Harris–SIFT hybrid + BoVW + SVM | 64.4% | 68.7% ± 9.4% |

With 5-fold SVM hyperparameter search, the holdout accuracy was 88.9% for
SIFT and 82.2% for the rotation-invariant Harris descriptor; the ranking is
unchanged.

**Cross-source generalisation (SIFT + BoVW)**:

| Train source | Test source | Accuracy |
|---|---|---:|
| Synthetic `set2` | Real `set1` | 25.0% |
| Real `set1` | Synthetic `set2` | 25.2% |
| Chance level | — | 16.7% |

**Part 2 - deep learning** (10-epoch verification runs, ImageNet pretrained):

| Model | Pooled holdout | Cross-source |
|---|---:|---:|
| ResNet-18 | 100.0% (seed 42); 95.6% (seed 123) | 31.7% `set2 -> set1`; 21.9% `set1 -> set2` |
| MViTv2-Tiny | 100.0% (seed 42) | — |

> **Important:** the pooled random split is optimistic. Both sources appear in
> train and test, the processed crops are clean and centred, and the test set
> has only 45 images. The 100% result is not the headline generalisation
> number. Use the cross-source / leave-one-source-out results as the
> generalisation estimate, and use `--seed` to check split stability. The
> script prints a caution note whenever a model gets a perfect score on a
> small pooled test set.

### Usage

Install the classical dependencies (already present in many Colab/Kaggle
environments):

```bash
pip install numpy pandas opencv-python scikit-learn matplotlib
```

Install the deep-learning dependencies for Part 2:

```bash
pip install torch torchvision timm
```

Run everything (downloads the dataset from GitHub on the first run):

```bash
python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py
```

Common options:

```bash
# classical part only (no torch/timm needed)
python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py --skip_dl

# deep learning only
python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py --skip_classical

# choose deep models / epochs
python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py --models resnet --epochs 10
python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py --models both --epochs 30

# train only the final layer (linear probe)
python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py --freeze_backbone

# different split / stability check
python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py --seed 123

# offline: use an existing clone
python PES1UG24CS568_PES1UG24CS197_PES1UG24CS226.py --data_root .
```

The first run downloads a repository archive (about 62 MB) and extracts only
the processed images and metadata (about 6 MB stored). Later runs reuse
`die_dataset_cache/`. `--download_mode api` fetches only the needed files
individually if you prefer lower bandwidth over speed.

### Outputs

- `results/cm_*.png` - confusion matrices for every model.
- `results/curves_*.png` - ResNet / MViT training curves.
- `results/harris_response.png` - Harris response and detected corners.
- `results/harris_sift_keypoints.png` - Harris vs SIFT keypoints.
- `results/results.json` - split, configuration, holdout results,
  cross-source results and 5-fold CV summary.

`results/` and `die_dataset_cache/` are generated locally and are ignored by
git.
