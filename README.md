# IP Mini Project: Die Face Dataset

Mini Project 1 for an image processing course. Photos of a single six-sided
die, one face up per photo. The preprocessing pipeline below turns the raw
photos into clean, uniform, grayscale crops of the die, ready for a possible
later stage that counts pips and classifies faces 1-6. No classifier is built
here, preprocessing only, using classical image processing (OpenCV + NumPy).

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
|-- capture.sh                  # phone capture script used for acquisition
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

