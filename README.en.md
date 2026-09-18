# Half-Frame Splitter

English | [简体中文](README.md)

Half-frame cameras (Olympus Pen, Kodak Ektar H35, Yashica Half and others) expose 18×24 mm
frames, and developing services frequently deliver two adjacent frames inside a single scan
file. Half-Frame Splitter locates the film base gap between them — dark, bright or
orange-tinted — and separates the file into two photographs while preserving EXIF, ICC
profile and DPI. It processes entire directory trees, recreating the same folder structure
under the output directory, and provides both a graphical interface and a command-line
interface. Only Pillow and NumPy are required.

- Place the scans to be processed in the `input` directory
- Results are written to the `output` directory
- The `input` directory is read-only: the tool never modifies, moves or deletes any original file

---

## 1. Directory Rules

Subfolders inside `input` are detected automatically, and folders of the same name are
created under `output`:

| Content placed in the input directory | Location of the results |
| --- | --- |
| `input/IMG_0001.jpg` (photo at the root) | `output/IMG_0001_1.jpg`, `output/IMG_0001_2.jpg` |
| `input/volume-1/` (a whole folder) | matching results inside `output/volume-1/` |
| `input/volume-1/night/` (nested folders) | `output/volume-1/night/`, structure mirrored in full |
| Photos and folders mixed together | each item is routed to its corresponding folder |

```
input/                        output/
├─ IMG_0001.jpg      -->      ├─ IMG_0001_1.jpg   IMG_0001_2.jpg
├─ volume-1/                   ├─ volume-1/
│  ├─ 000001.jpg     -->      │  ├─ 000001_1.jpg   000001_2.jpg
│  └─ 000002.jpg     -->      │  └─ 000002_1.jpg   000002_2.jpg
└─ volume-2/night/             └─ volume-2/night/
   └─ 0100.jpg       -->         └─ 0100_1.jpg   0100_2.jpg
```

- Preview images with the cut line are stored in the `_debug` subfolder of the **folder
  containing the results**, for example `output/volume-1/_debug/`
- To skip subfolders entirely, pass `--no-subdirs`, or clear the corresponding checkbox in
  the graphical interface
- Non-image files in the input directory (such as txt) are ignored
- Other directories may be specified on the command line, leaving `input` and `output` untouched:

```bat
python half_frame_splitter.py -i <input-dir> -o <output-dir> --debug
```

---

## 2. Usage

### 2.1 Graphical interface (recommended)

Run **`运行-图形界面.bat`** ("run — graphical interface"), select the directories in the
window and click the start button.

- The input and output directories default to `input` and `output` inside the tool folder
- A progress bar reports overall progress; the log area lists the result of every file and
  the folder it belongs to
- When processing finishes, all output folders are listed and the tool offers to open the
  output directory
- Options are saved automatically and restored on the next launch

### 2.2 Batch script

Run **`运行-命令行.bat`** ("run — command line"): all photos in `input`, including those in
subfolders, are cut into the corresponding folders under `output`.

### 2.3 Command line with options

```bat
python half_frame_splitter.py                    :: process the input directory into output
python half_frame_splitter.py --trim --debug     :: remove scan borders and write previews
python half_frame_splitter.py --no-subdirs       :: process the root level only
python half_frame_splitter.py -i <input-dir> -o <output-dir>
python half_frame_splitter.py --dry-run          :: detect only, write nothing (advised for large batches)
```

---

## 3. Command-line Options

| Option | Description |
| --- | --- |
| `-i, --input` | Input directory, defaults to `./input` |
| `-o, --output` | Output directory, defaults to `./output` |
| `--no-subdirs` | Process the root level of the input directory only (by default subfolders are detected and mirrored) |
| `--orient auto\|lr\|tb` | Frame arrangement: `auto` detect automatically (default), `lr` two frames side by side, `tb` two frames stacked |
| `--edge gap\|center` | Gap handling: `gap` discard the gap entirely (default, no dark edge remains); `center` cut through the middle of the gap |
| `--trim` | Remove uniform pure-black or pure-white scan borders (useful for full-strip scans including the film rebate) |
| `--aspect off\|auto\|value` | Centre-crop to a frame aspect ratio, for example `--aspect 0.75` (portrait half-frame) |
| `--frame-aspect` | Aspect ratio of a single frame, default `0.75` (18:24) |
| `--search LO HI` | Gap search range as a proportion, default `0.22 0.78`; adjust when the gap is noticeably off-centre |
| `--cut POSITION` | Manual cut position: `0~1` as a proportion, greater than `1` in pixels |
| `--format same\|jpg\|png\|tif\|webp` | Output format, defaults to the input format |
| `--quality` | JPEG/WEBP quality, default 95 |
| `--overwrite` | Overwrite existing output files (by default files of the same name are skipped) |
| `--skip-uncertain` | Skip files whose detection confidence is insufficient |
| `--min-confidence` | Confidence threshold, default 2.0 |
| `--debug` | Write preview images with the cut line into `_debug` inside the result folder |
| `--dry-run` | Detect only; no file is written |
| `--recursive` / `--keep-structure` | Legacy options; this behaviour is now the default and they are kept for compatibility |

Output naming: `<original-name>_1` (first frame: left or top) and `<original-name>_2`
(second frame: right or bottom). EXIF, ICC profile and DPI are carried over; the orientation
tag is normalised so the image is never rotated twice.

---

## 4. How the Gap Is Detected

The tool depends on Pillow and NumPy only; OpenCV is not required.

1. **Multi-scale bar detection.** A detector comparing a central bar against the two
   flanking bars slides along the luminance and chroma profiles to locate a narrow, uniform
   band that differs clearly from both sides. The gap may be darker than its surroundings
   (positive scans, lab JPEGs) or brighter and orange-tinted (negative film base).
2. **Uniformity weighting.** A real gap is highly uniform along its length, whereas strong
   vertical edges inside a picture (railings, door frames) are not; the latter are therefore
   penalised.
3. **Geometric constraints.** A half-frame image is 18×24 mm, so a pair placed side by side
   spans 24 mm × 1.5. This yields a plausible range for the cut position and a check on the
   proportions of the two resulting parts.
4. **Conservative fallback.** When the overall aspect ratio is close to 3:2, the two frames
   must be equally wide and the cut position is uniquely determined at the image centre;
   deviating from it then requires stronger evidence. When evidence is insufficient the file
   is cut along the centre line and a notice is written to the log.
5. **Optional post-processing.** `--trim` removes uniform pure-black or pure-white scan
   borders; `--aspect` corrects the composition to the frame aspect ratio.

The cut position keeps a 15% margin inside the gap so that picture content is never clipped.

---

## 5. FAQ

**The cut position is inaccurate, or the proportions of the two photos look wrong.**
Run once with `--debug` and open `_debug` inside the result folder: the red line is the
actual cut position and the green box is the detected gap. If the log reports that no clear
gap was detected and the centre line was used, either:
- specify the cut position manually, for example `--cut 0.5`; or
- rescan with both frames and the gap between them included (do not crop the gap away).

**When does detection fail?**
When a frame is severely overexposed or affected by light leaks so that the gap is no longer
distinguishable, the tool falls back to the centre line and reports it instead of cutting
through picture content. Such files are best reviewed manually.

**The two frames are stacked (the scan is rotated).**
The tool usually detects this automatically; use `--orient tb` to force it. Note that each
resulting photo from a stacked scan is oriented sideways and can be rotated in any image
viewer.

**I scan whole strips on a flatbed scanner, including black borders and sprocket holes.**
Use `--trim` to remove the borders and `--debug` to inspect the result.

**The log reports that the aspect ratio is close to 3:4 and the file may be a single frame.**
This notice is only issued when the gap could not be confirmed. A single half-frame is 3:4
(or 4:3), unlike the 3:2 of a side-by-side pair. If the file is in fact an already cut
single photo, move it out of the input directory; otherwise the scan was cropped tightly and
`--debug` will show what happened.

**Which file formats are supported?**
jpg, jpeg, png, tif, tiff, bmp, webp, ppm, pgm and gif, in either lower or upper case.
16-bit TIFF is supported as well and is stretched automatically for analysis. Very large
scans are handled without difficulty.

**Are the original files affected?**
No. The program only reads the input directory and only writes to the output directory; no
delete or move operation exists in the code. Existing output files of the same name are
skipped unless `--overwrite` is given.

**How long does a large batch take?**
About 0.2 seconds per file (at 2000×1300), so several hundred files usually finish within a
minute or two. The graphical interface stays responsive and processing can be stopped at any
time.

---

## 6. Self-test

```bat
python selftest.py
python selftest.py --photos <photo-A> <photo-B>   :: use real photographs as test material
```

The self-test composes 19 representative situations (black, white and orange gaps, stacked
frames, scan borders, sprocket holes, heavy JPEG compression, a very narrow gap, a strong
vertical line inside the picture, an absent gap, low-texture content and others), measures
the cut position error for each, and additionally verifies end-to-end writing, the
single-frame notice and **folder mirroring** (root level, A, A\B and B). Samples, previews
and results are written to the `_selftest` directory. The current version passes 19/19: with
real material the confidence of a confirmed gap is above 3.6, while interfering cases such as
an absent gap or low-texture content stay at or below 2.0 (threshold 2.0).

---

## 7. Files

| File / directory | Purpose |
| --- | --- |
| `input/` | Photos or photo folders to be processed (read-only, never modified) |
| `output/` | Cut results; subfolders share the input names, `_debug` holds previews |
| `half_frame_splitter.py` | Core algorithm and command-line entry point (importable by other scripts) |
| `gui.py` | Graphical interface |
| `selftest.py` | Self-test script |
| `check_env.py` | Pre-launch environment check |
| `运行-图形界面.bat` | Launch the graphical interface |
| `运行-命令行.bat` | Run input → output directly |
| `安装依赖.bat` | Install Pillow and NumPy |
| `_selftest/` | Samples and previews produced by the self-test; safe to delete |
| `_gui_settings.json` | Options saved by the graphical interface; safe to delete |

Requirements: Python 3.8 or later together with the Pillow and NumPy libraries
(run `安装依赖.bat`, or `python -m pip install -r requirements.txt`).
