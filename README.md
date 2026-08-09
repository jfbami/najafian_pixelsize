# Calibration Measurement Agent

Measures **nanometers per pixel** for electron-microscopy kidney-biopsy images by
finding the calibration diffraction-grating frame in each acquisition folder and
measuring its grid spacing to sub-pixel precision.

Calibration standard: **TedPella Prod. No. 607**, 2160 lines/mm. The pitch is
derived from the ruling (1/2160 mm = **0.4629630 µm per grid space**) rather
than the rounded 0.463 the SOP quotes, so there is one source of truth in
`src/calibration.py`:

```
nm_per_pixel = 462.9630 / D          # D = pixels per grid space
```

## Accuracy and what the numbers mean

| Term | Value | Source |
|---|---|---|
| Measurement precision | ~0.01-0.05% on well-sampled frames | comb-fit residuals, per measurement |
| Region-to-region repeatability | <1% | `precision_check.py` |
| **Grating tolerance** | **~0.2%** | manufacturer; **systematic, irreducible** |

Every result carries `nm_per_pixel_uncertainty`, which combines the fit
precision with the grating's own tolerance. The grating dominates: no amount of
FFT precision makes a result more accurate than the standard it is measured
against. The ruling is not certified traceable, so 0.2% is the accuracy floor
until the grating is measured against a traceable standard.

**A result is either valid or absent.** `MeasurementResult.valid` is the gate;
when it is False, `nm_per_pixel` is NaN. Nothing in the pipeline reports a
number it does not stand behind.

## Why FFT, not manual line-picking

The grid is periodic, so its Fourier transform is a comb of peaks at integer
multiples of the grid frequency. The subtleties the code handles:

1. **The brightest peak is usually a harmonic, not the fundamental.** Naively
   reading the strongest peak gives an answer wrong by an integer factor.
   `measurer.py` detects the whole harmonic comb along each grid axis and solves
   for the fundamental by weighted least squares.
2. **Higher harmonics sharpen precision.** Using the comb rather than one peak,
   over every pixel in the image, beats averaging ten manual Photoshop
   measurements with no operator bias.
3. **Aliasing corrupts the comb on small frames.** A grating of period D folds
   every harmonic above order D/2 back onto frequencies that are *not* integer
   multiples. On a thumbnail those folded peaks form a self-consistent comb with
   the wrong pitch. Frames with D < 12 px are refused rather than measured.
4. **The fundamental can be masked.** A fixed low-frequency cutoff (used to
   reject illumination gradients) also removes the fundamental of any grid whose
   period exceeds the cutoff radius. The cutoff is scaled to the image size so
   this cannot happen inside the measurable range, and the solver can still
   recover a fundamental from its harmonics if it does.

Tissue frames are rejected because they lack a strong, square, orthogonal comb.

## Detection vs measurement - they are different gates

Detection answers *which frame is the grid* and runs on cheap Drive thumbnails,
using only scale-invariant features. Measurement answers *what is the pitch* and
requires a well-sampled frame.

A correctly detected calibration **thumbnail** normally has `valid=False`: it is
too small to measure. That is intended. **nm/pixel must always come from
`measure_grid` on the full-resolution TIFF.** The agent is instructed
accordingly and `save_result` refuses implausible values.

## Magnification range

The measurement is magnification-agnostic - it measures whatever period is in
the frame. What bounds it is **pixels per grid space** (D), which depends on
frame size: D must be ≥12 px (below that, harmonic aliasing corrupts the comb)
and ≤ span/4 (four periods needed across the frame).

| Frame | Usable range |
|---|---|
| 1024 px | 1.81 - 38.6 nm/px |
| 2048 px | 0.90 - 38.6 nm/px |
| 4096 px | 0.45 - 38.6 nm/px |

Verified on real frames at both ends: a 2512×2304 frame at 15000× measured
**4.2339 nm/px** (D=109), and a 1024×1024 low-magnification frame measured
**22.09 nm/px** (D=21). Note that below D≈30 px the fit uncertainty (~0.5%)
overtakes the grating's 0.2% tolerance and becomes the dominant error term.

## Burned-in info bars

Acquisition software appends a strip carrying the file name, scale bar and
instrument settings. It is not part of the imaged field, and the hard full-width
edge between it and the image puts a strong ridge across one axis of the
transform. On a real 1024×1194 frame with a 21 px grid it made the two axes fit
21.1 px and 298.5 px, and the measurement was correctly refused - but it should
never have got that far.

`imaging.py` crops it before measuring. When the metadata records the imaged
field (`##fv3 2512 2304`) that is used directly; otherwise the bar is inferred
from lines that are almost entirely one value, bridging its text rows and
trimming only edge-anchored strips.

## Guarding against a silently wrong number

The FFT measurement is self-contained, so a mistake in it has nothing to
contradict it. `crosscheck.py` supplies independent references:

- **The magnification law.** `nm_per_pixel × magnification` is constant for one
  camera on one scope; a frame that breaks it by a large factor is wrong. This
  uses only the pipeline's own accepted results - no vendor metadata - and is
  the check that carries the weight.
- **The microscope's embedded scale** (`XpixCal`) - **off by default.** On this
  instrument it is not a scale at all: every frame carries a constant
  `XpixCal=150.000000, Unit=um` regardless of magnification, matching no unit
  reading of the true value. Trusting it made the 2.8× residual read as a
  suspected 3× harmonic error, flagging *every correct frame*. Enable
  `crosscheck.trust_embedded_scale` only after confirming the field tracks
  magnification on your scope.

The magnification law does not prove a result correct; it catches the
integer-factor blunder. The agent must run `cross_check_measurement` before
`save_result`, and is instructed never to "correct" a suspected factor itself.

## Quick start (local, no credentials)

The measurement core needs only `numpy`, `pillow`, `scipy`:

```bash
python -m pytest tests/ -q
```

```bash
python validate_core.py FRAME.tif
```

```bash
python precision_check.py CALIBRATION.tif
```

```bash
python run_bootstrap.py PATH/TO/FRAMES --out labels.csv
```

```bash
python revalidate_detector.py --labels labels.csv
```

The test suite runs against generated gratings of exactly known pitch, so it
checks the answer is *right*, not merely reproducible - the real reference
frames live on lab Drive accounts and cannot be committed.

## Full pipeline (Drive + agent)

1. **Google Cloud**: create a service account, enable the Drive API, download
   `service_account.json` into this folder.
2. **Share**: each collaborator shares their specimen folder with the service
   account email (one click).
3. **Configure**: fill in `config.yaml` (account `root_folder_id`s, output folder).
4. **Train the detector** (optional, improves edge cases): run `run_bootstrap.py`
   to make `labels.csv`, then `train.py` on a Colab GPU to produce weights.
5. **Run**: `export ANTHROPIC_API_KEY=...` then `python run.py --config config.yaml`.

Outputs: `calibration_results.csv`, `review_queue.csv`, and a run summary.
Folders are processed in batches, one conversation each, so a long run cannot
outgrow the context window and a failed batch leaves its cases retryable.

## Re-validating the thresholds

The detection thresholds in `config.yaml` were tuned on one set of
full-resolution frames and are **not self-validating**. Re-run
`revalidate_detector.py` after changing the image source (thumbnails vs TIFFs),
the scope, or the bit depth. It reports the separation margin between the worst
calibration frame and the best tissue frame and exits non-zero on any
misclassification.

Note `fft_confidence` saturates at 1.0 for any well-exposed grating, so it
separates "some strong periodicity" from "none" but does not grade grid
quality. `spectral_concentration` and `grid_uniformity` are the graded,
scale-invariant features.

## Status

| Component | State | Validated on |
|---|---|---|
| FFT grid measurement (`measurer.py`) | **working, tested** | synthetic gratings, exact pitch; real frame → 4.30 nm/pixel |
| Bit-depth-aware loading (`imaging.py`) | **working, tested** | 8-bit and 16-bit agree to 0.1% |
| Cross-checks (`crosscheck.py`) | **working, tested** | catches 2×/3×/4× errors |
| Metadata parser (`metadata.py`) | working, validated | magnification, date, dimensions |
| FFT calibration detector (`detect.py`) | **working, tested** | 222 frames: 2/2 found, 0 false positives |
| Auto-labeler (`bootstrap.py`) | working, validated | produced `labels.csv` over 222 local frames |
| State store (`state.py`) | **working, tested** | resumable; survives interruption |
| Matcher (`matcher.py`) | **working, tested, wired in** | exposed as `choose_calibration_frame` |
| Drive client (`drive_client.py`) | written | needs `service_account.json` |
| Learned detector + training (`detector.py`, `train.py`) | written | runs on Colab GPU |
| Agent loop + tools (`agent.py`, `tools.py`) | written | needs credentials + `ANTHROPIC_API_KEY` |

## Layout

```
src/
  calibration.py   physical constants, conversion, uncertainty budget
  imaging.py       bit-depth-aware TIFF loading (16-bit safe)
  measurer.py      FFT harmonic-comb sub-pixel measurement
  metadata.py      TIFF ImageDescription parser
  detect.py        FFT-based calibration-frame detector (no model)
  crosscheck.py    independent checks against embedded scale + magnification law
  bootstrap.py     auto-label a folder -> labels.csv
  detector.py      EfficientNet-B0 inference (Colab)
  train.py         EfficientNet-B0 fine-tuning (Colab)
  drive_client.py  Google Drive (service account, read-only)
  matcher.py       calibration-to-tissue pairing by magnification + date
  state.py         SQLite results / review queue / cache (resumable)
  tools.py         agent tool definitions + dispatcher
  agent.py         Anthropic tool-use loop (batched)
  models.py        shared data models
tests/
  synthetic.py     generated gratings with exactly known pitch
  test_measurer.py accuracy, edge cases, uncertainty coverage
  test_pipeline.py constants, detection, matching, cross-checks, state
run.py                  production entry point
revalidate_detector.py  re-check detection thresholds on labelled data
validate_core.py        report measurement + metadata for given frames
precision_check.py      region-to-region repeatability
run_bootstrap.py        auto-label a folder
diagnose_grid.py        inspect raw FFT peaks and autocorrelation
config.yaml             accounts, thresholds, paths
schema.sql              SQLite schema
```
