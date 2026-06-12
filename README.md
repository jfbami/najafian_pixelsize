# Calibration Measurement Agent

Measures **nanometers per pixel** for electron-microscopy kidney-biopsy images by
finding the calibration diffraction-grating frame in each acquisition folder and
measuring its grid spacing to sub-pixel precision.

Calibration standard: **TedPella Prod. No. 607**, 2160 lines/mm = **0.463 µm per
grid space**. The measurement applies the lab formula:

```
nm_per_pixel = (0.463 / D) * 1000      # D = pixels per grid space
```

## Status

| Component | State | Validated on |
|---|---|---|
| FFT grid measurement (`measurer.py`) | **working, validated** | real frame `17E00231-1_115.tif` → 4.30 nm/pixel |
| Metadata parser (`metadata.py`) | **working, validated** | magnification, date, dimensions parsed correctly |
| FFT calibration detector (`detect.py`) | **working, validated** | 222 frames: 2/2 calibration found, 0 false positives |
| Auto-labeler (`bootstrap.py`) | **working, validated** | produced `labels.csv` over all 222 local frames |
| State store (`state.py`) | **working, validated** | resumable; survives interruption |
| Matcher (`matcher.py`) | **working** | magnification + date pairing logic |
| Drive client (`drive_client.py`) | written | needs `service_account.json` |
| Learned detector + training (`detector.py`, `train.py`) | written | runs on Colab GPU |
| Agent loop + tools (`agent.py`, `tools.py`) | written | needs credentials + `ANTHROPIC_API_KEY` |

## Why FFT, not manual line-picking

The grid is periodic, so its Fourier transform is a comb of peaks at integer
multiples of the grid frequency. Two subtleties the code handles:

1. **The brightest peak is usually a harmonic, not the fundamental.** Naively
   reading the strongest peak gives an answer wrong by an integer factor (the
   first draft read 8.7 nm/pixel — exactly 2× the truth). `measurer.py` detects
   the whole harmonic comb along each grid axis and solves for the fundamental by
   least squares.
2. **Higher harmonics sharpen precision.** Using the comb (not one peak) and
   every pixel in the image yields ~0.7% region-to-region repeatability, better
   than averaging ten manual Photoshop measurements, with no operator bias.

Tissue frames are rejected because they lack a strong, square, orthogonal comb.

### Detection result on the local set (222 frames)

| | calibration score | strongest tissue score |
|---|---|---|
| value | 0.965 – 0.970 | 0.499 |

Both calibration frames (`17E00231-1_115.tif`, `17E00231-2 glom-1_107.tif`) were
found with no false positives — a ~0.47 margin over the closest tissue frame.
They measured 4.30 and 4.23 nm/pixel. The rule-based FFT detector alone is
sufficient on this data; the learned model is only needed for damaged grids.

## Quick start (local, no credentials)

The measurement core needs only `numpy`, `pillow`, `scipy`:

```bash
python validate_core.py      # measure a known calibration + tissue frame
python precision_check.py    # quantify precision across image regions
python run_bootstrap.py      # auto-label all local frames -> labels.csv
python test_state.py         # verify the resumable state store
```

## Full pipeline (Drive + agent)

1. **Google Cloud**: create a service account, enable the Drive API, download
   `service_account.json` into this folder.
2. **Share**: each collaborator shares their specimen folder with the service
   account email (one click).
3. **Configure**: fill in `config.yaml` (account `root_folder_id`s, output folder).
4. **Train the detector** (optional, improves edge cases): run `bootstrap.py`
   to make `labels.csv`, then `train.py` on a Colab GPU to produce weights.
5. **Run**: `export ANTHROPIC_API_KEY=...` then `python run.py --config config.yaml`.

Outputs: `calibration_results.csv`, `review_queue.csv`, and a run summary.

## Layout

```
src/
  calibration.py   physical constants + nm/pixel conversion
  measurer.py      FFT harmonic-comb sub-pixel measurement
  metadata.py      TIFF ImageDescription parser
  detect.py        FFT-based calibration-frame detector (no model)
  bootstrap.py     auto-label a folder -> labels.csv
  detector.py      EfficientNet-B0 inference (Colab)
  train.py         EfficientNet-B0 fine-tuning (Colab)
  drive_client.py  Google Drive (service account, read-only)
  matcher.py       calibration-to-tissue pairing by magnification + date
  state.py         SQLite results / review queue / cache (resumable)
  tools.py         agent tool definitions + dispatcher
  agent.py         Anthropic tool-use loop
  models.py        shared data models
run.py             production entry point
config.yaml        accounts, thresholds, paths
schema.sql         SQLite schema
```
