"""Re-validate the detection thresholds against a labelled reference set.

The thresholds in `config.yaml` were tuned on one set of full-resolution
frames. They are not self-validating: any change to the image source (Drive
thumbnails instead of TIFFs, a different scope, a different bit depth) can move
the feature distributions without moving the thresholds. Run this whenever any
of that changes.

Usage:
    python revalidate_detector.py --calibration DIR --tissue DIR
    python revalidate_detector.py --labels labels.csv

`--labels` re-scores the frames listed in a bootstrap CSV, using its
`is_calibration` column as truth. Correct any mislabelled rows by hand first --
bootstrap labels come from the same rule being tested, so an uncorrected CSV
measures self-consistency, not accuracy.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from src.bootstrap import tiff_files
from src.detect import Thresholds, detect_calibration


def score_frames(paths: list[Path], truth: bool, thresholds: Thresholds) -> list[dict]:
    rows = []
    for path in paths:
        try:
            detection = detect_calibration(str(path), thresholds)
        except Exception as error:
            print(f"  skipped {path.name}: {type(error).__name__}: {error}")
            continue
        rows.append(
            {
                "frame": path.name,
                "truth": truth,
                "predicted": detection.is_calibration,
                "score": detection.score,
                "confidence": detection.measurement.fft_confidence,
                "squareness": detection.measurement.grid_uniformity,
                "separation": detection.measurement.axis_separation_deg,
                "concentration": detection.measurement.spectral_concentration,
                "measurable": detection.measurement.valid,
            }
        )
    return rows


def report(rows: list[dict]) -> int:
    positives = [row for row in rows if row["truth"]]
    negatives = [row for row in rows if not row["truth"]]
    missed = [row for row in positives if not row["predicted"]]
    false_alarms = [row for row in negatives if row["predicted"]]

    print(f"\nCalibration frames : {len(positives)}  (missed {len(missed)})")
    print(f"Tissue frames      : {len(negatives)}  (false positives {len(false_alarms)})")

    for label, group in (("MISSED", missed), ("FALSE POSITIVE", false_alarms)):
        for row in group:
            print(f"  {label}: {row['frame']:32s} score={row['score']:.3f} "
                  f"conf={row['confidence']:.3f} sq={row['squareness']:.3f} "
                  f"sep={row['separation']:.1f} conc={row['concentration']:.3f}")

    if positives and negatives:
        worst_positive = min(row["score"] for row in positives)
        best_negative = max(row["score"] for row in negatives)
        margin = worst_positive - best_negative
        print(f"\nWorst calibration score : {worst_positive:.3f}")
        print(f"Best tissue score       : {best_negative:.3f}")
        print(f"Separation margin       : {margin:+.3f}")
        if margin <= 0:
            print("  FAIL: the classes overlap; the thresholds cannot separate them.")
        elif margin < 0.2:
            print("  WARNING: margin is narrow; expect misclassifications.")

    unmeasurable = [row for row in positives if not row["measurable"]]
    if unmeasurable:
        print(f"\n{len(unmeasurable)}/{len(positives)} calibration frames are detected but "
              f"not measurable at this resolution.")
        print("  Expected for thumbnails -- measure nm/pixel on the full TIFF.")

    return 1 if (missed or false_alarms) else 0


def load_from_labels(path: Path) -> tuple[list[Path], list[Path]]:
    calibration, tissue = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("error"):
                continue
            target = calibration if row["is_calibration"] == "1" else tissue
            target.append(Path(row["path"]))
    return calibration, tissue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, help="folder of known grid frames")
    parser.add_argument("--tissue", type=Path, help="folder of known tissue frames")
    parser.add_argument("--labels", type=Path, help="bootstrap CSV to re-score")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    thresholds = Thresholds.from_config(
        yaml.safe_load(args.config.read_text()) if args.config.exists() else None
    )
    print(f"Thresholds: confidence>={thresholds.confidence} "
          f"squareness>={thresholds.squareness} "
          f"|sep-90|<={thresholds.orthogonality_deg} "
          f"concentration>={thresholds.concentration}")

    if args.labels:
        calibration_paths, tissue_paths = load_from_labels(args.labels)
    elif args.calibration and args.tissue:
        calibration_paths = list(tiff_files(args.calibration))
        tissue_paths = list(tiff_files(args.tissue))
    else:
        parser.error("give --labels, or both --calibration and --tissue")

    rows = score_frames(calibration_paths, True, thresholds)
    rows += score_frames(tissue_paths, False, thresholds)
    raise SystemExit(report(rows))


if __name__ == "__main__":
    main()
