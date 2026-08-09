"""Auto-label a folder of frames and summarize the detection separation.

Usage:
    python run_bootstrap.py PATH/TO/FRAMES [--out labels.csv]

The separation margin printed at the end -- best calibration score minus best
tissue score -- is the number to watch. If it narrows, the FFT rule is no
longer cleanly separating the two classes on this data and the thresholds in
`config.yaml` need re-validating.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.bootstrap import label_folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="folder to scan recursively")
    parser.add_argument("--out", type=Path, default=Path("labels.csv"))
    args = parser.parse_args()

    rows = label_folder(args.root, args.out)
    if not rows:
        print(f"No TIFFs found under {args.root}")
        return

    failed = [row for row in rows if row["error"]]
    scored = [row for row in rows if not row["error"]]
    calibration = [row for row in scored if row["is_calibration"]]
    tissue = [row for row in scored if not row["is_calibration"]]

    print(f"Scanned {len(rows)} frames")
    print(f"  calibration: {len(calibration)}")
    print(f"  tissue:      {len(tissue)}")
    if failed:
        print(f"  unreadable:  {len(failed)}")
        for row in failed[:5]:
            print(f"    {row['frame']}: {row['error']}")

    print("\nDetected calibration frames:")
    for row in sorted(calibration, key=lambda r: r["score"], reverse=True):
        measurable = "" if row["measurable"] else "   [NOT MEASURABLE]"
        print(f"  {row['frame']:28s} score={row['score']:.3f}  "
              f"D={row['pixels_per_space']:.2f}px  {row['nm_per_pixel']:.4f} nm/px{measurable}")

    print("\nTop 8 frames by score (separation check):")
    for row in sorted(scored, key=lambda r: r["score"], reverse=True)[:8]:
        flag = "CAL " if row["is_calibration"] else "    "
        print(f"  {flag}{row['frame']:28s} score={row['score']:.3f}  "
              f"conf={row['confidence']:.3f}  sq={row['squareness']:.3f}  "
              f"sep={row['axis_separation_deg']:.1f}")

    if calibration and tissue:
        best_calibration = max(row["score"] for row in calibration)
        best_tissue = max(row["score"] for row in tissue)
        print(f"\nSeparation margin: {best_calibration:.3f} (calibration) "
              f"- {best_tissue:.3f} (tissue) = {best_calibration - best_tissue:.3f}")
        if best_calibration - best_tissue < 0.2:
            print("  WARNING: margin is narrow; re-validate the detection thresholds.")

    print(f"\nLabels written to {args.out}")


if __name__ == "__main__":
    main()
