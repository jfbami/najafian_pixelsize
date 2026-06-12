"""Auto-label all local test frames and summarize the detection separation."""

from pathlib import Path

from src.bootstrap import label_folder

DATA_ROOT = Path(r"C:\Users\jfbaa\OneDrive\Documents\test\17E00231")
OUTPUT_CSV = Path(r"C:\Users\jfbaa\OneDrive\Documents\test\calibration_agent\labels.csv")


def main() -> None:
    rows = label_folder(DATA_ROOT, OUTPUT_CSV)
    calibration = [r for r in rows if r["is_calibration"]]
    tissue = [r for r in rows if not r["is_calibration"]]

    print(f"Scanned {len(rows)} frames")
    print(f"  calibration: {len(calibration)}")
    print(f"  tissue:      {len(tissue)}")

    print("\nDetected calibration frames:")
    for row in sorted(calibration, key=lambda r: r["score"], reverse=True):
        print(f"  {row['frame']:28s} score={row['score']:.3f}  D={row['pixels_per_space']:.2f}px  {row['nm_per_pixel']:.4f} nm/px")

    print("\nTop 8 frames by score (separation check):")
    for row in sorted(rows, key=lambda r: r["score"], reverse=True)[:8]:
        flag = "CAL " if row["is_calibration"] else "    "
        print(f"  {flag}{row['frame']:28s} score={row['score']:.3f}  conf={row['confidence']:.3f}  sq={row['squareness']:.3f}  sep={row['axis_separation_deg']:.1f}")

    print(f"\nHighest-scoring tissue frame:")
    if tissue:
        worst = max(tissue, key=lambda r: r["score"])
        print(f"  {worst['frame']:28s} score={worst['score']:.3f}  conf={worst['confidence']:.3f}  sq={worst['squareness']:.3f}  sep={worst['axis_separation_deg']:.1f}")

    print(f"\nLabels written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
