"""
Week 2, Part B — baseline Tesseract OCR + quantified gap analysis.

The handout's version prints Tesseract's raw output for 5 images and asks
you to eyeball what went wrong. This script does the same job but adds a
number to back up the write-up: it runs Tesseract (`lang='urd'`) over your
whole labelled set and reports the Character Error Rate against your
ground truth, using the exact same CER metric your fine-tuned model will
later be judged on (see src/metrics.py) — so "Tesseract fails on Urdu
because..." in your README can cite a real percentage, and Week 4/5's
"how much better is our model" comparison is apples-to-apples.

Requires the Tesseract binary + Urdu language data, which the handout
already has you install:

    !apt-get install -y tesseract-ocr tesseract-ocr-urd
    !pip install pytesseract

Usage:
    python scripts/tesseract_baseline.py --labels-csv data/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from text_utils import normalize_urdu  # noqa: E402
from metrics import compute_cer, compute_metrics  # noqa: E402


def check_tesseract_available() -> None:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception as e:
        raise SystemExit(
            "Tesseract is not available. Install it first:\n"
            "  !apt-get install -y tesseract-ocr tesseract-ocr-urd\n"
            "  !pip install pytesseract\n"
            f"(original error: {e})"
        )


def run_baseline(labels_csv: Path | str, root_dir: Path | str, sample_limit: int | None,
                  report_csv: Path | str | None):
    import pandas as pd
    import pytesseract
    from PIL import Image

    check_tesseract_available()

    root_dir = Path(root_dir)
    df = pd.read_csv(labels_csv).dropna(subset=["image", "text"]).reset_index(drop=True)
    if sample_limit:
        df = df.head(sample_limit)

    predictions, references, rows = [], [], []
    for _, row in df.iterrows():
        img_path = Path(row["image"])
        if not img_path.is_absolute():
            img_path = root_dir / img_path
        if not img_path.exists():
            print(f"  [skip] missing image: {img_path}")
            continue

        actual = normalize_urdu(str(row["text"]))
        try:
            raw_pred = pytesseract.image_to_string(Image.open(img_path), lang="urd")
        except Exception as e:
            print(f"  [error] {img_path}: {e}")
            raw_pred = ""
        pred = normalize_urdu(raw_pred)

        predictions.append(pred)
        references.append(actual)
        rows.append({
            "image": str(row["image"]),
            "actual": actual,
            "tesseract_output": pred,
            "cer": round(compute_cer([pred], [actual]), 4),
        })

    metrics = compute_metrics(predictions, references)
    print("\n=== Tesseract Urdu Baseline ===")
    print(f"Samples:              {metrics['num_samples']}")
    print(f"Character Error Rate: {metrics['cer']*100:.2f}%")
    print(f"Word Error Rate:      {metrics['wer']*100:.2f}%")
    print(f"Exact-match accuracy: {metrics['exact_match_accuracy']*100:.2f}%")
    print(
        "\nA high CER here is expected and is *the point* of this step — it "
        "quantifies why a fine-tuned model is worth building. Compare this "
        "number directly against your fine-tuned model's CER from "
        "src/evaluate.py in your README's Results section."
    )

    if report_csv:
        report_csv = Path(report_csv)
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(report_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "actual", "tesseract_output", "cer"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-sample report written to: {report_csv}")

    worst = sorted(rows, key=lambda r: r["cer"], reverse=True)[:5]
    print("\n=== 5 examples for your 'gap analysis' write-up ===")
    for r in worst:
        print(f"\nImage:              {r['image']}")
        print(f"Actual text:        {r['actual']}")
        print(f"Tesseract output:   {r['tesseract_output']}")
        print(f"CER:                {r['cer']*100:.1f}%")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Tesseract Urdu OCR gap analysis.")
    parser.add_argument("--labels-csv", default=str(REPO_ROOT / "data" / "labels.csv"))
    parser.add_argument("--root-dir", default=str(REPO_ROOT))
    parser.add_argument("--sample-limit", type=int, default=None,
                         help="Only test the first N rows (handout default is 5).")
    parser.add_argument("--report-csv", default=str(REPO_ROOT / "reports" / "tesseract_baseline.csv"))
    args = parser.parse_args()

    run_baseline(args.labels_csv, args.root_dir, args.sample_limit, args.report_csv)
