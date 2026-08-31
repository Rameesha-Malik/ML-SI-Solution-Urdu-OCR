"""
Evaluate a fine-tuned Urdu TrOCR model on a held-out test set.

The Week 4 handout's Cell 4 prints every prediction and a single overall
exact-match percentage. This script keeps that (still useful for a quick
eyeball check) but adds what you actually need to write up Week 4's "note
your accuracy and 3-5 examples where the model got it wrong" deliverable
and the Week 5 README "Results" section:

  * CER / WER (see metrics.py) alongside exact-match accuracy.
  * A per-sample CSV report (image, ground truth, prediction, char error
    rate) you can sort/filter/skim instead of scrolling console output.
  * The worst-N samples by CER printed directly, so "find 3-5 wrong
    examples" is a one-command answer instead of manual scrolling.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from infer import UrduOCR
from metrics import compute_cer, compute_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent


def evaluate(
    model_path: str,
    test_csv: Path | str,
    root_dir: Path | str | None = None,
    num_beams: int = 4,
    batch_size: int = 8,
    report_csv: Path | str | None = None,
) -> dict:
    root_dir = Path(root_dir) if root_dir else REPO_ROOT
    df = pd.read_csv(test_csv).dropna(subset=["image", "text"]).reset_index(drop=True)

    ocr = UrduOCR(model_path, num_beams=num_beams)

    predictions: list[str] = []
    references = df["text"].astype(str).tolist()
    image_paths = df["image"].tolist()

    for start in range(0, len(df), batch_size):
        batch_rel = image_paths[start:start + batch_size]
        batch_abs = [
            p if Path(p).is_absolute() else (root_dir / p) for p in batch_rel
        ]
        preds = ocr.read_batch(batch_abs)
        predictions.extend(preds)
        print(f"  evaluated {min(start + batch_size, len(df))}/{len(df)}")

    per_sample_cer = [
        compute_cer([p], [r]) for p, r in zip(predictions, references)
    ]

    rows = []
    for img, ref, pred, cer in zip(image_paths, references, predictions, per_sample_cer):
        rows.append({
            "image": img,
            "actual": ref,
            "predicted": pred,
            "cer": round(cer, 4),
            "exact_match": ref.strip() == pred.strip(),
        })

    overall = compute_metrics(predictions, references)

    if report_csv:
        report_csv = Path(report_csv)
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(report_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "actual", "predicted", "cer", "exact_match"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-sample report written to: {report_csv}")

    print("\n=== Overall Results ===")
    print(f"Samples:               {overall['num_samples']}")
    print(f"Character Error Rate:  {overall['cer']*100:.2f}%")
    print(f"Word Error Rate:       {overall['wer']*100:.2f}%")
    print(f"Exact-match accuracy:  {overall['exact_match_accuracy']*100:.2f}%")

    worst = sorted(rows, key=lambda r: r["cer"], reverse=True)[:5]
    print("\n=== Worst 5 predictions (highest CER) — use these for your Week 4 write-up ===")
    for r in worst:
        print(f"\nImage:     {r['image']}")
        print(f"Actual:    {r['actual']}")
        print(f"Predicted: {r['predicted']}")
        print(f"CER:       {r['cer']*100:.1f}%")

    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Urdu OCR model.")
    parser.add_argument("model_path", help="Local path or HF Hub repo id of the model.")
    parser.add_argument("--test-csv", default=str(REPO_ROOT / "data" / "test.csv"))
    parser.add_argument("--root-dir", default=str(REPO_ROOT))
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--report-csv", default=str(REPO_ROOT / "reports" / "eval_report.csv"))
    args = parser.parse_args()

    evaluate(
        args.model_path, args.test_csv, args.root_dir,
        args.num_beams, args.batch_size, args.report_csv,
    )
