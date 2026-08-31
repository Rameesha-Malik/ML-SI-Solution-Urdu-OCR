"""
Evaluation metrics for the Urdu OCR pipeline.

The Week 4 handout scores the model with **exact full-string match only**:

    if pred.strip() == actual.strip(): correct += 1

That is a very harsh, low-information metric for OCR. A prediction that
gets 19 out of 20 characters right scores exactly the same (0) as one that
gets 0 out of 20 right — you cannot tell "almost there" from "completely
wrong", so you cannot tell whether an accuracy change between two training
runs is real progress or noise. It also means "accuracy" looks
artificially low even for a genuinely useful model, since any single
missed diacritic across the whole string zeroes the sample.

The standard OCR/ASR metrics are:

  * **CER** (Character Error Rate) — edit distance between predicted and
    reference text, divided by reference length, at the *character*
    level. This is the primary metric to optimise for OCR; lower is
    better; a well fine-tuned line-OCR model on printed text typically
    reaches single-digit CER.
  * **WER** (Word Error Rate) — same idea at the *word* level. Useful as
    a secondary/complementary number.

We report all three (CER, WER, and exact-match accuracy) so the exact-
match number stays comparable to the handout's baseline while CER/WER
give the granular signal you actually need to tell whether training is
working. All text is passed through `normalize_urdu()` first so cosmetic
Unicode differences never get counted as errors (see text_utils.py).
"""

from __future__ import annotations

import jiwer

from text_utils import normalize_urdu


def _prep(texts: list[str]) -> list[str]:
    return [normalize_urdu(t) for t in texts]


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """Character Error Rate in [0, +inf) (can exceed 1.0 on very bad
    predictions with many insertions). Returns 0.0 for an empty batch.
    """
    predictions, references = _prep(predictions), _prep(references)
    non_empty = [(p, r) for p, r in zip(predictions, references) if r]
    if not non_empty:
        return 0.0
    preds, refs = zip(*non_empty)
    return jiwer.cer(list(refs), list(preds))


def compute_wer(predictions: list[str], references: list[str]) -> float:
    predictions, references = _prep(predictions), _prep(references)
    non_empty = [(p, r) for p, r in zip(predictions, references) if r]
    if not non_empty:
        return 0.0
    preds, refs = zip(*non_empty)
    return jiwer.wer(list(refs), list(preds))


def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    predictions, references = _prep(predictions), _prep(references)
    if not references:
        return 0.0
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(references)


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """All headline metrics in one call, for logging / reports."""
    return {
        "cer": compute_cer(predictions, references),
        "wer": compute_wer(predictions, references),
        "exact_match_accuracy": compute_exact_match(predictions, references),
        "num_samples": len(references),
    }


def build_seq2seq_compute_metrics(processor):
    """Return a `compute_metrics` callable for HuggingFace's
    Seq2SeqTrainer (used with `predict_with_generate=True`).

    Usage:
        trainer = Seq2SeqTrainer(..., compute_metrics=build_seq2seq_compute_metrics(processor))
    """
    import numpy as np

    def _compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        pred_ids = np.where(pred_ids != -100, pred_ids, processor.tokenizer.pad_token_id)
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        metrics = compute_metrics(pred_str, label_str)
        # Trainer logging expects flat float values.
        metrics.pop("num_samples", None)
        return metrics

    return _compute_metrics


if __name__ == "__main__":
    preds = ["پاکستان زندہ باد", "شکریہ", "غلط جملہ"]
    refs = ["پاکستان زندہ باد", "شکریہ", "صحیح جملہ"]
    print(compute_metrics(preds, refs))
