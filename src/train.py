"""
Fine-tune TrOCR on Urdu — the Week 4 training step, redone for accuracy.

What changes vs. the handout's Cell 1-3, and why each one matters:

  * **Seq2SeqTrainer instead of a hand-rolled loop.** The handout's manual
    loop never evaluates with `generate()` during training, so you cannot
    see CER/WER improve (or plateau, or overfit) until the very end. Using
    `Seq2SeqTrainer` with `predict_with_generate=True` gets you a real
    CER/WER curve after every epoch, checkpointing on the best one, and
    early stopping — all things you need to actually tell whether training
    is working.
  * **Cosine LR schedule with warmup**, not a single flat `lr=5e-5` for
    the whole run. A short warmup avoids destabilising the pretrained
    encoder weights on step 1; cosine decay lets the model settle into a
    minimum instead of bouncing around one at a constant, relatively high
    LR for all 3 epochs.
  * **More epochs (default 12) with early stopping on CER**, not a fixed
    3. 3 epochs is rarely enough to adapt an English-pretrained decoder to
    Urdu; early stopping means you still don't overfit or waste compute —
    training stops itself once CER stops improving.
  * **Mixed precision (fp16) + gradient accumulation** so you can use a
    larger *effective* batch size than 4 on a single T4/Colab GPU without
    running out of memory, which stabilises gradients.
  * **`predict_with_generate` + beam search at eval time** (`num_beams=4`
    by default) — greedy decoding (what `model.generate()` does with no
    arguments, as in the handout) is measurably worse than a small beam
    search for seq2seq OCR.
  * Trains on `UrduOCRDataset` (see dataset.py), which applies text
    normalisation, image preprocessing, and train-time augmentation.

Usage (from the repo root, or from `src/` — paths are resolved relative to
the repo root by default):

    python src/train.py \
        --train-csv data/train.csv --test-csv data/test.csv \
        --output-dir models/urdu-trocr \
        --epochs 12 --batch-size 8

Colab tip: put `--output-dir /content/drive/MyDrive/SI26-urdu-ocr-model`
directly so checkpoints survive a disconnect, instead of copying at the end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from dataset import UrduOCRDataset, split_dataset
from metrics import build_seq2seq_compute_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_model_and_processor(base_model: str, device: str):
    processor = TrOCRProcessor.from_pretrained(base_model)
    model = VisionEncoderDecoderModel.from_pretrained(base_model)
    model.to(device)

    # Required generation config wiring (same as the handout, kept as-is
    # since it's correct) plus beam-search defaults used at eval/inference.
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 1.0
    model.config.num_beams = 4
    return model, processor


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on Urdu.")
    parser.add_argument("--labels-csv", default=str(REPO_ROOT / "data" / "labels.csv"),
                         help="Used only if --train-csv/--test-csv don't already exist.")
    parser.add_argument("--train-csv", default=str(REPO_ROOT / "data" / "train.csv"))
    parser.add_argument("--test-csv", default=str(REPO_ROOT / "data" / "test.csv"))
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--base-model", default="microsoft/trocr-base-printed")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "models" / "urdu-trocr"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2,
                         help="Effective batch size = batch_size * grad_accum.")
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--fp16", action="store_true", default=None,
                         help="Default: auto (on if CUDA available).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected. Training will be very slow — "
              "Runtime > Change runtime type > GPU in Colab.")
    fp16 = args.fp16 if args.fp16 is not None else (device == "cuda")

    if not Path(args.train_csv).exists() or not Path(args.test_csv).exists():
        print(f"{args.train_csv} / {args.test_csv} not found — splitting from {args.labels_csv}")
        n_train, n_test = split_dataset(
            args.labels_csv, args.train_csv, args.test_csv, args.test_size, args.seed
        )
        print(f"Split: {n_train} train / {n_test} test")

    model, processor = build_model_and_processor(args.base_model, device)

    train_dataset = UrduOCRDataset(args.train_csv, processor, train=True, seed=args.seed)
    eval_dataset = UrduOCRDataset(args.test_csv, processor, train=False)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        fp16=fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to=["none"],
        seed=args.seed,
    )

    # No custom data_collator / processing_class needed: UrduOCRDataset
    # already returns fixed-shape tensors (pixel_values from the feature
    # extractor's fixed square size, labels padded to --max_length with
    # -100 masking), so Trainer's default collator (a plain per-key
    # torch.stack) is exactly right and avoids the tokenizer-oriented
    # collators trying to reinterpret pixel_values as token sequences.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_seq2seq_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"Best model saved to: {final_dir}")


if __name__ == "__main__":
    main()
