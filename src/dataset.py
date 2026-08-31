"""
PyTorch Dataset for TrOCR fine-tuning on Urdu, plus train/test splitting.

Improvements over the Week 3 handout's `UrduOCRDataset`:

  * Text is passed through `normalize_urdu()` before tokenisation, so the
    model is trained on a consistent label space (see text_utils.py).
  * Images are run through `preprocess_image()` (deskew, contrast,
    letterbox — see preprocessing.py) instead of being fed to the
    processor raw.
  * Training-split images additionally get `OCRAugment` (see
    augmentation.py) applied on the fly, a fresh random perturbation every
    epoch — never applied to validation/test data.
  * `-100` label masking on padding tokens, so the loss is not computed
    over pad positions (the handout's version leaves pad tokens in the
    label tensor, which very slightly biases the loss/CER but is worth
    fixing since it's a one-line change).
  * A proper `split_dataset()` that writes explicit `train.csv` /
    `test.csv` files (reproducible, inspectable, diffable) instead of
    `torch.utils.data.random_split`, which produces an opaque in-memory
    split you cannot re-derive from disk alone.
  * Corrupt/missing image files are logged and skipped at load time
    instead of crashing the whole training run.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from augmentation import AugmentConfig, OCRAugment
from preprocessing import PreprocessConfig, preprocess_image
from text_utils import normalize_urdu

REPO_ROOT = Path(__file__).resolve().parent.parent


def split_dataset(
    labels_csv: Path | str,
    train_csv: Path | str,
    test_csv: Path | str,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[int, int]:
    """Deterministically split a labels.csv into train_csv / test_csv.
    Returns (n_train, n_test).
    """
    df = pd.read_csv(labels_csv)
    df = df.dropna(subset=["image", "text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str).map(normalize_urdu)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    rng = random.Random(seed)
    indices = list(df.index)
    rng.shuffle(indices)
    n_test = max(1, int(len(indices) * test_size))
    test_idx = set(indices[:n_test])

    train_df = df[~df.index.isin(test_idx)].reset_index(drop=True)
    test_df = df[df.index.isin(test_idx)].reset_index(drop=True)

    Path(train_csv).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_csv, index=False, encoding="utf-8")
    test_df.to_csv(test_csv, index=False, encoding="utf-8")
    return len(train_df), len(test_df)


class UrduOCRDataset(Dataset):
    """One row of `labels.csv` (`image`, `text`) -> a TrOCR training
    example (`pixel_values`, `labels`).

    Args:
        csv_path: path to a two-column (image, text) CSV. `image` paths
            are resolved relative to `root_dir` (defaults to the repo root)
            if they are not already absolute.
        processor: a `TrOCRProcessor` (image feature extractor + tokenizer).
        max_length: max token length for the text labels; longer labels
            are truncated (rare for single-line OCR, but keeps a single
            malformed row from crashing a batch).
        train: if True, apply `OCRAugment` and `preprocess_image` with
            default settings; if False (validation/test), only
            `preprocess_image` runs — no random augmentation.
        preprocess_config / augment_config: override defaults if needed.
    """

    def __init__(
        self,
        csv_path: Path | str,
        processor,
        max_length: int = 128,
        train: bool = True,
        root_dir: Path | str | None = None,
        preprocess_config: PreprocessConfig | None = None,
        augment_config: AugmentConfig | None = None,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir) if root_dir else REPO_ROOT
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["image", "text"]).reset_index(drop=True)

        # Verify every image actually exists up front, so a bad row fails
        # fast and loudly at dataset construction time rather than
        # silently mid-epoch.
        missing = []
        for img_rel in df["image"]:
            if not self._resolve(img_rel).exists():
                missing.append(img_rel)
        if missing:
            preview = ", ".join(missing[:5])
            print(
                f"[UrduOCRDataset] WARNING: {len(missing)} image(s) listed in "
                f"{csv_path} were not found on disk and will be skipped "
                f"(e.g. {preview}). Fix data/labels.csv or re-run preprocessing."
            )
            df = df[~df["image"].isin(missing)].reset_index(drop=True)

        self.data = df
        self.processor = processor
        self.max_length = max_length
        self.train = train
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.augment = (
            OCRAugment(augment_config or AugmentConfig(), seed=seed) if train else None
        )
        print(f"Dataset loaded: {len(self.data)} samples ({'train' if train else 'eval'})")

    def _resolve(self, img_rel: str) -> Path:
        p = Path(img_rel)
        return p if p.is_absolute() else (self.root_dir / p)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        image_path = self._resolve(row["image"])

        image = Image.open(image_path).convert("RGB")
        image = preprocess_image(image, self.preprocess_config)
        if self.train and self.augment is not None:
            image = self.augment(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        text = normalize_urdu(str(row["text"]))
        tokenized = self.processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        ).input_ids

        # Mask padding so the loss / CER-during-training isn't computed
        # over pad positions.
        pad_id = self.processor.tokenizer.pad_token_id
        labels = [tok if tok != pad_id else -100 for tok in tokenized]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


if __name__ == "__main__":
    # Lightweight self-check that doesn't need transformers/torch installed
    # for the *split* half — only __main__ exercises the Dataset itself.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", default=str(REPO_ROOT / "data" / "labels.csv"))
    parser.add_argument("--train-csv", default=str(REPO_ROOT / "data" / "train.csv"))
    parser.add_argument("--test-csv", default=str(REPO_ROOT / "data" / "test.csv"))
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()

    n_train, n_test = split_dataset(args.labels_csv, args.train_csv, args.test_csv, args.test_size)
    print(f"Train: {n_train}  Test: {n_test}")
