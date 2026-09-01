"""Dataset tests use a lightweight stub in place of `TrOCRProcessor` so
this suite runs without downloading model weights or needing `torch`'s
full transformers stack to hit the network (see conftest.py / CI config).
The stub matches the exact interface `dataset.py` calls:
`processor(image, return_tensors='pt').pixel_values` and
`processor.tokenizer(text, padding=..., truncation=..., max_length=...).input_ids`.
"""

import csv

import torch

from dataset import UrduOCRDataset, split_dataset


class _FakeTokenizer:
    pad_token_id = 1
    cls_token_id = 0
    sep_token_id = 2

    def __call__(self, text, padding="max_length", truncation=True, max_length=128):
        ids = [ord(c) % 500 + 10 for c in text][:max_length]
        ids = ids + [self.pad_token_id] * (max_length - len(ids))

        class _Result:
            pass

        r = _Result()
        r.input_ids = ids
        return r


class _FakeFeatureOutput:
    def __init__(self, pixel_values):
        self.pixel_values = pixel_values


class FakeProcessor:
    """Mimics TrOCRProcessor's call signature with a fixed 384x384 square
    output, matching what TrOCR's real feature extractor produces."""

    tokenizer = _FakeTokenizer()

    def __call__(self, image, return_tensors="pt"):
        return _FakeFeatureOutput(torch.rand(1, 3, 384, 384))


def _write_labels_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "text"])
        w.writeheader()
        w.writerows(rows)


def _make_dataset_fixture(tmp_path, n=20):
    """Creates n tiny real PNG files + a labels.csv pointing at them."""
    import numpy as np
    from PIL import Image

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rows = []
    for i in range(n):
        arr = np.full((40, 120, 3), 250, dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"img_{i}.png")
        rows.append({"image": f"imgs/img_{i}.png", "text": f"جملہ نمبر {i}"})

    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, rows)
    return labels_csv


def test_split_dataset_covers_all_rows_without_overlap(tmp_path):
    labels_csv = _make_dataset_fixture(tmp_path, n=40)
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"

    n_train, n_test = split_dataset(labels_csv, train_csv, test_csv, test_size=0.25, seed=1)

    assert n_train + n_test == 40
    assert n_test == 10  # 25% of 40

    with open(train_csv, newline="", encoding="utf-8") as f:
        train_images = {r["image"] for r in csv.DictReader(f)}
    with open(test_csv, newline="", encoding="utf-8") as f:
        test_images = {r["image"] for r in csv.DictReader(f)}

    assert len(train_images & test_images) == 0
    assert len(train_images) + len(test_images) == 40


def test_split_dataset_deterministic_given_seed(tmp_path):
    labels_csv = _make_dataset_fixture(tmp_path, n=30)
    a_train, a_test = tmp_path / "a_train.csv", tmp_path / "a_test.csv"
    b_train, b_test = tmp_path / "b_train.csv", tmp_path / "b_test.csv"

    split_dataset(labels_csv, a_train, a_test, test_size=0.2, seed=99)
    split_dataset(labels_csv, b_train, b_test, test_size=0.2, seed=99)

    assert a_train.read_text() == b_train.read_text()
    assert a_test.read_text() == b_test.read_text()


def test_urdu_ocr_dataset_getitem_shapes_and_masking(tmp_path):
    labels_csv = _make_dataset_fixture(tmp_path, n=5)
    ds = UrduOCRDataset(labels_csv, FakeProcessor(), train=True, root_dir=tmp_path, seed=0)

    assert len(ds) == 5
    sample = ds[0]
    assert sample["pixel_values"].shape == (3, 384, 384)
    assert sample["labels"].shape == (128,)
    assert sample["labels"].dtype == torch.long
    # padding must be masked to -100 so it's excluded from the loss
    assert (sample["labels"] == -100).any()


def test_urdu_ocr_dataset_train_vs_eval_augmentation(tmp_path):
    labels_csv = _make_dataset_fixture(tmp_path, n=3)
    train_ds = UrduOCRDataset(labels_csv, FakeProcessor(), train=True, root_dir=tmp_path)
    eval_ds = UrduOCRDataset(labels_csv, FakeProcessor(), train=False, root_dir=tmp_path)

    assert train_ds.augment is not None
    assert eval_ds.augment is None


def test_urdu_ocr_dataset_skips_missing_images(tmp_path, capsys):
    labels_csv = _make_dataset_fixture(tmp_path, n=3)
    rows = list(csv.DictReader(open(labels_csv, encoding="utf-8")))
    rows.append({"image": "imgs/does_not_exist.png", "text": "غائب تصویر"})
    _write_labels_csv(labels_csv, rows)

    ds = UrduOCRDataset(labels_csv, FakeProcessor(), train=False, root_dir=tmp_path)

    assert len(ds) == 3  # the missing row was dropped, not crashed on
    captured = capsys.readouterr()
    assert "does_not_exist.png" in captured.out or "WARNING" in captured.out
