import random

import numpy as np
import pytest
from PIL import Image

from synthetic import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_FONTS_DIR,
    HAS_RAQM,
    build_word_bank,
    discover_fonts,
    generate_dataset,
    generate_pseudo_sentences,
    load_corpus,
    merge_labels_csv,
    random_style,
    render_line_image,
)


def test_discover_fonts_finds_bundled_fonts():
    fonts = discover_fonts(DEFAULT_FONTS_DIR)
    assert len(fonts) >= 3
    names = {f.name for f in fonts}
    assert any("Nastaliq" in n for n in names)
    assert any("Naskh" in n for n in names)


def test_load_corpus_has_real_sentences():
    corpus = load_corpus(DEFAULT_CORPUS_PATH)
    assert len(corpus) >= 50
    # every line should actually contain Urdu/Arabic-block characters
    assert all(any("؀" <= c <= "ۿ" for c in line) for line in corpus)


def test_build_word_bank_and_pseudo_sentences():
    corpus = load_corpus(DEFAULT_CORPUS_PATH)
    bank = build_word_bank(corpus)
    assert len(bank) > 50

    rng = random.Random(0)
    pseudo = generate_pseudo_sentences(bank, n=10, rng=rng, min_words=3, max_words=6)
    assert len(pseudo) == 10
    for sentence in pseudo:
        words = sentence.split()
        assert 3 <= len(words) <= 6
        assert all(w in bank for w in words)


@pytest.mark.skipif(not HAS_RAQM, reason="raqm-specific rendering path")
@pytest.mark.parametrize("font_path", discover_fonts(DEFAULT_FONTS_DIR))
def test_render_line_image_draws_visible_content_for_every_font(font_path):
    """Regression test: earlier versions of this pipeline rendered Nastaliq
    as blank 'tofu' boxes under Pillow's non-raqm layout engine. This
    checks every bundled font actually draws non-background pixels."""
    rng = random.Random(1)
    style = random_style(rng, [font_path])
    img = render_line_image("پاکستان زندہ باد", style, rng)

    assert img.mode == "RGB"
    assert img.width > 20 and img.height > 20

    arr = np.asarray(img).astype(np.int16)
    bg = np.array(style.bg_color)
    diff = np.abs(arr - bg).sum(axis=-1)
    ink_pixel_fraction = (diff > 40).mean()

    # A rendered sentence should paint a meaningful fraction of the canvas
    # (letters aren't the whole image, but empty/failed renders paint ~0%).
    assert ink_pixel_fraction > 0.02, (
        f"font {font_path.name} rendered almost no visible ink "
        f"({ink_pixel_fraction:.4f} of pixels) -- likely a shaping failure"
    )


def test_generate_dataset_end_to_end(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    out_dir = tmp_path / "img"

    stats = generate_dataset(
        output_dir=out_dir,
        labels_csv=labels_csv,
        num_images=10,
        pseudo_fraction=0.3,
        seed=7,
    )

    assert stats.total == 10
    assert stats.from_corpus + stats.from_pseudo == 10
    assert len(stats.fonts_used) >= 1

    import csv
    with open(labels_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10
    for row in rows:
        assert row["text"].strip() != ""
        # image files are written directly under out_dir, addressed via a
        # repo-relative path in the CSV
        img_name = row["image"].split("/")[-1]
        assert (out_dir / img_name).exists()
        img = Image.open(out_dir / img_name)
        assert img.width > 0 and img.height > 0


def test_merge_labels_csv_dedupes_on_image(tmp_path):
    import csv

    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    out = tmp_path / "merged.csv"

    with open(csv_a, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "text"])
        w.writeheader()
        w.writerow({"image": "x.png", "text": "پہلا"})
        w.writerow({"image": "y.png", "text": "دوسرا"})

    with open(csv_b, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "text"])
        w.writeheader()
        w.writerow({"image": "x.png", "text": "اپڈیٹ شدہ"})  # overrides a.csv's x.png
        w.writerow({"image": "z.png", "text": "تیسرا"})

    n = merge_labels_csv([csv_a, csv_b], out)
    assert n == 3

    with open(out, newline="", encoding="utf-8") as f:
        rows = {r["image"]: r["text"] for r in csv.DictReader(f)}
    assert rows["x.png"] == "اپڈیٹ شدہ"
    assert rows["y.png"] == "دوسرا"
    assert rows["z.png"] == "تیسرا"
