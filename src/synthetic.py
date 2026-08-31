"""
Synthetic Urdu OCR data generator.

The Week 1 handout ships a 5-line demo that renders plain black-on-white
text with one font. That is fine as a proof of concept, but a model
fine-tuned on 5-20 near-identical images will not generalise — and hand
photographing 100-200 real images (the Week 1 / Week 3 ask) is slow and
gives you very little control over *variety*, which the handout itself
calls out as the thing that makes a model robust.

This module turns a modest, hand-checked sentence corpus
(`data/corpus/urdu_sentences.txt`) into a large, *varied* synthetic
training set by rendering every sentence in:

  * multiple fonts (Nastaliq / Naskh / Sans — different Urdu writing
    styles, see assets/fonts/),
  * multiple sizes,
  * multiple background styles (clean white, cream "newspaper", light
    grey "scan", subtle paper grain),
  * multiple ink colours (black, dark grey, dark blue — printers vary),
  * small random rotation, gaussian noise, blur and JPEG-recompression
    artifacts, which mimic a phone photo of a page far better than a
    pristine render.

It can also fabricate extra "pseudo-sentences" by recombining words
pulled from the real corpus, purely to widen glyph/ligature coverage
(documented, and kept a minority of the set — see `pseudo_fraction`).
Real, meaningful sentences always form the backbone of the dataset.

This is not a replacement for real photographed data — the Week 1/2/3
handouts are right that a model trained only on clean renders will
struggle on messy real photos. Use this to *bootstrap* a dataset that is
large enough to fine-tune on, then mix in every real image you can
collect (see `merge_labels_csv` below) — real + synthetic together is
what gets you a genuinely accurate model. See docs/ACCURACY_NOTES.md.
"""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

# Pillow can shape Arabic/Urdu script two ways:
#   1. Native complex-text layout via libraqm (Layout.RAQM) — this is the
#      *correct* way: it performs real OpenType contextual shaping, so
#      Nastaliq's dramatic diagonal stacking and letter joining render
#      properly (verified against all three bundled fonts).
#   2. A manual pre-shaping fallback using `arabic_reshaper` + `python-bidi`
#      to substitute presentation-form glyphs and reverse run order before
#      handing plain LTR-laid-out text to Pillow. This works for simpler
#      Naskh/Sans styles but visibly breaks Nastaliq (raqm-less Pillow has
#      no engine capable of the diagonal joins Nastaliq needs).
# We use (1) whenever libraqm is present (the normal case — it ships with
# modern Pillow wheels) and fall back to (2) only if it is genuinely
# unavailable, so a Colab/HF Spaces environment without libraqm still runs.
HAS_RAQM = features.check("raqm")

if not HAS_RAQM:  # pragma: no cover - exercised only on raqm-less installs
    import arabic_reshaper
    from bidi.algorithm import get_display

from text_utils import normalize_urdu

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FONTS_DIR = REPO_ROOT / "assets" / "fonts"
DEFAULT_CORPUS_PATH = REPO_ROOT / "data" / "corpus" / "urdu_sentences.txt"

# Background "paper" styles: (name, base RGB colour)
BACKGROUND_STYLES = [
    ("white", (255, 255, 255)),
    ("cream", (250, 240, 217)),      # newsprint / book paper
    ("light_gray", (235, 235, 235)),  # flatbed scan
    ("warm_white", (248, 246, 240)),
]

INK_COLORS = [
    (15, 15, 15),      # near-black print ink
    (40, 40, 45),       # dark grey ink
    (20, 25, 60),       # dark blue ("blue biro" signage)
]


# --------------------------------------------------------------------------
# Corpus / word bank
# --------------------------------------------------------------------------

def load_corpus(path: Path | str = DEFAULT_CORPUS_PATH) -> list[str]:
    """Read the curated sentence corpus, skipping blanks and comments."""
    path = Path(path)
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(normalize_urdu(line))
    return lines


def build_word_bank(sentences: Iterable[str]) -> list[str]:
    words: set[str] = set()
    for s in sentences:
        for w in s.split():
            if len(w) >= 2:
                words.add(w)
    return sorted(words)


def generate_pseudo_sentences(
    word_bank: list[str], n: int, rng: random.Random,
    min_words: int = 3, max_words: int = 9,
) -> list[str]:
    """Recombine real Urdu words into new word sequences.

    These are for *visual* coverage only (more ligature / word-shape
    combinations for the vision encoder to see) — they are not claimed to
    be grammatical sentences, and callers should keep them a minority of
    the final dataset. See module docstring.
    """
    out = []
    for _ in range(n):
        k = rng.randint(min_words, max_words)
        words = [rng.choice(word_bank) for _ in range(k)]
        out.append(" ".join(words))
    return out


# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------

def discover_fonts(fonts_dir: Path | str = DEFAULT_FONTS_DIR) -> list[Path]:
    fonts_dir = Path(fonts_dir)
    fonts = sorted(
        [p for p in fonts_dir.glob("*") if p.suffix.lower() in (".ttf", ".otf")]
    )
    if not fonts:
        raise FileNotFoundError(
            f"No .ttf/.otf fonts found in {fonts_dir}. "
            "Add at least one Urdu font (see assets/fonts/FONTS_LICENSE.txt)."
        )
    return fonts


def shape_for_render(text: str) -> str:
    """Fallback-only reshape + bidi-reorder for raqm-less Pillow installs.
    Not used when libraqm is available (see HAS_RAQM above) — with raqm,
    plain logical-order text plus direction='rtl' is drawn directly and
    shapes/joins correctly, Nastaliq included.
    """
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

@dataclass
class RenderStyle:
    font_path: Path
    font_size: int
    bg_color: tuple[int, int, int]
    ink_color: tuple[int, int, int]
    pad_x: int
    pad_y: int
    rotation_deg: float
    noise_sigma: float
    blur_radius: float
    jpeg_quality: int | None  # None = no recompression artifact


def random_style(rng: random.Random, fonts: list[Path]) -> RenderStyle:
    bg_name, bg_color = rng.choice(BACKGROUND_STYLES)
    return RenderStyle(
        font_path=rng.choice(fonts),
        font_size=rng.randint(28, 56),
        bg_color=bg_color,
        ink_color=rng.choice(INK_COLORS),
        pad_x=rng.randint(14, 40),
        pad_y=rng.randint(10, 26),
        rotation_deg=rng.uniform(-2.5, 2.5),
        noise_sigma=rng.uniform(0, 6),
        blur_radius=rng.choice([0, 0, 0, 0.4, 0.8]),
        jpeg_quality=rng.choice([None, None, 85, 70]),
    )


def _add_paper_grain(img: Image.Image, rng: random.Random, strength: float = 4.0) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    grain = rng_normal(rng, arr.shape[:2], strength)
    for c in range(arr.shape[2]):
        arr[..., c] += grain
    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def rng_normal(rng: random.Random, shape, sigma: float) -> np.ndarray:
    # random.Random has no gaussian-array method; use numpy seeded off it.
    seed = rng.randint(0, 2**31 - 1)
    local = np.random.default_rng(seed)
    return local.normal(0, sigma, size=shape)


def render_line_image(text: str, style: RenderStyle, rng: random.Random) -> Image.Image:
    """Render one line of (already-normalised) Urdu text to a PIL image."""
    if HAS_RAQM:
        # Native shaping: plain logical-order text, let raqm do contextual
        # joining + bidi reordering. text_kwargs applied to both the
        # measuring textbbox() call and the actual draw() call so they agree.
        render_text = text
        font = ImageFont.truetype(
            str(style.font_path), style.font_size, layout_engine=ImageFont.Layout.RAQM
        )
        text_kwargs = {"direction": "rtl", "language": "ur"}
    else:  # pragma: no cover
        render_text = shape_for_render(text)
        font = ImageFont.truetype(str(style.font_path), style.font_size)
        text_kwargs = {}

    # Measure text with a scratch image first.
    scratch = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), render_text, font=font, **text_kwargs)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    width = text_w + 2 * style.pad_x
    height = text_h + 2 * style.pad_y

    img = Image.new("RGB", (max(width, 20), max(height, 20)), color=style.bg_color)
    img = _add_paper_grain(img, rng, strength=2.0)
    draw = ImageDraw.Draw(img)
    # anchor at the measured bbox origin so padding is symmetric
    draw.text(
        (style.pad_x - bbox[0], style.pad_y - bbox[1]),
        render_text,
        font=font,
        fill=style.ink_color,
        **text_kwargs,
    )

    # Slight rotation, filled with background colour to simulate a
    # slightly skewed photo/scan rather than a perfectly axis-aligned crop.
    if abs(style.rotation_deg) > 0.05:
        img = img.rotate(
            style.rotation_deg,
            expand=True,
            fillcolor=style.bg_color,
            resample=Image.BICUBIC,
        )

    if style.blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(style.blur_radius))

    if style.noise_sigma > 0:
        img = _add_paper_grain(img, rng, strength=style.noise_sigma)

    if style.jpeg_quality is not None:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=style.jpeg_quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img


# --------------------------------------------------------------------------
# Dataset generation
# --------------------------------------------------------------------------

@dataclass
class GenerationStats:
    total: int = 0
    from_corpus: int = 0
    from_pseudo: int = 0
    fonts_used: list[str] = field(default_factory=list)


def generate_dataset(
    output_dir: Path | str,
    labels_csv: Path | str,
    num_images: int = 600,
    corpus_path: Path | str = DEFAULT_CORPUS_PATH,
    fonts_dir: Path | str = DEFAULT_FONTS_DIR,
    pseudo_fraction: float = 0.35,
    repeats_per_sentence: int | None = None,
    seed: int = 42,
    image_prefix: str = "synth",
) -> GenerationStats:
    """Generate a synthetic Urdu OCR dataset.

    Args:
        output_dir: folder to write PNG images into (created if missing).
        labels_csv: path to write the (image, text) label file to. Uses
            the same two-column ``image,text`` schema as the Week 1
            handout's `data/labels.csv`, so it merges directly with real
            hand-labelled rows (see `merge_labels_csv`).
        num_images: total number of synthetic images to generate.
        pseudo_fraction: fraction of `num_images` drawn from recombined
            pseudo-sentences rather than the real curated corpus (kept a
            minority by default — real sentences are more valuable).
        repeats_per_sentence: if set, cycle through the corpus this many
            times (each pass re-rendered with a fresh random style) instead
            of computing repeats from `num_images`. Useful for small,
            deterministic test runs.
        seed: RNG seed, for reproducibility.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = Path(labels_csv)
    labels_csv.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    fonts = discover_fonts(fonts_dir)
    corpus = load_corpus(corpus_path)
    if not corpus:
        raise ValueError(f"Corpus at {corpus_path} is empty.")
    word_bank = build_word_bank(corpus)

    n_pseudo = int(num_images * pseudo_fraction)
    n_real = num_images - n_pseudo

    real_lines: list[str] = []
    if repeats_per_sentence is not None:
        for _ in range(repeats_per_sentence):
            real_lines.extend(corpus)
    else:
        for i in range(n_real):
            real_lines.append(corpus[i % len(corpus)])
        rng.shuffle(real_lines)

    pseudo_lines = generate_pseudo_sentences(word_bank, n_pseudo, rng)

    all_lines = [(t, "corpus") for t in real_lines] + [(t, "pseudo") for t in pseudo_lines]
    rng.shuffle(all_lines)

    stats = GenerationStats()
    rows = []
    fonts_used: set[str] = set()

    for idx, (text, source) in enumerate(all_lines, start=1):
        style = random_style(rng, fonts)
        fonts_used.add(style.font_path.name)
        img = render_line_image(text, style, rng)

        fname = f"{image_prefix}_{idx:05d}.png"
        rel_path = f"data/raw/synthetic/{fname}"
        (output_dir / fname).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_dir / fname)

        rows.append({"image": rel_path, "text": text})
        stats.total += 1
        if source == "corpus":
            stats.from_corpus += 1
        else:
            stats.from_pseudo += 1

    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "text"])
        writer.writeheader()
        writer.writerows(rows)

    stats.fonts_used = sorted(fonts_used)
    return stats


def merge_labels_csv(csv_paths: list[Path | str], output_path: Path | str) -> int:
    """Concatenate several labels.csv files (e.g. real + synthetic) into
    one, de-duplicating on the `image` column. Returns row count written.
    """
    seen: dict[str, str] = {}
    for p in csv_paths:
        p = Path(p)
        if not p.exists():
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen[row["image"]] = normalize_urdu(row["text"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "text"])
        writer.writeheader()
        for image, text in seen.items():
            writer.writerow({"image": image, "text": text})
    return len(seen)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic Urdu OCR data.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data" / "raw" / "synthetic"))
    parser.add_argument("--labels-csv", default=str(REPO_ROOT / "data" / "labels_synthetic.csv"))
    parser.add_argument("--num-images", type=int, default=600)
    parser.add_argument("--pseudo-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stats = generate_dataset(
        output_dir=args.output_dir,
        labels_csv=args.labels_csv,
        num_images=args.num_images,
        pseudo_fraction=args.pseudo_fraction,
        seed=args.seed,
    )
    print(f"Generated {stats.total} images "
          f"({stats.from_corpus} corpus + {stats.from_pseudo} pseudo)")
    print(f"Fonts used: {', '.join(stats.fonts_used)}")
    print(f"Labels written to: {args.labels_csv}")
