Urdu OCR Project | Code Saviours SI-26 | Rameesha Malik

# Urdu OCR

[![Tests](https://github.com/Rameesha-Malik/ML-SI-Solution-Urdu-OCR/actions/workflows/tests.yml/badge.svg)](https://github.com/Rameesha-Malik/ML-SI-Solution-Urdu-OCR/actions/workflows/tests.yml)

A fine-tuned TrOCR model that extracts text from images of Urdu — printed
pages, signage, screenshots, and photographed documents — built end to end
(data collection, preprocessing, training, evaluation, and deployment)
during the Code Saviours ML/AI Internship, Batch SI-26.

> **Status:** this repo ships a complete, tested pipeline (data generation,
> preprocessing, training, evaluation, deployment) ready to run end to end.
> The **Live demo**, **Results**, and dataset-size numbers below are filled
> in once a model has actually been trained on real+synthetic data and
> deployed — see [How to run it locally](#how-to-run-it-locally) to produce
> them. Everything marked `[ FILL IN … ]` is where those run-specific
> numbers/links go.

---

## Table of contents

1. [What problem this solves and why it matters](#what-problem-this-solves-and-why-it-matters)
2. [How it works](#how-it-works)
3. [Live demo](#live-demo)
4. [How to run it locally](#how-to-run-it-locally)
5. [Dataset details](#dataset-details)
6. [Results](#results)
7. [Why We Need a Better Model (Tesseract gap analysis)](#why-we-need-a-better-model-tesseract-gap-analysis)
8. [Testing](#testing)
9. [Project structure](#project-structure)
10. [Credit](#credit)

---

## What problem this solves and why it matters

Optical Character Recognition (OCR) turns an image containing text — a
scanned page, a photo of a signboard, a screenshot — into machine-readable,
editable text. It's a solved problem for English printed text, but **Urdu
OCR is meaningfully harder**: Urdu is normally written in the Nastaliq
calligraphic style, where letters change shape depending on their position
in a word and are frequently stacked diagonally instead of sitting on one
flat baseline the way Latin letters do, so there's no single consistent
"line" to segment. Many letters are distinguished only by the number and
placement of small dots, several different Unicode code points render as
visually identical letters, and — unlike English — large, clean, publicly
available labelled Urdu OCR datasets are rare. General-purpose OCR engines
like Tesseract were never trained on enough Urdu data to do this well (see
the [gap analysis](#why-we-need-a-better-model-tesseract-gap-analysis)
below for a measured comparison).

Two concrete real-world use cases this addresses:

1. **Digitising Urdu documents** — government records, legal judgments,
   historical newspapers and books — so they become searchable and
   preservable instead of sitting as untouched paper or scanned images.
2. **Everyday accessibility and automation** — e.g. extracting and
   translating Urdu text from a photographed signboard or menu for a
   traveller, or reading Urdu text aloud from a photo for a visually
   impaired user.

## How it works

1. **TrOCR** ([microsoft/trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed))
   is a transformer OCR model made of two parts: a **vision encoder** that
   reads the image and a **text decoder** that turns what it sees into
   characters, one at a time. Microsoft pretrained it on huge amounts of
   printed English text.
2. **Fine-tuning** takes that pretrained model and continues training it —
   at a much lower learning rate, for far fewer steps — on a new, smaller
   dataset (here, Urdu images + their correct text), so it keeps the
   general "how to read text from an image" skill and adapts it to Urdu's
   script instead of learning to read from scratch. This is why the model
   needs comparatively little Urdu data to become useful, versus training
   an OCR model from nothing.
3. **The dataset** is a mix of real photographed/downloaded Urdu images and
   a large synthetically rendered set (see [Dataset details](#dataset-details))
   — every image is paired with its correct ground-truth text in
   `data/labels.csv`.
4. **Training, evaluation, and deployment** are described fully in
   [`docs/ACCURACY_NOTES.md`](docs/ACCURACY_NOTES.md); short version: text is
   Unicode-normalised, images are deskewed/contrast-enhanced/letterboxed
   (not squashed or hard-thresholded), training uses a cosine learning-rate
   schedule with early stopping on Character Error Rate (CER), and
   inference uses beam-search decoding — all specifically because each of
   these measurably improves accuracy over the naive version of this
   pipeline (details + reasoning for each choice are in that doc).

## Live demo

**[ FILL IN — paste your HuggingFace Space URL here once deployed, e.g.
`https://huggingface.co/spaces/yourusername/urdu-ocr-codesaviours-si26-rameesha` ]**

*(Screenshot of the working demo — add after you deploy:)*
`[ FILL IN — drag your demo screenshot into this README, or link it, e.g. ![demo](docs/demo_screenshot.png) ]`

*(2-3 minute walkthrough video — see the Week 8 handout:)*
`[ FILL IN — Loom video link ]`

## How to run it locally

```bash
git clone https://github.com/Rameesha-Malik/ML-SI-Solution-Urdu-OCR.git
cd ML-SI-Solution-Urdu-OCR
pip install -r requirements.txt
```

**Run the demo app** (needs a trained model — either train your own with
the steps below, or point `MODEL_PATH` at a published HuggingFace Hub
model):

```bash
export MODEL_PATH=yourusername/urdu-trocr-si26   # or a local folder, e.g. models/urdu-trocr/final
python app.py
```

Then open the local URL Gradio prints (typically `http://127.0.0.1:7860`).

**Reproduce the full pipeline from scratch** (bootstrap data -> preprocess
-> train -> evaluate — see [`docs/ACCURACY_NOTES.md`](docs/ACCURACY_NOTES.md)
for what each step does and why):

```bash
# 1. Generate a synthetic starter dataset (see Dataset details below for
#    why this alone isn't enough — add real photos too).
python src/synthetic.py --num-images 600

# 2. Merge with any real images you've labelled in data/labels_real.csv,
#    then preprocess everything.
python -c "
import sys; sys.path.insert(0, 'src')
from synthetic import merge_labels_csv
merge_labels_csv(['data/labels_synthetic.csv', 'data/labels_real.csv'], 'data/labels.csv')
"
python src/preprocessing.py --src-dir data/raw --dst-dir data/processed

# 3. Split, then train (GPU strongly recommended — see notebooks/ for a
#    ready-to-run Colab version of every step below).
python src/dataset.py --labels-csv data/labels.csv
python src/train.py --train-csv data/train.csv --test-csv data/test.csv \
    --output-dir models/urdu-trocr --epochs 12 --batch-size 8

# 4. Evaluate (CER / WER / exact-match + a per-sample error report).
python src/evaluate.py models/urdu-trocr/final --test-csv data/test.csv
```

The `notebooks/` folder has the same steps as five ready-to-run Colab
notebooks (`SI26_Week1_DataCollection.ipynb` through
`SI26_Week5_Gradio_Deploy.ipynb`), matching the internship's weekly
structure.

## Dataset details

`[ FILL IN once you've run the pipeline — src/dataset.py and
src/synthetic.py print exact counts. Template: ]`

- **Total images:** `[ FILL IN, e.g. 850 ]` (`[ FILL IN ]` real +
  `[ FILL IN ]` synthetic)
- **Sources:**
  - Real: `[ FILL IN — e.g. photographed newspaper/book/signboard photos,
    screenshots from Dawn Urdu / BBC Urdu / Jang, downloaded dataset(s)
    from Kaggle / Mendeley / urduhack ]`
  - Synthetic: rendered via `src/synthetic.py` from
    `data/corpus/urdu_sentences.txt` (a curated, hand-checked corpus
    spanning greetings, news/headline style, proverbs, signage, book
    prose, and numeric/date sentences)
- **Variety collected:**
  - Fonts/styles: Nastaliq, Naskh, and Sans-style Urdu (bundled: Noto
    Nastaliq Urdu, Noto Naskh Arabic, Noto Sans Arabic —
    `assets/fonts/`) `[ + any additional real fonts you photographed ]`
  - Backgrounds: white, cream/newsprint, light-grey scan, warm-white, plus
    `[ FILL IN real backgrounds you collected, e.g. dark signboards ]`
  - Sizes: small captions through large headlines
- **Train/test split:** 85% / 15%, seeded for reproducibility
  (`src/dataset.py::split_dataset`)

See [`docs/ACCURACY_NOTES.md`](docs/ACCURACY_NOTES.md) §4 for why the
dataset mixes real and synthetic data rather than using either alone.

## Results

`[ FILL IN with the numbers src/evaluate.py prints after training —
template + how to read them below: ]`

| Metric | Score |
|---|---|
| Character Error Rate (CER) | `[ FILL IN ]`% |
| Word Error Rate (WER) | `[ FILL IN ]`% |
| Exact-match accuracy | `[ FILL IN ]`% |
| Test set size | `[ FILL IN ]` images |

**Why three numbers, not one:** exact-match accuracy (the Week 4 handout's
metric) counts a prediction as wrong even if it's off by a single
character, which makes it hard to tell "almost right" from "completely
wrong." CER/WER measure *how far off* a wrong prediction is, which is the
standard way OCR/ASR accuracy is reported — see
[`docs/ACCURACY_NOTES.md`](docs/ACCURACY_NOTES.md) §1.

**If accuracy is lower than you'd like:** the two highest-leverage next
steps are (1) more real photographed data — synthetic data alone
plateaus, and (2) trying `microsoft/trocr-large-printed` as the base model
if your GPU budget allows. See "Where to push further" at the bottom of
[`docs/ACCURACY_NOTES.md`](docs/ACCURACY_NOTES.md) for the full list.

**A few examples the model got wrong**, for context on typical failure
modes (full list in `reports/eval_report.csv` after running `evaluate.py`,
sorted by CER):

`[ FILL IN 3-5 examples, e.g.: ]`

| Image | Actual | Predicted |
|---|---|---|
| `[ FILL IN ]` | `[ FILL IN ]` | `[ FILL IN ]` |

## Why We Need a Better Model (Tesseract gap analysis)

Run `python scripts/tesseract_baseline.py` to reproduce this. Tesseract
fails on Urdu because its Urdu language model was trained on a much
smaller and less varied dataset than its English one, and it doesn't
handle Nastaliq's diagonal letter-stacking and heavy contextual joining
well — it tends to segment the image into a flat left-to-right strip of
isolated glyph guesses, which breaks down whenever letters overlap
vertically or a word's letterforms change with position.

`[ FILL IN Tesseract's measured CER/WER from
reports/tesseract_baseline.csv, and compare directly against this
project's fine-tuned-model numbers above — that comparison is the actual
evidence for "why we need a better model." ]`

## Testing

The data/preprocessing/dataset logic has an automated test suite (`tests/`,
46 tests) that runs on every push via GitHub Actions (badge at the top of
this README) — it doesn't need a GPU or model weights, so it runs in a few
seconds and catches regressions in the parts of the pipeline that don't
require training to verify (Unicode normalisation, synthetic rendering
across all three bundled fonts, the preprocessing pipeline, augmentation,
and dataset splitting/shape/masking correctness).

```bash
pip install -r requirements-dev.txt
pytest -v
```

`app.py`'s demo also ships with 4 example images (`examples/`, with
ground truth in `examples/ground_truth.csv`) pre-loaded into the Gradio
interface, so a visitor to your Space has something to click before
uploading their own image.

## Project structure

```
.
├── app.py                    # Gradio demo (Week 5) — same code path for local, Colab, and HF Spaces
├── requirements.txt, requirements-dev.txt, pytest.ini
├── .github/workflows/tests.yml     # CI: runs tests/ on every push
├── assets/fonts/              # Bundled Urdu fonts for synthetic data generation
├── examples/                  # Sample images + ground_truth.csv for the Gradio demo
├── data/
│   ├── corpus/urdu_sentences.txt   # Curated real Urdu sentences (synthetic data source)
│   ├── raw/                        # Collected images, organised by source (Week 1)
│   ├── processed/                  # Preprocessed images (Week 2)
│   └── labels.csv, train.csv, test.csv
├── src/
│   ├── text_utils.py          # Urdu Unicode normalisation
│   ├── synthetic.py           # Synthetic Urdu OCR image generator
│   ├── preprocessing.py       # Deskew / contrast / letterbox pipeline
│   ├── augmentation.py        # Train-time image augmentation
│   ├── dataset.py             # PyTorch Dataset + train/test split
│   ├── metrics.py             # CER / WER / exact-match
│   ├── train.py               # Fine-tuning (Week 4)
│   ├── evaluate.py            # Test-set evaluation + error report
│   └── infer.py               # Shared inference (used by evaluate.py and app.py)
├── tests/                     # Automated test suite (see Testing above)
├── scripts/tesseract_baseline.py   # Week 2 baseline gap analysis
├── notebooks/                 # SI26_Week1..5_*.ipynb — Colab versions of the whole pipeline
└── docs/ACCURACY_NOTES.md     # Full explanation of every accuracy decision made in this repo
```

## Credit

**Rameesha Malik**

*Built during the Code Saviours ML/AI Internship — Batch SI-26.*
