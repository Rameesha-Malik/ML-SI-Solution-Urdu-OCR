# Accuracy notes

This project's code is a from-scratch rewrite of the Week 1-5 handout
pipeline, aimed specifically at Character Error Rate (CER) — not just "does
it run in Colab." This doc is the single place that explains *why* each
piece exists, so it isn't scattered as a wall of code comments. Every claim
below is also explained inline in the relevant module's docstring
(`src/*.py`).

## 1. Measure the right thing first

The handout scores the model with **exact full-string match only**. That
metric cannot distinguish "almost right" from "completely wrong," so you
cannot tell whether a training change actually helped. `src/metrics.py`
adds **CER** (Character Error Rate) and **WER** (Word Error Rate) — the
standard OCR/ASR metrics — computed with `jiwer`, alongside exact-match for
comparability with the handout's baseline. `src/train.py` reports CER after
every epoch during training (`predict_with_generate=True`), not just at the
very end.

## 2. Make ground truth internally consistent

Urdu text collected from different sources (keyboards, websites, OCR tools)
is rarely written with one consistent Unicode encoding — e.g. `ي`
(U+064A, Arabic yeh) vs. `ی` (U+06CC, Urdu "farsi yeh") look identical but
are different code points. If training labels mix both forms, the model is
effectively asked to learn two "correct" spellings for the same glyph, and
CER scoring penalises a visually-perfect prediction that happens to use the
other form. `src/text_utils.py::normalize_urdu()` canonicalises letter
forms, digits, diacritics, and tatweel before text ever reaches the
tokenizer or the scorer. Every module that touches text (dataset loading,
synthetic generation, training, evaluation, the Tesseract baseline) routes
through it.

## 3. Preprocess for a vision transformer, not for a binary mask

The handout's preprocessing (grayscale -> squash-resize to a fixed 512x128
-> hard global threshold) actively hurts a TrOCR-style model:

- **Squash-resize** distorts every glyph's aspect ratio differently
  depending on line length, which destroys exactly the joins/dots Urdu
  script depends on.
- **Hard global threshold** discards anti-aliasing and, on anything but a
  perfectly even white background, reliably wipes out thin strokes or
  floods the image solid. It also makes the input look nothing like the
  natural photos the pretrained ViT encoder inside TrOCR was trained on,
  throwing away most of the value of transfer learning.

`src/preprocessing.py` instead: corrects EXIF rotation, deskews, applies
**local adaptive contrast** (CLAHE on the luminance channel only, so colour
and gradients survive), denoises with an edge-preserving filter, and
**letterboxes to a square** (pads with the image's own background colour)
instead of squashing — since TrOCRProcessor's own resize step is a fixed
square anyway, letterboxing first means *it* isn't the thing distorting
your glyphs.

## 4. Bootstrap a large, varied dataset

Weeks 1 and 3 correctly identify that variety (fonts, backgrounds, sizes)
is what makes a model robust — but hand-collecting hundreds of varied real
images in a few weeks is slow. `src/synthetic.py` renders a curated,
hand-checked Urdu sentence corpus (`data/corpus/urdu_sentences.txt`,
covering greetings/news/proverbs/signage/book-style/numeric text) across:

- 3 structurally different Urdu writing styles (Nastaliq / Naskh / Sans —
  bundled fonts, see `assets/fonts/`),
- 4 background styles, 3 ink colours, randomised sizes/padding,
- small random rotation, gaussian noise, blur, and JPEG-recompression
  artifacts, to look more like a phone photo than a pristine render.

It renders with Pillow's native `libraqm` shaping engine when available
(verified against all three bundled fonts — Nastaliq's dramatic diagonal
joins render correctly), falling back to `arabic_reshaper` + `python-bidi`
pre-shaping otherwise.

It can also recombine real corpus words into extra "pseudo-sentences" purely
for additional glyph/ligature visual coverage (`pseudo_fraction`, kept a
minority) — these are documented as non-semantic and are not a substitute
for real sentences or real photographed data.

**This is a bootstrap, not a replacement for real data.** Mix in every real
photo/screenshot/downloaded-dataset image you can (Week 1/3's Source 1 & 2)
via `data/labels_real.csv` + `synthetic.merge_labels_csv` — a model trained
on synthetic-only data will not generalise to real photos as well as one
trained on a real+synthetic mix. Real data should grow every week; synthetic
data is there so you're never blocked waiting on it.

## 5. Augment at train time

`src/augmentation.py::OCRAugment` applies mild, OCR-safe random rotation,
perspective jitter, brightness/contrast, blur, noise, and small occlusion
patches — a different perturbation every epoch, **train split only**. With
a dataset in the hundreds-to-low-thousands (realistic even after synthetic
bootstrapping), this is one of the highest-leverage single changes against
overfitting to the exact fonts/backgrounds/noise the model happened to see.

## 6. Train longer, more carefully, and check progress as you go

`src/train.py` replaces the handout's hand-rolled 3-epoch loop
(`lr=5e-5` flat, no eval) with `Seq2SeqTrainer`:

- **Cosine LR schedule with warmup** instead of one flat LR for the whole
  run — avoids destabilising pretrained weights on step 1, lets the model
  settle into a minimum instead of oscillating.
- **More epochs (default 12) with early stopping on CER** instead of a
  fixed 3 — 3 epochs is rarely enough to adapt an English-pretrained
  decoder to Urdu; early stopping means this doesn't cost you overfitting
  or wasted compute.
- **Mixed precision (fp16) + gradient accumulation** — a larger effective
  batch size than a single T4 could otherwise hold, which stabilises
  gradients.
- **`predict_with_generate` + beam search at eval time** (not just at the
  very end) so you can watch CER move epoch to epoch.

## 7. Decode better at inference time too

Greedy decoding (`model.generate(pixel_values)` with no arguments, as in
the handout) is measurably worse for seq2seq OCR than a small beam search.
`src/infer.py::UrduOCR` uses `num_beams=4` by default and is the single
inference path shared by `evaluate.py` (scoring) and `app.py` (the live
demo) — so the model is never evaluated under different decoding settings
than it's demoed with.

## Where to push further from here

- **More real data is the single highest-value thing you can add.**
  Synthetic data plateaus in usefulness; real photographed variety does not.
- **Try `microsoft/trocr-large-printed`** as `--base-model` if your GPU/time
  budget allows — larger encoder-decoder, generally lower CER, slower to
  train and to run inference.
- **Domain-match your backgrounds.** If your real use case is newspapers,
  weight the corpus/backgrounds toward that; if it's signage, weight
  toward that instead.
- **Tune `--num-beams` up (e.g. 6-8)** at inference time if latency isn't
  critical — usually a small further CER improvement for more compute.
- **Watch the worst-CER examples** (`reports/eval_report.csv`, sorted by
  `cer`) — they usually point at a systematic gap (a font style, a
  background type, a punctuation pattern) worth adding more of to the
  training set, rather than a random one-off.
