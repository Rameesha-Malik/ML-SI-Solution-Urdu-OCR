"""
Urdu / Arabic-script text normalisation.

Why this file exists (accuracy matters):
-----------------------------------------
Urdu text collected from different keyboards, websites, and OCR tools is
almost never written with a single consistent Unicode encoding. The same
*visible* word can arrive as several different byte sequences, e.g.:

    ی   U+06CC  ARABIC LETTER FARSI YEH   (correct Urdu "yeh")
    ي   U+064A  ARABIC LETTER YEH         (Arabic "yeh", looks identical)
    ک   U+06A9  ARABIC LETTER KEHEH       (correct Urdu "kaf")
    ك   U+0643  ARABIC LETTER KAF         (Arabic "kaf")
    ہ   U+06C1  ARABIC LETTER HEH GOAL    (correct Urdu "choti heh")
    ه   U+0647  ARABIC LETTER HEH         (Arabic "heh")

If half your training labels use one form and half use the other, the
model is being asked to learn two different "correct answers" for the
same glyph shape — this directly hurts both training convergence and the
character error rate (CER) at evaluation time, because a prediction that
is visually perfect can still be marked wrong.

`normalize_urdu()` canonicalises all of this *before* it ever reaches the
tokenizer or the CER/WER scorer, so:
  1. training labels are internally consistent,
  2. predictions vs. ground truth are compared fairly,
  3. Tesseract's baseline output (Week 2) is compared on equal footing.

Always run collected labels (labels.csv) through this once, and always run
model predictions through it before scoring.
"""

from __future__ import annotations

import re
import unicodedata

# Presentation-form / alternate-script letters -> canonical Urdu letters.
# Left side: Arabic Unicode block forms that visually match an Urdu letter
# but are a different code point. Right side: the Urdu-correct code point.
_LETTER_MAP = {
    "ي": "ی",  # ARABIC LETTER YEH        -> FARSI YEH (ی)
    "ى": "ی",  # ARABIC LETTER ALEF MAKSURA -> FARSI YEH (ی)
    "ك": "ک",  # ARABIC LETTER KAF        -> KEHEH (ک)
    "ه": "ہ",  # ARABIC LETTER HEH        -> HEH GOAL (ہ)
    "ۀ": "ہ",  # HEH WITH YEH ABOVE       -> HEH GOAL (ہ)
    "ءٔ": "ء",  # stray hamza combos -> plain hamza
}

# Arabic-Indic and Extended Arabic-Indic (Urdu) digits -> ASCII digits.
# Kept optional (see `digits="keep"|"ascii"`) because publication-style
# Urdu text conventionally uses ۰۱۲۳۴۵۶۷۸۹, not 0123456789.
_URDU_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_ASCII_DIGITS = "0123456789"
_DIGIT_TO_ASCII = {
    ch: str(i) for i, ch in enumerate(_URDU_DIGITS)
} | {ch: str(i) for i, ch in enumerate(_ARABIC_INDIC_DIGITS)}
_DIGIT_TO_URDU = {str(i): _URDU_DIGITS[i] for i in range(10)} | {
    ch: _URDU_DIGITS[i] for i, ch in enumerate(_ARABIC_INDIC_DIGITS)
}

# Arabic diacritics / harakat (fatha, kasra, damma, tanwin, shadda, sukun...)
_DIACRITICS_RE = re.compile(
    "[" + "".join(chr(c) for c in range(0x064B, 0x0653)) + "ٰۖ-ۭ" + "]"
)

# Tatweel / kashida (elongation character) — a pure typographic stretch
# character with no phonetic value; safe to strip.
_TATWEEL = "ـ"

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_urdu(
    text: str,
    strip_diacritics: bool = True,
    digits: str = "ascii",
) -> str:
    """Canonicalise an Urdu string for training / evaluation.

    Args:
        text: raw Urdu string.
        strip_diacritics: remove zabar/zer/pesh/tanwin/shadda etc. Almost
            all real-world Urdu text (news, books, signage, casual
            writing) omits these, so keeping them in ground truth just
            adds noisy characters a model trained on unvocalised text
            will never predict. Set False only if your dataset is
            deliberately fully-vocalised (e.g. Qur'anic text).
        digits: "ascii" to normalise all digit variants to 0-9 (recommended
            — keeps the label space small and matches how most printed
            Urdu documents actually mix digits), "urdu" to normalise
            everything to ۰-۹ instead, or "keep" to leave digits untouched.

    Returns:
        Normalised string: NFC-composed, canonical letter forms, single
        spaces, no leading/trailing whitespace.
    """
    if text is None:
        return ""

    # 1. Unicode-normalise first so combining marks compose consistently.
    text = unicodedata.normalize("NFC", text)

    # 2. Canonical letter forms.
    for src, dst in _LETTER_MAP.items():
        text = text.replace(src, dst)

    # 3. Diacritics.
    if strip_diacritics:
        text = _DIACRITICS_RE.sub("", text)

    # 4. Tatweel.
    text = text.replace(_TATWEEL, "")

    # 5. Digits.
    if digits == "ascii":
        text = "".join(_DIGIT_TO_ASCII.get(ch, ch) for ch in text)
    elif digits == "urdu":
        text = "".join(_DIGIT_TO_URDU.get(ch, ch) for ch in text)
    # digits == "keep" -> no-op

    # 6. Whitespace: collapse runs, strip ends. Do NOT strip zero-width
    #    joiner/non-joiner (U+200C/200D) — they can change letter shaping.
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


def is_urdu_text(text: str, threshold: float = 0.5) -> bool:
    """Heuristic check: are at least `threshold` fraction of the
    non-space characters in the Arabic Unicode block (U+0600-U+06FF) or
    Arabic Presentation Forms? Useful for sanity-checking a labels.csv
    for rows that accidentally contain English/garbage text.
    """
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    arabic = sum(
        1
        for c in chars
        if "؀" <= c <= "ۿ" or "ﭐ" <= c <= "﻿"
    )
    return (arabic / len(chars)) >= threshold


if __name__ == "__main__":
    samples = [
        "پاکستان زندہ باد",
        "يه مثال کا متن", # arabic-form letters
        "آج بروز جمعہ  ہے۔۔۔",
        "قیمت ٥٠٠ روپے",
    ]
    for s in samples:
        print(repr(s), "->", repr(normalize_urdu(s)))
