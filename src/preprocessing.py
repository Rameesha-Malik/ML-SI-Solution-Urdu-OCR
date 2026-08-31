"""
Image preprocessing for the Urdu OCR pipeline.

This intentionally does **not** follow the Week 2 handout's pipeline
(grayscale -> squash-resize to 512x128 -> hard global threshold) because
each of those three steps actively hurts a TrOCR-style model:

  * **Squash-resize to a fixed (w, h)** distorts every glyph's aspect
    ratio differently depending on the source line's length — a short
    word gets stretched wide, a long line gets squeezed thin. Urdu's
    Nastaliq/Naskh joins and dots are exactly the kind of fine detail
    that distortion destroys.
  * **Hard global binary threshold** (`cv2.threshold(..., 127, 255,
    THRESH_BINARY)`) throws away anti-aliasing and, on anything but a
    perfectly clean white-background scan (a yellowed newspaper, a photo
    with uneven lighting, a dark signboard), reliably wipes out thin
    strokes or floods the image to solid black/white. It also produces
    images that look nothing like the natural photos TrOCR's vision
    encoder was pretrained on, which throws away most of that
    pretraining's value.
  * **Grayscale-only** discards colour contrast that can matter (e.g.
    dark blue ink vs. dark background).

Instead this module:
  1. corrects EXIF rotation and converts to RGB,
  2. (optionally) deskews the text line,
  3. applies **local, adaptive** contrast enhancement (CLAHE) instead of a
     global threshold — this improves legibility without flattening the
     image to pure black/white,
  4. denoises gently with an edge-preserving filter,
  5. **letterboxes to a square** (pads with the image's own background
     colour) instead of squash-resizing. TrOCRProcessor's own feature
     extractor resizes to a fixed square (e.g. 384x384) internally — if
     you hand it a very wide, short line crop, *that* resize does the
     squashing. Letterboxing first means the aspect-ratio distortion
     TrOCRProcessor introduces is at most the padding, not the glyphs.

See docs/ACCURACY_NOTES.md for the reasoning and references.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class PreprocessConfig:
    deskew: bool = True
    max_deskew_angle: float = 15.0       # ignore skew estimates beyond this (likely noise)
    enhance_contrast: bool = True
    clahe_clip_limit: float = 2.0
    denoise: bool = True
    denoise_strength: int = 7            # cv2.fastNlMeansDenoisingColored `h`
    letterbox: bool = True
    target_size: int = 384               # matches TrOCR's default square input
    border_pad_frac: float = 0.04        # extra breathing room around text


def _pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def estimate_skew_angle(gray: np.ndarray, max_angle: float) -> float:
    """Estimate the dominant text-line skew via the minimum-area rectangle
    of dark (ink) pixels. Returns 0.0 if no confident estimate is found.
    """
    # Otsu threshold *only* for skew estimation (not for the output image).
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 20:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect angle convention varies; normalise to [-45, 45].
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > max_angle:
        return 0.0
    return float(angle)


def deskew(img_bgr: np.ndarray, max_angle: float) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew_angle(gray, max_angle)
    if abs(angle) < 0.2:
        return img_bgr
    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    border_color = _estimate_background_color(img_bgr)
    return cv2.warpAffine(
        img_bgr, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color,
    )


def _estimate_background_color(img_bgr: np.ndarray) -> tuple[int, int, int]:
    """Sample the image border to guess the page/background colour, used
    to fill padding so letterboxing/rotation don't introduce a jarring
    black or white frame that isn't representative of the source image.
    """
    h, w = img_bgr.shape[:2]
    border_px = np.concatenate([
        img_bgr[0, :, :], img_bgr[-1, :, :],
        img_bgr[:, 0, :], img_bgr[:, -1, :],
    ])
    median = np.median(border_px, axis=0)
    return tuple(int(c) for c in median)


def enhance_contrast(img_bgr: np.ndarray, clip_limit: float) -> np.ndarray:
    """CLAHE (local adaptive histogram equalisation) on the luminance
    channel only, so colour is preserved and contrast improves evenly
    across uneven lighting/shadows — unlike a single global threshold.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(img_bgr: np.ndarray, strength: int) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, strength, strength, 7, 21)


def letterbox_to_square(img_bgr: np.ndarray, size: int, pad_frac: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    bg = _estimate_background_color(img_bgr)

    # Shrink the usable area slightly so we always have a small margin.
    usable = int(size * (1 - 2 * pad_frac))
    scale = usable / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size, 3), bg, dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def preprocess_image(
    image: Image.Image | str | Path,
    config: PreprocessConfig | None = None,
) -> Image.Image:
    """Run the full preprocessing pipeline and return a PIL RGB image.

    Accepts either a PIL Image or a path. Does not write anything to
    disk — see `preprocess_file` / `preprocess_folder` for that.
    """
    config = config or PreprocessConfig()

    if isinstance(image, (str, Path)):
        pil_img = Image.open(image)
    else:
        pil_img = image
    pil_img = ImageOps.exif_transpose(pil_img)  # respect camera-photo rotation
    img_bgr = _pil_to_cv(pil_img)

    if config.deskew:
        img_bgr = deskew(img_bgr, config.max_deskew_angle)
    if config.enhance_contrast:
        img_bgr = enhance_contrast(img_bgr, config.clahe_clip_limit)
    if config.denoise:
        img_bgr = denoise(img_bgr, config.denoise_strength)
    if config.letterbox:
        img_bgr = letterbox_to_square(img_bgr, config.target_size, config.border_pad_frac)

    return _cv_to_pil(img_bgr)


def preprocess_file(src_path: Path | str, dst_path: Path | str,
                     config: PreprocessConfig | None = None) -> None:
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out = preprocess_image(src_path, config)
    out.save(dst_path)


def preprocess_folder(
    src_dir: Path | str,
    dst_dir: Path | str,
    config: PreprocessConfig | None = None,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> int:
    """Preprocess every image under `src_dir` (recursively) into `dst_dir`,
    flattening into a single folder (matching the Week 2 handout layout:
    data/raw/**/*.jpg -> data/processed/<filename>). Returns count processed.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(src_dir.rglob("*")):
        if path.suffix.lower() not in extensions:
            continue
        try:
            preprocess_file(path, dst_dir / path.name, config)
            count += 1
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"  [skip] {path}: {e}")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Urdu OCR images.")
    parser.add_argument("--src-dir", default="data/raw")
    parser.add_argument("--dst-dir", default="data/processed")
    parser.add_argument("--no-deskew", action="store_true")
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--target-size", type=int, default=384)
    args = parser.parse_args()

    cfg = PreprocessConfig(
        deskew=not args.no_deskew,
        enhance_contrast=not args.no_clahe,
        denoise=not args.no_denoise,
        target_size=args.target_size,
    )
    n = preprocess_folder(args.src_dir, args.dst_dir, cfg)
    print(f"Preprocessed {n} images -> {args.dst_dir}")
