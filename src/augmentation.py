"""
Train-time image augmentation for the Urdu OCR dataset.

The Week 4 handout trains on raw images with zero augmentation. With a
dataset in the hundreds-to-low-thousands of images (realistic for a 5-week
internship project, even after synthetic bootstrapping), a model with no
augmentation memorises the exact fonts/backgrounds/noise levels it saw and
degrades sharply on anything slightly different — e.g. a phone photo taken
at a different angle or under different lighting than the training set.

`OCRAugment` applies a *mild*, OCR-safe set of photometric and geometric
perturbations (never anything that would make the text itself unreadable,
like heavy occlusion or extreme rotation) on every training epoch, so the
model sees a different version of each image each time. This is one of
the single highest-leverage changes for small-dataset OCR accuracy.

Only apply this to the **training** split — never to validation/test
images, which must stay exactly as collected so evaluation numbers are
meaningful.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class AugmentConfig:
    rotation_deg: float = 2.0            # max +/- rotation
    perspective_strength: float = 0.02   # max corner jitter, as a fraction of size
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    blur_prob: float = 0.15
    blur_radius_range: tuple[float, float] = (0.3, 1.0)
    noise_prob: float = 0.25
    noise_sigma_range: tuple[float, float] = (2.0, 8.0)
    erasing_prob: float = 0.10           # simulate small stains / occlusions
    erasing_area_frac: tuple[float, float] = (0.01, 0.04)


class OCRAugment:
    """Callable: `augmented = OCRAugment(cfg)(pil_image)`."""

    def __init__(self, config: AugmentConfig | None = None, seed: int | None = None):
        self.cfg = config or AugmentConfig()
        self.rng = random.Random(seed)

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        img = self._geometric(img)
        img = self._photometric(img)
        img = self._noise(img)
        img = self._erasing(img)
        return img

    # -- geometric -----------------------------------------------------
    def _geometric(self, img: Image.Image) -> Image.Image:
        cfg = self.cfg
        angle = self.rng.uniform(-cfg.rotation_deg, cfg.rotation_deg)
        bg = _border_color(img)
        img = img.rotate(angle, expand=False, fillcolor=bg, resample=Image.BICUBIC)

        if cfg.perspective_strength > 0:
            img = _random_perspective(img, cfg.perspective_strength, self.rng, bg)
        return img

    # -- photometric -----------------------------------------------------
    def _photometric(self, img: Image.Image) -> Image.Image:
        cfg = self.cfg
        img = ImageEnhance.Brightness(img).enhance(self.rng.uniform(*cfg.brightness_range))
        img = ImageEnhance.Contrast(img).enhance(self.rng.uniform(*cfg.contrast_range))
        if self.rng.random() < cfg.blur_prob:
            radius = self.rng.uniform(*cfg.blur_radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius))
        return img

    # -- noise -----------------------------------------------------
    def _noise(self, img: Image.Image) -> Image.Image:
        if self.rng.random() >= self.cfg.noise_prob:
            return img
        sigma = self.rng.uniform(*self.cfg.noise_sigma_range)
        arr = np.asarray(img).astype(np.float32)
        noise = np.random.default_rng(self.rng.randint(0, 2**31 - 1)).normal(0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    # -- random erasing -----------------------------------------------------
    def _erasing(self, img: Image.Image) -> Image.Image:
        if self.rng.random() >= self.cfg.erasing_prob:
            return img
        w, h = img.size
        area_frac = self.rng.uniform(*self.cfg.erasing_area_frac)
        patch_area = area_frac * w * h
        patch_w = int(min(w * 0.2, (patch_area * self.rng.uniform(0.5, 2.0)) ** 0.5))
        patch_h = int(min(h * 0.4, patch_area / max(patch_w, 1)))
        if patch_w < 2 or patch_h < 2:
            return img
        x0 = self.rng.randint(0, max(0, w - patch_w))
        y0 = self.rng.randint(0, max(0, h - patch_h))
        arr = np.array(img)
        fill = _border_color(img)
        arr[y0:y0 + patch_h, x0:x0 + patch_w] = fill
        return Image.fromarray(arr)


def _border_color(img: Image.Image) -> tuple[int, int, int]:
    arr = np.asarray(img.convert("RGB"))
    border = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]])
    return tuple(int(c) for c in np.median(border, axis=0))


def _random_perspective(img: Image.Image, strength: float, rng: random.Random,
                         fill: tuple[int, int, int]) -> Image.Image:
    """Small random perspective warp implemented with PIL's QUAD transform
    (keeps this module dependency-free — no OpenCV needed here)."""
    w, h = img.size
    jx, jy = strength * w, strength * h

    def jitter():
        return rng.uniform(-1, 1)

    # Four source corners, each nudged slightly -> mild "photographed at an angle" look.
    quad = [
        0 + jitter() * jx, 0 + jitter() * jy,
        0 + jitter() * jx, h + jitter() * jy,
        w + jitter() * jx, h + jitter() * jy,
        w + jitter() * jx, 0 + jitter() * jy,
    ]
    return img.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC, fillcolor=fill)
