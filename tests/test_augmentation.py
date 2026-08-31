import numpy as np
from PIL import Image

from augmentation import AugmentConfig, OCRAugment


def _sample_image(w=300, h=80) -> Image.Image:
    arr = np.full((h, w, 3), 245, dtype=np.uint8)
    arr[h // 3: 2 * h // 3, 10:w - 10] = 20
    return Image.fromarray(arr)


def test_ocr_augment_returns_rgb_image_same_size():
    img = _sample_image()
    aug = OCRAugment(seed=0)
    out = aug(img)
    assert out.mode == "RGB"
    assert out.size == img.size


def test_ocr_augment_is_deterministic_given_seed():
    img = _sample_image()
    out1 = OCRAugment(seed=42)(img)
    out2 = OCRAugment(seed=42)(img)
    assert np.array_equal(np.asarray(out1), np.asarray(out2))


def test_ocr_augment_changes_the_image_at_least_sometimes():
    img = _sample_image()
    # with a strong config, at least one of several seeded runs should differ
    cfg = AugmentConfig(rotation_deg=5.0, blur_prob=1.0, noise_prob=1.0, erasing_prob=1.0)
    changed = False
    for seed in range(5):
        out = OCRAugment(cfg, seed=seed)(img)
        if not np.array_equal(np.asarray(out), np.asarray(img)):
            changed = True
            break
    assert changed


def test_ocr_augment_handles_many_seeds_without_crashing():
    img = _sample_image()
    for seed in range(20):
        out = OCRAugment(seed=seed)(img)
        assert out.size == img.size


def test_ocr_augment_accepts_non_rgb_input():
    img = _sample_image().convert("L")
    out = OCRAugment(seed=1)(img)
    assert out.mode == "RGB"
