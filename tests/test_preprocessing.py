import numpy as np
from PIL import Image

from preprocessing import (
    PreprocessConfig,
    estimate_skew_angle,
    letterbox_to_square,
    preprocess_file,
    preprocess_folder,
    preprocess_image,
)


def _sample_image(w=300, h=80, text_band=True) -> Image.Image:
    """A quick synthetic 'line of text' stand-in: a light background with
    a dark horizontal band, so there's real structure to preprocess."""
    arr = np.full((h, w, 3), 245, dtype=np.uint8)
    if text_band:
        arr[h // 3: 2 * h // 3, 10:w - 10] = 20
    return Image.fromarray(arr)


def test_preprocess_image_returns_square_rgb():
    img = _sample_image()
    out = preprocess_image(img, PreprocessConfig(target_size=200))
    assert out.mode == "RGB"
    assert out.size == (200, 200)


def test_preprocess_image_does_not_squash_aspect_ratio():
    """A wide, short line crop should be letterboxed (padded), not
    stretched -- so the 'ink band' should still look like a horizontal
    band relative to its own bounding box, not be stretched square."""
    wide_img = _sample_image(w=600, h=60)
    out = preprocess_image(wide_img, PreprocessConfig(
        target_size=256, deskew=False, denoise=False,
    ))
    arr = np.asarray(out.convert("L"))
    dark_rows = np.where(arr.min(axis=1) < 100)[0]
    assert len(dark_rows) > 0
    band_height = dark_rows.max() - dark_rows.min()
    # the original band was 20/60 = ~33% of image height; letterboxing
    # should roughly preserve that proportion instead of stretching it
    # toward 256 (a squash-resize would inflate this dramatically).
    assert band_height < 256 * 0.6


def test_preprocess_image_disable_all_steps_still_letterboxes():
    img = _sample_image()
    cfg = PreprocessConfig(deskew=False, enhance_contrast=False, denoise=False, letterbox=True, target_size=128)
    out = preprocess_image(img, cfg)
    assert out.size == (128, 128)


def test_preprocess_image_no_letterbox_keeps_aspect_ratio():
    img = _sample_image(w=300, h=80)
    cfg = PreprocessConfig(deskew=False, enhance_contrast=False, denoise=False, letterbox=False)
    out = preprocess_image(img, cfg)
    assert out.size == (300, 80)


def test_estimate_skew_angle_near_zero_for_straight_text():
    img = _sample_image()
    gray = np.asarray(img.convert("L"))
    angle = estimate_skew_angle(gray, max_angle=15.0)
    assert abs(angle) < 5.0


def test_estimate_skew_angle_handles_blank_image():
    blank = np.full((80, 300), 250, dtype=np.uint8)
    angle = estimate_skew_angle(blank, max_angle=15.0)
    assert angle == 0.0


def test_letterbox_to_square_preserves_background_estimate():
    arr = np.full((60, 300, 3), 250, dtype=np.uint8)
    out = letterbox_to_square(arr, size=128, pad_frac=0.05)
    assert out.shape == (128, 128, 3)
    # corners (padding) should stay close to the original background colour
    corner = out[0, 0]
    assert abs(int(corner[0]) - 250) < 20


def test_preprocess_file_writes_output(tmp_path):
    src = tmp_path / "in.png"
    dst = tmp_path / "nested" / "out.png"
    _sample_image().save(src)

    preprocess_file(src, dst, PreprocessConfig(target_size=128))
    assert dst.exists()
    assert Image.open(dst).size == (128, 128)


def test_preprocess_folder_processes_all_images(tmp_path):
    src_dir = tmp_path / "raw" / "sub"
    src_dir.mkdir(parents=True)
    for i in range(3):
        _sample_image().save(src_dir / f"img_{i}.png")
    (src_dir / "not_an_image.txt").write_text("hello")

    dst_dir = tmp_path / "processed"
    count = preprocess_folder(tmp_path / "raw", dst_dir, PreprocessConfig(target_size=96))

    assert count == 3
    assert len(list(dst_dir.glob("*.png"))) == 3
