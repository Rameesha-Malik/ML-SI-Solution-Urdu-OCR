"""
Shared single/batch-image inference for a fine-tuned Urdu TrOCR model.

Used by both `evaluate.py` (scoring the test set) and `app.py` (the
Gradio demo), so the exact same decoding settings are used everywhere —
there is no separate "greedy for the demo, beam search for the paper"
inconsistency.

The Week 5 handout's `extract_urdu_text()` calls `model.generate(pixel_values)`
with no arguments, i.e. **greedy decoding**. Beam search (`num_beams>1`)
reliably reduces CER for seq2seq OCR because it doesn't commit to the
single highest-probability token at every step — it keeps several
candidate continuations and picks the best complete sequence. The cost is
a bit more compute per image, which is a good trade for a demo/eval where
you generate one line of text at a time (still comfortably sub-second
on CPU, and near-instant on GPU).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from preprocessing import PreprocessConfig, preprocess_image
from text_utils import normalize_urdu


class UrduOCR:
    """Loads once, call `.read(image)` (or `.read_batch(images)`) many times."""

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        num_beams: int = 4,
        max_length: int = 128,
        preprocess: bool = True,
        preprocess_config: PreprocessConfig | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.num_beams = num_beams
        self.max_length = max_length
        self.preprocess = preprocess
        self.preprocess_config = preprocess_config or PreprocessConfig()

    @torch.no_grad()
    def read(self, image: Image.Image | str | Path) -> str:
        return self.read_batch([image])[0]

    @torch.no_grad()
    def read_batch(self, images: list[Image.Image | str | Path]) -> list[str]:
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img)
            img = img.convert("RGB")
            if self.preprocess:
                img = preprocess_image(img, self.preprocess_config)
            pil_images.append(img)

        pixel_values = self.processor(pil_images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        generated_ids = self.model.generate(
            pixel_values,
            num_beams=self.num_beams,
            max_length=self.max_length,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [normalize_urdu(t) for t in texts]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Urdu OCR on one or more images.")
    parser.add_argument("model_path", help="Local path or HF Hub repo id of the fine-tuned model.")
    parser.add_argument("images", nargs="+", help="Image file path(s).")
    parser.add_argument("--num-beams", type=int, default=4)
    args = parser.parse_args()

    ocr = UrduOCR(args.model_path, num_beams=args.num_beams)
    for path, text in zip(args.images, ocr.read_batch(args.images)):
        print(f"{path}\t{text}")
