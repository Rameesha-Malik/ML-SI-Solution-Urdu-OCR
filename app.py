"""
Gradio demo app — the Week 5 deployment step, in a form that runs both in
Colab (`interface.launch(share=True)`) and as a standalone HuggingFace
Space (`python app.py`).

Differences from the Week 5 handout's version:
  * Uses `UrduOCR` (src/infer.py) — beam search decoding + the same
    preprocessing pipeline the model was trained/evaluated with, instead
    of greedy `model.generate(pixel_values)` on a raw, un-preprocessed
    image. Consistency between train/eval/demo preprocessing matters:
    a model evaluated on preprocessed images but demoed on raw ones will
    look worse live than its reported CER/WER suggests.
  * Model location is one environment variable (`MODEL_PATH`), so the
    same file works locally, in Colab, and on Spaces — no drive.mount()
    branch to delete before deploying.
  * Loads example images from `examples/` automatically, if present.

Deploying to HuggingFace Spaces:
  1. Create a Space (SDK: Gradio) named
     `urdu-ocr-codesaviours-si26-[yourfirstname]` (see README "Live Demo").
  2. Push this whole repo to it (or upload app.py, requirements.txt,
     src/, assets/, examples/, and your trained model folder).
  3. Set the Space variable `MODEL_PATH` to your HF Hub model repo id
     (e.g. `yourusername/urdu-trocr-si26`) if you pushed the model there,
     otherwise commit the model folder into the Space and point
     `MODEL_PATH` at its relative path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from infer import UrduOCR  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", str(REPO_ROOT / "models" / "urdu-trocr" / "final"))
NUM_BEAMS = int(os.environ.get("NUM_BEAMS", "4"))

print(f"Loading Urdu OCR model from: {MODEL_PATH}")
try:
    ocr = UrduOCR(MODEL_PATH, num_beams=NUM_BEAMS)
    LOAD_ERROR = None
except Exception as e:  # noqa: BLE001
    ocr = None
    LOAD_ERROR = str(e)
    print(f"WARNING: could not load model from {MODEL_PATH}: {e}")
    print("Set the MODEL_PATH environment variable to a valid local path or "
          "HuggingFace Hub model repo id (see README > How to run it locally).")


def extract_urdu_text(image):
    """Takes an image, returns extracted Urdu text."""
    if image is None:
        return "Please upload an image"
    if ocr is None:
        return (
            "Model failed to load. Set MODEL_PATH to your trained model "
            f"(local folder or HF Hub repo id).\nDetails: {LOAD_ERROR}"
        )
    text = ocr.read(image)
    return text if text else "Could not extract text from this image"


def _example_images() -> list[str]:
    examples_dir = REPO_ROOT / "examples"
    if not examples_dir.exists():
        return []
    exts = (".png", ".jpg", ".jpeg")
    return [str(p) for p in sorted(examples_dir.glob("*")) if p.suffix.lower() in exts]


interface = gr.Interface(
    fn=extract_urdu_text,
    inputs=gr.Image(type="pil", label="Upload Urdu Image"),
    outputs=gr.Textbox(label="Extracted Urdu Text", rtl=True),
    title="Urdu OCR -- Code Saviours SI-26",
    description=(
        "Upload an image containing Urdu text (printed, signage, or a photo of a page) "
        "and get the extracted text. Fine-tuned from microsoft/trocr-base-printed."
    ),
    examples=_example_images() or None,
    flagging_mode="never",
)

if __name__ == "__main__":
    interface.launch(share=os.environ.get("GRADIO_SHARE", "false").lower() == "true")
