"""Neutral VLM auditor (Qwen2-VL-2B-Instruct).

The prompt states what the inputs are and asks for a JSON verdict. It contains NO rule
that dictates the answer (the earlier prompt told the model to answer "STEALTHY
ADVERSARIAL ATTACK" under given conditions, which made its verdicts uninformative).
Every call is logged verbatim by the caller.
"""
from __future__ import annotations

import json
import re

import cv2
import numpy as np
from PIL import Image

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

PROMPT_WITH_NUMBERS = (
    "You are auditing a histopathology image for tampering.\n"
    "Image 1: the H&E image as received.\n"
    "Image 2: a watermark integrity map of the same image. Each 16x16 block is coloured by how far its "
    "embedded watermark is from the expected value: dark = intact, bright = damaged. An untouched image is dark "
    "everywhere; any pixel edit damages the watermark where it was made.\n"
    "Measurements: watermark agreement NC = {nc:.3f} (about 0.7-0.9 for an untouched image, about 0 when the "
    "watermark is absent or destroyed); share of blocks flagged as damaged = {flag:.3f}; classifier prediction = "
    "{pred} with confidence {conf:.3f}.\n"
    "Decide whether the image was modified after watermarking. Answer with JSON only, in this form: "
    '{{"verdict": "TAMPERED" or "AUTHENTIC", "confidence": a number from 0 to 1, "reason": "one short sentence"}}'
)

PROMPT_IMAGES_ONLY = (
    "You are auditing a histopathology image for tampering.\n"
    "Image 1: the H&E image as received.\n"
    "Image 2: a watermark integrity map of the same image: dark = intact, bright = damaged. An untouched "
    "image is dark everywhere.\n"
    "Decide whether the image was modified after watermarking. Answer with JSON only, in this form: "
    '{"verdict": "TAMPERED" or "AUTHENTIC", "confidence": a number from 0 to 1, "reason": "one short sentence"}'
)


PROMPT_TEXT_ONLY = (
    "You are auditing a histopathology image for tampering, using only these measurements.\n"
    "Watermark agreement NC = {nc:.3f} (about 0.7-0.9 for an untouched image, about 0 when the watermark is absent "
    "or destroyed); share of 16x16 blocks whose watermark is damaged = {flag:.3f}; classifier prediction = {pred} "
    "with confidence {conf:.3f}.\n"
    "Decide whether the image was modified after watermarking. Answer with JSON only, in this form: "
    '{{"verdict": "TAMPERED" or "AUTHENTIC", "confidence": a number from 0 to 1, "reason": "one short sentence"}}'
)


def integrity_map_image(score_map: np.ndarray, size: int = 448) -> Image.Image:
    """Block score map (0 intact .. 0.25 random) -> colour image."""
    norm = np.clip(score_map / 0.25, 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


class VLMAuditor:
    def __init__(self, model_id: str = MODEL_ID, prompt_mode: str = "with_numbers", examples=None):
        """prompt_mode: with_numbers | images_only | text_only | fewshot (with_numbers prompt preceded by
        worked examples: list of dicts with rgb, score_map, nc, flag, pred, conf, answer)."""
        self.examples = examples or []
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        kwargs = {"device_map": "auto"}
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                               bnb_4bit_quant_type="nf4")
            self.precision = "4-bit nf4"
        except Exception:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            self.precision = str(kwargs["torch_dtype"])
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=128 * 28 * 28, max_pixels=448 * 448)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
        self.model_id = model_id
        self.prompt_mode = prompt_mode

    def build_prompt(self, nc, flag, pred, conf):
        if self.prompt_mode == "images_only":
            return PROMPT_IMAGES_ONLY
        if self.prompt_mode == "text_only":
            return PROMPT_TEXT_ONLY.format(nc=nc, flag=flag, pred=pred, conf=conf)
        return PROMPT_WITH_NUMBERS.format(nc=nc, flag=flag, pred=pred, conf=conf)

    def audit(self, rgb: np.ndarray, score_map: np.ndarray, nc: float, flag: float, pred: str, conf: float) -> dict:
        import torch
        prompt = self.build_prompt(nc, flag, pred, conf)
        if self.prompt_mode == "text_only":
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.model.device)
        else:
            messages, images = [], []
            for ex in (self.examples if self.prompt_mode == "fewshot" else []):
                messages.append({"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text",
                                 "text": PROMPT_WITH_NUMBERS.format(nc=ex["nc"], flag=ex["flag"], pred=ex["pred"], conf=ex["conf"])}]})
                messages.append({"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]})
                images += [Image.fromarray(ex["rgb"]).resize((448, 448)), integrity_map_image(ex["score_map"])]
            messages.append({"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt}]})
            images += [Image.fromarray(rgb).resize((448, 448)), integrity_map_image(score_map)]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        raw = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return {"prompt": prompt, "raw": raw, **parse_verdict(raw)}


def parse_verdict(raw: str) -> dict:
    mv = re.search(r"VERDICT\s*[:=]\s*\**\s*(TAMPERED|AUTHENTIC)([^\n]*)", raw, re.I)
    if mv and re.search(r"TAMPERED|AUTHENTIC", mv.group(2), re.I):      # template echoed back
        return {"verdict": "UNPARSED", "vlm_conf": np.nan, "reason": raw.strip()[:200]}
    if mv:
        mc = re.search(r"CONFIDENCE\s*[:=]\s*([01](?:\.\d+)?)", raw, re.I)
        mr = re.search(r"REASON\s*[:=]\s*(.+)", raw, re.I)
        return {"verdict": mv.group(1).upper(), "vlm_conf": float(mc.group(1)) if mc else np.nan,
                "reason": mr.group(1).strip()[:200] if mr else ""}
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = str(obj.get("verdict", "")).upper()
            if v in ("TAMPERED", "AUTHENTIC"):
                return {"verdict": v, "vlm_conf": float(obj.get("confidence", np.nan)), "reason": str(obj.get("reason", ""))}
        except (ValueError, TypeError):
            pass
    up = raw.upper()
    if "TAMPERED" in up and "AUTHENTIC" not in up:
        return {"verdict": "TAMPERED", "vlm_conf": np.nan, "reason": raw.strip()[:200]}
    if "AUTHENTIC" in up and "TAMPERED" not in up:
        return {"verdict": "AUTHENTIC", "vlm_conf": np.nan, "reason": raw.strip()[:200]}
    return {"verdict": "UNPARSED", "vlm_conf": np.nan, "reason": raw.strip()[:200]}


# ----------------------------------------------------------------------------- Quilt-LLaVA
QUILT_SYSTEM = ("A chat between a curious human and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the human's questions.")
QUILT_MAP = (
    "The picture has two panels. Left: an H&E histopathology image as received. Right: a watermark integrity map "
    "of the same image, where each block is dark if its embedded watermark is intact and bright if it is damaged. "
    "An untouched image gives a dark map everywhere; any pixel edit makes the map bright where it was made.\n"
    "Measurements: watermark agreement NC = {nc:.3f} (about 0.7-0.9 for an untouched image, about 0 when the "
    "watermark is destroyed); share of damaged blocks = {flag:.3f}; classifier prediction = {pred} "
    "(confidence {conf:.3f}).\n"
    "Was the image modified after watermarking? Reply exactly in this form:\n"
    "VERDICT: TAMPERED or AUTHENTIC\nCONFIDENCE: a number from 0 to 1\nREASON: one short sentence")
QUILT_TISSUE = (
    "This is an H&E histopathology image. A classifier labelled it {pred} with confidence {conf:.3f}. "
    "A watermark check gives agreement NC = {nc:.3f} (about 0.7-0.9 for an untouched image, about 0 when the "
    "watermark is destroyed) and a share of damaged blocks of {flag:.3f}.\n"
    "Looking at the tissue and these measurements, was the image digitally modified (for example by an "
    "adversarial perturbation, a pasted region or local noise)? Reply exactly in this form:\n"
    "VERDICT: TAMPERED or AUTHENTIC\nCONFIDENCE: a number from 0 to 1\nREASON: one short sentence")
CLASS_NAMES = {"colon_aca": "colon adenocarcinoma", "colon_n": "benign colon tissue", "lung_aca": "lung adenocarcinoma"}


def side_by_side(rgb: np.ndarray, score_map: np.ndarray, size: int = 672) -> Image.Image:
    """Square canvas (LLaVA-1.5 crops non-square inputs): H&E left, integrity map right, grey bands top and bottom."""
    half = size // 2
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(Image.fromarray(rgb).resize((half, half)), (0, half // 2))
    canvas.paste(integrity_map_image(score_map, half), (half, half // 2))
    return canvas


class QuiltAuditor:
    """Quilt-LLaVA v1.5 7B (pathology instruction-tuned LLaVA-1.5), converted to the transformers format by
    convert_quilt_llava.py. mode "map": H&E + integrity map in one picture; mode "tissue": H&E only."""

    def __init__(self, model_path: str, mode: str = "map"):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        kwargs = {"device_map": "auto"}
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                               bnb_4bit_quant_type="nf4")
            self.precision = "4-bit nf4"
        except Exception:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            self.precision = str(kwargs["torch_dtype"])
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_path, **kwargs).eval()
        self.model_id = "Quilt-LLaVA-v1.5-7b"
        self.mode = mode

    def audit(self, rgb, score_map, nc, flag, pred, conf) -> dict:
        import torch
        pred = CLASS_NAMES.get(pred, pred)
        if self.mode == "map":
            img, q = side_by_side(rgb, score_map), QUILT_MAP.format(nc=nc, flag=flag, pred=pred, conf=conf)
        else:
            img, q = Image.fromarray(rgb), QUILT_TISSUE.format(nc=nc, flag=flag, pred=pred, conf=conf)
        text = f"{QUILT_SYSTEM} USER: <image>\n{q} ASSISTANT:"
        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype if self.model.dtype != torch.int8 else torch.float16)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        raw = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return {"prompt": q, "raw": raw, **parse_verdict(raw)}
