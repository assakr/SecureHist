"""One-time conversion of Quilt-LLaVA (original LLaVA-1.5 format) to the standard Hugging Face
LlavaForConditionalGeneration format, so it runs with plain `transformers` on Windows
(the original LLaVA code base does not support Windows).

  python securehist/convert_quilt_llava.py                      # -> models/quilt-llava-v1.5-7b-hf
  python securehist/convert_quilt_llava.py --src <local folder of the original checkpoint>

Needs about 16 GB of free RAM and about 30 GB of disk (download + converted copy). Run once.
Quilt-LLaVA licence: research use (CC BY-NC 3.0 / Llama 2 Community License).
"""
import argparse
import glob
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
SRC_REPO = "wisdomik/Quilt-Llava-v1.5-7b"
CLIP_ID = "openai/clip-vit-large-patch14-336"
DEFAULT_OUT = os.path.join(PROJECT, "models", "quilt-llava-v1.5-7b-hf")


def load_original(folder):
    sd = {}
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    if files:
        from safetensors.torch import load_file
        for f in files:
            sd.update(load_file(f))
    else:
        for f in sorted(glob.glob(os.path.join(folder, "pytorch_model*.bin"))):
            sd.update(torch.load(f, map_location="cpu", weights_only=True))
    if not sd:
        raise SystemExit(f"No weights found in {folder}")
    return sd


def legacy_name(k):
    """Original LLaVA key -> transformers legacy LLaVA key (or None to drop)."""
    if k.startswith("model.vision_tower.vision_tower.vision_model."):
        return "vision_tower." + k[len("model.vision_tower.vision_tower.vision_model."):]
    if k.startswith("model.vision_tower.vision_tower."):
        return "vision_tower." + k[len("model.vision_tower.vision_tower."):]
    if k.startswith("model.mm_projector.0."):
        return "multi_modal_projector.linear_1." + k.split(".")[-1]
    if k.startswith("model.mm_projector.2."):
        return "multi_modal_projector.linear_2." + k.split(".")[-1]
    if k == "lm_head.weight":
        return "language_model.lm_head.weight"
    if k.startswith("model."):
        return "language_model.model." + k[len("model."):]
    return None


def to_model_layout(legacy_key, model_keys):
    """transformers >= 4.52 nests modules under `model.`; map legacy names onto whatever this version uses."""
    if legacy_key in model_keys:
        return legacy_key
    if legacy_key.startswith("language_model.model."):
        k = "model.language_model." + legacy_key[len("language_model.model."):]
    elif legacy_key == "language_model.lm_head.weight":
        k = "lm_head.weight"
    else:
        k = "model." + legacy_key
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="local folder of the original checkpoint (default: download)")
    ap.add_argument("--clip", default=CLIP_ID)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    from transformers import (AddedToken, AutoTokenizer, CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel,
                              LlamaConfig, LlavaConfig, LlavaForConditionalGeneration, LlavaProcessor)

    src = args.src
    if src is None:
        from huggingface_hub import snapshot_download
        src = snapshot_download(SRC_REPO, allow_patterns=["*.json", "*.model", "*.safetensors", "*.bin", "*.txt"])
    raw_cfg = json.load(open(os.path.join(src, "config.json")))
    print("[convert] source:", src, "| mm_vision_tower:", raw_cfg.get("mm_vision_tower"),
          "| projector:", raw_cfg.get("mm_projector_type"), "| select layer:", raw_cfg.get("mm_vision_select_layer"))
    if raw_cfg.get("mm_projector_type", "mlp2x_gelu") != "mlp2x_gelu":
        raise SystemExit("Unexpected projector type; this converter handles LLaVA-1.5 (mlp2x_gelu) only.")

    text_cfg = LlamaConfig(**{k: v for k, v in raw_cfg.items() if not k.startswith("mm_") and k not in
                              ("architectures", "model_type", "auto_map", "_name_or_path", "image_aspect_ratio",
                               "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", "use_mm_proj", "tokenizer_padding_side",
                               "tokenizer_model_max_length")})
    vis_cfg = CLIPVisionConfig.from_pretrained(args.clip)
    config = LlavaConfig(text_config=text_cfg, vision_config=vis_cfg, image_token_index=32000,
                         projector_hidden_act="gelu", vision_feature_layer=raw_cfg.get("mm_vision_select_layer", -2),
                         vision_feature_select_strategy="default", image_seq_length=576)
    config.pad_token_id = 32001

    with torch.device("meta"):
        model = LlavaForConditionalGeneration(config)
    model_keys = set(model.state_dict().keys())

    state = {}
    # CLIP vision weights first (LLaVA-1.5 keeps CLIP frozen; the checkpoint may or may not include them)
    clip = CLIPVisionModel.from_pretrained(args.clip, torch_dtype=torch.float16)
    for k, v in clip.state_dict().items():
        state[to_model_layout("vision_tower." + k, model_keys)] = v
    del clip
    orig = load_original(src)
    n_vis_ckpt = 0
    for k, v in orig.items():
        lk = legacy_name(k)
        if lk is None or "rotary_emb.inv_freq" in k:
            continue
        n_vis_ckpt += lk.startswith("vision_tower.")
        state[to_model_layout(lk, model_keys)] = v.to(torch.float16)
    del orig
    print(f"[convert] vision weights taken from the checkpoint: {n_vis_ckpt} (0 = CLIP weights used as released)")

    missing = sorted(model_keys - set(state)); unexpected = sorted(set(state) - model_keys)
    if missing or unexpected:
        raise SystemExit(f"Key mismatch. missing={missing[:8]} ({len(missing)}), unexpected={unexpected[:8]} ({len(unexpected)})")
    model.load_state_dict(state, strict=True, assign=True)

    tok = AutoTokenizer.from_pretrained(src, use_fast=False)
    tok.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    tok.add_special_tokens({"pad_token": "<pad>"})
    assert tok.convert_tokens_to_ids("<image>") == 32000, tok.convert_tokens_to_ids("<image>")
    model.resize_token_embeddings(len(tok), pad_to_multiple_of=64)   # new rows: mean of existing embeddings
    proc = LlavaProcessor(image_processor=CLIPImageProcessor.from_pretrained(args.clip), tokenizer=tok,
                          patch_size=14, vision_feature_select_strategy="default",
                          num_additional_image_tokens=1)   # CLS token, as in llava-hf/llava-1.5-7b-hf
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    proc.save_pretrained(args.out)
    print("[convert] saved to", args.out)


if __name__ == "__main__":
    main()
