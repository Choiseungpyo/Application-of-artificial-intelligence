# -*- coding: utf-8 -*-
"""
Inspect the current GameUIModel structure for the v6 dataset schema.

This script checks:
- SigLIP2 backbone structure
- GameUIModel output heads
- Optional checkpoint vocab sizes
- Output tensor shapes from a dummy forward pass

Usage:
    python inspect_model.py
    python inspect_model.py --checkpoint output/best_model.pth
    python inspect_model.py --unfreeze_last_n 2
"""

import argparse
import os
from typing import Any, Dict, List

import torch

from model import GameUIModel


DEFAULT_MODEL_NAME = "google/siglip2-base-patch16-224"

DEFAULT_PRIMARY_SCREEN_TYPES = [
    "gameplay_hud",
    "main_menu",
    "lobby",
    "inventory",
    "equipment",
    "map",
    "shop",
    "settings",
    "dialogue",
    "quest",
    "character_screen",
    "skill_tree",
    "crafting",
    "battle_result",
    "loading_screen",
    "title_screen",
    "tutorial",
    "other",
]

DEFAULT_VISUAL_STYLE_TAGS = [
    "realistic", "cartoon", "anime", "pixel_art", "retro", "modern", "minimal", "clean", "skeuomorphic", "flat", "neon", "gritty", "cute"
]

DEFAULT_THEME_TAGS = [
    "fantasy", "dark_fantasy", "medieval", "sci_fi", "cyberpunk", "military", "horror", "post_apocalyptic", "historical", "modern_world"
]

DEFAULT_LAYOUT_TOKENS = [
    "top_left:health_bar:health",
    "top_right:minimap:navigation",
    "bottom_center:skill_bar:combat",
    "right:panel:quest",
    "center:popup:system",
    "left:menu:navigation",
]


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}


def print_parameter_summary(model: torch.nn.Module) -> None:
    counts = count_parameters(model)
    print("\n[Parameter Summary]")
    print(f"Total parameters    : {counts['total']:,}")
    print(f"Trainable parameters: {counts['trainable']:,}")
    print(f"Frozen parameters   : {counts['frozen']:,}")


def print_head_summary(model: GameUIModel) -> None:
    print("\n[Head Summary]")
    print(f"Hidden size              : {model.hidden_size}")
    print(f"Primary screen head out  : {model.primary_screen_head.out_features}")
    print(f"Visual Style head out    : {model.visual_style_head.out_features}")
    print(f"Theme head out           : {model.theme_head.out_features}")
    print(f"Layout head out          : {model.layout_head.out_features}")


def print_backbone_summary(model: GameUIModel) -> None:
    print("\n[Backbone Summary]")
    backbone = model.backbone
    print(f"Backbone class: {backbone.__class__.__name__}")

    if hasattr(backbone, "vision_model"):
        print("Has vision_model: yes")
        vision_model = backbone.vision_model
        if hasattr(vision_model, "encoder") and hasattr(vision_model.encoder, "layers"):
            layers = vision_model.encoder.layers
            print(f"Encoder layers: {len(layers)}")
        else:
            print("Encoder layers: not found")
    else:
        print("Has vision_model: no")


def load_checkpoint_vocab(path: str) -> Dict[str, List[str]]:
    if not path:
        return {}

    if not os.path.exists(path):
        print(f"[!] Checkpoint not found: {path}")
        return {}

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    vocab = ckpt.get("vocab", {})
    if not isinstance(vocab, dict):
        return {}
    return vocab


def get_label_spaces(args: argparse.Namespace) -> Dict[str, List[str]]:
    vocab = load_checkpoint_vocab(args.checkpoint) if args.checkpoint else {}

    primary_screen_types = vocab.get("primary_screen_types") or DEFAULT_PRIMARY_SCREEN_TYPES
    visual_style_tags = vocab.get("visual_style_tags") or DEFAULT_VISUAL_STYLE_TAGS
    theme_tags = vocab.get("theme_tags") or DEFAULT_THEME_TAGS
    layout_tokens = vocab.get("layout_tokens") or DEFAULT_LAYOUT_TOKENS

    return {
        "primary_screen_types": list(primary_screen_types),
        "visual_style_tags": list(visual_style_tags),
        "theme_tags": list(theme_tags),
        "layout_tokens": list(layout_tokens),
    }


def print_vocab_summary(label_spaces: Dict[str, List[str]]) -> None:
    print("\n[Vocab Summary]")
    print(f"Primary screen types: {len(label_spaces['primary_screen_types'])}")
    print(f"Visual Style tags   : {len(label_spaces['visual_style_tags'])}")
    print(f"Theme tags          : {len(label_spaces['theme_tags'])}")
    print(f"Layout tokens       : {len(label_spaces['layout_tokens'])}")

    for key in ["primary_screen_types", "visual_style_tags", "theme_tags", "layout_tokens"]:
        preview = label_spaces[key][:8]
        suffix = " ..." if len(label_spaces[key]) > 8 else ""
        print(f"- {key}: {preview}{suffix}")


def load_model_state_if_available(model: GameUIModel, checkpoint_path: str) -> None:
    if not checkpoint_path:
        return

    if not os.path.exists(checkpoint_path):
        return

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict")
    if not state:
        print("[!] Checkpoint has no model_state_dict. Skipping state load.")
        return

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("\n[Checkpoint Load]")
    print(f"Loaded from: {checkpoint_path}")
    print(f"Missing keys   : {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"Missing preview: {missing[:5]}")
    if unexpected:
        print(f"Unexpected preview: {unexpected[:5]}")


def run_dummy_forward(model: GameUIModel, batch_size: int) -> None:
    print("\n[Dummy Forward]")
    model.eval()
    dummy = torch.randn(batch_size, 3, 224, 224)
    with torch.inference_mode():
        outputs = model(dummy)

    expected_keys = [
        "logits_primary_screen_type",
        "logits_visual_style_tags",
        "logits_theme_tags",
        "logits_layout_tokens",
    ]

    for key in expected_keys:
        if key not in outputs:
            print(f"[!] Missing output key: {key}")
            continue
        print(f"{key}: {tuple(outputs[key].shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--unfreeze_last_n", type=int, default=0)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    args = parser.parse_args()

    label_spaces = get_label_spaces(args)
    print_vocab_summary(label_spaces)

    model = GameUIModel(
        model_name=args.model_name,
        num_primary_screen_types=len(label_spaces["primary_screen_types"]),
        num_visual_style_tags=len(label_spaces["visual_style_tags"]),
        num_theme_tags=len(label_spaces["theme_tags"]),
        num_layout_tokens=len(label_spaces["layout_tokens"]),
        freeze_backbone=args.freeze_backbone,
    )

    if args.unfreeze_last_n > 0:
        model.unfreeze_backbone(args.unfreeze_last_n)

    load_model_state_if_available(model, args.checkpoint)

    print_backbone_summary(model)
    print_head_summary(model)
    print_parameter_summary(model)
    run_dummy_forward(model, args.batch_size)

    print("\n[+] Model inspection completed.")


if __name__ == "__main__":
    main()
