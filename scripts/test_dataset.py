# -*- coding: utf-8 -*-
"""
Dataset smoke test for the Game UI schema v6.0 pipeline.

This script checks whether data/metadata.csv can be loaded with the new
primary_screen_type + style_tags + layout_tokens structure.
"""

import json
import os
from typing import Any, List, Tuple

import pandas as pd
from transformers import AutoProcessor

from dataset import GameUIDataset


CSV_FILE = "data/metadata.csv"
IMG_DIR = "data/images"
MODEL_NAME = "google/siglip2-base-patch16-224"

DEFAULT_PRIMARY_SCREEN_TYPES = [
    "gameplay_hud",
    "main_menu",
    "title_screen",
    "lobby",
    "inventory",
    "equipment",
    "character_screen",
    "skill_tree",
    "map",
    "quest",
    "shop",
    "crafting",
    "dialogue",
    "settings",
    "pause_menu",
    "battle_result",
    "loading_screen",
    "tutorial",
    "other",
]

DEFAULT_STYLE_TAGS = [
    "fantasy",
    "dark_fantasy",
    "medieval",
    "sci_fi",
    "cyberpunk",
    "military",
    "horror",
    "realistic",
    "cartoon",
    "pixel_art",
    "anime",
    "minimal",
    "clean",
    "skeuomorphic",
    "flat",
    "neon",
    "retro",
    "modern",
    "cute",
    "gritty",
]

DEFAULT_LAYOUT_TOKENS = [
    "top_left:health_bar:health",
    "top_left:bar:status",
    "top_center:bar:status",
    "top_right:minimap:navigation",
    "top_right:resource_bar:resource",
    "left:menu:navigation",
    "left:panel:social",
    "center:popup:system",
    "center:preview:character",
    "center:grid:inventory",
    "right:panel:quest",
    "right:panel:character",
    "bottom_left:chat_box:social",
    "bottom_center:skill_bar:combat",
    "bottom_center:dialogue_box:dialogue",
    "bottom_right:slot_group:inventory",
    "full_screen:menu:navigation",
    "full_screen:panel:settings",
]


REQUIRED_COLUMNS = [
    "file_name",
    "is_game_ui",
    "ui_quality",
    "primary_screen_type",
    "style_tags",
    "layout_tokens",
    "components",
    "review_status",
]


def parse_json_list(value: Any) -> List[str]:
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass

    return [item.strip() for item in text.split(",") if item.strip()]


def get_vocabs_from_metadata(csv_file: str) -> Tuple[List[str], List[str], List[str]]:
    if not os.path.exists(csv_file):
        print(f"[!] Metadata file not found: {csv_file}")
        print("[!] Run build_mobygames_dataset.py and labeling_tool.py first.")
        return DEFAULT_PRIMARY_SCREEN_TYPES, DEFAULT_STYLE_TAGS, DEFAULT_LAYOUT_TOKENS

    df = pd.read_csv(csv_file)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    primary_screen_types = sorted(
        {
            str(value).strip()
            for value in df["primary_screen_type"].dropna().tolist()
            if str(value).strip()
        }
    )
    if "other" not in primary_screen_types:
        primary_screen_types.append("other")
    if not primary_screen_types:
        primary_screen_types = DEFAULT_PRIMARY_SCREEN_TYPES

    style_tags = sorted(
        {
            tag
            for value in df["style_tags"].dropna().tolist()
            for tag in parse_json_list(value)
        }
    )
    if not style_tags:
        style_tags = DEFAULT_STYLE_TAGS

    layout_tokens = sorted(
        {
            token
            for value in df["layout_tokens"].dropna().tolist()
            for token in parse_json_list(value)
        }
    )
    if not layout_tokens:
        layout_tokens = DEFAULT_LAYOUT_TOKENS

    return primary_screen_types, style_tags, layout_tokens


def test_dataset() -> None:
    primary_screen_types, style_tags, layout_tokens = get_vocabs_from_metadata(CSV_FILE)

    print(f"[*] Primary screen types: {len(primary_screen_types)}")
    print(f"[*] Style tags: {len(style_tags)}")
    print(f"[*] Layout tokens: {len(layout_tokens)}")

    if not os.path.exists(CSV_FILE):
        return

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    dataset = GameUIDataset(
        csv_file=CSV_FILE,
        img_dir=IMG_DIR,
        processor=processor,
        primary_screen_types=primary_screen_types,
        style_tags=style_tags,
        layout_tokens=layout_tokens,
    )

    print(f"[+] Dataset loaded successfully. Length: {len(dataset)}")

    sample = dataset[0]
    print(f"[+] Sample file: {sample['file_name']}")
    print(f"[+] pixel_values shape: {tuple(sample['pixel_values'].shape)}")
    print(f"[+] primary_screen_label shape: {tuple(sample['primary_screen_label'].shape)}")
    print(f"[+] style_label shape: {tuple(sample['style_label'].shape)}")
    print(f"[+] layout_label shape: {tuple(sample['layout_label'].shape)}")

    assert sample["style_label"].shape[0] == len(style_tags)
    assert sample["layout_label"].shape[0] == len(layout_tokens)
    print("[+] Dataset smoke test passed.")


if __name__ == "__main__":
    try:
        test_dataset()
    except Exception as exc:
        print(f"[!] Dataset test failed: {exc}")
        raise
