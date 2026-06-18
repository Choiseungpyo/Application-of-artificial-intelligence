# -*- coding: utf-8 -*-
"""
Prompt builder for the new Game UI dataset schema.

This module converts structured metadata into a generation prompt.
It is mainly for the future UI generation feature, not for the current
search-only pipeline.

Expected schema version: 6.x
Key fields:
- primary_screen_type: str
- secondary_screen_types: list[str]
- visual_style_tags: list[str]
- theme_tags: list[str]
- layout_blocks: list[dict(position, element_type, role)]
- layout_tokens: list[str]
- components: list[str]
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping


QUALITY_SUFFIX = (
    "high fidelity game UI concept, clean readable typography, "
    "clear hierarchy, production-ready interface mockup, no real game logo, "
    "no copyrighted character, no watermark"
)


def parse_list(value: Any) -> List[str]:
    """Parse JSON array, comma-separated text, or Python list into list[str]."""
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

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


def parse_layout_blocks(value: Any) -> List[Dict[str, str]]:
    """Parse layout_blocks into normalized dictionaries."""
    if value is None:
        return []

    raw_blocks: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            raw_blocks = json.loads(text.replace("'", '"'))
        except Exception:
            return []

    if not isinstance(raw_blocks, list):
        return []

    blocks: List[Dict[str, str]] = []
    for block in raw_blocks:
        if not isinstance(block, Mapping):
            continue
        position = str(block.get("position", "")).strip()
        element_type = str(block.get("element_type", "")).strip()
        role = str(block.get("role", "")).strip()
        if not position and not element_type and not role:
            continue
        blocks.append(
            {
                "position": position,
                "element_type": element_type,
                "role": role,
            }
        )
    return blocks


def prettify_token(value: str) -> str:
    """Convert snake_case-like labels into generation-friendly words."""
    return str(value).replace("_", " ").replace(":", " ").strip()


def unique_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def layout_block_to_phrase(block: Mapping[str, str]) -> str:
    """Convert a layout block into a short phrase."""
    position = prettify_token(str(block.get("position", "")))
    element_type = prettify_token(str(block.get("element_type", "")))
    role = prettify_token(str(block.get("role", "")))

    pieces = []
    if position:
        pieces.append(f"{position} area")
    if element_type:
        pieces.append(element_type)
    if role:
        pieces.append(f"for {role}")

    return " ".join(pieces).strip()


def build_layout_phrases(metadata: Mapping[str, Any]) -> List[str]:
    """Build readable layout phrases from layout_blocks or layout_tokens."""
    blocks = parse_layout_blocks(metadata.get("layout_blocks"))
    phrases = [layout_block_to_phrase(block) for block in blocks]
    phrases = [phrase for phrase in phrases if phrase]

    if phrases:
        return unique_keep_order(phrases)

    # Fallback for metadata that only has layout_tokens.
    tokens = parse_list(metadata.get("layout_tokens"))
    fallback = [prettify_token(token) for token in tokens]
    return unique_keep_order(fallback)


def build_generation_prompt(metadata: Mapping[str, Any]) -> str:
    """
    Build a text prompt from the new structured Game UI metadata.

    Args:
        metadata: dictionary-like object using the new schema.

    Returns:
        A prompt string suitable for an image generation model.
    """
    primary_screen_type = str(metadata.get("primary_screen_type", "")).strip()
    secondary_screen_types = parse_list(metadata.get("secondary_screen_types"))
    visual_style_tags = parse_list(metadata.get("visual_style_tags"))
    theme_tags = parse_list(metadata.get("theme_tags"))
    components = parse_list(metadata.get("components"))
    layout_phrases = build_layout_phrases(metadata)

    prompt_parts: List[str] = []

    if primary_screen_type:
        prompt_parts.append(f"A professional game {prettify_token(primary_screen_type)} interface")
    else:
        prompt_parts.append("A professional game user interface")

    if secondary_screen_types:
        secondary_text = ", ".join(prettify_token(item) for item in secondary_screen_types)
        prompt_parts.append(f"including secondary UI elements such as {secondary_text}")

    combined_styles = unique_keep_order(theme_tags + visual_style_tags)
    if combined_styles:
        style_text = ", ".join(prettify_token(tag) for tag in combined_styles)
        prompt_parts.append(f"with a {style_text} visual style")

    if layout_phrases:
        layout_text = "; ".join(layout_phrases)
        prompt_parts.append(f"layout structure: {layout_text}")

    if components:
        component_text = ", ".join(prettify_token(component) for component in components)
        prompt_parts.append(f"featuring {component_text}")

    prompt_parts.append(QUALITY_SUFFIX)

    return ". ".join(part for part in prompt_parts if part).strip()


def build_negative_prompt() -> str:
    """Return a negative prompt for UI image generation."""
    return (
        "blurry, low resolution, unreadable text, cluttered layout, broken UI, "
        "real copyrighted logo, real copyrighted character, watermark, photo of a monitor"
    )


if __name__ == "__main__":
    sample_metadata = {
        "primary_screen_type": "gameplay_hud",
        "secondary_screen_types": ["quest_panel"],
        "visual_style_tags": ["realistic"],
        "theme_tags": ["dark_fantasy"],
        "layout_blocks": [
            {"position": "top_left", "element_type": "health_bar", "role": "health"},
            {"position": "top_right", "element_type": "minimap", "role": "navigation"},
            {"position": "bottom_center", "element_type": "skill_bar", "role": "combat"},
        ],
        "components": ["health_bar", "minimap", "skill_icons", "quest_list"],
    }

    print("Prompt:")
    print(build_generation_prompt(sample_metadata))
    print("\nNegative prompt:")
    print(build_negative_prompt())
