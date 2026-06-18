# -*- coding: utf-8 -*-
"""
Common layout vocabulary and normalization for Game UI classification.
Positions, Element Types, and Roles are standardized across the project.
"""

from typing import Any, Dict, List, Optional

POSITIONS = [
    "top_left", "top_center", "top_right", "left", "center_left", "center",
    "center_right", "right", "bottom_left", "bottom_center", "bottom_right",
    "full_screen", "overlay",
]

ELEMENT_TYPES = [
    "bar", "panel", "popup", "menu", "tab_bar", "grid", "list", "card_group",
    "slot_group", "button_group", "preview", "dialogue_box", "minimap",
    "skill_bar", "health_bar", "resource_bar", "tooltip", "notification",
    "portrait_group", "progress_bar", "chat_box",
]

ROLES = [
    "status", "health", "resource", "combat", "navigation", "quest",
    "inventory", "equipment", "character", "skill", "shop", "crafting",
    "dialogue", "settings", "social", "notification", "tutorial", "result",
    "loading", "selection", "system", "unknown",
]

# Common aliases for normalization
LAYOUT_ALIASES = {
    "hp_bar": "health_bar",
    "text_box": "dialogue_box",
    "conversation_box": "dialogue_box",
    "button": "button_group",
    "buttons": "button_group",
    "item_grid": "grid",
    "stats": "status",
    "battle": "combat",
    "hp": "health",
    "gold": "resource",
    "money": "resource",
    "items": "inventory",
    "journal": "quest",
    "mission": "quest",
    "map": "navigation",
    "abilities": "skill",
    "store": "shop",
    "forge": "crafting",
    "options": "settings",
    "config": "settings",
    "reward": "result",
    "victory": "result",
    "defeat": "result",
}


def normalize_layout_value(value: Any, axis: str) -> str:
    """
    Standardize a layout value based on the axis.
    axis must be one of: 'position', 'element_type', 'role'
    """
    if axis == "position":
        allowed = POSITIONS
        default = "center"
    elif axis == "element_type":
        allowed = ELEMENT_TYPES
        default = "panel"
    elif axis == "role":
        allowed = ROLES
        default = "unknown"
    else:
        return str(value).strip().lower()

    if value is None:
        return default
    
    # Basic cleanup
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return default

    # Apply aliases
    text = LAYOUT_ALIASES.get(text, text)

    if text in allowed:
        return text
    
    return default
