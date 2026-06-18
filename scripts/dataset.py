# -*- coding: utf-8 -*-
"""
Game UI dataset loader for schema v6.0.

Expected metadata columns:
- file_name
- is_game_ui
- ui_quality
- primary_screen_type
- secondary_screen_types
- style_tags
- layout_blocks
- layout_tokens
- components
- confidence
- needs_review
- review_status

This dataset is designed for the new pipeline:
primary_screen_type: single-label classification
style_tags: multi-label classification
layout_tokens: multi-label classification
"""

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from layout_vocab import POSITIONS, ELEMENT_TYPES, ROLES, normalize_layout_value


REQUIRED_COLUMNS = [
    "file_name",
    "is_game_ui",
    "ui_quality",
    "primary_screen_type",
    "visual_style_tags",
    "theme_tags",
    "layout_tokens",
    "components",
    "review_status",
]


VALID_QUALITY_FOR_TRAINING = {"keep"}
VALID_STATUS_FOR_TRAINING = {"labeled"}




class GameUIDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        processor,
        primary_screen_types: List[str],
        visual_style_tags: List[str],
        theme_tags: List[str],
        layout_positions: List[str],
        layout_element_types: List[str],
        layout_roles: List[str],
        transform=None,
        require_image_exists: bool = True,
        use_grouped: bool = False,
    ):
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.processor = processor
        self.primary_screen_types = [str(x).strip() for x in primary_screen_types if str(x).strip()]
        self.visual_style_tags = [str(x).strip() for x in visual_style_tags if str(x).strip()]
        self.theme_tags = [str(x).strip() for x in theme_tags if str(x).strip()]
        self.layout_positions = [str(x).strip() for x in layout_positions if str(x).strip()]
        self.layout_element_types = [str(x).strip() for x in layout_element_types if str(x).strip()]
        self.layout_roles = [str(x).strip() for x in layout_roles if str(x).strip()]
        self.transform = transform
        self.require_image_exists = require_image_exists
        self.use_grouped = use_grouped

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Metadata file not found: {csv_file}")

        self.df = pd.read_csv(csv_file)
        self._validate_schema()
        self.df = self._filter_trainable_rows(self.df)

        if len(self.df) == 0:
            raise ValueError(
                f"No trainable samples found in {csv_file}. "
                "Expected rows with review_status='labeled', is_game_ui=true, "
                "ui_quality='keep', and non-empty primary_screen_type."
            )

        self.primary_screen_to_idx = {t: i for i, t in enumerate(self.primary_screen_types)}
        self.visual_style_tag_to_idx = {t: i for i, t in enumerate(self.visual_style_tags)}
        self.theme_tag_to_idx = {t: i for i, t in enumerate(self.theme_tags)}
        self.layout_position_to_idx = {t: i for i, t in enumerate(self.layout_positions)}
        self.layout_element_type_to_idx = {t: i for i, t in enumerate(self.layout_element_types)}
        self.layout_role_to_idx = {t: i for i, t in enumerate(self.layout_roles)}

        fallback_cls = "flow_other" if self.use_grouped else "other"
        if fallback_cls not in self.primary_screen_to_idx:
            raise ValueError(f"primary_screen_types must include '{fallback_cls}'.")

    def _validate_schema(self) -> None:
        missing = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(
                "Invalid metadata schema. Missing columns: " + ", ".join(missing)
            )

    @staticmethod
    def parse_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None or pd.isna(value):
            return False
        text = str(value).strip().lower()
        return text in {"true", "1", "yes", "y"}

    @staticmethod
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
                    result = []
                    for item in parsed:
                        if isinstance(item, dict):
                            result.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
                        else:
                            item_text = str(item).strip()
                            if item_text:
                                result.append(item_text)
                    return result
            except Exception:
                pass

        return [item.strip() for item in text.split(",") if item.strip()]

    @staticmethod
    def parse_json_object_list(value: Any) -> List[Dict[str, Any]]:

        if value is None or pd.isna(value):
            return []

        text = str(value).strip()
        if not text:
            return []

        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except Exception:
            pass

        return []

    def _filter_trainable_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered = df.copy()

        filtered["_is_game_ui_bool"] = filtered["is_game_ui"].apply(self.parse_bool)
        filtered["_review_status_norm"] = filtered["review_status"].fillna("").astype(str).str.strip()
        filtered["_ui_quality_norm"] = filtered["ui_quality"].fillna("").astype(str).str.strip()
        filtered["_primary_norm"] = filtered["primary_screen_type"].fillna("").astype(str).str.strip()

        mask = (
            filtered["_is_game_ui_bool"]
            & filtered["_review_status_norm"].isin(VALID_STATUS_FOR_TRAINING)
            & filtered["_ui_quality_norm"].isin(VALID_QUALITY_FOR_TRAINING)
            & (filtered["_primary_norm"] != "")
        )
        filtered = filtered[mask].copy()

        if self.require_image_exists:
            exists_mask = filtered["file_name"].apply(
                lambda name: os.path.exists(os.path.join(self.img_dir, str(name)))
            )
            filtered = filtered[exists_mask].copy()

        filtered = filtered.reset_index(drop=True)
        return filtered

    def __len__(self) -> int:
        return len(self.df)

    def _make_primary_label(self, value: Any) -> torch.Tensor:
        fallback_cls = "flow_other" if self.use_grouped else "other"
        primary = str(value).strip() if value is not None and not pd.isna(value) else fallback_cls
        index = self.primary_screen_to_idx.get(primary, self.primary_screen_to_idx[fallback_cls])
        return torch.tensor(index, dtype=torch.long)

    def _make_multi_hot(self, values: List[str], mapping: Dict[str, int]) -> torch.Tensor:
        label = torch.zeros(len(mapping), dtype=torch.float)
        for value in values:
            if value in mapping:
                label[mapping[value]] = 1.0
        return label

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        file_name = str(row["file_name"])
        img_path = os.path.join(self.img_dir, file_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)

        style_col = "grouped_visual_style" if self.use_grouped else "visual_style_tags"
        screen_col = "grouped_primary_screen" if self.use_grouped else "primary_screen_type"
        layout_col = "grouped_layout_blocks" if self.use_grouped else "layout_blocks"

        visual_style_values = self.parse_json_list(row.get(style_col, "[]"))
        theme_values = self.parse_json_list(row.get("theme_tags", "[]"))
        layout_blocks = self.parse_json_object_list(row.get(layout_col, "[]"))

        if self.use_grouped:
            pos_values = [str(b.get("position")).strip() for b in layout_blocks if b.get("position")]
            elem_values = [str(b.get("element_type")).strip() for b in layout_blocks if b.get("element_type")]
            role_values = [str(b.get("role")).strip() for b in layout_blocks if b.get("role")]
        else:
            pos_values = [normalize_layout_value(b.get("position"), "position") for b in layout_blocks if b.get("position")]
            elem_values = [normalize_layout_value(b.get("element_type"), "element_type") for b in layout_blocks if b.get("element_type")]
            role_values = [normalize_layout_value(b.get("role"), "role") for b in layout_blocks if b.get("role")]

        return {
            "pixel_values": pixel_values,
            "primary_screen_label": self._make_primary_label(row.get(screen_col, "other")),
            "visual_style_label": self._make_multi_hot(visual_style_values, self.visual_style_tag_to_idx),
            "theme_label": self._make_multi_hot(theme_values, self.theme_tag_to_idx),
            "layout_position_label": self._make_multi_hot(pos_values, self.layout_position_to_idx),
            "layout_element_type_label": self._make_multi_hot(elem_values, self.layout_element_type_to_idx),
            "layout_role_label": self._make_multi_hot(role_values, self.layout_role_to_idx),
            "file_name": file_name,
        }

    def save_vocab(self, save_path: str) -> None:
        vocab = {
            "primary_screen_types": self.primary_screen_types,
            "visual_style_tags": self.visual_style_tags,
            "theme_tags": self.theme_tags,
            "layout_positions": self.layout_positions,
            "layout_element_types": self.layout_element_types,
            "layout_roles": self.layout_roles,
            "primary_screen_to_idx": self.primary_screen_to_idx,
            "visual_style_tag_to_idx": self.visual_style_tag_to_idx,
            "theme_tag_to_idx": self.theme_tag_to_idx,
            "layout_position_to_idx": self.layout_position_to_idx,
            "layout_element_type_to_idx": self.layout_element_type_to_idx,
            "layout_role_to_idx": self.layout_role_to_idx,
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"[+] Vocab saved to {save_path}")
