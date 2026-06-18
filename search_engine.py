# -*- coding: utf-8 -*-
"""
=======================================================================
Game UI semantic search engine

New schema only:
- primary_screen_type
- secondary_screen_types
- visual_style_tags
- theme_tags
- layout_blocks
- layout_tokens
- components

Search modes:
- pure text embedding search
- pure image embedding search
- guided text search with label bonuses
- guided image search with label bonuses
=======================================================================
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EMBEDDING_CACHE = "data/embeddings.pt"
IMAGE_DIR = "data/images"

def normalize_label(label):
    if label is None:
        return ""
    s = str(label).strip().lower()
    if ":" in s:
        return ":".join(normalize_label(part) for part in s.split(":"))
    normalized = s.replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")

PRIMARY_SCREEN_MAPPING = {
    "main_menu": "menu_lobby",
    "menu_lobby": "menu_lobby",
    "title_screen": "menu_lobby",
    "lobby": "menu_lobby",
    "shop": "menu_lobby",
    "settings": "menu_lobby",

    "quest": "gameplay_panel",
    "inventory": "gameplay_panel",
    "character_screen": "gameplay_panel",
    "skill_tree": "gameplay_panel",
    "gameplay_panel": "gameplay_panel",

    "gameplay_hud": "gameplay_hud",
    "hud": "gameplay_hud",

    "battle_result": "flow_other",
    "game_over": "flow_other",
    "loading": "flow_other",
    "loading_screen": "flow_other",
    "tutorial": "flow_other",
    "pause_menu": "flow_other",

    "dialogue": "dialogue_story",
    "dialogue_story": "dialogue_story",
    "cutscene": "dialogue_story",

    "map": "map_screen",
    "world_map": "map_screen",
    "stage_select": "map_screen",
}
PRIMARY_SCREEN_MAPPING = {normalize_label(k): normalize_label(v) for k, v in PRIMARY_SCREEN_MAPPING.items()}

def to_primary_group(label):
    key = normalize_label(label)
    return PRIMARY_SCREEN_MAPPING.get(key, key)

PRIMARY_SCREEN_BONUS = 0.35
SECONDARY_SCREEN_BONUS = 0.10
STYLE_BONUS_PER_MATCH = 0.08
STYLE_BONUS_MAX = 0.24
THEME_BONUS_PER_MATCH = 0.12
THEME_BONUS_MAX = 0.36
LAYOUT_TOKEN_BONUS_PER_MATCH = 0.05
LAYOUT_TOKEN_BONUS_MAX = 0.25
COMPONENT_BONUS_PER_MATCH = 0.04
COMPONENT_BONUS_MAX = 0.12

LAYOUT_POSITION_BONUS_PER_MATCH = 0.05
LAYOUT_ELEMENT_BONUS_PER_MATCH = 0.05
LAYOUT_ROLE_BONUS_PER_MATCH = 0.05
LAYOUT_SPLIT_BONUS_MAX = 0.20

ADULT_KEYWORDS = [
    "adult",
    "hentai",
    "eroge",
    "explicit",
    "sexual"
]


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------
def safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return default
    return text if text else default


def parse_json_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)

    text = safe_text(value)
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            try:
                parsed = json.loads(text.replace("'", '"'))
                return parsed if isinstance(parsed, list) else []
            except Exception:
                pass

    return [item.strip() for item in text.split(",") if item.strip()]


def parse_string_list(value: Any) -> List[str]:
    result: List[str] = []
    for item in parse_json_list(value):
        if isinstance(item, dict):
            continue
        text = safe_text(item)
        if text and text not in result:
            result.append(text)
    return result


def parse_layout_blocks(value: Any) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    for item in parse_json_list(value):
        if not isinstance(item, dict):
            continue
        position = safe_text(item.get("position"))
        element_type = safe_text(item.get("element_type"))
        role = safe_text(item.get("role"), "general")
        if not position or not element_type:
            continue
        block = {"position": position, "element_type": element_type, "role": role}
        if block not in result:
            result.append(block)
    return result


def layout_block_to_token(block: Dict[str, Any]) -> str:
    position = safe_text(block.get("position"))
    element_type = safe_text(block.get("element_type"))
    role = safe_text(block.get("role"), "general")
    if not position or not element_type:
        return ""
    return f"{position}:{element_type}:{role}"


def normalize_layout_tokens(tokens: Optional[Sequence[Any]] = None, blocks: Optional[Sequence[Dict[str, Any]]] = None) -> List[str]:
    result: List[str] = []

    for token in tokens or []:
        text = normalize_label(token)
        if text and text not in result:
            result.append(text)

    for block in blocks or []:
        token = normalize_label(layout_block_to_token(block))
        if token and token not in result:
            result.append(token)

    return result


def token_parts(token: str) -> Tuple[str, str, str]:
    parts = token.split(":")
    pos = normalize_label(parts[0]) if len(parts) >= 1 else ""
    elem = normalize_label(parts[1]) if len(parts) >= 2 else ""
    role = normalize_label(parts[2]) if len(parts) >= 3 else ""
    return pos, elem, role


def layout_token_match_score(query_tokens: Sequence[str], row_tokens: Sequence[str]) -> Tuple[List[str], float]:
    """Return matched layout tokens and a small normalized bonus.

    Exact matches get full credit. If a query token omits the role, position + element
    matches still get partial credit. This lets text queries like top_left:health_bar
    match top_left:health_bar:health.
    """
    query = [normalize_label(x) for x in query_tokens if safe_text(x)]
    row = [normalize_label(x) for x in row_tokens if safe_text(x)]
    if not query or not row:
        return [], 0.0

    row_set = set(row)
    matched: List[str] = []
    bonus = 0.0

    for q in query:
        if q in row_set:
            if q not in matched:
                matched.append(q)
            bonus += LAYOUT_TOKEN_BONUS_PER_MATCH
            continue

        q_pos, q_type, q_role = token_parts(q)
        for r in row:
            r_pos, r_type, r_role = token_parts(r)
            if q_pos and q_type and q_pos == r_pos and q_type == r_type:
                if q_role and r_role and q_role != r_role:
                    continue
                if r not in matched:
                    matched.append(r)
                bonus += LAYOUT_TOKEN_BONUS_PER_MATCH * 0.75
                break

    return matched, min(LAYOUT_TOKEN_BONUS_MAX, bonus)


# ---------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------
class GameUISearchEngine:
    def __init__(self, processor, model, device):
        self.processor = processor
        self.model = model
        self.device = device
        self.cache_data: Optional[Dict[str, Any]] = None
        self._load_db()

    def is_adult_row(self, idx: int) -> bool:
        if self.cache_data is None:
            return False
        
        # safely get string values with fallback
        title = self._get_text_value("game_titles", idx)
        genre = self._get_text_value("genres", idx)
        # Use display_labels or other available cached data since tags/description might be missing
        theme_tags = " ".join(self._get_theme_tags(idx))
        v_styles = " ".join(self._get_visual_style_tags(idx))
        
        text = f"{title} {genre} {theme_tags} {v_styles}".lower()
        return any(keyword in text for keyword in ADULT_KEYWORDS)

    def _load_db(self) -> None:
        if not os.path.exists(EMBEDDING_CACHE):
            print("[!] Vector DB not found.")
            return

        self.cache_data = torch.load(EMBEDDING_CACHE, weights_only=False)
        n = len(self.cache_data.get("file_names", []))
        schema = self.cache_data.get("schema_version", "unknown")
        print(f"[+] Vector DB loaded: {n} images (schema {schema})")

    def _count(self) -> int:
        if self.cache_data is None:
            return 0
        return len(self.cache_data.get("file_names", []))

    def _get_list_value(self, key: str, idx: int, default: Any = None) -> Any:
        if self.cache_data is None:
            return default
        values = self.cache_data.get(key, [])
        if idx < len(values):
            return values[idx]
        return default

    def _get_text_value(self, key: str, idx: int, default: str = "") -> str:
        return safe_text(self._get_list_value(key, idx, default), default)

    def _get_primary_screen_type(self, idx: int) -> str:
        return self._get_text_value("primary_screen_types", idx)

    def _get_secondary_screen_types(self, idx: int) -> List[str]:
        return parse_string_list(self._get_list_value("secondary_screen_types", idx, []))

    def _get_visual_style_tags(self, idx: int) -> List[str]:
        tags = self._get_list_value("visual_style_tags", idx)
        if tags is None:
            tags = self._get_list_value("style_tags", idx, [])
        return parse_string_list(tags)

    def _get_theme_tags(self, idx: int) -> List[str]:
        return parse_string_list(self._get_list_value("theme_tags", idx, []))

    def _get_layout_blocks(self, idx: int) -> List[Dict[str, str]]:
        return parse_layout_blocks(self._get_list_value("layout_blocks", idx, []))

    def _get_layout_tokens(self, idx: int) -> List[str]:
        blocks = self._get_layout_blocks(idx)
        raw_tokens = parse_string_list(self._get_list_value("layout_tokens", idx, []))
        return normalize_layout_tokens(raw_tokens, blocks)

    def _get_layout_positions(self, idx: int) -> List[str]:
        return parse_string_list(self._get_list_value("layout_positions", idx, []))

    def _get_layout_element_types(self, idx: int) -> List[str]:
        return parse_string_list(self._get_list_value("layout_element_types", idx, []))

    def _get_layout_roles(self, idx: int) -> List[str]:
        return parse_string_list(self._get_list_value("layout_roles", idx, []))

    def _get_components(self, idx: int) -> List[str]:
        return parse_string_list(self._get_list_value("components", idx, []))

    def _get_display_label(self, idx: int) -> str:
        label = self._get_text_value("display_labels", idx)
        if label:
            return label

        primary = self._get_primary_screen_type(idx)
        v_styles = self._get_visual_style_tags(idx)
        themes = self._get_theme_tags(idx)
        layouts = self._get_layout_tokens(idx)
        parts = []
        if primary:
            parts.append(f"[{primary}]")
        if v_styles or themes:
            parts.append("styles: " + ", ".join((v_styles + themes)[:4]))
        if layouts:
            parts.append("layout: " + ", ".join(layouts[:4]))
        return " / ".join(parts) if parts else "unlabeled"

    def _build_result(
        self,
        idx: int,
        score: float,
        *,
        base_score: Optional[float] = None,
        primary_match: Optional[bool] = None,
        secondary_match: Optional[bool] = None,
        matched_visual_style_tags: Optional[List[str]] = None,
        matched_theme_tags: Optional[List[str]] = None,
        matched_layout_tokens: Optional[List[str]] = None,
        matched_layout_positions: Optional[List[str]] = None,
        matched_layout_element_types: Optional[List[str]] = None,
        matched_layout_roles: Optional[List[str]] = None,
        matched_components: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.cache_data is None:
            raise RuntimeError("Vector DB is not loaded.")

        file_name = self.cache_data["file_names"][idx]
        return {
            "file_name": file_name,
            "game_title": self.cache_data["game_titles"][idx],
            "score": float(score),
            "base_score": float(base_score if base_score is not None else score),
            "image_path": os.path.join(IMAGE_DIR, file_name),
            "genre": self.cache_data["genres"][idx],
            "platform": self.cache_data["platforms"][idx],
            "source_api": self._get_text_value("source_apis", idx, "Unknown"),
            "moby_url": self._get_text_value("moby_urls", idx),
            "screenshot_url": self._get_text_value("screenshot_urls", idx),
            "screenshot_caption": self._get_text_value("screenshot_captions", idx),
            "display_label": self._get_display_label(idx),
            "ui_label": self._get_display_label(idx),
            "primary_screen_type": self._get_primary_screen_type(idx),
            "secondary_screen_types": self._get_secondary_screen_types(idx),
            "visual_style_tags": self._get_visual_style_tags(idx),
            "style_tags": self._get_visual_style_tags(idx), # Compatibility
            "theme_tags": self._get_theme_tags(idx),
            "layout_blocks": self._get_layout_blocks(idx),
            "layout_tokens": self._get_layout_tokens(idx),
            "layout_positions": self._get_layout_positions(idx),
            "layout_element_types": self._get_layout_element_types(idx),
            "layout_roles": self._get_layout_roles(idx),
            "components": self._get_components(idx),
            "confidence": self._get_list_value("confidence", idx, 0.0),
            "needs_review": self._get_list_value("needs_review", idx, False),
            "primary_match": primary_match,
            "secondary_match": secondary_match,
            "matched_visual_style_tags": matched_visual_style_tags or [],
            "matched_theme_tags": matched_theme_tags or [],
            "matched_layout_tokens": matched_layout_tokens or [],
            "matched_layout_positions": matched_layout_positions or [],
            "matched_layout_element_types": matched_layout_element_types or [],
            "matched_layout_roles": matched_layout_roles or [],
            "matched_components": matched_components or [],
        }

    def _encode_text(self, query_text: str) -> torch.Tensor:
        text_inputs = self.processor(text=[query_text], padding="max_length", return_tensors="pt").to(self.device)
        text_features = self.model.get_text_features(**text_inputs)
        text_features = getattr(text_features, "text_embeds", text_features)
        if not isinstance(text_features, torch.Tensor):
            text_features = getattr(text_features, "pooler_output", text_features)
        return F.normalize(text_features, p=2, dim=-1)

    def _encode_image(self, query_image) -> torch.Tensor:
        image_inputs = self.processor(images=query_image, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**image_inputs)
        image_features = getattr(image_features, "image_embeds", image_features)
        if not isinstance(image_features, torch.Tensor):
            image_features = getattr(image_features, "pooler_output", image_features)
        return F.normalize(image_features, p=2, dim=-1)

    def _rank_from_similarities(
        self,
        similarities: Sequence[float],
        top_k: int,
        *,
        primary_screen_type: str = "",
        secondary_screen_types: Optional[Sequence[str]] = None,
        visual_style_tags: Optional[Sequence[str]] = None,
        theme_tags: Optional[Sequence[str]] = None,
        layout_tokens: Optional[Sequence[str]] = None,
        layout_blocks: Optional[Sequence[Dict[str, Any]]] = None,
        layout_positions: Optional[Sequence[str]] = None,
        layout_element_types: Optional[Sequence[str]] = None,
        layout_roles: Optional[Sequence[str]] = None,
        components: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        primary = safe_text(primary_screen_type)
        q_secondary = {normalize_label(x) for x in (secondary_screen_types or []) if safe_text(x)}
        v_style_set: Set[str] = {normalize_label(x) for x in (visual_style_tags or []) if safe_text(x)}
        theme_set: Set[str] = {normalize_label(x) for x in (theme_tags or []) if safe_text(x)}
        component_set: Set[str] = {normalize_label(x) for x in (components or []) if safe_text(x)}
        query_layout_tokens = normalize_layout_tokens(layout_tokens or [], layout_blocks or [])

        ranked = []
        total = self._count()
        for idx in range(total):
            if self.is_adult_row(idx):
                continue
                
            base_sim = float(similarities[idx])
            row_primary = self._get_primary_screen_type(idx)
            row_secondary = {normalize_label(x) for x in self._get_secondary_screen_types(idx)}
            row_components = {normalize_label(x) for x in self._get_components(idx)}
            row_layout_tokens = self._get_layout_tokens(idx)
            row_layout_pos = {normalize_label(x) for x in self._get_layout_positions(idx)}
            row_layout_elem = {normalize_label(x) for x in self._get_layout_element_types(idx)}
            row_layout_role = {normalize_label(x) for x in self._get_layout_roles(idx)}

            # Group-based primary screen matching
            q_group = to_primary_group(primary)
            row_group = to_primary_group(row_primary)
            primary_bonus = PRIMARY_SCREEN_BONUS if q_group and row_group and q_group == row_group else 0.0

            secondary_match = bool(q_secondary.intersection(row_secondary))
            secondary_bonus = SECONDARY_SCREEN_BONUS if secondary_match else 0.0

            row_v_styles = {normalize_label(x) for x in self._get_visual_style_tags(idx)}
            row_themes = {normalize_label(x) for x in self._get_theme_tags(idx)}

            matched_v_styles = sorted(v_style_set.intersection(row_v_styles))
            matched_themes = sorted(theme_set.intersection(row_themes))
            
            style_bonus = min(STYLE_BONUS_MAX, STYLE_BONUS_PER_MATCH * len(matched_v_styles))
            theme_bonus = min(THEME_BONUS_MAX, THEME_BONUS_PER_MATCH * len(matched_themes))

            matched_layouts, layout_bonus = layout_token_match_score(query_layout_tokens, row_layout_tokens)

            # New split axis bonuses
            q_pos_set = {normalize_label(x) for x in (layout_positions or []) if safe_text(x)}
            q_elem_set = {normalize_label(x) for x in (layout_element_types or []) if safe_text(x)}
            q_role_set = {normalize_label(x) for x in (layout_roles or []) if safe_text(x)}

            matched_pos = sorted(q_pos_set.intersection(row_layout_pos))
            matched_elem = sorted(q_elem_set.intersection(row_layout_elem))
            matched_role = sorted(q_role_set.intersection(row_layout_role))

            split_bonus = min(LAYOUT_SPLIT_BONUS_MAX, 
                              LAYOUT_POSITION_BONUS_PER_MATCH * len(matched_pos) +
                              LAYOUT_ELEMENT_BONUS_PER_MATCH * len(matched_elem) +
                              LAYOUT_ROLE_BONUS_PER_MATCH * len(matched_role))

            matched_components = sorted(component_set.intersection(row_components))
            component_bonus = min(COMPONENT_BONUS_MAX, COMPONENT_BONUS_PER_MATCH * len(matched_components))

            final_rank_score = base_sim + primary_bonus + secondary_bonus + style_bonus + theme_bonus + layout_bonus + split_bonus + component_bonus
            display_score = max(0.0, min(100.0, final_rank_score * 100.0))
            base_score = max(0.0, min(100.0, base_sim * 100.0))

            ranked.append(
                (
                    final_rank_score,
                    idx,
                    display_score,
                    base_score,
                    bool(primary_bonus > 0.0),
                    secondary_match,
                    matched_v_styles,
                    matched_themes,
                    matched_layouts,
                    matched_pos,
                    matched_elem,
                    matched_role,
                    matched_components,
                )
            )

        ranked.sort(key=lambda x: x[0], reverse=True)
        ranked = ranked[: min(top_k, len(ranked))]

        results: List[Dict[str, Any]] = []
        for _, idx, display_score, base_score, primary_match, secondary_match, m_v_styles, m_themes, m_layouts, m_pos, m_elem, m_role, m_components in ranked:
            results.append(
                self._build_result(
                    idx,
                    display_score,
                    base_score=base_score,
                    primary_match=primary_match,
                    secondary_match=secondary_match,
                    matched_visual_style_tags=m_v_styles,
                    matched_theme_tags=m_themes,
                    matched_layout_tokens=m_layouts,
                    matched_layout_positions=m_pos,
                    matched_layout_element_types=m_elem,
                    matched_layout_roles=m_role,
                    matched_components=m_components,
                )
            )
        return results

    # -----------------------------------------------------------------
    # Public search APIs
    # -----------------------------------------------------------------
    def search_by_text(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.cache_data is None:
            return []

        self.model.eval()
        with torch.inference_mode():
            text_features = self._encode_text(query_text)
            image_embeddings = self.cache_data["embeddings"].to(self.device)
            similarities = torch.matmul(text_features, image_embeddings.T).squeeze(0).cpu().tolist()

        return self._rank_from_similarities(similarities, top_k)

    def search_by_image(self, query_image, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.cache_data is None:
            return []

        self.model.eval()
        with torch.inference_mode():
            image_features = self._encode_image(query_image)
            image_embeddings = self.cache_data["embeddings"].to(self.device)
            similarities = torch.matmul(image_features, image_embeddings.T).squeeze(0).cpu().tolist()

        return self._rank_from_similarities(similarities, top_k)

    def search_by_text_guided(
        self,
        query_text: str,
        primary_screen_type: str = "",
        visual_style_tags: Optional[Sequence[str]] = None,
        theme_tags: Optional[Sequence[str]] = None,
        layout_tokens: Optional[Sequence[str]] = None,
        *,
        secondary_screen_types: Optional[Sequence[str]] = None,
        layout_blocks: Optional[Sequence[Dict[str, Any]]] = None,
        layout_positions: Optional[Sequence[str]] = None,
        layout_element_types: Optional[Sequence[str]] = None,
        layout_roles: Optional[Sequence[str]] = None,
        components: Optional[Sequence[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if self.cache_data is None:
            return []

        self.model.eval()
        with torch.inference_mode():
            text_features = self._encode_text(query_text)
            image_embeddings = self.cache_data["embeddings"].to(self.device)
            similarities = torch.matmul(text_features, image_embeddings.T).squeeze(0).cpu().tolist()

        return self._rank_from_similarities(
            similarities,
            top_k,
            primary_screen_type=primary_screen_type,
            secondary_screen_types=secondary_screen_types,
            visual_style_tags=visual_style_tags,
            theme_tags=theme_tags,
            layout_tokens=layout_tokens,
            layout_blocks=layout_blocks,
            layout_positions=layout_positions,
            layout_element_types=layout_element_types,
            layout_roles=layout_roles,
            components=components,
        )

    def search_by_filters(
        self,
        primary_screen_type: str = "",
        visual_style_tags: Optional[Sequence[str]] = None,
        theme_tags: Optional[Sequence[str]] = None,
        layout_roles: Optional[Sequence[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if self.cache_data is None:
            return []
            
        # No text or image query, so base similarity is 0.0 for all items
        total_items = self._count()
        similarities = [0.0] * total_items
        
        return self._rank_from_similarities(
            similarities,
            top_k,
            primary_screen_type=primary_screen_type,
            visual_style_tags=visual_style_tags,
            theme_tags=theme_tags,
            layout_roles=layout_roles,
        )


    def search_by_image_guided(
        self,
        query_image,
        primary_screen_type: str = "",
        visual_style_tags: Optional[Sequence[str]] = None,
        theme_tags: Optional[Sequence[str]] = None,
        layout_tokens: Optional[Sequence[str]] = None,
        *,
        secondary_screen_types: Optional[Sequence[str]] = None,
        layout_blocks: Optional[Sequence[Dict[str, Any]]] = None,
        layout_positions: Optional[Sequence[str]] = None,
        layout_element_types: Optional[Sequence[str]] = None,
        layout_roles: Optional[Sequence[str]] = None,
        components: Optional[Sequence[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if self.cache_data is None:
            return []

        self.model.eval()
        with torch.inference_mode():
            image_features = self._encode_image(query_image)
            image_embeddings = self.cache_data["embeddings"].to(self.device)
            similarities = torch.matmul(image_features, image_embeddings.T).squeeze(0).cpu().tolist()

        return self._rank_from_similarities(
            similarities,
            top_k,
            primary_screen_type=primary_screen_type,
            secondary_screen_types=secondary_screen_types,
            visual_style_tags=visual_style_tags,
            theme_tags=theme_tags,
            layout_tokens=layout_tokens,
            layout_blocks=layout_blocks,
            layout_positions=layout_positions,
            layout_element_types=layout_element_types,
            layout_roles=layout_roles,
            components=components,
        )

    def find_game_ui_set(self, query_image, top_k: int = 1) -> Dict[str, Dict[str, Any]]:
        results = self.search_by_image(query_image, top_k=30)
        if not results:
            return {}

        game_scores: Dict[str, List[float]] = {}
        for item in results:
            game_scores.setdefault(item["game_title"], []).append(float(item["score"]))

        game_avg = {title: sum(scores) / len(scores) for title, scores in game_scores.items()}
        top_games = sorted(game_avg.items(), key=lambda x: x[1], reverse=True)[:top_k]

        result_sets: Dict[str, Dict[str, Any]] = {}
        for title, avg_score in top_games:
            result_sets[title] = {
                "avg_score": avg_score,
                "screenshots": self.get_game_screenshots(title)[:12],
            }
        return result_sets

    def get_game_screenshots(self, game_title: str) -> List[Dict[str, Any]]:
        if self.cache_data is None:
            return []

        results: List[Dict[str, Any]] = []
        for idx, title in enumerate(self.cache_data.get("game_titles", [])):
            if title != game_title:
                continue
            results.append(self._build_result(idx, 0.0, base_score=0.0))
        return results


if __name__ == "__main__":
    from transformers import AutoModel, AutoProcessor

    MODEL_NAME = "google/siglip2-base-patch16-224"
    print("[*] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    engine = GameUISearchEngine(processor, model, device)
    tests = [
        {
            "query": "dark fantasy inventory screen with a character preview",
            "primary_screen_type": "inventory",
            "visual_style_tags": ["realistic"],
            "theme_tags": ["dark_fantasy", "fantasy"],
            "layout_tokens": ["center:preview:character", "left:grid:inventory"],
        },
        {
            "query": "gameplay hud with top health bar and bottom skill bar",
            "primary_screen_type": "gameplay_hud",
            "layout_tokens": ["top_left:health_bar:health", "bottom_center:skill_bar:combat"],
        },
    ]

    for test in tests:
        print("=" * 60)
        print(test["query"])
        results = engine.search_by_text_guided(top_k=3, **test)
        for i, result in enumerate(results, start=1):
            print(f"{i}. {result['game_title']} / {result['score']:.1f} / {result['display_label']}")
