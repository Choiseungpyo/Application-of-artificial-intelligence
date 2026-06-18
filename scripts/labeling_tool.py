# -*- coding: utf-8 -*-
"""
=======================================================================
Game UI Labeling Tool - Schema 6.0 - Fixed Version
=======================================================================

핵심 수정 사항
1. Gradio output 개수와 callback return 개수를 고정했습니다.
2. GPT 실패 시 모든 필드가 "오류"로 채워지지 않고 status에만 메시지가 표시됩니다.
3. CSV 저장용 타입과 Gradio 표시용 타입을 분리했습니다.
4. CheckboxGroup은 list[str], JSON Textbox는 str, Checkbox는 bool을 받도록 정리했습니다.
5. Save & Next, Skip, Retry, Restore, Auto Labeling의 반환 순서를 통일했습니다.
=======================================================================
"""

import base64
import json
import os
import signal
import sys
import time
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from layout_vocab import ELEMENT_TYPES, POSITIONS, ROLES, normalize_layout_value

# LangGraph Agent Integration (Optional)
try:
    from langgraph_labeling_agent import run_langgraph_analysis
    LANGGRAPH_AVAILABLE = True
except Exception as e:
    run_langgraph_analysis = None
    LANGGRAPH_AVAILABLE = False

load_dotenv()
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ======================== Paths and settings ========================
IMAGE_DIR = "data/images"
METADATA_FILE = "data/metadata.csv"
EVENT_LOG_FILE = "data/labeling_events.jsonl"
SCHEMA_VERSION = "6.0"
OPENAI_MODEL = "gpt-4o"
API_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_AUTO_DELAY = 1.5
VISUAL_FEEDBACK_DELAY = 0.35
MAX_LAYOUT_BLOCKS = 8
MAX_COMPONENTS = 12
MIN_COMPONENT_COUNT = 1

DATA_OUTPUT_COUNT = 15 + 24
AI_OUTPUT_COUNT = 13 + 24
SAVE_OUTPUT_COUNT = 16 + 24
AUTO_OUTPUT_COUNT = 17 + 24

# ======================== Label vocabularies ========================
PRIMARY_SCREEN_TYPES = [
    "gameplay_hud",
    "main_menu",
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
    "title_screen",
    "tutorial",
    "other",
]

SECONDARY_SCREEN_TYPES = [
    "popup",
    "shop_popup",
    "settings_popup",
    "quest_panel",
    "dialogue_overlay",
    "map_overlay",
    "inventory_overlay",
    "tooltip_overlay",
    "notification_overlay",
    "character_preview",
    "party_panel",
    "minimap_overlay",
    "chat_panel",
]

VISUAL_STYLE_TAGS = [
    "realistic",
    "cartoon",
    "anime",
    "pixel_art",
    "retro",
    "modern",
    "minimal",
    "clean",
    "skeuomorphic",
    "flat",
    "neon",
    "gritty",
    "cute",
]

THEME_TAGS = [
    "fantasy",
    "dark_fantasy",
    "medieval",
    "sci_fi",
    "cyberpunk",
    "military",
    "horror",
    "post_apocalyptic",
    "historical",
    "modern_world",
]

# POSITIONS, ELEMENT_TYPES, ROLES are imported from layout_vocab.py

UI_QUALITY_VALUES = ["keep", "weak", "reject"]
REVIEW_STATUS_VALUES = ["unlabeled", "labeled", "needs_review", "skipped_permanent", "retry_pending"]

REQUIRED_COLUMNS = [
    "source_api",
    "moby_game_id",
    "game_title",
    "moby_url",
    "platform_id",
    "platform_name",
    "screenshot_caption",
    "screenshot_url",
    "file_name",
    "image_hash",
    "genre",
    "collected_at",
    "schema_version",
    "is_game_ui",
    "ui_quality",
    "primary_screen_type",
    "secondary_screen_types",
    "visual_style_tags",
    "theme_tags",
    "layout_blocks",
    "layout_tokens",
    "components",
    "confidence",
    "needs_review",
    "review_status",
    "ui_score",
    "ui_score_reason",
    "source_target_screen_type",
    "notes",
]

JSON_LIST_COLUMNS = [
    "secondary_screen_types",
    "visual_style_tags",
    "theme_tags",
    "layout_blocks",
    "layout_tokens",
    "components",
]

# ======================== Conversion helpers ========================
def is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip() == ""


def parse_json_list(value: Any) -> List[Any]:
    """CSV 문자열, Python list, comma-separated 문자열을 안전하게 list로 변환."""
    if is_blank(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    text = str(value).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
        return []
    except Exception:
        return [x.strip() for x in text.split(",") if x.strip()]


def load_json_textbox(value: Any) -> List[Any]:
    return parse_json_list(value)


def parse_json_list_for_checkbox(value: Any, allowed: Optional[List[str]] = None) -> List[str]:
    items = parse_json_list(value)
    allowed_set = set(allowed) if allowed else None
    result: List[str] = []
    for item in items:
        text = str(item).strip().lower().replace(" ", "_").replace("-", "_")
        if not text:
            continue
        if allowed_set is not None and text not in allowed_set:
            continue
        if text not in result:
            result.append(text)
    return result


def format_json_for_textbox(value: Any) -> str:
    items = parse_json_list(value)
    return json.dumps(items, ensure_ascii=False, indent=2)


def compact_json_text(value: Any) -> str:
    items = parse_json_list(value)
    return json.dumps(items, ensure_ascii=False, indent=2)


def parse_bool_for_ui(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if is_blank(value):
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return default


def serialize_bool_for_csv(value: Any) -> str:
    return "True" if parse_bool_for_ui(value) else "False"


def serialize_json_for_csv(value: Any) -> str:
    if isinstance(value, str):
        # 문자열이 JSON list/dict면 그대로 정규화해서 저장한다.
        try:
            parsed = json.loads(value)
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            # 일반 문자열이면 list 안에 넣어 저장한다.
            if value.strip() == "":
                return "[]"
            return json.dumps([value.strip()], ensure_ascii=False)
    return json.dumps(value if value is not None else [], ensure_ascii=False)


def normalize_choice(value: Any, allowed: List[str], default: str) -> str:
    if is_blank(value):
        return default
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return text if text in allowed else default


def normalize_string_list(value: Any, allowed: Optional[List[str]] = None, max_items: Optional[int] = None) -> List[str]:
    items = parse_json_list(value)
    allowed_set = set(allowed) if allowed else None
    result: List[str] = []
    for item in items:
        text = str(item).strip().lower().replace(" ", "_").replace("-", "_")
        if not text:
            continue
        if allowed_set is not None and text not in allowed_set:
            continue
        if text not in result:
            result.append(text)
        if max_items is not None and len(result) >= max_items:
            break
    return result


def normalize_components(value: Any) -> List[str]:
    items = parse_json_list(value)
    result: List[str] = []
    for item in items:
        if isinstance(item, dict):
            text = " ".join(str(v).strip() for v in item.values() if str(v).strip())
        else:
            text = str(item).strip()
        text = text.replace("\n", " ").strip()
        if text and text not in result:
            result.append(text)
        if len(result) >= MAX_COMPONENTS:
            break
    return result


def normalize_layout_blocks(value: Any) -> List[Dict[str, str]]:
    items = parse_json_list(value)
    result: List[Dict[str, str]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        position = normalize_layout_value(item.get("position"), "position")
        element_type = normalize_layout_value(item.get("element_type"), "element_type")
        role = normalize_layout_value(item.get("role"), "role")
        key = (position, element_type, role)
        if key in seen:
            continue
        seen.add(key)
        result.append({"position": position, "element_type": element_type, "role": role})
        if len(result) >= MAX_LAYOUT_BLOCKS:
            break
    return result


def make_layout_tokens(layout_blocks: List[Dict[str, str]]) -> List[str]:
    tokens: List[str] = []
    for block in layout_blocks:
        position = block.get("position", "center")
        element_type = block.get("element_type", "panel")
        role = block.get("role", "unknown")
        token = f"{position}:{element_type}:{role}"
        if token not in tokens:
            tokens.append(token)
    return tokens


def unchanged_data_updates() -> Tuple[Any, ...]:
    return tuple(gr.update() for _ in range(DATA_OUTPUT_COUNT))


def assert_return_length(name: str, values: Tuple[Any, ...], expected: int) -> Tuple[Any, ...]:
    actual = len(values)
    if actual != expected:
        print(f"[ERROR] {name} return length mismatch: expected={expected}, actual={actual}")
    return values

# ======================== OpenAI client ========================
_client: Optional[OpenAI] = None


def get_openai_client() -> Optional[OpenAI]:
    global _client
    if _client is None:
        if not API_KEY:
            print("[!] OPENAI_API_KEY가 설정되지 않았습니다.")
            return None
        _client = OpenAI(api_key=API_KEY)
    return _client

# ======================== Utilities ========================
def append_event(event_type: str, payload: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(EVENT_LOG_FILE), exist_ok=True)
        row = {"event_type": event_type, **payload}
        with open(EVENT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[!] 이벤트 로그 기록 실패: {e}")


def encode_image(image: Image.Image, max_size: int = 1024, quality: int = 82) -> str:
    img = image.copy().convert("RGB")
    width, height = img.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        img = img.resize(new_size, Image.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def normalize_ai_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    is_game_ui = parse_bool_for_ui(raw.get("is_game_ui"), default=False)
    ui_quality = normalize_choice(raw.get("ui_quality"), UI_QUALITY_VALUES, "reject")
    primary = normalize_choice(raw.get("primary_screen_type"), PRIMARY_SCREEN_TYPES, "other")
    secondary = normalize_string_list(raw.get("secondary_screen_types", []), SECONDARY_SCREEN_TYPES)
    visual_styles = normalize_string_list(raw.get("visual_style_tags", []), VISUAL_STYLE_TAGS)
    themes = normalize_string_list(raw.get("theme_tags", []), THEME_TAGS)
    layout_blocks = normalize_layout_blocks(raw.get("layout_blocks", []))
    layout_tokens = make_layout_tokens(layout_blocks)
    components = normalize_components(raw.get("components", []))

    try:
        confidence = float(raw.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    needs_review = parse_bool_for_ui(raw.get("needs_review"), default=False)
    notes = str(raw.get("notes", "") or "").strip()

    if not is_game_ui:
        ui_quality = "reject"
        primary = "other"
        secondary = []
        visual_styles = []
        themes = []
        layout_blocks = []
        layout_tokens = []
        components = []
        if not notes:
            notes = "No meaningful game UI detected."

    if ui_quality == "reject":
        is_game_ui = False
        primary = "other"

    if primary == "other" or confidence < 0.70 or len(components) < MIN_COMPONENT_COUNT:
        needs_review = True

    return {
        "is_game_ui": is_game_ui,
        "ui_quality": ui_quality,
        "primary_screen_type": primary,
        "secondary_screen_types": secondary,
        "visual_style_tags": visual_styles,
        "theme_tags": themes,
        "layout_blocks": layout_blocks,
        "layout_tokens": layout_tokens,
        "components": components,
        "confidence": confidence,
        "needs_review": needs_review,
        "notes": notes,
    }

# ======================== GPT analysis ========================
# ======================== GPT analysis ========================
def get_ai_analysis_direct(image: Image.Image, row_info: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """기존의 직접적인 OpenAI API 호출 및 정규화 로직 (LangGraph 노드에서 호출됨)"""
    client = get_openai_client()
    if client is None or image is None:
        return None

    base64_image = encode_image(image)
    caption = ""
    if row_info:
        caption = str(row_info.get("screenshot_caption", "") or "").strip()

    system_prompt = f"""
You are a strict game UI dataset labeler.
Return valid JSON only.

Allowed primary_screen_type values:
{PRIMARY_SCREEN_TYPES}

Allowed secondary_screen_types values:
{SECONDARY_SCREEN_TYPES}

Allowed visual_style_tags (Visual appearance only):
{VISUAL_STYLE_TAGS}

Allowed theme_tags (World/Genre/Setting only):
{THEME_TAGS}

Allowed layout block positions:
{POSITIONS}

Allowed layout block element_type values:
{ELEMENT_TYPES}

Allowed layout block role values:
{ROLES}

Rules:
1. If the screenshot is mostly environment, cutscene, key art, character render, or gameplay without meaningful UI, set is_game_ui false and ui_quality reject.
2. If the UI is visible but small or incomplete, set is_game_ui true and ui_quality weak.
3. If the UI is clear and useful for UI reference search, set is_game_ui true and ui_quality keep.
4. Choose exactly one primary_screen_type.
5. Use secondary_screen_types only for overlays or mixed screens.
6. visual_style_tags: Describe ONLY how the UI elements look visually (e.g., flat, skeuomorphic, neon).
7. theme_tags: Describe ONLY the game world/setting (e.g., fantasy, sci_fi, cyberpunk).
8. Do not infer visual style from the game theme alone.
9. Prefer 2 to 4 tags max per field.
10. layout_blocks must describe up to {MAX_LAYOUT_BLOCKS} visible UI regions. Each block must have position, element_type, and role.
11. components must be short concrete UI elements, not long sentences.
12. confidence must be between 0 and 1.
13. needs_review should be true if the label is uncertain, mixed, or weak.
14. character_screen: Includes character selection, creation, stats, portraits, level, job, and status. EXCLUDE if the focus is equipment slots (use equipment), item list (use inventory), or skill tree (use skill_tree).
""".strip()

    user_text = "Analyze this game screenshot for a game UI search dataset."
    if caption:
        user_text += f"\nMobyGames screenshot caption: {caption}"

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=900,
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            return None
        parsed = json.loads(content)
        return normalize_ai_result(parsed)
    except Exception as e:
        err_msg = str(e).lower()
        if "insufficient_quota" in err_msg:
            print("\n[CRITICAL] OpenAI API 쿼터가 부족합니다. 결제 정보나 한도를 확인해주세요.")
        elif "rate_limit" in err_msg:
            print("\n[WARN] OpenAI API 속도 제한에 걸렸습니다. 잠시 후 시도해주세요.")
        else:
            print(f"\n[!] GPT 분석 실패: {e}")
        traceback.print_exc()
        return None


def get_ai_analysis(image: Image.Image, row_info: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """분석 진입점: LangGraph 에이전트를 사용하거나 직접 호출로 Fallback 함"""
    if LANGGRAPH_AVAILABLE and run_langgraph_analysis is not None:
        return run_langgraph_analysis(
            image=image,
            row_info=row_info or {},
            analyzer_func=get_ai_analysis_direct,
            normalizer_func=None  # get_ai_analysis_direct 내에서 이미 normalize_ai_result 호출함
        )
    
    if LANGGRAPH_AVAILABLE:
        # print("[LANGGRAPH] Agent available but function is None. Using direct GPT analysis.")
        pass
    else:
        # print("[LANGGRAPH] Unavailable. Using direct GPT analysis.")
        pass
        
    return get_ai_analysis_direct(image, row_info=row_info)

# ======================== Labeling app state ========================
class LabelingApp:
    def __init__(self):
        # 엑셀 락 체크
        temp_file = os.path.join(os.path.dirname(METADATA_FILE), f"~${os.path.basename(METADATA_FILE)}")
        if os.path.exists(temp_file):
            print(f"\n[!] 주의: '{METADATA_FILE}'가 엑셀에서 열려 있는 것 같습니다.")
            print("[!] 저장이 불가능할 수 있으니 엑셀을 닫고 다시 시작하는 것을 권장합니다.\n")

        if os.path.exists(METADATA_FILE):
            self.df = pd.read_csv(METADATA_FILE, dtype=str, encoding="utf-8-sig").fillna("")
        elif os.path.exists("metadata.csv"):
            self.df = pd.read_csv("metadata.csv", dtype=str, encoding="utf-8-sig").fillna("")
        else:
            self.df = pd.DataFrame(columns=REQUIRED_COLUMNS)

        self.current_idx = 0
        self.is_processing = False
        self.filter_mode = "unlabeled"  # "all", "unlabeled", "suspicious", "targeted"
        self.target_filter = "All"
        
        self.target_classes = [
            "equipment", "skill_tree", "crafting", "pause_menu", 
            "loading_screen", "quest", "other"
        ]
        
        self.suspicious_files = set()
        if os.path.exists("targeted_collection_report.csv"):
            try:
                susp_df = pd.read_csv("targeted_collection_report.csv", dtype=str)
                if "file_name" in susp_df.columns:
                    self.suspicious_files = set(susp_df["file_name"].tolist())
                    print(f"[INIT] Loaded {len(self.suspicious_files)} suspicious files for filtering.")
            except Exception as e:
                print(f"[!] Failed to load targeted_collection_report.csv: {e}")

        self._ensure_schema()
        self._find_next_matching_item()

    def _ensure_schema(self) -> None:
        modified = False
        for col in REQUIRED_COLUMNS:
            if col not in self.df.columns:
                self.df[col] = ""
                modified = True

        if len(self.df) > 0:
            # review_status 기본값
            review = self.df["review_status"].astype(str).str.strip()
            mask_review = review.eq("") | ~review.isin(REVIEW_STATUS_VALUES)
            if mask_review.any():
                self.df.loc[mask_review, "review_status"] = "unlabeled"
                modified = True

            # schema_version 기본값
            mask_schema = self.df["schema_version"].astype(str).str.strip().eq("")
            if mask_schema.any():
                self.df.loc[mask_schema, "schema_version"] = SCHEMA_VERSION
                modified = True

            # JSON 필드 기본값
            for col in JSON_LIST_COLUMNS:
                mask_blank = self.df[col].astype(str).str.strip().eq("")
                if mask_blank.any():
                    self.df.loc[mask_blank, col] = "[]"
                    modified = True

            # 안전 기본값
            defaults = {
                "is_game_ui": "False",
                "needs_review": "True",
                "confidence": "0.0",
                "ui_score": "0",
                "ui_score_reason": "",
                "source_target_screen_type": "",
                "notes": "",
                "ui_quality": "",
                "primary_screen_type": "",
            }
            for col, default in defaults.items():
                mask = self.df[col].astype(str).str.strip().eq("")
                if mask.any() and default != "":
                    self.df.loc[mask, col] = default
                    modified = True

        if modified:
            self.flush()

    def flush(self) -> None:
        try:
            os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
            self.df.to_csv(METADATA_FILE, index=False, encoding="utf-8-sig")
        except PermissionError:
            print(f"\n[!] ERROR: '{METADATA_FILE}' 파일에 접근할 수 없습니다.")
            print("[!] 파일이 엑셀(Excel) 등 다른 프로그램에서 열려 있는지 확인하고 닫아주세요.")
            # 엑셀 임시 파일 존재 여부 확인
            temp_file = os.path.join(os.path.dirname(METADATA_FILE), f"~${os.path.basename(METADATA_FILE)}")
            if os.path.exists(temp_file):
                print(f"[!] 탐지됨: 엑셀 임시 파일 '{temp_file}'이 존재합니다.")
            raise  # 다시 발생시켜서 상위 콜백에서 에러 메시지로 표시되게 함
        except Exception as e:
            print(f"[!] 저장 중 알 수 없는 오류 발생: {e}")
            raise

    def _find_next_matching_item(self, advance: bool = False) -> None:
        if "review_status" not in self.df.columns or len(self.df) == 0:
            self.current_idx = len(self.df)
            return
            
        mask = pd.Series([True] * len(self.df))
        
        if self.filter_mode == "unlabeled":
            mask &= self.df["review_status"].astype(str).str.strip().eq("unlabeled")
        elif self.filter_mode == "needs_review":
            mask &= self.df["review_status"].astype(str).str.strip().eq("needs_review")
        elif self.filter_mode == "suspicious":
            mask &= self.df["file_name"].isin(self.suspicious_files)
        elif self.filter_mode == "personal":
            mask &= self.df["review_status"].astype(str).str.strip().eq("unlabeled")
            mask &= self.df["source_api"].astype(str).str.strip().eq("Personal")
        elif self.filter_mode == "targeted":
            # Priority: unlabeled targeted images
            mask &= self.df["review_status"].astype(str).str.strip().eq("unlabeled")
            
            # Check if file is in the targeted collection report (self.suspicious_files)
            in_report = self.df["file_name"].isin(self.suspicious_files)
            
            if self.target_filter == "All":
                # Include items in report, OR with target type set, OR with priority primary type set
                has_source = self.df["source_target_screen_type"].astype(str).str.strip().ne("")
                has_priority_primary = self.df["primary_screen_type"].isin(self.target_classes)
                mask &= (in_report | has_source | has_priority_primary)
            else:
                # Include rows where source_target matches target, or primary matches target
                target = self.target_filter
                source_match = self.df["source_target_screen_type"].astype(str).str.strip().eq(target)
                primary_match = self.df["primary_screen_type"].astype(str).str.strip().eq(target)
                mask &= (source_match | primary_match)
        
        filtered = self.df[mask]
        if filtered.empty:
            self.current_idx = len(self.df)
            return
            
        # Sorting logic for 'targeted' mode with 'All'
        if self.filter_mode == "targeted" and self.target_filter == "All":
            def get_priority_val(row_idx):
                row = self.df.iloc[row_idx]
                st = str(row.get("source_target_screen_type", "")).strip()
                pr = str(row.get("primary_screen_type", "")).strip()
                # Check priority classes
                for i, cls in enumerate(self.target_classes):
                    if st == cls or pr == cls:
                        return i
                return len(self.target_classes) # Others at the end

            # Sort indices based on priority and then filename
            indices = filtered.index.tolist()
            indices.sort(key=lambda idx: (get_priority_val(idx), str(self.df.at[idx, "file_name"])))
        else:
            # Default sorting by index
            indices = filtered.index.tolist()

        if advance:
            # Find the next item in the sorted list after current_idx
            try:
                curr_pos = indices.index(self.current_idx)
                if curr_pos + 1 < len(indices):
                    self.current_idx = indices[curr_pos + 1]
                else:
                    # Wrap around
                    self.current_idx = indices[0]
            except ValueError:
                # current_idx not in filtered list (maybe status changed), pick first
                self.current_idx = indices[0]
        else:
            # Initial load or filter change: stay on current if valid, else first
            if self.current_idx in indices:
                pass
            else:
                self.current_idx = indices[0]
            
    def set_filter_mode(self, mode: str, target: str = "All") -> None:
        self.filter_mode = mode
        self.target_filter = target
        self._find_next_matching_item()

    def get_current_row(self) -> Optional[pd.Series]:
        if self.current_idx >= len(self.df):
            return None
        return self.df.iloc[self.current_idx]

    def get_current_row_info(self) -> Dict[str, Any]:
        row = self.get_current_row()
        if row is None:
            return {}
        return {col: row.get(col, "") for col in self.df.columns}

    def get_progress_summary(self) -> str:
        total = len(self.df)
        labeled = int((self.df["review_status"] == "labeled").sum()) if total else 0
        needs_review = int((self.df["review_status"] == "needs_review").sum()) if total else 0
        unlabeled = int((self.df["review_status"] == "unlabeled").sum()) if total else 0
        skipped = int((self.df["review_status"] == "skipped_permanent").sum()) if total else 0
        retry = int((self.df["review_status"] == "retry_pending").sum()) if total else 0
        percent = (labeled / total * 100.0) if total else 0.0
        
        # Calculate target class progress
        target_counts = {}
        for cat in self.target_classes:
            # labeled_count: primary_screen_type is cat AND status is labeled
            l_count = int(((self.df["primary_screen_type"] == cat) & (self.df["review_status"] == "labeled")).sum())
            # pending_count: status is unlabeled AND source_target_screen_type is cat
            p_count = int(((self.df["review_status"] == "unlabeled") & (self.df["source_target_screen_type"] == cat)).sum())
            target_counts[cat] = (l_count, p_count)

        def get_pill_style(cat, l_count):
            if l_count < 30:
                if cat in ["crafting", "skill_tree", "shop"]:
                    return "background:#fef2f2;color:#ef4444;border:1px solid #fecaca;font-weight:800;"
                return "background:#fff7ed;color:#f97316;border:1px solid #ffedd5;"
            return "background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0;"

        pills_html = "".join([
            f"<div style='padding:4px 10px;border-radius:999px;font-size:11px;{get_pill_style(c, target_counts[c][0])}'>{c}: {target_counts[c][0]} / {target_counts[c][1]}</div>"
            for c in self.target_classes
        ])

        return f"""
        <div style='padding:16px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;margin-bottom:14px;box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
            <div style='display:flex;gap:20px;align-items:center;margin-bottom:12px;'>
                <div style='color:#000000;'><b style='color:#111827;font-size:11px;text-transform:uppercase;'>Total</b><br><span style='font-size:18px;font-weight:800;color:#000000;'>{total}</span></div>
                <div style='color:#000000;'><b style='color:#111827;font-size:11px;text-transform:uppercase;'>Labeled</b><br><span style='font-size:18px;font-weight:800;color:#000000;'>{labeled}</span></div>
                <div style='color:#000000;'><b style='color:#111827;font-size:11px;text-transform:uppercase;'>Needs Review</b><br><span style='font-size:18px;font-weight:800;color:#000000;'>{needs_review}</span></div>
                <div style='color:#000000;'><b style='color:#111827;font-size:11px;text-transform:uppercase;'>Unlabeled</b><br><span style='font-size:18px;font-weight:800;color:#000000;'>{unlabeled}</span></div>
                <div style='flex:1;'>
                    <div style='display:flex;justify-content:space-between;font-weight:800;color:#000000;margin-bottom:6px;'>
                        <span style='font-size:13px;'>Dataset Progress</span>
                        <span style='font-size:13px;'>{percent:.1f}%</span>
                    </div>
                    <div style='height:12px;background:#e2e8f0;border-radius:999px;overflow:hidden;'>
                        <div style='height:12px;background:linear-gradient(90deg, #3b82f6, #2563eb);width:{percent:.1f}%;'></div>
                    </div>
                </div>
            </div>
            <div style='display:flex;flex-wrap:wrap;gap:6px;'>
                <b style='color:#64748b;font-size:11px;width:100%;margin-bottom:4px;text-transform:uppercase;'>Targeted Progress (Labeled / Pending)</b>
                {pills_html}
            </div>
        </div>
        """

    def get_info_html(self, row: Optional[pd.Series]) -> str:
        if row is None:
            return "<div style='padding:40px;text-align:center;background:#f8fafc;border-radius:14px;border:1px solid #e2e8f0;color:#64748b;font-weight:600;'>모든 unlabeled 작업이 완료되었습니다. 🎉</div>"

        def value(key: str, default: str = "-") -> str:
            raw = row.get(key, default)
            if is_blank(raw):
                return default
            return str(raw)

        return f"""
        <div style='padding:20px;background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);'>
            <div style='font-size:11px;color:#3b82f6;font-weight:800;letter-spacing:0.05em;margin-bottom:4px;'>TARGET GAME</div>
            <div style='font-size:22px;font-weight:900;color:#0f172a;margin-bottom:14px;line-height:1.2;'>{value('game_title', 'Unknown Title')}</div>
            
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:13px;'>
                <div style='background:#f1f5f9;padding:8px 12px;border-radius:8px;'><b style='color:#64748b;font-size:10px;text-transform:uppercase;'>File Name</b><br><span style='color:#334155;font-weight:600;'>{value('file_name')}</span></div>
                <div style='background:#fef2f2;padding:8px 12px;border-radius:8px;border:1px solid #fecaca;'><b style='color:#ef4444;font-size:10px;text-transform:uppercase;'>Target Category (Ref)</b><br><span style='color:#b91c1c;font-weight:800;'>{value('source_target_screen_type', 'None')}</span></div>
                
                <div style='background:#f1f5f9;padding:8px 12px;border-radius:8px;'><b style='color:#64748b;font-size:10px;text-transform:uppercase;'>Primary Screen</b><br><span style='color:#0f172a;font-weight:600;'>{value('primary_screen_type', '-')}</span></div>
                <div style='background:#f1f5f9;padding:8px 12px;border-radius:8px;'><b style='color:#64748b;font-size:10px;text-transform:uppercase;'>UI Quality</b><br><span style='color:#059669;font-weight:700;'>{value('ui_quality', '-')}</span></div>
                
                <div style='background:#f1f5f9;padding:8px 12px;border-radius:8px;'><b style='color:#64748b;font-size:10px;text-transform:uppercase;'>Visual Styles</b><br><span style='color:#334155;'>{value('visual_style_tags', '[]')}</span></div>
                <div style='background:#f1f5f9;padding:8px 12px;border-radius:8px;'><b style='color:#64748b;font-size:10px;text-transform:uppercase;'>Theme Tags</b><br><span style='color:#334155;'>{value('theme_tags', '[]')}</span></div>
                
                <div style='background:#f1f5f9;padding:8px 12px;border-radius:8px;grid-column:span 2;'><b style='color:#64748b;font-size:10px;text-transform:uppercase;'>Caption</b><br><span style='color:#334155;'>{value('screenshot_caption', '-')}</span></div>
            </div>
        </div>
        """

    def resolve_image_path(self, file_name: str) -> str:
        candidates = [
            os.path.join(IMAGE_DIR, file_name),
            os.path.join("images", file_name),
            file_name,
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def get_current_data(self) -> Tuple[Any, ...]:
        row = self.get_current_row()
        if row is None:
            values = (
                None,  # image
                self.get_info_html(None),
                False,  # is_game_ui
                "reject",  # ui_quality
                "other",  # primary
                [],  # secondary
                [],  # visual
                [],  # theme
                "[]",  # layout_blocks
                "[]",  # layout_tokens
                "[]",  # components
                0.0,  # confidence
                True,  # needs_review
                "",  # notes
            )
            values += ("",) * 24  # 8 blocks * 3 dropdowns
            values += (self.get_progress_summary(),)
            return assert_return_length("get_current_data_done", values, DATA_OUTPUT_COUNT)

        file_name = str(row.get("file_name", "") or "")
        img_path = self.resolve_image_path(file_name)
        image = None
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[!] 이미지 로드 실패: {img_path} / {e}")

        # Robust parsing for UI components
        is_game_ui = parse_bool_for_ui(row.get("is_game_ui", False))
        ui_quality = normalize_choice(row.get("ui_quality"), UI_QUALITY_VALUES, "reject")
        primary = normalize_choice(row.get("primary_screen_type"), PRIMARY_SCREEN_TYPES, "other")
        secondary = parse_json_list_for_checkbox(row.get("secondary_screen_types", "[]"), SECONDARY_SCREEN_TYPES)
        visual_styles = parse_json_list_for_checkbox(row.get("visual_style_tags", "[]"), VISUAL_STYLE_TAGS)
        themes = parse_json_list_for_checkbox(row.get("theme_tags", "[]"), THEME_TAGS)
        layout_blocks = load_json_textbox(row.get("layout_blocks", "[]"))
        layout_tokens = load_json_textbox(row.get("layout_tokens", "[]"))
        components = load_json_textbox(row.get("components", "[]"))

        try:
            confidence = float(row.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        needs_review = parse_bool_for_ui(row.get("needs_review", True), default=True)
        notes = str(row.get("notes", "") or "")

        values = (
            image,
            self.get_info_html(row),
            is_game_ui,
            ui_quality,
            primary,
            secondary,
            visual_styles,
            themes,
            format_json_for_textbox(layout_blocks),
            format_json_for_textbox(layout_tokens),
            format_json_for_textbox(components),
            confidence,
            needs_review,
            notes,
        )
        
        # Populate 24 dropdown values from layout_blocks
        dropdown_vals = []
        for i in range(MAX_LAYOUT_BLOCKS):
            if i < len(layout_blocks):
                block = layout_blocks[i]
                dropdown_vals.append(block.get("position", ""))
                dropdown_vals.append(block.get("element_type", ""))
                dropdown_vals.append(block.get("role", ""))
            else:
                dropdown_vals.extend(["", "", ""])
        
        values += tuple(dropdown_vals)
        values += (self.get_progress_summary(),)
        
        return assert_return_length("get_current_data", values, DATA_OUTPUT_COUNT)

    def save_and_next(
        self,
        is_game_ui: Any,
        ui_quality: Any,
        primary_screen_type: Any,
        secondary_screen_types: Any,
        visual_style_tags: Any,
        theme_tags: Any,
        layout_blocks_text: Any,
        components_text: Any,
        confidence: Any,
        needs_review: Any,
        notes: Any,
        *layout_dropdowns: Any,
        forced_status: Optional[str] = None
    ) -> Tuple[Tuple[Any, ...], str]:
        if self.current_idx >= len(self.df):
            return self.get_current_data(), "상태: 더 이상 작업할 데이터가 없습니다."

        file_name = self.df.iloc[self.current_idx].get("file_name", "unknown")
        print(f"\n[DEBUG] Saving file={file_name}")
        try:
            final_is_game_ui = parse_bool_for_ui(is_game_ui)
            final_ui_quality = normalize_choice(ui_quality, UI_QUALITY_VALUES, "reject")
            final_primary = normalize_choice(primary_screen_type, PRIMARY_SCREEN_TYPES, "other")
            final_secondary = normalize_string_list(secondary_screen_types, SECONDARY_SCREEN_TYPES)
            final_visual_styles = normalize_string_list(visual_style_tags, VISUAL_STYLE_TAGS)
            final_themes = normalize_string_list(theme_tags, THEME_TAGS)
            
            # Combine dropdowns into layout blocks if they are not empty
            dropdown_blocks = []
            for i in range(0, len(layout_dropdowns), 3):
                if i + 2 < len(layout_dropdowns):
                    p, t, r = layout_dropdowns[i], layout_dropdowns[i+1], layout_dropdowns[i+2]
                    if p or t or r:
                        dropdown_blocks.append({"position": p, "element_type": t, "role": r})
            
            # If dropdowns are empty, fallback to JSON text
            if not dropdown_blocks:
                final_layout_blocks = normalize_layout_blocks(layout_blocks_text)
            else:
                final_layout_blocks = normalize_layout_blocks(dropdown_blocks)
                
            final_layout_tokens = make_layout_tokens(final_layout_blocks)
            final_components = normalize_components(components_text)

            try:
                final_conf = float(confidence)
            except Exception:
                final_conf = 0.0
            final_conf = max(0.0, min(1.0, final_conf))
            final_needs_review = parse_bool_for_ui(needs_review)

            print("[DEBUG] Values:")
            debug_values = {
                "is_game_ui": final_is_game_ui,
                "ui_quality": final_ui_quality,
                "primary_screen_type": final_primary,
                "secondary_screen_types": final_secondary,
                "visual_style_tags": final_visual_styles,
                "theme_tags": final_themes,
                "layout_blocks": final_layout_blocks,
                "layout_tokens": final_layout_tokens,
                "components": final_components,
                "confidence": final_conf,
                "needs_review": final_needs_review,
                "notes": str(notes or "").strip(),
            }
            for k, v in debug_values.items():
                print(f"  - {k}: {type(v).__name__} = {v}")

            self.df.at[self.current_idx, "schema_version"] = SCHEMA_VERSION
            self.df.at[self.current_idx, "is_game_ui"] = serialize_bool_for_csv(final_is_game_ui)
            self.df.at[self.current_idx, "ui_quality"] = final_ui_quality
            self.df.at[self.current_idx, "primary_screen_type"] = final_primary
            self.df.at[self.current_idx, "secondary_screen_types"] = serialize_json_for_csv(final_secondary)
            self.df.at[self.current_idx, "visual_style_tags"] = serialize_json_for_csv(final_visual_styles)
            self.df.at[self.current_idx, "theme_tags"] = serialize_json_for_csv(final_themes)
            self.df.at[self.current_idx, "layout_blocks"] = serialize_json_for_csv(final_layout_blocks)
            self.df.at[self.current_idx, "layout_tokens"] = serialize_json_for_csv(final_layout_tokens)
            self.df.at[self.current_idx, "components"] = serialize_json_for_csv(final_components)
            self.df.at[self.current_idx, "confidence"] = str(final_conf)
            self.df.at[self.current_idx, "needs_review"] = serialize_bool_for_csv(final_needs_review)
            
            # Status logic: user button determines status
            if forced_status:
                self.df.at[self.current_idx, "review_status"] = forced_status
            else:
                # Fallback to automated logic if no forced status (e.g. from auto labeling)
                if final_needs_review or final_conf < 0.7:
                    self.df.at[self.current_idx, "review_status"] = "needs_review"
                else:
                    self.df.at[self.current_idx, "review_status"] = "labeled"
                
            saved_status = self.df.at[self.current_idx, "review_status"]
            self.df.at[self.current_idx, "notes"] = str(notes or "").strip()

            if saved_status in ["labeled", "needs_review"]:
                notes_text = self.df.at[self.current_idx, "notes"]
                if "original_path:" in notes_text:
                    orig_path = notes_text.split("original_path:")[1].strip()
                    if os.path.exists(orig_path) and "UnSelected" in orig_path:
                        import shutil
                        new_path = orig_path.replace("UnSelected", "Selected")
                        os.makedirs(os.path.dirname(new_path), exist_ok=True)
                        try:
                            shutil.move(orig_path, new_path)
                            self.df.at[self.current_idx, "notes"] = notes_text.replace("original_path:", "moved_to:")
                        except Exception as move_e:
                            print(f"[!] Failed to move file: {move_e}")

            self.flush()
            print(f"[OK] Saved {file_name} as {saved_status}")
            self._find_next_matching_item(advance=True)
            return self.get_current_data(), f"상태: 저장 완료 ({saved_status})"
        except Exception as e:
            print(f"\n[!!!] Error saving {file_name}: {e}")
            traceback.print_exc()
            return self.get_current_data(), f"오류 발생: {str(e)}"

    def skip(self) -> Tuple[Any, ...]:
        if self.current_idx < len(self.df):
            self.df.at[self.current_idx, "schema_version"] = SCHEMA_VERSION
            self.df.at[self.current_idx, "review_status"] = "skipped_permanent"
            self.flush()
            self._find_next_matching_item(advance=True)
        return self.get_current_data()

    def mark_retry_pending(self) -> Tuple[Any, ...]:
        if self.current_idx < len(self.df):
            self.df.at[self.current_idx, "schema_version"] = SCHEMA_VERSION
            self.df.at[self.current_idx, "review_status"] = "retry_pending"
            self.flush()
            self._find_next_matching_item(advance=True)
        return self.get_current_data()

    def restore_retry_pending(self) -> str:
        if "review_status" not in self.df.columns:
            return "review_status 컬럼이 없습니다."
        mask = self.df["review_status"] == "retry_pending"
        count = int(mask.sum())
        if count > 0:
            self.df.loc[mask, "review_status"] = "unlabeled"
            self.flush()
            self._find_next_matching_item()
        return f"{count}개 항목을 unlabeled로 복구했습니다."

    def generate_final_report(self) -> str:
        df = self.df.copy()
        
        # statistics based on review_status
        labeled_mask = df["review_status"] == "labeled"
        needs_review_mask = df["review_status"] == "needs_review"
        
        stats_labeled = df[labeled_mask]["primary_screen_type"].value_counts().to_dict()
        stats_needs = df[needs_review_mask]["primary_screen_type"].value_counts().to_dict()
        
        report_rows = []
        for cat in PRIMARY_SCREEN_TYPES:
            count_l = stats_labeled.get(cat, 0)
            count_n = stats_needs.get(cat, 0)
            report_rows.append({
                "category": cat,
                "labeled": count_l,
                "needs_review": count_n,
                "status": "OK" if count_l >= 30 else ("DEFICIT" if count_l >= 15 else "CRITICAL")
            })
            
        report_df = pd.DataFrame(report_rows)
        report_path = os.path.abspath("class_balance_report_after_labeling.csv")
        report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
        
        summary = f"### 📊 데이터 분포 보고서\n"
        summary += f"- **저장 위치**: `{report_path}`\n\n"
        
        total_l = int(labeled_mask.sum())
        total_n = int(needs_review_mask.sum())
        summary += f"| 항목 | 개수 |\n"
        summary += f"| :--- | :--- |\n"
        summary += f"| **Total Labeled (Ready)** | **{total_l}** |\n"
        summary += f"| Needs Review | {total_n} |\n\n"
        
        deficit = [r["category"] for r in report_rows if r["labeled"] < 30]
        summary += f"#### ⚠️ 부족한 클래스 (Goal 30 미달)\n"
        summary += f"- {', '.join(deficit) if deficit else '없음 (모두 달성!)'}"
        return summary


app = LabelingApp()

# ======================== UI callbacks ========================
def apply_ai_result(img: Image.Image):
    row_info = app.get_current_row_info()
    result = get_ai_analysis(img, row_info=row_info)
    if result is None:
        append_event("gpt_error", {"row": row_info})
        base_updates = (
            gr.update(),  # is_game_ui
            gr.update(),  # ui_quality
            gr.update(),  # primary_screen_type
            gr.update(),  # secondary_screen_types
            gr.update(),  # visual_style_tags
            gr.update(),  # theme_tags
            gr.update(),  # layout_blocks
            gr.update(),  # layout_tokens
            gr.update(),  # components
            gr.update(),  # confidence
            gr.update(),  # needs_review
            gr.update(),  # notes
        )
        dropdown_updates = (gr.update(),) * 24
        status = "상태: GPT 분석 실패. API 키, 네트워크, 모델 응답을 확인해줘."
        values = base_updates + dropdown_updates + (status,)
        return assert_return_length("apply_ai_result_error", values, AI_OUTPUT_COUNT)

    append_event("gpt_analysis", {"row": row_info, "result": result})
    base_updates = (
        gr.update(value=result["is_game_ui"]),
        gr.update(value=result["ui_quality"]),
        gr.update(value=result["primary_screen_type"]),
        gr.update(value=result["secondary_screen_types"]),
        gr.update(value=result["visual_style_tags"]),
        gr.update(value=result["theme_tags"]),
        gr.update(value=compact_json_text(result["layout_blocks"])),
        gr.update(value=compact_json_text(result["layout_tokens"])),
        gr.update(value=compact_json_text(result["components"])),
        gr.update(value=result["confidence"]),
        gr.update(value=result["needs_review"]),
        gr.update(value=result["notes"]),
    )
    
    # Populate dropdown updates
    dropdown_updates = []
    blocks = result.get("layout_blocks", [])
    for i in range(MAX_LAYOUT_BLOCKS):
        if i < len(blocks):
            b = blocks[i]
            dropdown_updates.extend([
                gr.update(value=b.get("position", "")),
                gr.update(value=b.get("element_type", "")),
                gr.update(value=b.get("role", ""))
            ])
        else:
            dropdown_updates.extend([gr.update(value=""), gr.update(value=""), gr.update(value="")])
    
    status = "상태: GPT 분석 완료. 결과를 확인한 뒤 저장하면 됩니다."
    values = base_updates + tuple(dropdown_updates) + (status,)
    return assert_return_length("apply_ai_result_success", values, AI_OUTPUT_COUNT)


def save_callback(is_ui, quality, primary, secondary, v_styles, themes, layout_text, components_text, confidence, needs_review, notes, *layout_dropdowns):
    next_data, status = app.save_and_next(is_ui, quality, primary, secondary, v_styles, themes, layout_text, components_text, confidence, needs_review, notes, *layout_dropdowns, forced_status="labeled")
    values = (*next_data, status)
    return assert_return_length("save_callback", values, SAVE_OUTPUT_COUNT)


def save_needs_review_callback(is_ui, quality, primary, secondary, v_styles, themes, layout_text, components_text, confidence, needs_review, notes, *layout_dropdowns):
    next_data, status = app.save_and_next(is_ui, quality, primary, secondary, v_styles, themes, layout_text, components_text, confidence, needs_review, notes, *layout_dropdowns, forced_status="needs_review")
    values = (*next_data, status)
    return assert_return_length("save_needs_review_callback", values, SAVE_OUTPUT_COUNT)


def skip_callback():
    values = (*app.skip(), "상태: 건너뛰기 완료")
    return assert_return_length("skip_callback", values, SAVE_OUTPUT_COUNT)


def retry_callback():
    values = (*app.mark_retry_pending(), "상태: retry_pending 처리 완료")
    return assert_return_length("retry_callback", values, SAVE_OUTPUT_COUNT)




def restore_retry_callback():
    msg = app.restore_retry_pending()
    values = (*app.get_current_data(), f"상태: {msg}")
    return assert_return_length("restore_retry_callback", values, SAVE_OUTPUT_COUNT)


def auto_step(mode, phase, img, is_ui, quality, primary, secondary, v_styles, themes, layout_text, components_text, confidence, needs_review, notes, *layout_dropdowns):
    """
    자동 라벨링의 2단계 프로세스:
    Phase 0: 현재 이미지를 GPT로 분석하여 UI에 결과를 표시하고 Phase 1로 변경.
    Phase 1: 표시된 결과를 저장하고 다음 이미지로 넘어가며 Phase 0으로 변경.
    """
    if not mode:
        return gr.update(active=False), 0, "상태: 자동 라벨링 중지", *unchanged_data_updates()

    if app.is_processing:
        return gr.update(), phase, "상태: 작업 처리 중...", *unchanged_data_updates()

    if img is None:
        return gr.update(active=False), 0, "상태: 작업 완료 또는 이미지 없음", *unchanged_data_updates()

    app.is_processing = True
    try:
        # Phase 0: 분석 및 결과 표시
        if phase == 0:
            row_info = app.get_current_row_info()
            result = get_ai_analysis(img, row_info=row_info)
            
            if result is None:
                # 분석 실패 시 retry_pending 처리 후 중단
                next_data = app.mark_retry_pending()
                return gr.update(active=False), 0, "상태: GPT 분석 실패. 중지됨.", *next_data

            append_event("auto_gpt_analysis", {"row": row_info, "result": result})

            # UI가 아니라고 판단되면 즉시 건너뛰고 다시 Phase 0 유지
            if result["ui_quality"] == "reject" or not result["is_game_ui"]:
                next_data = app.skip()
                return gr.update(active=True), 0, f"상태: UI 아님({result['ui_quality']}) - 건너뜀", *next_data

            # 분석 결과를 UI에 표시 (사용자가 볼 수 있도록)
            # data_outputs 형식에 맞춰 반환
            values = (
                img,
                app.get_info_html(app.get_current_row()),
                result["is_game_ui"],
                result["ui_quality"],
                result["primary_screen_type"],
                result["secondary_screen_types"],
                result["visual_style_tags"],
                result["theme_tags"],
                compact_json_text(result["layout_blocks"]),
                compact_json_text(result["layout_tokens"]),
                compact_json_text(result["components"]),
                result["confidence"],
                result["needs_review"],
                result["notes"],
            )
            
            # Populate dropdown values for auto phase 0
            dropdown_vals = []
            blocks = result.get("layout_blocks", [])
            for i in range(MAX_LAYOUT_BLOCKS):
                if i < len(blocks):
                    b = blocks[i]
                    dropdown_vals.extend([b.get("position", ""), b.get("element_type", ""), b.get("role", "")])
                else:
                    dropdown_vals.extend(["", "", ""])
            
            values += tuple(dropdown_vals)
            values += (app.get_progress_summary(),)
            
            return gr.update(active=True), 1, "상태: GPT 분석 완료 (결과 확인 중...)", *values

        # Phase 1: 저장 및 다음으로 이동
        else:
            next_data, _status = app.save_and_next(
                is_ui, quality, primary, secondary, v_styles, themes, 
                layout_text, components_text, confidence, needs_review, notes,
                *layout_dropdowns
            )
            return gr.update(active=True), 0, "상태: 자동 저장 완료 (다음 이미지로)", *next_data

    except Exception as e:
        print(f"[!] 자동 라벨링 오류: {e}")
        traceback.print_exc()
        next_data = app.mark_retry_pending()
        return gr.update(active=False), 0, f"상태: 오류 발생({e}). 중지됨.", *next_data
    finally:
        app.is_processing = False

# ======================== Gradio UI ========================
custom_css = """
body { background-color: #f1f5f9 !important; }
.gradio-container { max-width: 1400px !important; margin: 20px auto !important; background-color: transparent !important; }
textarea { font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, monospace !important; font-size: 13px !important; }
.gr-button-primary { background: linear-gradient(135deg, #2563eb, #1d4ed8) !important; border: none !important; box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important; }
.gr-button-primary:hover { background: linear-gradient(135deg, #1d4ed8, #1e40af) !important; transform: translateY(-1px); }
.gr-block { border-radius: 16px !important; border: 1px solid #e2e8f0 !important; background-color: #ffffff !important; padding: 16px !important; }
.gr-input, .gr-select, .gr-checkbox, .gr-radio { border-radius: 8px !important; border: 1px solid #e2e8f0 !important; }
.gr-padded { padding: 20px !important; }
h1 { color: #ffffff !important; font-weight: 900 !important; letter-spacing: -0.025em !important; }
.prose h1, .markdown h1, .gradio-container h1 { color: #ffffff !important; }
"""

with gr.Blocks(title="Game UI Labeling Tool", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# Game UI Labeling Tool - Schema 6.0")
    gr.Markdown("MobyGames 기반 스크린샷을 새 데이터셋 구조로 라벨링합니다.")

    progress_display = gr.HTML()
    status_display = gr.Markdown("상태: 대기 중")
    auto_mode = gr.State(value=False)
    auto_phase = gr.State(value=0)  # 0: 분석 대기, 1: 결과 표시 중
    timer = gr.Timer(value=DEFAULT_AUTO_DELAY, active=False)

    with gr.Row():
        with gr.Column(scale=2):
            img_display = gr.Image(label="Screenshot", type="pil", height=620)
            info_display = gr.HTML()
        with gr.Column(scale=1):
            with gr.Row():
                ai_btn = gr.Button("GPT 분석", variant="secondary")
                save_btn = gr.Button("확정 저장", variant="primary")
            with gr.Row():
                save_needs_btn = gr.Button("재검토로 저장")
                skip_btn = gr.Button("건너뛰기")
                retry_btn = gr.Button("retry_pending")

            with gr.Accordion("데이터 필터 및 도구", open=True):
                filter_mode = gr.Radio(
                    choices=[
                        ("전체 미검수", "unlabeled"), 
                        ("재검토 필요", "needs_review"),
                        ("개인 리소스 우선", "personal"),
                        ("타겟 수집물 우선", "targeted"),
                        ("의심 데이터 우선", "suspicious"), 
                        ("모두 보기", "all")
                    ],
                    value="unlabeled",
                    label="검수 필터"
                )
                target_select = gr.Dropdown(
                    choices=["All"] + app.target_classes,
                    value="All",
                    label="타겟 클래스 선택 (타겟 수집물 우선 모드 전용)",
                    interactive=True
                )
                report_btn = gr.Button("최종 분포 보고서 생성", variant="secondary")
                report_display = gr.Markdown("보고서가 생성되면 여기에 표시됩니다.")
                restore_retry_btn = gr.Button("retry 복구")

            with gr.Accordion("자동 라벨링", open=False):
                with gr.Row():
                    auto_start_btn = gr.Button("자동 시작")
                    auto_stop_btn = gr.Button("자동 중지")
                auto_delay = gr.Slider(0.5, 10.0, value=DEFAULT_AUTO_DELAY, step=0.1, label="자동 처리 간격")

            is_game_ui = gr.Checkbox(label="is_game_ui", value=False)
            ui_quality = gr.Dropdown(choices=UI_QUALITY_VALUES, label="ui_quality", value="reject")
            primary_screen_type = gr.Dropdown(choices=PRIMARY_SCREEN_TYPES, label="primary_screen_type", value="other")
            secondary_screen_types = gr.CheckboxGroup(choices=SECONDARY_SCREEN_TYPES, label="secondary_screen_types")
            visual_style_tags = gr.CheckboxGroup(choices=VISUAL_STYLE_TAGS, label="visual_style_tags")
            theme_tags = gr.CheckboxGroup(choices=THEME_TAGS, label="theme_tags")
            
            with gr.Accordion("Layout Blocks (Visual Editor)", open=True):
                layout_dropdowns = []
                for i in range(MAX_LAYOUT_BLOCKS):
                    with gr.Row():
                        p = gr.Dropdown(choices=[""] + POSITIONS, label=f"Pos {i+1}", scale=1)
                        t = gr.Dropdown(choices=[""] + ELEMENT_TYPES, label=f"Type {i+1}", scale=1)
                        r = gr.Dropdown(choices=[""] + ROLES, label=f"Role {i+1}", scale=1)
                        layout_dropdowns.extend([p, t, r])
            
            layout_blocks = gr.Textbox(label="layout_blocks JSON", lines=3, value="[]")
            layout_tokens = gr.Textbox(label="layout_tokens JSON", lines=3, value="[]", interactive=False)
            components = gr.Textbox(label="components JSON", lines=5, value="[]")
            confidence = gr.Number(label="confidence", value=0.0, precision=2)
            needs_review = gr.Checkbox(label="needs_review", value=True)
            notes = gr.Textbox(label="notes", lines=3)

            shutdown_btn = gr.Button("서버 종료", variant="stop")

    data_outputs = [
        img_display,
        info_display,
        is_game_ui,
        ui_quality,
        primary_screen_type,
        secondary_screen_types,
        visual_style_tags,
        theme_tags,
        layout_blocks,
        layout_tokens,
        components,
        confidence,
        needs_review,
        notes,
        *layout_dropdowns,
        progress_display,
    ]

    ai_outputs = [
        is_game_ui,
        ui_quality,
        primary_screen_type,
        secondary_screen_types,
        visual_style_tags,
        theme_tags,
        layout_blocks,
        layout_tokens,
        components,
        confidence,
        needs_review,
        notes,
        *layout_dropdowns,
        status_display,
    ]

    ai_btn.click(apply_ai_result, inputs=[img_display], outputs=ai_outputs)

    save_btn.click(
        save_callback,
        inputs=[is_game_ui, ui_quality, primary_screen_type, secondary_screen_types, visual_style_tags, theme_tags, layout_blocks, components, confidence, needs_review, notes, *layout_dropdowns],
        outputs=[*data_outputs, status_display],
    )

    save_needs_btn.click(
        save_needs_review_callback,
        inputs=[is_game_ui, ui_quality, primary_screen_type, secondary_screen_types, visual_style_tags, theme_tags, layout_blocks, components, confidence, needs_review, notes, *layout_dropdowns],
        outputs=[*data_outputs, status_display],
    )

    skip_btn.click(skip_callback, outputs=[*data_outputs, status_display])
    retry_btn.click(retry_callback, outputs=[*data_outputs, status_display])
    
    def change_filter(mode, target):
        app.set_filter_mode(mode, target)
        return (*app.get_current_data(), f"상태: 필터 변경({mode} / {target})")

    filter_mode.change(change_filter, inputs=[filter_mode, target_select], outputs=[*data_outputs, status_display])
    target_select.change(change_filter, inputs=[filter_mode, target_select], outputs=[*data_outputs, status_display])
    
    report_btn.click(lambda: app.generate_final_report(), outputs=[report_display])
    restore_retry_btn.click(restore_retry_callback, outputs=[*data_outputs, status_display])

    auto_start_btn.click(
        lambda: (True, 0, gr.update(active=True), "상태: 자동 라벨링 시작"),
        outputs=[auto_mode, auto_phase, timer, status_display],
    )
    auto_stop_btn.click(
        lambda: (False, 0, gr.update(active=False), "상태: 자동 라벨링 중지 요청"),
        outputs=[auto_mode, auto_phase, timer, status_display],
    )
    auto_delay.change(lambda v: gr.update(value=v), inputs=auto_delay, outputs=timer)

    timer.tick(
        auto_step, 
        inputs=[auto_mode, auto_phase, img_display, is_game_ui, ui_quality, primary_screen_type, secondary_screen_types, visual_style_tags, theme_tags, layout_blocks, components, confidence, needs_review, notes, *layout_dropdowns], 
        outputs=[timer, auto_phase, status_display, *data_outputs]
    )

    def terminate_server():
        app.flush()
        os.kill(os.getpid(), signal.SIGTERM)

    shutdown_btn.click(terminate_server)

    demo.load(lambda: (*app.get_current_data(), "상태: 대기 중"), outputs=[*data_outputs, status_display])

if __name__ == "__main__":
    print(f"\n[INIT] Starting Labeling Tool...")
    print(f"[INIT] Path: {os.path.abspath(__file__)}")
    print(f"[INIT] style_tags in REQUIRED_COLUMNS: {'style_tags' in REQUIRED_COLUMNS}")
    
    checkbox_labels = []
    # Collect labels from CheckboxGroups manually by searching the Blocks structure
    # or just print the constants we used.
    print(f"[INIT] Checkbox Groups:")
    print(f"  - secondary_screen_types (count: {len(SECONDARY_SCREEN_TYPES)})")
    print(f"  - visual_style_tags (count: {len(VISUAL_STYLE_TAGS)})")
    print(f"  - theme_tags (count: {len(THEME_TAGS)})")
    
    if 'style_tags' in globals() or 'style_tags' in locals():
        print("[WARNING] 'style_tags' variable still exists in global/local scope!")

    demo.launch(inbrowser=False, share=False)
