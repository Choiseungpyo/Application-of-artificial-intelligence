# -*- coding: utf-8 -*-
"""
=======================================================================
Game UI Discovery Studio - schema 6.0 web app

- Text query: interprets text into primary_screen_type, style_tags, layout_tokens
- Image query: uses fine-tuned model prediction when a checkpoint exists
- Search ranking: uses SigLIP2 similarity + label/layout guided bonuses
=======================================================================
"""

import base64
import html
import json
import os
import re
import sys
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gradio as gr
import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoModel, AutoProcessor

from model import GameUIModel
from search_engine import GameUISearchEngine, PRIMARY_SCREEN_MAPPING, normalize_label, to_primary_group

load_dotenv()
try:
    # sys.stdout.reconfigure(encoding="utf-8")
    pass
except Exception:
    pass

MODEL_NAME = "google/siglip2-base-patch16-224"
STYLE_THRESHOLD = 0.50
LAYOUT_THRESHOLD = 0.45
BASE_STYLE_TOP_K = 3
BASE_LAYOUT_TOP_K = 5
ENABLE_BASELINE_COMPARE = True

CHECKPOINT_CANDIDATES = [
    "outputs/retrain_grouped_v2/best_model.pth",
    "output_retrain_midcheck/best_model.pth",
    "output_run3_epoch15/best_model.pth",
    "output_run2_epoch10/best_model.pth",
    "output/best_model.pth",
]

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
    "pause_menu",
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
    "top_left:bar:status",
    "top_center:bar:status",
    "top_right:minimap:navigation",
    "top_right:resource_bar:resource",
    "left:menu:navigation",
    "left:panel:party",
    "center:popup:system",
    "center:preview:character",
    "center_left:grid:inventory",
    "center_right:panel:character",
    "right:panel:quest",
    "right:panel:character",
    "bottom_left:chat_box:social",
    "bottom_center:skill_bar:combat",
    "bottom_center:dialogue_box:dialogue",
    "bottom_right:slot_group:inventory",
    "full_screen:menu:navigation",
    "full_screen:panel:settings",
]

PRIMARY_SCREEN_KEYWORDS = {
    "gameplay_hud": [
        "hud", "gameplay", "in game", "ingame", "combat hud", "battle hud", "플레이", "인게임", "전투", "전투 hud", "체력바", "스킬바",
    ],
    "main_menu": ["main menu", "menu screen", "메인 메뉴", "메뉴 화면", "시작 메뉴"],
    "lobby": ["lobby", "home", "hub", "room", "waiting room", "로비", "홈", "대기실", "방 목록"],
    "inventory": ["inventory", "bag", "items", "item grid", "인벤토리", "가방", "아이템창", "아이템"],
    "equipment": ["equipment", "gear", "loadout", "equip", "장비", "장착", "로드아웃"],
    "map": ["map", "world map", "mini map", "지도", "월드맵", "맵"],
    "shop": ["shop", "store", "vendor", "market", "상점", "스토어", "구매", "판매"],
    "settings": ["settings", "options", "config", "설정", "옵션"],
    "dialogue": ["dialogue", "dialog", "conversation", "npc", "대화", "대사", "대화창"],
    "quest": ["quest", "mission", "journal", "quest log", "퀘스트", "임무", "저널"],
    "character_screen": ["character", "status", "stats", "profile", "캐릭터", "스탯", "능력치", "프로필"],
    "skill_tree": ["skill tree", "skills", "talent", "perks", "스킬트리", "스킬", "특성"],
    "crafting": ["crafting", "craft", "forge", "recipe", "제작", "합성", "레시피"],
    "battle_result": ["result", "victory", "defeat", "reward", "score", "결과", "승리", "패배", "보상", "점수"],
    "loading_screen": ["loading", "load screen", "로딩"],
    "title_screen": ["title", "start screen", "press start", "타이틀", "시작 화면"],
    "tutorial": ["tutorial", "guide", "help", "튜토리얼", "가이드", "도움말"],
    "pause_menu": ["pause", "pause menu", "esc", "일시정지", "정지 메뉴"],
}

STYLE_KEYWORDS = {
    # Themes
    "fantasy": ["fantasy", "magic", "rpg", "판타지", "마법"],
    "dark_fantasy": ["dark fantasy", "gothic", "dark rpg", "다크 판타지", "어두운 판타지"],
    "medieval": ["medieval", "middle age", "중세"],
    "sci_fi": ["sci fi", "sci-fi", "science fiction", "space", "sf", "공상과학", "우주"],
    "cyberpunk": ["cyberpunk", "neon", "future noir", "사이버펑크", "네온"],
    "military": ["military", "tactical", "army", "war", "군사", "밀리터리", "전술"],
    "horror": ["horror", "scary", "blood", "공포", "호러"],
    "historical": ["historical", "history", "역사", "사극"],
    "modern_world": ["modern world", "contemporary world", "현대 세계", "현대물"],
    "post_apocalyptic": ["post apocalyptic", "apocalypse", "포스트 아포칼립스", "세기말", "종말"],
    "adult": ["adult", "hentai", "성인", "19금"],
    
    # Visual Styles (Grouped)
    "modern_clean": ["modern clean", "modern", "contemporary", "현대적", "모던", "clean", "neat", "깔끔", "정돈", "minimal", "미니멀", "flat", "플랫"],
    "realistic_gritty": ["realistic gritty", "realistic", "real", "photoreal", "사실적", "리얼", "gritty", "dirty", "worn", "거친", "낡은"],
    "retro_pixel": ["retro pixel", "retro", "arcade", "oldschool", "레트로", "고전", "pixel", "pixel art", "픽셀", "도트"],
    "stylized_cartoon": ["stylized cartoon", "cartoon", "toon", "카툰", "만화", "anime", "애니", "cute", "casual", "귀여운", "아기자기", "skeuomorphic", "스큐어모피즘"],
    
    # Fallback individual styles (just in case)
    "realistic": ["realistic", "real", "photoreal", "사실적", "리얼"],
    "cartoon": ["cartoon", "toon", "카툰", "만화"],
    "pixel_art": ["pixel", "pixel art", "픽셀", "도트"],
    "anime": ["anime", "애니"],
    "minimal": ["minimal", "minimalist", "미니멀", "간결"],
    "clean": ["clean", "neat", "깔끔", "정돈"],
    "skeuomorphic": ["skeuomorphic", "ornate", "embossed", "스큐어모피즘", "입체", "장식적"],
    "flat": ["flat", "플랫"],
    "neon": ["neon", "glow", "glowing", "네온", "발광"],
    "retro": ["retro", "arcade", "oldschool", "레트로", "고전"],
    "modern": ["modern", "contemporary", "현대적", "모던"],
    "cute": ["cute", "casual", "귀여운", "아기자기"],
    "gritty": ["gritty", "dirty", "worn", "거친", "낡은"],
}

POSITION_KEYWORDS = {
    "top_left": ["top left", "upper left", "좌측 상단", "왼쪽 상단"],
    "top_center": ["top center", "upper center", "상단 중앙", "위쪽 중앙"],
    "top_right": ["top right", "upper right", "우측 상단", "오른쪽 상단"],
    "left": ["left side", "left panel", "좌측", "왼쪽"],
    "center_left": ["center left", "middle left", "중앙 좌측"],
    "center": ["center", "middle", "중앙", "가운데"],
    "center_right": ["center right", "middle right", "중앙 우측"],
    "right": ["right side", "right panel", "우측", "오른쪽"],
    "bottom_left": ["bottom left", "lower left", "좌측 하단", "왼쪽 하단"],
    "bottom_center": ["bottom center", "lower center", "하단 중앙", "아래 중앙"],
    "bottom_right": ["bottom right", "lower right", "우측 하단", "오른쪽 하단"],
    "full_screen": ["full screen", "entire screen", "전체 화면", "풀스크린"],
    "overlay": ["overlay", "오버레이", "겹쳐진"],
}

ELEMENT_KEYWORDS = {
    "bar": ["bar", "status bar", "바", "상태바", "정보바"],
    "panel": ["panel", "side panel", "패널"],
    "popup": ["popup", "modal", "pop up", "팝업", "모달"],
    "menu": ["menu", "메뉴"],
    "tab_bar": ["tab", "tab bar", "탭"],
    "grid": ["grid", "격자", "그리드"],
    "list": ["list", "목록", "리스트"],
    "card_group": ["card", "cards", "카드"],
    "slot_group": ["slot", "slots", "quick slot", "슬롯", "퀵슬롯"],
    "button_group": ["button", "buttons", "버튼"],
    "preview": ["preview", "character preview", "미리보기", "프리뷰"],
    "dialogue_box": ["dialogue box", "text box", "대화창", "대사창"],
    "minimap": ["minimap", "mini map", "미니맵"],
    "skill_bar": ["skill bar", "hotbar", "skill icons", "스킬바", "스킬 아이콘"],
    "health_bar": ["health bar", "hp bar", "체력바", "hp"],
    "resource_bar": ["resource bar", "currency", "gold", "재화", "자원", "골드"],
    "tooltip": ["tooltip", "툴팁", "설명 박스"],
    "notification": ["notification", "alert", "알림", "경고"],
    "portrait_group": ["portrait", "portraits", "초상화"],
    "progress_bar": ["progress bar", "loading bar", "진행바", "로딩바"],
    "chat_box": ["chat", "chat box", "채팅", "채팅창"],
}

ROLE_KEYWORDS = {
    "status": ["status", "상태"],
    "health": ["health", "hp", "체력"],
    "resource": ["resource", "currency", "gold", "mana", "자원", "재화", "골드", "마나"],
    "combat": ["combat", "skill", "attack", "전투", "스킬", "공격"],
    "navigation": ["navigation", "map", "nav", "탐색", "이동", "지도"],
    "quest": ["quest", "mission", "퀘스트", "임무"],
    "inventory": ["inventory", "item", "인벤토리", "아이템"],
    "equipment": ["equipment", "gear", "장비"],
    "character": ["character", "hero", "캐릭터", "영웅"],
    "skill": ["skill", "스킬"],
    "shop": ["shop", "store", "상점"],
    "crafting": ["crafting", "제작"],
    "dialogue": ["dialogue", "대화"],
    "settings": ["settings", "설정"],
    "social": ["social", "party", "chat", "소셜", "파티", "채팅"],
    "notification": ["notification", "알림"],
    "tutorial": ["tutorial", "튜토리얼"],
    "result": ["result", "reward", "결과", "보상"],
    "loading": ["loading", "로딩"],
    "selection": ["selection", "select", "선택"],
    "system": ["system", "confirm", "cancel", "시스템", "확인", "취소"],
}

KOREAN_KEYWORD_MAP = {
    "타이틀 화면": "menu_lobby",
    "타이틀화면": "menu_lobby",
    "타이틀": "menu_lobby",
    "시작 화면": "menu_lobby",
    "시작화면": "menu_lobby",
    "메인 화면": "menu_lobby",
    "메인화면": "menu_lobby",
    "로비 화면": "menu_lobby",
    "로비화면": "menu_lobby",
    "대기실": "menu_lobby",
    "설정 화면": "settings",
    "옵션 화면": "settings",
    "설정창": "settings",
    "옵션창": "settings",
    "퀘스트 창": "quest",
    "임무 창": "quest",
    "퀘스트": "quest",
    "임무": "quest",
    "인벤토리": "inventory",
    "가방": "inventory",
    "상점": "shop",
    "구매 화면": "shop",
    "전투 결과": "battle_result",
    "결과창": "battle_result",
    "게임오버": "flow_other",
    "사망 화면": "flow_other",
    "hud": "gameplay_hud",
    "플레이 화면": "gameplay_hud",
    "상태창": "gameplay_panel",
    "캐릭터 창": "gameplay_panel",
    "스킬 창": "gameplay_panel",
    "패널 화면": "gameplay_panel",
    "판타지": "fantasy",
    "중세": "medieval",
    "현대": "modern",
    "sf": "sci_fi",
    "에스에프": "sci_fi",
    "어두운": "dark",
    "픽셀": "pixel_art",
    "레트로": "retro",
    "카툰": "cartoon",
    "깔끔한": "clean",
    "미니멀": "minimal",
}

print("[*] 시스템 초기화 중...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
base_model.eval()
engine = GameUISearchEngine(processor, base_model, device)

classifier_model: Optional[GameUIModel] = None
classifier_primary_screen_types: List[str] = []
classifier_visual_style_tags: List[str] = []
classifier_theme_tags: List[str] = []
classifier_layout_positions: List[str] = []
classifier_layout_element_types: List[str] = []
classifier_layout_roles: List[str] = []
active_checkpoint_path: Optional[str] = None
base_primary_text_embeds = None
base_visual_style_text_embeds = None
base_theme_text_embeds = None
base_layout_text_embeds = None


def escape(text: Any) -> str:
    return html.escape(str(text), quote=True)


def normalize_query(text: str) -> str:
    lowered = str(text).lower().replace("_", " ").replace("-", " ")
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def prettify_label_name(text: Any) -> str:
    return str(text).replace("_", " ").replace(":", " : ").strip()


LABEL_TRANSLATIONS = {
    # Primary Screen Types
    "dialogue_story": "대화/스토리",
    "flow_other": "기타 흐름/로딩/결과",
    "gameplay_hud": "인게임 HUD",
    "gameplay_panel": "게임 플레이 패널",
    "menu_lobby": "메뉴/로비",
    "main_menu": "메인 메뉴",
    "lobby": "로비",
    "title_screen": "타이틀 화면",
    "settings": "설정/옵션",
    "inventory": "인벤토리/가방",
    "equipment": "장비 장착",
    "shop": "상점/구매",
    "battle_result": "전투 결과",
    "loading_screen": "로딩 화면",
    "tutorial": "튜토리얼/가이드",
    "pause_menu": "일시정지 메뉴",
    "quest": "퀘스트/임무",
    "map": "지도/월드맵",
    "map_screen": "지도/월드맵",
    "character_screen": "캐릭터 상태창",
    "skill_tree": "스킬 트리",
    "crafting": "제작/합성",
    "other": "기타",
    # Styles
    "modern_clean": "모던/깔끔함",
    "realistic_gritty": "실사풍/거침",
    "retro_pixel": "레트로/픽셀",
    "stylized_cartoon": "카툰/애니메이션",
    "realistic": "실사풍",
    "cartoon": "카툰풍",
    "anime": "애니메이션풍",
    "pixel_art": "픽셀 아트",
    "retro": "레트로",
    "modern": "현대풍",
    "minimal": "미니멀",
    "clean": "깔끔함",
    "skeuomorphic": "스큐어모피즘",
    "flat": "플랫",
    "neon": "네온",
    "gritty": "거친 느낌",
    "cute": "귀여움",
    # Themes
    "fantasy": "판타지",
    "dark_fantasy": "다크 판타지",
    "medieval": "중세",
    "sci_fi": "SF/공상과학",
    "cyberpunk": "사이버펑크",
    "military": "밀리터리/군사",
    "horror": "공포/호러",
    "post_apocalyptic": "포스트 아포칼립스",
    "historical": "역사/시대극",
    "modern_world": "현대 월드",
    "adult": "성인/다크",
    # Positions
    "top": "상단",
    "bottom": "하단",
    "center": "중앙",
    "left": "좌측",
    "right": "우측",
    "full_screen": "전체 화면",
    "top_left": "좌측 상단",
    "top_center": "상단 중앙",
    "top_right": "우측 상단",
    "bottom_left": "좌측 하단",
    "bottom_center": "하단 중앙",
    "bottom_right": "우측 하단",
    "center_left": "중앙 좌측",
    "center_right": "중앙 우측",
    # Elements
    "panel_menu": "패널/메뉴",
    "container_grid": "그리드/컨테이너",
    "button": "버튼 그룹",
    "preview_avatar": "캐릭터 프리뷰",
    "text_box": "텍스트 박스",
    "hud_bar_indicator": "HUD 바/인디케이터",
    # Roles
    "navigation_quest": "네비게이션/퀘스트",
    "status_resource": "상태/자원",
    "combat_skill": "전투/스킬",
    "inventory_shop": "인벤토리/상점",
    "character_select": "캐릭터 선택",
    "system_narrative": "시스템/내러티브",
}
LABEL_TRANSLATIONS = {normalize_label(k): v for k, v in LABEL_TRANSLATIONS.items()}


def translate_label(eng_label: str) -> str:
    norm = normalize_label(eng_label)
    if not norm:
        return ""
    # Check if this is a layout token (contains colon ':')
    if ":" in norm:
        parts = norm.split(":")
        translated_parts = []
        for p in parts:
            p_norm = normalize_label(p)
            translated_parts.append(LABEL_TRANSLATIONS.get(p_norm, prettify_label_name(p_norm)))
        return f"{prettify_label_name(norm)} (" + " : ".join(translated_parts) + ")"

    korean = LABEL_TRANSLATIONS.get(norm)
    if korean:
        return f"{prettify_label_name(norm)} ({korean})"
    return prettify_label_name(norm)


def safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    text = str(value).strip()
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
    return [x.strip() for x in text.split(",") if x.strip()]


def safe_str_list(value: Any) -> List[str]:
    result: List[str] = []
    for item in safe_list(value):
        if isinstance(item, dict):
            continue
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def layout_block_to_token(block: Dict[str, Any]) -> str:
    position = str(block.get("position", "")).strip()
    element_type = str(block.get("element_type", "")).strip()
    role = str(block.get("role", "general")).strip() or "general"
    if not position or not element_type:
        return ""
    return f"{position}:{element_type}:{role}"


def get_base64_image(image_path: str, size=(1280, 720)) -> str:
    if not os.path.exists(image_path):
        return ""
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail(size)
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=90)
            return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print(f"[!] 이미지 인코딩 실패: {e}")
        return ""


def resolve_checkpoint_path() -> Optional[str]:
    for path in CHECKPOINT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def get_cache_label_space() -> Tuple[List[str], List[str], List[str], List[str]]:
    if engine.cache_data is None:
        return [], [], [], []

    primary_values = engine.cache_data.get("primary_screen_types", [])
    v_style_values = engine.cache_data.get("visual_style_tags", [])
    theme_values = engine.cache_data.get("theme_tags", [])
    layout_pos_values = engine.cache_data.get("layout_positions", [])
    layout_elem_values = engine.cache_data.get("layout_element_types", [])
    layout_role_values = engine.cache_data.get("layout_roles", [])
    layout_tokens = engine.cache_data.get("layout_tokens", [])

    primary = sorted({str(x).strip() for x in primary_values if str(x).strip()})

    v_styles = set()
    for row in v_style_values:
        for tag in safe_str_list(row):
            v_styles.add(tag)

    themes = set()
    for row in theme_values:
        for tag in safe_str_list(row):
            themes.add(tag)

    layouts = set()
    for row in layout_tokens:
        for token in safe_str_list(row):
            layouts.add(token)

    # Note: Search guided still uses layout_tokens for legacy support, 
    # but we also need the split axes from cache if available
    pos = set()
    for row in layout_pos_values:
        for p in safe_str_list(row): pos.add(p)
    elem = set()
    for row in layout_elem_values:
        for e in safe_str_list(row): elem.add(e)
    role = set()
    for row in layout_role_values:
        for r in safe_str_list(row): role.add(r)

    return primary, sorted(v_styles), sorted(themes), sorted(layouts)


def get_effective_label_space() -> Tuple[List[str], List[str], List[str], List[str]]:
    if classifier_primary_screen_types and classifier_visual_style_tags and classifier_theme_tags:
        return classifier_primary_screen_types, classifier_visual_style_tags, classifier_theme_tags, []

    cache_primary, cache_v_styles, cache_themes, cache_layouts = get_cache_label_space()
    return (
        cache_primary or DEFAULT_PRIMARY_SCREEN_TYPES[:],
        cache_v_styles or DEFAULT_VISUAL_STYLE_TAGS[:],
        cache_themes or DEFAULT_THEME_TAGS[:],
        cache_layouts or DEFAULT_LAYOUT_TOKENS[:],
    )


def build_zero_shot_prompt_lists(primary_types: List[str], v_style_tags: List[str], theme_tags: List[str], layout_tokens: List[str]):
    primary_prompts = [f"a game UI screenshot of a {prettify_label_name(x)} screen" for x in primary_types]
    v_style_prompts = [f"a game UI with {prettify_label_name(x)} style" for x in v_style_tags]
    theme_prompts = [f"a game UI with {prettify_label_name(x)} theme" for x in theme_tags]
    layout_prompts = [f"a game UI layout with {prettify_label_name(x)}" for x in layout_tokens]
    return primary_prompts, v_style_prompts, theme_prompts, layout_prompts


def encode_text_prompts(prompts: List[str]) -> Optional[torch.Tensor]:
    if not prompts:
        return None
    with torch.inference_mode():
        inputs = processor(text=prompts, padding="max_length", return_tensors="pt").to(device)
        outputs = base_model.get_text_features(**inputs)
        features = getattr(outputs, "text_embeds", outputs)
        if not isinstance(features, torch.Tensor):
            features = getattr(outputs, "pooler_output", features)
        return torch.nn.functional.normalize(features, p=2, dim=-1)


def rebuild_zero_shot_embeddings() -> None:
    global base_primary_text_embeds, base_visual_style_text_embeds, base_theme_text_embeds, base_layout_text_embeds
    primary_types, v_style_tags, theme_tags, layout_tokens = get_effective_label_space()
    primary_prompts, v_style_prompts, theme_prompts, layout_prompts = build_zero_shot_prompt_lists(primary_types, v_style_tags, theme_tags, layout_tokens)
    base_primary_text_embeds = encode_text_prompts(primary_prompts)
    base_visual_style_text_embeds = encode_text_prompts(v_style_prompts)
    base_theme_text_embeds = encode_text_prompts(theme_prompts)
    base_layout_text_embeds = encode_text_prompts(layout_prompts)


def load_classifier() -> None:
    global classifier_model, classifier_primary_screen_types, classifier_visual_style_tags, classifier_theme_tags, \
           classifier_layout_positions, classifier_layout_element_types, classifier_layout_roles, active_checkpoint_path

    checkpoint_path = resolve_checkpoint_path()
    active_checkpoint_path = checkpoint_path

    if checkpoint_path is None:
        print("[!] 분류 모델 체크포인트를 찾지 못했습니다. 텍스트 라벨 해석과 기본 SigLIP2 검색만 사용합니다.")
        rebuild_zero_shot_embeddings()
        return

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        vocab = ckpt.get("vocab", {})
        classifier_primary_screen_types = vocab.get("primary_screen_types", []) or DEFAULT_PRIMARY_SCREEN_TYPES[:]
        classifier_visual_style_tags = vocab.get("visual_style_tags", []) or DEFAULT_VISUAL_STYLE_TAGS[:]
        classifier_theme_tags = vocab.get("theme_tags", []) or DEFAULT_THEME_TAGS[:]
        classifier_layout_positions = vocab.get("layout_positions", [])
        classifier_layout_element_types = vocab.get("layout_element_types", [])
        classifier_layout_roles = vocab.get("layout_roles", [])

        model = GameUIModel(
            model_name=MODEL_NAME,
            num_primary_screen_types=len(classifier_primary_screen_types),
            num_visual_style_tags=len(classifier_visual_style_tags),
            num_theme_tags=len(classifier_theme_tags),
            num_layout_positions=len(classifier_layout_positions),
            num_layout_element_types=len(classifier_layout_element_types),
            num_layout_roles=len(classifier_layout_roles),
            freeze_backbone=True,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        classifier_model = model

        print(f"[+] 분류 모델 로드 완료: {checkpoint_path}")
        print(f"[*] primary screen types: {len(classifier_primary_screen_types)}")
        print(f"[*] visual style tags: {len(classifier_visual_style_tags)}")
        print(f"[*] theme tags: {len(classifier_theme_tags)}")
        print(f"[*] layout positions: {len(classifier_layout_positions)}")
        print(f"[*] layout element types: {len(classifier_layout_element_types)}")
        print(f"[*] layout roles: {len(classifier_layout_roles)}")
    except Exception as e:
        print(f"[!] 분류 모델 로드 실패: {e}")
        classifier_model = None

    rebuild_zero_shot_embeddings()

load_classifier()

# -------------------------------------------------------------
# SBERT Semantic Search Integration
# -------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None

sbert_model = None
sbert_tag_embeddings = {}

SBERT_PROMPTS = {
    # primary
    "menu_lobby": "메인 메뉴, 타이틀 화면, 시작 화면, 로비, 상점, 설정처럼 게임 시작 전후에 기능을 선택하는 UI",
    "gameplay_panel": "인벤토리, 퀘스트 창, 캐릭터 상태창, 스킬창처럼 화면 위에 열리는 정보 패널",
    "gameplay_hud": "플레이 중 계속 보이는 체력바, 자원바, 미니맵, 스킬 아이콘, 전투 HUD",
    "flow_other": "게임오버, 로딩, 전투 결과, 스코어보드, 대화, 컷신처럼 흐름이 전환되는 화면",
    "map_screen": "월드맵, 지역 선택, 스테이지 선택, 던전 선택 화면",
    
    # styles
    "pixel_art": "픽셀 아트, 도트 그래픽, 레트로 아케이드 8비트 느낌의 시각 스타일",
    "modern_clean": "현대적이고 깔끔한, 심플한, 세련된 모던 스타일 UI, modern clean",
    "cartoon": "애니메이션, 카툰 렌더링, 만화 같은 일러스트 풍의 스타일",
    "skeuomorphic": "실제 사물과 같은 질감, 가죽이나 나무 느낌, 입체감이 사실적인 스큐어모픽 스타일",
    "retro": "고전적인, 낡은 오락실, 클래식한 과거 느낌의 레트로 스타일",

    # themes
    "fantasy": "검과 마법, 중세 판타지, 기사, 드래곤, 마법사가 나오는 배경",
    "sci_fi": "우주, 미래, 공상과학, SF 느낌, 사이버네틱, 메카닉, 첨단 기술, 로봇",
    "cyberpunk": "사이버펑크, 네온사인, 어두운 기계도시, 해커, 디스토피아 미래 도시",
    "military": "군대, 밀리터리, 총기, 병사, 현대전 특수부대",  # 전투(전술) 단어 제거하여 오탐 방지
    "horror": "공포, 좀비, 괴물, 어둡고 무서운 호러 스릴러 분위기",
    "casual": "캐주얼하고 가벼운, 퍼즐, 보드게임, 아기자기한 테마",
    "sports": "스포츠, 레이싱, 축구, 농구 경기 테마",
    "post_apocalyptic": "포스트 아포칼립스, 멸망한 세계, 황무지, 생존 테마"
}

def load_sbert():
    global sbert_model, sbert_tag_embeddings
    if SentenceTransformer is None:
        print("[!] sentence-transformers 라이브러리가 설치되지 않아 의미 검색을 건너뜁니다.")
        return
    if sbert_model is None:
        print("[*] SBERT 모델 로딩 중 (snunlp/KR-SBERT-V40K-klueNLI-augSTS)...")
        # CPU로 로드하여 가볍게 구동
        sbert_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS", device="cpu")
        
        primary, styles, themes, _ = get_effective_label_space()
        
        for t in primary + styles + themes:
            prompt = SBERT_PROMPTS.get(t, t.replace("_", " "))
            emb = sbert_model.encode(prompt, convert_to_tensor=True)
            sbert_tag_embeddings[t] = emb
        print("[+] SBERT 의미 매칭 모델 초기화 완료!")

load_sbert()
print("[+] 준비 완료!")


def score_keyword_hits(query: str, keywords: Sequence[str]) -> int:
    score = 0
    for keyword in keywords:
        key = normalize_query(keyword)
        if key and key in query:
            score += max(1, len(key.split()))
    return score


def pick_by_keywords(query: str, candidates: Sequence[str], keyword_map: Dict[str, List[str]], allow_multi: bool) -> List[str]:
    norm_keyword_map = {normalize_label(k): v for k, v in keyword_map.items()}
    scored: List[Tuple[int, str]] = []
    for candidate in candidates:
        norm_candidate = normalize_label(candidate)
        keywords = [candidate, prettify_label_name(candidate)] + norm_keyword_map.get(norm_candidate, [])
        score = score_keyword_hits(query, keywords)
        if score > 0:
            scored.append((score, candidate))
    scored.sort(key=lambda x: x[0], reverse=True)
    if allow_multi:
        return [item for _, item in scored]
    return [scored[0][1]] if scored else []


def infer_layout_tokens_from_query(query: str, available_tokens: Sequence[str]) -> List[str]:
    q = normalize_query(query)

    positions = pick_by_keywords(q, POSITION_KEYWORDS.keys(), POSITION_KEYWORDS, allow_multi=True)
    elements = pick_by_keywords(q, ELEMENT_KEYWORDS.keys(), ELEMENT_KEYWORDS, allow_multi=True)
    roles = pick_by_keywords(q, ROLE_KEYWORDS.keys(), ROLE_KEYWORDS, allow_multi=True)

    direct_matches: List[str] = []
    for token in available_tokens:
        normalized_token = normalize_query(prettify_label_name(token))
        token_parts = token.split(":")
        readable = " ".join(token_parts)
        if normalized_token in q or normalize_query(readable) in q:
            direct_matches.append(token)

    inferred: List[str] = []
    norm_available_map = {normalize_label(t): t for t in available_tokens}

    if positions and elements:
        for position in positions[:4]:
            for element in elements[:4]:
                matched_role = ""
                for role in roles:
                    exact = f"{position}:{element}:{role}"
                    if exact in norm_available_map:
                        matched_role = role
                        break
                if matched_role:
                    exact_key = f"{position}:{element}:{matched_role}"
                    token = norm_available_map.get(exact_key, exact_key)
                else:
                    exact_key = f"{position}:{element}"
                    token = norm_available_map.get(exact_key, exact_key)
                if token not in inferred:
                    inferred.append(token)

    for token in direct_matches:
        if token not in inferred:
            inferred.append(token)

    return inferred[:8]


def interpret_text_query(query: str) -> Dict[str, Any]:
    primary_types, v_style_tags, theme_tags, layout_tokens = get_effective_label_space()
    q = normalize_query(query)

    # 1. 1차 해석: 기존 키워드 매칭
    korean_primary = ""
    korean_styles = []
    korean_themes = []

    norm_styles = {normalize_label(s): s for s in v_style_tags}
    norm_themes = {normalize_label(t): t for t in theme_tags}

    clean_query = re.sub(r"\s+", " ", query).strip()
    for kr_word, eng_label in KOREAN_KEYWORD_MAP.items():
        kr_clean = re.sub(r"\s+", " ", kr_word).strip()
        if kr_clean in clean_query:
            norm_eng = normalize_label(eng_label)
            # Check if this maps to a primary screen type group
            is_primary_screen = (
                norm_eng in PRIMARY_SCREEN_MAPPING or
                norm_eng in ["menu_lobby", "gameplay_panel", "gameplay_hud", "flow_other", "map_screen", "dialogue_story"]
            )
            if is_primary_screen:
                korean_primary = to_primary_group(eng_label)
            elif norm_eng in norm_styles:
                style = norm_styles[norm_eng]
                if style not in korean_styles:
                    korean_styles.append(style)
            elif norm_eng in norm_themes:
                theme = norm_themes[norm_eng]
                if theme not in korean_themes:
                    korean_themes.append(theme)

    # Standard keyword pickers
    picked_primary_list = pick_by_keywords(q, primary_types, PRIMARY_SCREEN_KEYWORDS, allow_multi=False)
    picked_v_styles = pick_by_keywords(q, v_style_tags, STYLE_KEYWORDS, allow_multi=True)
    picked_themes = pick_by_keywords(q, theme_tags, STYLE_KEYWORDS, allow_multi=True)
    picked_layout_tokens = infer_layout_tokens_from_query(q, layout_tokens)

    # Blend Korean results
    final_primary = korean_primary if korean_primary else (picked_primary_list[0] if picked_primary_list else "")
    
    final_v_styles = picked_v_styles[:]
    for style in korean_styles:
        if style not in final_v_styles:
            final_v_styles.insert(0, style)
            
    final_themes = picked_themes[:]
    for theme in korean_themes:
        if theme not in final_themes:
            final_themes.insert(0, theme)

    match_source = {
        "primary_screen_type": "키워드 매칭" if final_primary else "",
        "visual_style_tags": "키워드 매칭" if final_v_styles else "",
        "theme_tags": "키워드 매칭" if final_themes else "",
        "layout_tokens": "키워드 매칭" if picked_layout_tokens else "",
    }

    # 2. 2차 해석: SBERT 의미 매칭 (빈 항목만 보완)
    if sbert_model is not None and sbert_tag_embeddings:
        try:
            from sentence_transformers import util
            query_emb = sbert_model.encode(clean_query, convert_to_tensor=True)
            
            # Primary Screen (Threshold 0.45)
            if not final_primary:
                best_score = -1.0
                best_tag = ""
                for t in primary_types:
                    if t in sbert_tag_embeddings:
                        score = util.cos_sim(query_emb, sbert_tag_embeddings[t]).item()
                        if score > best_score and score >= 0.45:
                            best_score = score
                            best_tag = t
                if best_tag:
                    final_primary = best_tag
                    match_source["primary_screen_type"] = "SBERT 의미 매칭"
            
            # Visual Styles (Threshold 0.42)
            if not final_v_styles:
                best_score = -1.0
                best_tag = ""
                for t in v_style_tags:
                    if t in sbert_tag_embeddings:
                        score = util.cos_sim(query_emb, sbert_tag_embeddings[t]).item()
                        if score > best_score and score >= 0.42:
                            best_score = score
                            best_tag = t
                if best_tag:
                    final_v_styles.append(best_tag)
                    match_source["visual_style_tags"] = "SBERT 의미 매칭"

            # Theme Tags (Threshold 0.42)
            if not final_themes:
                best_score = -1.0
                best_tag = ""
                for t in theme_tags:
                    if t in sbert_tag_embeddings:
                        score = util.cos_sim(query_emb, sbert_tag_embeddings[t]).item()
                        if score > best_score and score >= 0.42:
                            best_score = score
                            best_tag = t
                if best_tag:
                    final_themes.append(best_tag)
                    match_source["theme_tags"] = "SBERT 의미 매칭"

            # Layout Tokens SBERT is omitted as keyword is highly robust for layout
            
        except Exception as e:
            print(f"[!] SBERT 의미 매칭 중 오류 발생: {e}")

    return {
        "original_query": query,
        "primary_screen_type": final_primary,
        "visual_style_tags": final_v_styles,
        "theme_tags": final_themes,
        "layout_tokens": picked_layout_tokens,
        "layout_positions": pick_by_keywords(q, POSITION_KEYWORDS.keys(), POSITION_KEYWORDS, allow_multi=True),
        "layout_element_types": pick_by_keywords(q, ELEMENT_KEYWORDS.keys(), ELEMENT_KEYWORDS, allow_multi=True),
        "layout_roles": pick_by_keywords(q, ROLE_KEYWORDS.keys(), ROLE_KEYWORDS, allow_multi=True),
        "mode": "schema_6_label_interpretation",
        "match_source": match_source,
    }


def predict_finetuned(image: Image.Image) -> Dict[str, Any]:
    if classifier_model is None:
        return {
            "primary_screen_type": "학습 모델 없음",
            "primary_conf": 0.0,
            "primary_top3": [],
            "style_top3": [],
            "theme_top3": [],
            "visual_style_tags": [],
            "theme_tags": [],
            "layout_positions": [],
            "layout_element_types": [],
            "layout_roles": [],
            "source": "fine_tuned",
        }

    if image is None:
        return {
            "primary_screen_type": "이미지 없음",
            "primary_conf": 0.0,
            "primary_top3": [],
            "style_top3": [],
            "theme_top3": [],
            "visual_style_tags": [],
            "theme_tags": [],
            "layout_positions": [],
            "layout_element_types": [],
            "layout_roles": [],
            "source": "fine_tuned",
        }

    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            outputs = classifier_model(pixel_values)

            # 1. Primary Screen Top-3
            primary_probs = torch.softmax(outputs["logits_primary_screen_type"], dim=1)[0]
            top_k_val, top_k_idx = torch.topk(primary_probs, k=min(3, len(classifier_primary_screen_types)))
            primary_top3 = []
            for val, idx in zip(top_k_val, top_k_idx):
                prob = float(val.item())
                if prob >= 0.01:
                    primary_top3.append((classifier_primary_screen_types[int(idx.item())], prob))
            
            primary_type = primary_top3[0][0] if primary_top3 else "미해석"
            primary_conf = primary_top3[0][1] if primary_top3 else 0.0

            # Sigmoid probabilities for tags
            style_probs = torch.sigmoid(outputs["logits_visual_style_tags"])[0]
            theme_probs = torch.sigmoid(outputs["logits_theme_tags"])[0]

            # Style Top-3
            top_style_val, top_style_idx = torch.topk(style_probs, k=min(3, len(classifier_visual_style_tags)))
            style_top3 = []
            for val, idx in zip(top_style_val, top_style_idx):
                prob = float(val.item())
                if prob >= 0.01:
                    style_top3.append((classifier_visual_style_tags[int(idx.item())], prob))

            # Theme Top-3
            top_theme_val, top_theme_idx = torch.topk(theme_probs, k=min(3, len(classifier_theme_tags)))
            theme_top3 = []
            for val, idx in zip(top_theme_val, top_theme_idx):
                prob = float(val.item())
                if prob >= 0.01:
                    theme_top3.append((classifier_theme_tags[int(idx.item())], prob))

            def get_picked(probs, vocab, threshold):
                picked = []
                for i, prob in enumerate(probs):
                    if float(prob.item()) >= threshold:
                        picked.append(vocab[i])
                if not picked and len(vocab) > 0:
                    top_idx = int(torch.argmax(probs).item())
                    picked = [vocab[top_idx]]
                return picked

            picked_v_styles = get_picked(style_probs, classifier_visual_style_tags, STYLE_THRESHOLD)
            picked_themes = get_picked(theme_probs, classifier_theme_tags, STYLE_THRESHOLD)

            picked_positions = get_picked(torch.sigmoid(outputs["logits_layout_positions"])[0], classifier_layout_positions, LAYOUT_THRESHOLD)
            picked_elements = get_picked(torch.sigmoid(outputs["logits_layout_element_types"])[0], classifier_layout_element_types, LAYOUT_THRESHOLD)
            picked_roles = get_picked(torch.sigmoid(outputs["logits_layout_roles"])[0], classifier_layout_roles, LAYOUT_THRESHOLD)

        return {
            "primary_screen_type": primary_type,
            "primary_conf": primary_conf,
            "primary_top3": primary_top3,
            "style_top3": style_top3,
            "theme_top3": theme_top3,
            "visual_style_tags": picked_v_styles,
            "theme_tags": picked_themes,
            "layout_positions": picked_positions,
            "layout_element_types": picked_elements,
            "layout_roles": picked_roles,
            "source": "fine_tuned",
        }
    except Exception as e:
        print(f"[!] 학습 모델 예측 실패: {e}")
        return {
            "primary_screen_type": "예측 실패",
            "primary_conf": 0.0,
            "primary_top3": [],
            "style_top3": [],
            "theme_top3": [],
            "visual_style_tags": [],
            "theme_tags": [],
            "layout_positions": [],
            "layout_element_types": [],
            "layout_roles": [],
            "source": "fine_tuned",
        }


def predict_base_zero_shot(image: Image.Image) -> Dict[str, Any]:
    primary_types, v_style_tags, theme_tags, layout_tokens = get_effective_label_space()
    if image is None:
        return {
            "primary_screen_type": "이미지 없음",
            "primary_conf": 0.0,
            "visual_style_tags": [],
            "theme_tags": [],
            "layout_tokens": [],
            "layout_scores": [],
            "source": "base_siglip2",
        }

    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        with torch.inference_mode():
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            image_outputs = base_model.get_image_features(**image_inputs)
            image_features = getattr(image_outputs, "image_embeds", image_outputs)
            if not isinstance(image_features, torch.Tensor):
                image_features = getattr(image_outputs, "pooler_output", image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)

            primary_scores = torch.matmul(image_features, base_primary_text_embeds.T).squeeze(0)
            primary_probs = torch.softmax(primary_scores * 10.0, dim=-1)
            primary_idx = int(torch.argmax(primary_probs).item())

            v_style_scores = torch.matmul(image_features, base_visual_style_text_embeds.T).squeeze(0)
            v_style_probs = torch.softmax(v_style_scores * 10.0, dim=-1)
            v_style_idx = int(torch.argmax(v_style_probs).item())

            theme_scores = torch.matmul(image_features, base_theme_text_embeds.T).squeeze(0)
            theme_probs = torch.softmax(theme_scores * 10.0, dim=-1)
            theme_idx = int(torch.argmax(theme_probs).item())

            layout_tokens_out: List[str] = []
            layout_scores_out: List[float] = []
            if base_layout_text_embeds is not None and len(layout_tokens) > 0:
                layout_scores_raw = torch.matmul(image_features, base_layout_text_embeds.T).squeeze(0)
                layout_probs = torch.softmax(layout_scores_raw * 10.0, dim=-1)
                layout_top_k = min(BASE_LAYOUT_TOP_K, len(layout_tokens))
                layout_vals, layout_idxs = torch.topk(layout_probs, k=layout_top_k)
                layout_tokens_out = [layout_tokens[int(i)] for i in layout_idxs]
                layout_scores_out = [float(v.item()) for v in layout_vals]

        return {
            "primary_screen_type": primary_types[primary_idx],
            "primary_conf": float(primary_probs[primary_idx].item()),
            "visual_style_tags": [v_style_tags[v_style_idx]],
            "theme_tags": [theme_tags[theme_idx]],
            "layout_tokens": layout_tokens_out,
            "layout_scores": layout_scores_out,
            "source": "base_siglip2",
        }
    except Exception as e:
        print(f"[!] Base SigLIP2 zero-shot 예측 실패: {e}")
        return {
            "primary_screen_type": "예측 실패",
            "primary_conf": 0.0,
            "visual_style_tags": [],
            "theme_tags": [],
            "layout_tokens": [],
            "layout_scores": [],
            "source": "base_siglip2",
        }


def make_chip_row(items: Sequence[str], variant: str = "primary", limit: int = 8) -> str:
    values = [str(x) for x in items if str(x).strip()]
    if not values:
        return '<span class="chip chip-muted">해당 없음</span>'
    shown = values[:limit]
    chips = "".join([f'<span class="chip chip-{variant}">{escape(translate_label(item))}</span>' for item in shown])
    # 발표 화면에서는 +1, +2 같은 축약 칩이 의미가 불명확하므로 표시하지 않는다.
    return chips


def _conf_level(score: float) -> str:
    if score >= 90.0: return "매우 높음"
    if score >= 70.0: return "높음"
    if score >= 40.0: return "보통"
    return "낮음"


# ─────────────────────────────────────────────
#  새 레이아웃: 입력 요약 바
# ─────────────────────────────────────────────
def create_input_summary_bar(parsed_or_fine: Dict[str, Any], mode: str, query: str = "", count: int = 0, elapsed: float = 0.0) -> str:
    if mode == "text":
        primary_str = translate_label(parsed_or_fine.get("primary_screen_type", "")) or "해당 없음"
        styles_str  = ", ".join([translate_label(s) for s in parsed_or_fine.get("visual_style_tags", [])]) or "해당 없음"
        themes_str  = ", ".join([translate_label(t) for t in parsed_or_fine.get("theme_tags", [])]) or "해당 없음"
        layout_str  = ", ".join([translate_label(t) for t in parsed_or_fine.get("layout_tokens", [])]) or "해당 없음"
        inner = f"""
        <div class="summary-item"><span class="summary-key">입력 문장</span><span class="summary-val">"{escape(query)}"</span></div>
        <div class="summary-item"><span class="summary-key">화면 유형</span><span class="summary-val">{escape(primary_str)}</span></div>
        <div class="summary-item"><span class="summary-key">스타일</span><span class="summary-val">{escape(styles_str)}</span></div>
        <div class="summary-item"><span class="summary-key">테마</span><span class="summary-val">{escape(themes_str)}</span></div>
        <div class="summary-item"><span class="summary-key">레이아웃</span><span class="summary-val">{escape(layout_str)}</span></div>
        <div class="summary-item summary-item--right"><span class="summary-key">결과 수</span><span class="summary-val">{count}개 ({elapsed:.2f}s)</span></div>
        """
    else:
        top3 = parsed_or_fine.get("primary_top3", [])
        primary_str = f"{translate_label(top3[0][0])} ({top3[0][1]*100:.1f}%)" if top3 else "해당 없음"
        styles_str  = ", ".join([translate_label(s) for s in parsed_or_fine.get("visual_style_tags", [])]) or "해당 없음"
        themes_str  = ", ".join([translate_label(t) for t in parsed_or_fine.get("theme_tags", [])]) or "해당 없음"
        inner = f"""
        <div class="summary-item"><span class="summary-key">화면 유형 분석</span><span class="summary-val">{escape(primary_str)}</span></div>
        <div class="summary-item"><span class="summary-key">스타일</span><span class="summary-val">{escape(styles_str)}</span></div>
        <div class="summary-item"><span class="summary-key">테마</span><span class="summary-val">{escape(themes_str)}</span></div>
        <div class="summary-item summary-item--right"><span class="summary-key">결과 수</span><span class="summary-val">{count}개 ({elapsed:.2f}s)</span></div>
        """
    return f'<div class="summary-bar">{inner}</div>'


# ─────────────────────────────────────────────
#  새 레이아웃: 모델 비교 카드 (상단 두 박스)
# ─────────────────────────────────────────────
def create_model_compare_block(fine_pred: Dict[str, Any], base_pred: Dict[str, Any]) -> str:
    def _render_card(pred: Dict[str, Any], title: str, subtitle: str, accent: str, is_base: bool, status_badge: str, diff_note: str) -> str:
        raw_primary = pred.get("primary_screen_type", "")
        if is_base:
            primary_label_str = "구조화 해석 없음"
        else:
            is_primary_empty = raw_primary in ["", "미해석", "예측 실패", "이미지 없음", "해석 후보 없음"]
            is_styles_empty  = len(pred.get("visual_style_tags", [])) == 0
            is_themes_empty  = len(pred.get("theme_tags", [])) == 0
            is_layouts_empty = (
                (len(pred.get("layout_positions", [])) == 0 and
                 len(pred.get("layout_element_types", [])) == 0 and
                 len(pred.get("layout_roles", [])) == 0)
                if "layout_positions" in pred else
                len(pred.get("layout_tokens", [])) == 0
            )
            all_empty = is_primary_empty and is_styles_empty and is_themes_empty and is_layouts_empty
            if all_empty:
                primary_label_str = "해석 후보 없음"
            else:
                primary_label_str = translate_label(raw_primary) if not is_primary_empty else "해당 없음"

        conf_val = float(pred.get("primary_conf", 0.0))
        if conf_val > 0.0:
            score = conf_val * 100
            level = _conf_level(score)
            conf_str = f"{score:.1f}점 / {level}"
            conf_section = f"""
            <div class="mc-conf">
                <div class="mc-conf-label">분류 신뢰도</div>
                <div class="mc-conf-value">{escape(conf_str)}</div>
            </div>
            """
        else:
            conf_section = ""

        # Top-3
        top3_list = pred.get("primary_top3", [])
        if top3_list:
            rows = []
            for i, (name, val) in enumerate(top3_list):
                s = val * 100
                rows.append(f'<div class="mc-top3-row"><span class="mc-rank">{i+1}위</span><span>{escape(translate_label(name))}</span><span class="mc-pct">{s:.1f}%</span></div>')
            top3_html = f'<div class="mc-top3">{"".join(rows)}</div>'
        else:
            top3_html = ""

        # Layout display
        if "layout_positions" in pred:
            pos_list  = [translate_label(x) for x in pred.get("layout_positions", [])]
            elem_list = [translate_label(x) for x in pred.get("layout_element_types", [])]
            layout_section = f"""
            <div class="mc-field-label">레이아웃 구조</div>
            <div style="font-size:12px; color:var(--muted);">
                위치: {escape(", ".join(pos_list) or "해당 없음")} / 요소: {escape(", ".join(elem_list) or "해당 없음")}
            </div>
            """
        else:
            layout_section = f"""
            <div class="mc-field-label">레이아웃 토큰</div>
            <div class="chip-row">{make_chip_row(pred.get("layout_tokens", []), "secondary", limit=4)}</div>
            """

        return f"""
        <div class="mc-card mc-{accent}">
            <div class="mc-accent-bar mc-bar-{accent}"></div>
            <div class="mc-header">
                {status_badge}
                <div>
                    <div class="mc-subtitle">{escape(subtitle)}</div>
                    <div class="mc-title">{escape(title)}</div>
                </div>
            </div>
            <div class="mc-diff-note">{escape(diff_note)}</div>
            <div class="mc-primary-row">
                <div>
                    <div class="mc-field-label">대표 화면 유형</div>
                    <div class="mc-primary-val">{escape(primary_label_str)}</div>
                </div>
                {conf_section}
            </div>
            {top3_html}
            <div class="mc-field-label" style="margin-top:12px;">비주얼 스타일</div>
            <div class="chip-row">{make_chip_row(pred.get("visual_style_tags", []), accent, limit=4)}</div>
            <div class="mc-field-label" style="margin-top:10px;">테마</div>
            <div class="chip-row">{make_chip_row(pred.get("theme_tags", []), accent, limit=4)}</div>
            {layout_section}
        </div>
        """

    fine_card = _render_card(
        fine_pred, "내 학습 모델", "Fine-tuned SigLIP2", "primary", False,
        status_badge='<span class="mc-status-badge mc-status-ok">✓ 구조화 해석 성공</span>',
        diff_note="화면 유형·스타일·레이아웃을 구조화 분석하여 맞춤 보너스를 적용한 결과",
    )
    base_card = _render_card(
        base_pred, "기본 SigLIP2", "Base SigLIP2 (비교용)", "secondary", True,
        status_badge='<span class="mc-status-badge mc-status-none">✕ 구조화 해석 없음</span>',
        diff_note="Base SigLIP2는 라벨을 직접 예측하지 않고, 임베딩 유사도로 검색된 이미지의 기존 metadata를 표시합니다.",
    )
    return f'<div class="mc-grid">{fine_card}{base_card}</div>'


# ─────────────────────────────────────────────
#  새 레이아웃: 추천 이미지 카드 (3열 그리드)
# ─────────────────────────────────────────────
def _calibrated_card_score(result: Dict[str, Any], mode: str) -> float:
    """
    검색 엔진의 내부 score는 라벨 보너스가 합산되어 100점으로 포화되기 쉽다.
    발표용 카드에는 포화 점수 대신 base similarity와 매칭 정보를 섞은 보정 점수를 표시한다.
    """
    raw = float(result.get("score", 0.0))
    if raw <= 1.0:
        raw *= 100.0
    base = float(result.get("base_score", raw))
    if base <= 1.0:
        base *= 100.0

    if mode == "guided":
        bonus = 0.0
        if result.get("primary_match", False):
            bonus += 10.0
        bonus += min(6.0, 3.0 * len(result.get("matched_theme_tags", []) or []))
        bonus += min(4.0, 2.0 * len(result.get("matched_visual_style_tags", []) or []))
        bonus += min(4.0, 2.0 * len(result.get("matched_layout_tokens", []) or []))
        score = 0.82 * base + bonus
        return max(0.0, min(96.0, score))

    return max(0.0, min(100.0, raw))

def create_result_card_new(result: Dict[str, Any], mode: str, rank: int) -> str:
    image_b64 = get_base64_image(result["image_path"])
    score_val = _calibrated_card_score(result, mode)
    level = _conf_level(score_val)
    score_text = f"{score_val:.1f}점"

    primary  = result.get("primary_screen_type", "")
    v_styles = result.get("visual_style_tags", [])
    themes   = result.get("theme_tags", [])
    game_title = result.get("game_title", "Unknown")
    platform   = result.get("platform", "")
    genre      = result.get("genre", "")

    # ── 매칭 배지 (guided 모드만) ──
    match_badge = ""
    if mode == "guided":
        is_match  = result.get("primary_match", False)
        badge_cls = "match-badge-yes" if is_match else "match-badge-no"
        badge_txt = "✓ 화면 일치" if is_match else "✕ 화면 불일치"
        theme_cnt = len(result.get("matched_theme_tags", []))
        style_cnt = len(result.get("matched_visual_style_tags", []))
        match_badge = f"""
        <div class="rc-match-row">
            <span class="match-badge {badge_cls}">{badge_txt}</span>
            {f'<span class="match-badge match-badge-neutral">스타일 {style_cnt}</span>' if style_cnt > 0 else ''}
            {f'<span class="match-badge match-badge-neutral">테마 {theme_cnt}</span>' if theme_cnt > 0 else ''}
        </div>
        """

    img_html = (f'<img src="data:image/jpeg;base64,{image_b64}" '
                f'alt="{escape(game_title)}" loading="lazy">') if image_b64 else \
               '<div class="rc-no-img">이미지 없음</div>'

    # ── 정보 순서: 이미지 → 대표화면유형 → 게임명 → 점수 → 태그 → 매칭 ──
    return f"""
    <div class="rc-card">
        <div class="rc-rank">#{rank}</div>
        <div class="rc-thumb">{img_html}</div>
        <div class="rc-body">
            <div class="rc-primary">{escape(translate_label(primary) or "미지정")}</div>
            <div class="rc-game" title="{escape(game_title)}">{escape(game_title)}</div>
            <div class="rc-score">{score_text} <span class="rc-level">{level}</span></div>
            <div class="chip-row rc-tags">
                {make_chip_row(v_styles, "secondary", limit=2)}
                {make_chip_row(themes, "primary", limit=2)}
            </div>
            {match_badge}
            <div class="rc-meta">{escape(platform)} · <span class="rc-genre">{escape(genre)}</span></div>
        </div>
    </div>
    """


def create_results_section(fine_results: List[Dict[str, Any]], base_results: List[Dict[str, Any]]) -> str:
    fine_cards = "".join([create_result_card_new(r, "guided", i+1) for i, r in enumerate(fine_results)])
    if not fine_cards:
        fine_cards = '<div class="empty-box">추천 결과가 없습니다.</div>'

    base_cards = "".join([create_result_card_new(r, "base", i+1) for i, r in enumerate(base_results)])
    if not base_cards:
        base_cards = '<div class="empty-box">추천 결과가 없습니다.</div>'

    return f"""
    <div class="results-section">
        <!-- ① 내 학습 모델 결과 -->
        <div class="results-section-header">
            <span class="results-section-kicker">FINE-TUNED MODEL</span>
            <h2 class="results-section-title">내 학습 모델 추천 결과</h2>
        </div>
        <div class="rc-grid">{fine_cards}</div>

        <!-- ② Base SigLIP2 결과: 접지 않고 비교용 섹션으로 바로 표시 -->
        <div class="base-result-panel">
            <div class="base-result-header">
                <div>
                    <span class="results-section-kicker base-kicker">BASE SIGLIP2</span>
                    <h2 class="results-section-title">기본 SigLIP2 임베딩 검색 결과</h2>
                </div>
                <span class="base-summary-badge">비교용 · 구조화 해석 없음</span>
            </div>
            <p class="base-note">Base SigLIP2는 라벨을 직접 예측하지 않고, 입력 임베딩과 이미지 임베딩의 유사도만으로 검색합니다. 카드의 라벨은 검색된 이미지의 기존 metadata입니다.</p>
            <div class="rc-grid">{base_cards}</div>
        </div>
    </div>
    """


# ─────────────────────────────────────────────
#  최상위 HTML 조립 함수
# ─────────────────────────────────────────────
def create_image_compare_html(fine_pred, base_pred, fine_results, base_results, elapsed: float) -> str:
    summary_bar = create_input_summary_bar(fine_pred, "image", count=len(fine_results), elapsed=elapsed)
    compare_block = create_model_compare_block(fine_pred, base_pred)
    results_block = create_results_section(fine_results, base_results)
    return f"""
    <div class="pres-shell">
        <div class="pres-section-label">이미지 입력 기반 UI 추천 결과</div>
        {summary_bar}
        {compare_block}
        {results_block}
    </div>
    """


def create_text_compare_html(parsed, guided_results, base_results, elapsed: float) -> str:
    base_pred = {
        "primary_screen_type": "구조화 해석 없음",
        "primary_conf": 0.0,
        "visual_style_tags": [],
        "theme_tags": [],
        "layout_tokens": [],
    }
    summary_bar  = create_input_summary_bar(parsed, "text",
                                            query=parsed.get("original_query", ""),
                                            count=len(guided_results), elapsed=elapsed)
    compare_block = create_model_compare_block(parsed, base_pred)
    results_block = create_results_section(guided_results, base_results)
    
    return f"""
    <div class="pres-shell">
        <div class="pres-section-label">텍스트 입력 기반 UI 추천 결과</div>
        {summary_bar}
        {compare_block}
        {results_block}
    </div>
    """


# ─────────────────────────────────────────────
#  run_ 함수 (Gradio 이벤트 핸들러)
# ─────────────────────────────────────────────
def run_image_compare(img: Image.Image):
    count = 6
    if img is None:
        return '<div class="empty-box">이미지를 업로드해주세요.</div>'

    start = time.time()
    fine_pred = predict_finetuned(img)
    fine_results = engine.search_by_image_guided(
        query_image=img,
        primary_screen_type=fine_pred.get("primary_screen_type", ""),
        visual_style_tags=fine_pred.get("visual_style_tags", []),
        theme_tags=fine_pred.get("theme_tags", []),
        layout_tokens=[],
        layout_positions=fine_pred.get("layout_positions", []),
        layout_element_types=fine_pred.get("layout_element_types", []),
        layout_roles=fine_pred.get("layout_roles", []),
        top_k=int(count),
    )
    base_pred    = predict_base_zero_shot(img)
    base_results = engine.search_by_image(query_image=img, top_k=int(count))
    elapsed      = time.time() - start

    return create_image_compare_html(fine_pred, base_pred, fine_results, base_results, elapsed)


def run_text_compare(query: str):
    count = 6
    query = str(query).strip()
    if not query:
        return '<div class="empty-box">검색어를 입력해주세요.</div>'

    start  = time.time()
    parsed = interpret_text_query(query)
    guided_results = engine.search_by_text_guided(
        query_text=query,
        primary_screen_type=parsed.get("primary_screen_type", ""),
        visual_style_tags=parsed.get("visual_style_tags", []),
        theme_tags=parsed.get("theme_tags", []),
        layout_tokens=parsed.get("layout_tokens", []),
        layout_positions=parsed.get("layout_positions", []),
        layout_element_types=parsed.get("layout_element_types", []),
        layout_roles=parsed.get("layout_roles", []),
        top_k=int(count),
    )
    base_results = engine.search_by_text(query_text=query, top_k=int(count))
    elapsed      = time.time() - start

    return create_text_compare_html(parsed, guided_results, base_results, elapsed)


# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

:root {
    --bg:        #eef4fb;
    --panel:     #ffffff;
    --panel-2:   #f4f8ff;
    --text:      #0f172a;
    --muted:     #64748b;
    --line:      #d2ddf0;
    --blue:      #2563eb;
    --blue-2:    #1d4ed8;
    --accent:    #f97316;
    --green:     #16a34a;
    --red:       #dc2626;
    --shadow-sm: 0 4px 12px rgba(15,23,42,.04);
    --shadow:    0 12px 30px rgba(15,23,42,.08);
    --shadow-lg: 0 18px 50px rgba(15,23,42,.12);
    --r:         20px;
}

* { box-sizing: border-box; }
body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: var(--text) !important;
}
footer { display:none !important; }

/* ── Gradio 기본 배경/테두리 제거 ── */
.gradio-container label span {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    color: var(--muted) !important;
}
/* Remove input-card-wrap CSS */

/* ── App root ── */
#app-root { max-width:1440px; margin:0 auto; padding:0 12px 48px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #2563eb 100%);
    color:#fff; border-radius:28px; padding:38px 44px;
    box-shadow: var(--shadow-lg); margin-bottom:24px;
}
.hero .eyebrow { font-size:11px; font-weight:900; letter-spacing:.18em; opacity:.8; margin-bottom:10px; }
.hero h1 { font-size:34px; line-height:1.1; margin:0 0 10px; font-weight:900; }
.hero p  { margin:0; font-size:15px; opacity:.9; max-width:680px; }
.hero .hero-meta { margin-top:14px; font-size:12px; opacity:.8; }
.hero .hero-tags { display:flex; gap:10px; margin-top:14px; flex-wrap:wrap; }
.hero .hero-tag {
    background:rgba(255,255,255,.15); border:1px solid rgba(255,255,255,.25);
    border-radius:999px; padding:5px 14px; font-size:12px; font-weight:700;
}

/* ── Summary Bar ── */
.summary-bar {
    display:flex; flex-wrap:wrap; gap:8px 20px;
    background:var(--panel); border:1px solid var(--line);
    border-radius:var(--r); padding:16px 22px; margin-bottom:20px;
    box-shadow:var(--shadow-sm);
}
.summary-item { display:flex; flex-direction:column; min-width:120px; }
.summary-item--right { margin-left:auto; align-items:flex-end; }
.summary-key { font-size:10px; font-weight:900; letter-spacing:.08em; color:var(--blue); text-transform:uppercase; margin-bottom:3px; }
.summary-val { font-size:13px; font-weight:700; color:var(--text); }

/* ── Model Compare Grid ── */
.mc-grid {
    display:grid; grid-template-columns:1fr 1fr; gap:16px;
    margin-bottom:24px; align-items:stretch;
}
.mc-card {
    border-radius:var(--r); padding:22px; border:1.5px solid var(--line);
    background:var(--panel); box-shadow:var(--shadow-sm);
    position:relative; display:flex; flex-direction:column;
}
.mc-primary  { border-top:4px solid #2563eb; }
.mc-secondary { border-top:4px solid var(--accent); }
.mc-accent-bar { height:4px; border-radius:2px; margin-bottom:18px; }
.mc-bar-primary   { background:linear-gradient(90deg,#2563eb,#7c3aed); }
.mc-bar-secondary { background:linear-gradient(90deg,#f97316,#ef4444); }
.mc-header { display:flex; align-items:flex-start; gap:10px; margin-bottom:8px; flex-wrap:wrap; }
.mc-subtitle { font-size:10px; font-weight:900; letter-spacing:.12em; color:var(--blue); text-transform:uppercase; margin-bottom:2px; }
.mc-secondary .mc-subtitle { color:var(--accent); }
.mc-title { font-size:17px; font-weight:900; color:var(--text); }
/* ── Base / Forms ── */
textarea,
input[type="text"],
.gr-textbox textarea,
.gr-textbox input {
    color: #111827 !important;
    background-color: #ffffff !important;
}

textarea::placeholder,
input::placeholder {
    color: #6b7280 !important;
}

#start-btn { max-width: 400px !important; margin: 0 auto !important; display: block !important; }

/* Safe Text Colors */
.gr-textbox textarea, .gr-textbox input, .gr-textbox label span {
    color: #0f172a !important;
}
.gr-textbox textarea::placeholder, .gr-textbox input::placeholder {
    color: #64748b !important;
}

.mc-diff-note {
    font-size:12px; color:var(--muted); font-weight:600;
    background:var(--panel-2); border-radius:8px; padding:8px 12px;
    margin-bottom:14px; line-height:1.5;
}
.mc-status-badge {
    display:inline-flex; align-items:center; gap:4px;
    padding:5px 12px; border-radius:999px; font-size:11px; font-weight:900;
    white-space:nowrap;
}
.mc-status-ok   { background:#dcfce7; color:#15803d; border:1px solid #bbf7d0; }
.mc-status-none { background:#fee2e2; color:#b91c1c; border:1px solid #fecaca; }
.mc-primary-row { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px; }
.mc-field-label { font-size:10px; font-weight:900; letter-spacing:.08em; color:var(--muted); text-transform:uppercase; margin-bottom:5px; }
.mc-primary-val { font-size:20px; font-weight:900; color:var(--text); }
.mc-conf { text-align:right; }
.mc-conf-label { font-size:10px; font-weight:900; letter-spacing:.08em; color:var(--muted); text-transform:uppercase; margin-bottom:3px; }
.mc-conf-value { font-size:15px; font-weight:800; color:var(--blue); }
.mc-top3 { background:var(--panel-2); border-radius:10px; padding:10px 12px; margin-bottom:12px; }
.mc-top3-row { display:flex; align-items:center; gap:8px; font-size:12px; padding:3px 0; }
.mc-rank { background:var(--blue); color:#fff; border-radius:999px; padding:1px 7px; font-size:10px; font-weight:900; }
.mc-secondary .mc-rank { background:var(--accent); }
.mc-pct { margin-left:auto; font-weight:700; color:var(--muted); }

/* ── Results Section ── */
.results-section { margin-top:4px; }
.results-section-header { margin-bottom:14px; }
.results-section-kicker { font-size:10px; font-weight:900; letter-spacing:.14em; color:var(--blue); }
.results-section-title { font-size:22px; font-weight:900; color:var(--text); margin:4px 0 0; }

/* ── Base SigLIP2 접기 영역 ── */
.base-details {
    margin-top:28px;
    border:1px solid var(--line);
    border-radius:var(--r);
    background:var(--panel);
    box-shadow:var(--shadow-sm);
    overflow:hidden;
}
.base-summary {
    display:flex; align-items:center; gap:10px;
    padding:16px 22px; cursor:pointer;
    font-weight:800; font-size:15px; color:var(--text);
    list-style:none; user-select:none;
    background:var(--panel-2);
    border-bottom:1px solid transparent;
    transition: background .15s;
}
.base-summary:hover { background:#eef4fb; }
.base-details[open] .base-summary { border-bottom-color:var(--line); }
.base-summary-icon { font-size:18px; }
.base-summary-badge {
    margin-left:auto; background:#fee2e2; color:#b91c1c;
    border:1px solid #fecaca; border-radius:999px;
    padding:3px 10px; font-size:10px; font-weight:900;
}
.base-summary-arrow { font-size:10px; color:var(--muted); transition:transform .2s; }
.base-details[open] .base-summary-arrow { transform:rotate(180deg); }
.base-content { padding:20px 22px; }
.base-note {
    font-size:13px; color:var(--muted); font-weight:600;
    background:var(--panel-2); border-radius:8px; padding:10px 14px;
    margin-bottom:16px; line-height:1.6;
}

/* ── Result Cards Grid ── */
.rc-grid {
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:14px;
}
.rc-card {
    background:var(--panel); border:1px solid var(--line);
    border-radius:16px; overflow:hidden; box-shadow:var(--shadow-sm);
    display:flex; flex-direction:column; position:relative;
    transition:transform .15s, box-shadow .15s;
}
.rc-card:hover { transform:translateY(-3px); box-shadow:var(--shadow); }
.rc-rank {
    position:absolute; top:10px; left:10px;
    background:rgba(15,31,56,.85); color:#fff;
    border-radius:999px; padding:3px 10px; font-size:11px; font-weight:900;
    z-index:1;
}
.rc-thumb { width:100%; height:185px; background:#162033; overflow:hidden; }
.rc-thumb img { width:100%; height:100%; object-fit:contain; display:block; }
.rc-no-img { width:100%; height:100%; display:flex; align-items:center; justify-content:center; color:#8899aa; font-size:13px; }
.rc-body { padding:14px; flex:1; display:flex; flex-direction:column; gap:5px; }
/* 정보 순서: 대표화면유형 → 게임명 → 점수 → 태그 → 매칭 → 메타 */
.rc-primary { font-size:13px; font-weight:800; color:var(--text); }
.rc-game    { font-size:12px; font-weight:700; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.rc-score   { font-size:15px; font-weight:900; color:var(--blue); }
.rc-level   { font-size:11px; font-weight:700; color:var(--muted); margin-left:4px; }
.rc-tags    { margin:0; }
.rc-match-row { display:flex; gap:5px; flex-wrap:wrap; }
.rc-meta    { font-size:11px; color:#8899aa; }
.rc-genre   { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; display:inline-block; max-width:120px; vertical-align:bottom; }

/* ── Match Badges ── */
.match-badge { display:inline-block; padding:3px 9px; border-radius:999px; font-size:10px; font-weight:800; }
.match-badge-yes     { background:#dcfce7; color:var(--green); }
.match-badge-no      { background:#fee2e2; color:var(--red); }
.match-badge-neutral { background:#e0e7ff; color:var(--blue-2); }

/* ── Chips ── */
.chip-row { display:flex; flex-wrap:wrap; gap:6px; }
.chip { display:inline-flex; align-items:center; padding:4px 10px; border-radius:999px; font-size:11px; font-weight:800; border:1px solid transparent; }
.chip-primary   { background:#dbeafe; color:#1d4ed8; border-color:#bfdbfe; }
.chip-secondary { background:#ffedd5; color:#b45309; border-color:#fed7aa; }
.chip-muted     { background:#eef2f7; color:#475569; border-color:#d8e1eb; }

/* ── Misc ── */
.pres-shell { display:flex; flex-direction:column; gap:0; }
.pres-section-label { font-size:11px; font-weight:900; letter-spacing:.14em; color:var(--blue); text-transform:uppercase; margin-bottom:12px; }
.empty-box { background:var(--panel); border:1.5px dashed #b8c7da; border-radius:16px; padding:32px; text-align:center; color:var(--muted); font-size:15px; font-weight:700; }

/* ── Responsive ── */
@media (max-width:1100px) {
    .rc-grid { grid-template-columns:repeat(2,1fr); }
    .mc-grid { grid-template-columns:1fr; }
}
@media (max-width:680px) {
    .rc-grid { grid-template-columns:1fr; }
    .hero h1 { font-size:24px; }
    .hero { padding:24px 20px; }
}


/* ── Final visibility and clean-up overrides ── */
.gradio-container,
.gradio-container p,
.gradio-container label,
.gradio-container .wrap,
.gradio-container .form,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .dropdown,
.gradio-container .container,
.input-card-wrap,
.input-card-wrap * {
    color: #0f172a !important;
}

.hero, .hero *,
button, button *,
.gr-button-primary, .gr-button-primary * {
    color: #ffffff !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

.gradio-container .tab-nav button {
    color: #64748b !important;
}
.gradio-container .tab-nav button.selected {
    color: #f97316 !important;
}

.empty-box, .empty-box * {
    color: #334155 !important;
}

/* 결과 수 드롭다운/입력 컨트롤이 흐리게 보이는 문제 방지 */
.input-card-wrap [data-testid],
.input-card-wrap [class*="dropdown"],
.input-card-wrap [class*="input"],
.input-card-wrap [class*="container"] {
    color: #0f172a !important;
    opacity: 1 !important;
}

/* 이미지 입력도 텍스트 입력처럼 상단 카드 안에 배치 */
.image-input-card {
    margin-bottom: 18px !important;
}
.image-input-card button {
    min-height: 46px !important;
    border-radius: 14px !important;
    font-weight: 900 !important;
}

/* Base 결과를 접지 않고 하나의 비교 섹션으로 표시 */
.base-result-panel {
    margin-top: 34px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--r);
    box-shadow: var(--shadow-sm);
    padding: 22px;
}
.base-result-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 14px;
    margin-bottom: 14px;
}
.base-kicker { color: var(--accent) !important; }

/* +1, +2 제거 후 칩 가독성 보강 */
.chip-muted {
    background: #eef2f7 !important;
    color: #334155 !important;
}

/* 이미지 카드의 흐린 메타 정보도 검정 계열로 보정 */
.rc-meta, .rc-meta *, .rc-genre {
    color: #64748b !important;
}

/* 모델 비교 카드 세로 과확대 방지 */
.mc-card {
    min-height: 0 !important;
}
.mc-primary-val {
    word-break: keep-all;
    line-height: 1.25;
}


/* ── Critical final fixes: text visibility, intro, tabs, radio, image input ── */
.intro-page, .intro-page * {
    color: #0f172a !important;
    opacity: 1 !important;
}
.intro-page h1, .intro-page h2, .intro-page h3, .intro-page strong {
    color: #0f172a !important;
}
.intro-page [style*="color:#2563eb"],
.intro-page [style*="color: #2563eb"] {
    color: #2563eb !important;
}
.intro-page [style*="color:#64748b"],
.intro-page [style*="color: #64748b"] {
    color: #475569 !important;
}

/* 이미지 에디터 툴바 제거 (취소, 되돌리기 등) */
.gradio-container [data-testid="image"] button[aria-label="Clear"],
.gradio-container [data-testid="image"] button[aria-label="Remove Image"],
.gradio-container [data-testid="image"] .icon-button,
.gradio-container [data-testid="image"] [aria-label*="edit"],
.gradio-container [data-testid="image"] [aria-label*="crop"],
.gradio-container [data-testid="image"] .actions {
    display: none !important;
}


.fixed-count-note {
    background:#f8fbff;
    border:1px solid #d8e4f2;
    border-radius:14px;
    padding:14px 16px;
    color:#0f172a !important;
    font-weight:700;
    text-align:center;
}
.fixed-count-note * { color:#0f172a !important; }

/* 이미지 업로드 컴포넌트가 사이드 패널처럼 보이지 않도록 보정 */
.image-input-card {
    padding: 24px !important;
}
.image-input-card .wrap,
.image-input-card .block,
.image-input-card .form {
    background: transparent !important;
}
.image-input-card [data-testid*="image"] {
    background:#f8fbff !important;
    border:1.5px dashed #cbd8ea !important;
    border-radius:16px !important;
    overflow:hidden !important;
}
.image-input-card button {
    color:#ffffff !important;
}

/* 카드 텍스트 최종 가시성 */
.mc-card, .mc-card *, .summary-bar, .summary-bar *,
.rc-card, .rc-card *, .base-result-panel, .base-result-panel * {
    opacity:1 !important;
}
.rc-primary, .rc-game, .rc-meta, .rc-genre, .mc-primary-val, .summary-val {
    color:#0f172a !important;
}
.rc-game, .rc-meta, .rc-genre {
    color:#475569 !important;
}

/* 버튼만 흰 글씨 */
.gradio-container button:not([role="tab"]),
.gradio-container button:not([role="tab"]) * {
    color: #ffffff !important;
}


"""

# ─────────────────────────────────────────────
#  Hero HTML
# ─────────────────────────────────────────────
hero_html = f"""
<div class="hero">
    <div class="eyebrow">AI-BASED GAME UI DISCOVERY STUDIO</div>
    <h1>다중 작업 학습 모델 기반 게임 UI 지능형 추천</h1>
    <p>사용자 입력을 기반으로 화면 유형, 스타일, 테마 및 세부 레이아웃 구조를 분석하여 유사한 게임 UI 레퍼런스를 추천합니다.</p>
    <div class="hero-tags">
        <span class="hero-tag">🎓 Fine-tuned SigLIP2</span>
        <span class="hero-tag">📐 다중 작업 학습</span>
        <span class="hero-tag">🔍 텍스트 / 이미지 검색</span>
        <span class="hero-tag">⚖️ Base SigLIP2 비교</span>
    </div>
    <div class="hero-meta">서빙 모델: <code>{escape(active_checkpoint_path or '감지 안 됨')}</code></div>
</div>
"""

# ─────────────────────────────────────────────
#  Intro Page HTML
# ─────────────────────────────────────────────
intro_html = f"""
<div class="intro-page" style="max-width:1100px; margin:36px auto; color:#0f172a;">
    <!-- Hero Card -->
    <div style="background:#ffffff; border-radius:20px; box-shadow:0 12px 30px rgba(15,23,42,0.08); padding:50px 40px; text-align:center; margin-bottom:30px;">
        <h1 style="font-size:32px; font-weight:900; margin-bottom:16px; color:#0f172a;">다중 작업 학습 모델 기반 <span style="color:#2563eb;">게임 UI 지능형 추천</span></h1>
        <p style="font-size:16px; color:#64748b; margin-bottom:28px;">
            게임 UI를 화면 유형, 스타일, 테마, 레이아웃 구조로 분석하여 사용자 입력과 유사한 UI 레퍼런스를 추천합니다.
        </p>
        <div style="display:flex; justify-content:center; gap:8px; flex-wrap:wrap; margin-bottom:32px;">
            <span style="background:#eef4fb; color:#2563eb; padding:6px 14px; border-radius:999px; font-size:13px; font-weight:700;">Fine-tuned SigLIP2</span>
            <span style="background:#eef4fb; color:#2563eb; padding:6px 14px; border-radius:999px; font-size:13px; font-weight:700;">다중 작업 학습</span>
            <span style="background:#eef4fb; color:#2563eb; padding:6px 14px; border-radius:999px; font-size:13px; font-weight:700;">텍스트·이미지 검색</span>
            <span style="background:#eef4fb; color:#2563eb; padding:6px 14px; border-radius:999px; font-size:13px; font-weight:700;">Base SigLIP2 비교</span>
        </div>
    </div>

    <!-- 핵심 기능 4개 카드 -->
    <h3 style="font-size:18px; font-weight:800; margin-bottom:16px; margin-top:40px; padding-left:8px;">✨ 핵심 기능</h3>
    <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:16px; margin-bottom:40px;">
        <div style="background:#ffffff; border-radius:20px; box-shadow:0 12px 30px rgba(15,23,42,0.08); padding:24px;">
            <div style="font-size:24px; margin-bottom:12px;">📝</div>
            <div style="font-size:15px; font-weight:800; margin-bottom:8px;">텍스트 기반 검색</div>
            <div style="font-size:13px; color:#64748b; line-height:1.5;">한국어 자연어 입력을 UI 라벨 구조로 변환</div>
        </div>
        <div style="background:#ffffff; border-radius:20px; box-shadow:0 12px 30px rgba(15,23,42,0.08); padding:24px;">
            <div style="font-size:24px; margin-bottom:12px;">🖼️</div>
            <div style="font-size:15px; font-weight:800; margin-bottom:8px;">이미지 기반 검색</div>
            <div style="font-size:13px; color:#64748b; line-height:1.5;">업로드한 게임 UI 이미지를 학습 모델로 분석</div>
        </div>
        <div style="background:#ffffff; border-radius:20px; box-shadow:0 12px 30px rgba(15,23,42,0.08); padding:24px;">
            <div style="font-size:24px; margin-bottom:12px;">📐</div>
            <div style="font-size:15px; font-weight:800; margin-bottom:8px;">다중 작업 분류</div>
            <div style="font-size:13px; color:#64748b; line-height:1.5;">화면 유형, 스타일, 테마, 레이아웃을 동시에 예측</div>
        </div>
        <div style="background:#ffffff; border-radius:20px; box-shadow:0 12px 30px rgba(15,23,42,0.08); padding:24px;">
            <div style="font-size:24px; margin-bottom:12px;">⚖️</div>
            <div style="font-size:15px; font-weight:800; margin-bottom:8px;">Base SigLIP2 비교</div>
            <div style="font-size:13px; color:#64748b; line-height:1.5;">기본 임베딩 검색 방식과 학습 모델 결과 비교</div>
        </div>
    </div>

    <!-- 분석 범주 4개 카드 -->
    <h3 style="font-size:18px; font-weight:800; margin-bottom:16px; margin-top:40px; padding-left:8px;">📊 분석 범주</h3>
    <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:16px; margin-bottom:40px;">
        <div style="background:#ffffff; border-radius:20px; box-shadow:0 12px 30px rgba(15,23,42,0.08); padding:24px;">
            <div style="font-size:14px; font-weight:800; margin-bottom:12px; color:#2563eb;">대표 화면 유형</div>
            <div style="font-size:12px; color:#64748b; line-height:1.6;">
                menu_lobby<br>gameplay_panel<br>gameplay_hud<br>flow_other<br>map_screen
            </div>
        </div>
        <div style="background:#ffffff; border-radius:20px; box-shadow:0 12px 30px rgba(15,23,42,0.08); padding:24px;">
            <div style="font-size:14px; font-weight:800; margin-bottom:12px; color:#2563eb;">비주얼 스타일</div>
            <div style="font-size:12px; color:#64748b; line-height:1.6;">
                pixel art, modern clean, stylized cartoon, retro, minimal, realistic, skeuomorphic
            </div>
        </div>
        <div style="background:#ffffff; border-radius:20px; box-shadow:0 12px 30px rgba(15,23,42,0.08); padding:24px;">
            <div style="font-size:14px; font-weight:800; margin-bottom:12px; color:#2563eb;">테마</div>
            <div style="font-size:12px; color:#64748b; line-height:1.6;">
                fantasy, sci fi, modern world, historical, cyberpunk
            </div>
        </div>
        <div style="background:#ffffff; border-radius:20px; box-shadow:0 12px 30px rgba(15,23,42,0.08); padding:24px;">
            <div style="font-size:14px; font-weight:800; margin-bottom:12px; color:#2563eb;">레이아웃 구조</div>
            <div style="font-size:12px; color:#64748b; line-height:1.6;">
                위치: top, bottom, left...<br>
                요소: bar, panel, popup...<br>
                역할: resource, quest...
            </div>
        </div>
    </div>

    <!-- 성능 요약 4개 카드 -->
    <h3 style="font-size:18px; font-weight:800; margin-bottom:16px; margin-top:40px; padding-left:8px;">🚀 성능 요약</h3>
    <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:16px; margin-bottom:40px;">
        <div style="background:#ffffff; border-radius:20px; box-shadow:0 12px 30px rgba(15,23,42,0.08); padding:24px; text-align:center;">
            <div style="font-size:28px; font-weight:900; color:#2563eb; margin-bottom:4px;">93.08%</div>
            <div style="font-size:12px; color:#64748b; font-weight:700;">Primary Screen</div>
        </div>
        <div style="background:#ffffff; border-radius:20px; box-shadow:0 12px 30px rgba(15,23,42,0.08); padding:24px; text-align:center;">
            <div style="font-size:28px; font-weight:900; color:#2563eb; margin-bottom:4px;">96.27%</div>
            <div style="font-size:12px; color:#64748b; font-weight:700;">Style Tags</div>
        </div>
        <div style="background:#ffffff; border-radius:20px; box-shadow:0 12px 30px rgba(15,23,42,0.08); padding:24px; text-align:center;">
            <div style="font-size:28px; font-weight:900; color:#2563eb; margin-bottom:4px;">85.59%</div>
            <div style="font-size:12px; color:#64748b; font-weight:700;">Theme Tags</div>
        </div>
        <div style="background:#ffffff; border-radius:20px; box-shadow:0 12px 30px rgba(15,23,42,0.08); padding:24px; text-align:center;">
            <div style="font-size:28px; font-weight:900; color:#2563eb; margin-bottom:4px;">90.58%</div>
            <div style="font-size:12px; color:#64748b; font-weight:700;">Layout Avg</div>
        </div>
    </div>
</div>
"""

# ─────────────────────────────────────────────
#  Gradio Layout
# ─────────────────────────────────────────────
with gr.Blocks(title="Game UI Discovery Studio") as demo:
    
    # --- Intro Page ---
    with gr.Column(visible=True) as intro_page:
        gr.HTML(intro_html)
        with gr.Row():
            start_btn = gr.Button("🚀 UI 추천 시작하기", variant="primary", size="lg", elem_id="start-btn")
            
    # --- Main App Page ---
    with gr.Column(visible=False, elem_id="app-root") as main_page:
        gr.HTML(hero_html)

        with gr.Tabs():
            # ── 텍스트 검색 탭 ──
            with gr.Tab("📝 텍스트 검색"):
                with gr.Group():
                    gr.HTML('''
                    <div style="margin-bottom:12px; font-size:13px; color:#64748b; padding:12px;">
                        원하는 게임 UI의 특징을 자연어나 키워드로 입력해주세요. 아래 대표 화면 유형을 참고하세요.<br><br>
                        • <b style="color:#0f172a;">menu_lobby</b>: 메인 메뉴, 타이틀 화면, 게임 로비, 대기실, 상점, 설정창<br>
                        • <b style="color:#0f172a;">gameplay_panel</b>: 인벤토리(가방), 퀘스트/임무 창, 캐릭터 상세 정보, 스킬 트리<br>
                        • <b style="color:#0f172a;">gameplay_hud</b>: 전투 중 기본 화면, 체력바/자원바, 미니맵, 스킬 아이콘<br>
                        • <b style="color:#0f172a;">flow_other</b>: 전투 결과(승리/패배), 게임오버, 로딩 화면, 스토리 대화창<br>
                        • <b style="color:#0f172a;">map_screen</b>: 월드맵 전체 화면, 스테이지/지역 선택
                    </div>
                    ''')
                    with gr.Row():
                        txt_input = gr.Textbox(
                            label="원하는 게임 UI 특징 입력",
                            placeholder="예: 판타지 풍의 타이틀 화면, 중세 전투 HUD ...",
                            lines=1,
                            scale=6,
                        )
                        txt_btn = gr.Button("🔍  검색", variant="primary", scale=1)
                txt_output_html = gr.HTML(
                    value='<div class="empty-box" style="margin-top:16px;">검색어를 입력하고 검색 버튼을 누르면 추천 결과가 표시됩니다.</div>'
                )

            # ── 이미지 검색 탭 ──
            with gr.Tab("🖼️ 이미지 검색"):
                with gr.Group():
                    with gr.Row(equal_height=True):
                        img_input = gr.Image(
                            type="pil",
                            label="게임 UI 이미지 업로드",
                            show_label=False,
                            height=240,
                            scale=5,
                        )
                        with gr.Column(scale=2):
                            img_btn = gr.Button("🔍  검색", variant="primary")
                img_output_html = gr.HTML(
                    value='<div class="empty-box" style="margin-top:16px;">이미지를 업로드하고 검색 버튼을 누르면 추천 결과가 표시됩니다.</div>'
                )

        txt_btn.click(run_text_compare,  inputs=[txt_input],  outputs=[txt_output_html])
        img_btn.click(run_image_compare, inputs=[img_input],  outputs=[img_output_html])

    def show_main_page():
        return gr.update(visible=False), gr.update(visible=True)
        
    start_btn.click(show_main_page, inputs=None, outputs=[intro_page, main_page])

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=custom_css)
