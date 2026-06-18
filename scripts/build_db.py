# -*- coding: utf-8 -*-
"""
=======================================================================
Game UI vector database builder

New schema only:
- primary_screen_type
- secondary_screen_types
- style_tags
- layout_blocks
- layout_tokens
- components
- is_game_ui
- ui_quality
- review_status
=======================================================================
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from layout_vocab import normalize_layout_value

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL_NAME = "google/siglip2-base-patch16-224"
IMAGE_DIR = "data/images"
METADATA_FILE = "data/metadata.csv"
EMBEDDING_CACHE = "data/embeddings.pt"
BATCH_SIZE = 8
INCLUDE_WEAK_UI = False

REQUIRED_COLUMNS = [
    "file_name",
    "game_title",
    "is_game_ui",
    "ui_quality",
    "primary_screen_type",
    "secondary_screen_types",
    "visual_style_tags",
    "theme_tags",
    "layout_blocks",
    "layout_tokens",
    "components",
    "review_status",
]


def safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    return text if text else default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = safe_text(value).lower()
    return text in {"true", "1", "yes", "y", "keep"}


def parse_float(value: Any, default: float = 0.0) -> float:
    text = safe_text(value)
    if not text:
        return default
    try:
        return float(text)
    except Exception:
        return default


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
    blocks: List[Dict[str, str]] = []
    for item in parse_json_list(value):
        if not isinstance(item, dict):
            continue
        position = safe_text(item.get("position"))
        element_type = safe_text(item.get("element_type"))
        role = safe_text(item.get("role"))
        if not position or not element_type:
            continue
        block = {
            "position": position,
            "element_type": element_type,
            "role": role,
        }
        if block not in blocks:
            blocks.append(block)
    return blocks


def make_layout_token(block: Dict[str, str]) -> str:
    position = safe_text(block.get("position"))
    element_type = safe_text(block.get("element_type"))
    role = safe_text(block.get("role"), "general")
    if not position or not element_type:
        return ""
    return f"{position}:{element_type}:{role}"


def build_layout_tokens(layout_blocks: List[Dict[str, str]], raw_tokens: Any) -> List[str]:
    tokens = parse_string_list(raw_tokens)
    for block in layout_blocks:
        token = make_layout_token(block)
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def pick_platform(row: pd.Series) -> str:
    for key in ["platform_name", "platform", "platforms"]:
        value = safe_text(row.get(key))
        if value:
            return value
    return "Unknown"


def pick_genre(row: pd.Series) -> str:
    value = safe_text(row.get("genre"))
    return value if value else "Unknown"


def is_usable_row(row: pd.Series) -> bool:
    if not parse_bool(row.get("is_game_ui")):
        return False

    if safe_text(row.get("review_status")) != "labeled":
        return False

    quality = safe_text(row.get("ui_quality"))
    allowed_quality = {"keep", "weak"} if INCLUDE_WEAK_UI else {"keep"}
    if quality not in allowed_quality:
        return False

    if not safe_text(row.get("primary_screen_type")):
        return False

    return True


def build_display_label(
    primary_screen_type: str,
    visual_style_tags: List[str],
    layout_tokens: List[str],
) -> str:
    parts = []
    if primary_screen_type:
        parts.append(f"[{primary_screen_type}]")
    if visual_style_tags:
        parts.append("styles: " + ", ".join(visual_style_tags[:4]))
    if layout_tokens:
        parts.append("layout: " + ", ".join(layout_tokens[:4]))
    return " / ".join(parts) if parts else "unlabeled"


def load_model():
    print(f"[*] Loading model: {MODEL_NAME}")
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"[+] Model loaded. ({time.time() - start:.1f}s)")
    return processor, model, device


def extract_features(processor, model, device, batch_images: List[Image.Image]) -> torch.Tensor:
    inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
    outputs = model.get_image_features(**inputs)
    features = getattr(outputs, "image_embeds", outputs)
    if not isinstance(features, torch.Tensor):
        features = getattr(outputs, "pooler_output", features)
    return F.normalize(features, p=2, dim=-1)


def validate_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "metadata.csv does not match the new schema. Missing columns: "
            + ", ".join(missing)
        )


def append_meta(cache: Dict[str, List[Any]], row: pd.Series, file_name: str) -> None:
    primary = safe_text(row.get("primary_screen_type"))
    secondary = parse_string_list(row.get("secondary_screen_types"))
    styles = parse_string_list(row.get("visual_style_tags"))
    themes = parse_string_list(row.get("theme_tags"))
    layout_blocks = parse_layout_blocks(row.get("layout_blocks"))
    layout_tokens = build_layout_tokens(layout_blocks, row.get("layout_tokens"))
    components = parse_string_list(row.get("components"))
    confidence = parse_float(row.get("confidence"), 0.0)
    needs_review = parse_bool(row.get("needs_review"))

    cache["file_names"].append(file_name)
    cache["game_titles"].append(safe_text(row.get("game_title"), "Unknown"))
    cache["genres"].append(pick_genre(row))
    cache["platforms"].append(pick_platform(row))
    cache["source_apis"].append(safe_text(row.get("source_api"), "Unknown"))
    cache["moby_urls"].append(safe_text(row.get("moby_url")))
    cache["screenshot_urls"].append(safe_text(row.get("screenshot_url")))
    cache["screenshot_captions"].append(safe_text(row.get("screenshot_caption")))

    cache["is_game_ui"].append(parse_bool(row.get("is_game_ui")))
    cache["ui_quality"].append(safe_text(row.get("ui_quality")))
    cache["primary_screen_types"].append(primary)
    cache["secondary_screen_types"].append(secondary)
    cache["visual_style_tags"].append(styles)
    cache["style_tags"].append(styles) # Compatibility
    cache["theme_tags"].append(themes)
    cache["layout_blocks"].append(layout_blocks)
    cache["layout_tokens"].append(layout_tokens)
    
    # Store split layout axes for search guidances
    positions = [normalize_layout_value(b.get("position"), "position") for b in layout_blocks if b.get("position")]
    elements = [normalize_layout_value(b.get("element_type"), "element_type") for b in layout_blocks if b.get("element_type")]
    roles = [normalize_layout_value(b.get("role"), "role") for b in layout_blocks if b.get("role")]
    cache["layout_positions"].append(sorted(set(positions)))
    cache["layout_element_types"].append(sorted(set(elements)))
    cache["layout_roles"].append(sorted(set(roles)))

    cache["components"].append(components)
    cache["confidence"].append(confidence)
    cache["needs_review"].append(needs_review)
    cache["review_status"].append(safe_text(row.get("review_status")))
    cache["notes"].append(safe_text(row.get("notes")))
    cache["display_labels"].append(build_display_label(primary, styles, layout_tokens))


def build_embeddings(processor, model, device) -> None:
    if not os.path.exists(METADATA_FILE):
        print(f"[!] {METADATA_FILE} not found.")
        return

    df = pd.read_csv(METADATA_FILE)
    validate_schema(df)
    print(f"[*] Metadata loaded: {len(df)} rows")

    valid_rows = []
    for _, row in df.iterrows():
        file_name = safe_text(row.get("file_name"))
        if not file_name:
            continue
            
        if file_name.startswith("http://") or file_name.startswith("https://"):
            img_path = file_name
        else:
            img_path = os.path.join(IMAGE_DIR, file_name)
            if not os.path.exists(img_path):
                continue
                
        if not is_usable_row(row):
            continue
        valid_rows.append(row)

    print(f"[*] Usable labeled UI images: {len(valid_rows)}")
    if not valid_rows:
        print("[!] No usable labeled UI images found.")
        return

    embeddings: List[torch.Tensor] = []
    cache: Dict[str, List[Any]] = {
        "file_names": [],
        "game_titles": [],
        "genres": [],
        "platforms": [],
        "source_apis": [],
        "moby_urls": [],
        "screenshot_urls": [],
        "screenshot_captions": [],
        "is_game_ui": [],
        "ui_quality": [],
        "primary_screen_types": [],
        "secondary_screen_types": [],
        "visual_style_tags": [],
        "style_tags": [],
        "theme_tags": [],
        "layout_blocks": [],
        "layout_tokens": [],
        "layout_positions": [],
        "layout_element_types": [],
        "layout_roles": [],
        "components": [],
        "confidence": [],
        "needs_review": [],
        "review_status": [],
        "notes": [],
        "display_labels": [],
    }

    print(f"[*] Extracting image embeddings. Batch size: {BATCH_SIZE}")

    with torch.inference_mode():
        batch_images: List[Image.Image] = []
        batch_meta: List[pd.Series] = []

        for row in tqdm(valid_rows, desc="Embedding"):
            file_name = safe_text(row.get("file_name"))
            img_path = os.path.join(IMAGE_DIR, file_name)
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[!] Failed to open image: {file_name} ({e})")
                continue

            batch_images.append(image)
            batch_meta.append(row)

            if len(batch_images) < BATCH_SIZE:
                continue

            features = extract_features(processor, model, device, batch_images)
            for i, meta in enumerate(batch_meta):
                embeddings.append(features[i : i + 1].cpu())
                append_meta(cache, meta, safe_text(meta.get("file_name")))

            batch_images.clear()
            batch_meta.clear()

        if batch_images:
            features = extract_features(processor, model, device, batch_images)
            for i, meta in enumerate(batch_meta):
                embeddings.append(features[i : i + 1].cpu())
                append_meta(cache, meta, safe_text(meta.get("file_name")))

    if not embeddings:
        print("[!] No embeddings were created.")
        return

    cache_data: Dict[str, Any] = {
        "schema_version": "6.0",
        "model_name": MODEL_NAME,
        "embeddings": torch.cat(embeddings, dim=0),
        **cache,
    }

    os.makedirs(os.path.dirname(EMBEDDING_CACHE), exist_ok=True)
    torch.save(cache_data, EMBEDDING_CACHE)

    print(f"\n[+] Vector DB build complete: {len(cache['file_names'])} images")
    print(f"[*] Saved to: {EMBEDDING_CACHE}")


def main() -> None:
    print("=" * 60)
    print("  Game UI Vector Database Builder")
    print("=" * 60)
    processor, model, device = load_model()
    build_embeddings(processor, model, device)


if __name__ == "__main__":
    main()
