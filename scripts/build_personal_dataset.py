# -*- coding: utf-8 -*-
"""
Personal screenshot dataset builder.

Reads images from '개인 리소스' folder and adds them to data/metadata.csv 
with the new Game UI schema.
"""

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Set, Tuple

import sys

# configuration
if len(sys.argv) > 1:
    PERSONAL_RESOURCE_DIR = sys.argv[1]
else:
    PERSONAL_RESOURCE_DIR = os.path.join("개인 리소스", "UnSelected")
IMAGE_DIR = "data/images"
METADATA_FILE = "data/metadata.csv"
SCHEMA_VERSION = "6.0"

FIELDNAMES = [
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
    "release_year",
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

def json_array(items=None) -> str:
    return json.dumps(items or [], ensure_ascii=False)

def get_image_hash(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def normalize_existing_row(row: Dict[str, str]) -> Dict[str, str]:
    normalized = {key: str(row.get(key, "") or "") for key in FIELDNAMES}
    return normalized

def load_existing_metadata() -> Tuple[List[Dict[str, str]], Set[str]]:
    rows: List[Dict[str, str]] = []
    existing_hashes: Set[str] = set()

    if not os.path.exists(METADATA_FILE):
        return rows, existing_hashes

    with open(METADATA_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = normalize_existing_row(raw_row)
            rows.append(row)
            image_hash = row.get("image_hash", "")
            if image_hash:
                existing_hashes.add(image_hash)

    return rows, existing_hashes

def make_new_row(
    *,
    game_title: str,
    screenshot_caption: str,
    file_name: str,
    image_hash: str,
    original_path: str,
) -> Dict[str, str]:
    return {
        "source_api": "Personal",
        "moby_game_id": "",
        "game_title": str(game_title),
        "moby_url": "",
        "platform_id": "",
        "platform_name": "PC",
        "screenshot_caption": str(screenshot_caption),
        "screenshot_url": "",
        "file_name": str(file_name),
        "image_hash": str(image_hash),
        "genre": "Action/RPG", # 기본값
        "release_year": "",
        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "is_game_ui": "",
        "ui_quality": "",
        "primary_screen_type": "",
        "secondary_screen_types": json_array(),
        "visual_style_tags": json_array(),
        "theme_tags": json_array(),
        "layout_blocks": json_array(),
        "layout_tokens": json_array(),
        "components": json_array(),
        "confidence": "",
        "needs_review": "true",
        "review_status": "unlabeled",
        "ui_score": "10",
        "ui_score_reason": "+personal",
        "source_target_screen_type": game_title.lower(),
        "notes": f"original_path: {original_path}",
    }

def save_metadata(rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def main():
    print("=" * 60)
    print("Personal Resource Dataset Builder")
    print("=" * 60)

    if not os.path.exists(PERSONAL_RESOURCE_DIR):
        print(f"[!] Directory not found: {PERSONAL_RESOURCE_DIR}")
        return

    os.makedirs(IMAGE_DIR, exist_ok=True)
    all_rows, existing_hashes = load_existing_metadata()
    initial_count = len(all_rows)
    added_count = 0
    skipped_count = 0

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for game_folder in os.listdir(PERSONAL_RESOURCE_DIR):
        folder_path = os.path.join(PERSONAL_RESOURCE_DIR, game_folder)
        if not os.path.isdir(folder_path):
            continue
            
        print(f"[INFO] Processing Game: {game_folder}")
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in valid_extensions:
                    continue
    
                filepath = os.path.join(root, file)
                image_hash = get_image_hash(filepath)
                
                if image_hash in existing_hashes:
                    skipped_count += 1
                    continue
                    
                clean_game_title = "".join(c if c.isalnum() else "_" for c in game_folder).strip("_")
                new_file_name = f"personal_{clean_game_title}_{image_hash[:8]}{ext}"
                new_filepath = os.path.join(IMAGE_DIR, new_file_name)
                
                # Copy file
                shutil.copy2(filepath, new_filepath)
                
                # Make metadata row
                row = make_new_row(
                    game_title=game_folder,
                    screenshot_caption=file,
                    file_name=new_file_name,
                    image_hash=image_hash,
                    original_path=filepath
                )
                
                all_rows.append(row)
                existing_hashes.add(image_hash)
                added_count += 1

    if added_count > 0:
        save_metadata(all_rows)

    print("=" * 60)
    print(f"Finished processing personal resources.")
    print(f"Total existing rows before: {initial_count}")
    print(f"New images added: {added_count}")
    print(f"Skipped duplicates: {skipped_count}")
    print(f"Total rows now: {len(all_rows)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
