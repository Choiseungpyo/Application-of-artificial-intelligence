# -*- coding: utf-8 -*-
"""
Upgraded Targeted MobyGames screenshot dataset builder.
Focuses on collecting images for categories that are specifically lacking in 'labeled' status.
Goal: Reach at least 30 human-labeled samples per class.
"""

import csv
import hashlib
import json
import os
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_KEY = os.getenv("MOBYGAMES_API_KEY")
BASE_URL = "https://api.mobygames.com/v1"
IMAGE_DIR = "data/images"
METADATA_FILE = "data/metadata.csv"
BALANCE_REPORT = "class_balance_report_after_labeling.csv"
TARGET_REPORT = "targeted_collection_report.csv"
SCHEMA_VERSION = "6.0"

# --- Targeted Collection Settings ---
MIN_LABELED_GOAL = 30
COLLECTION_MULTIPLIER = 1.5
MAX_COLLECT_PER_CLASS = 50
MIN_COLLECT_PER_CLASS = 10
REQUEST_DELAY_SECONDS = 1.6
REQUEST_TIMEOUT = 30

# --- Expanded Keywords (Requested by USER) ---
TARGET_CLASSES_KEYWORDS = {
    "lobby": [
        "game lobby screen", "multiplayer lobby UI", "game room lobby", "character lobby screen"
    ],
    "equipment": [
        "game equipment screen", "RPG equipment UI", "inventory equipment screen", "gear menu UI"
    ],
    "skill_tree": [
        "game skill tree UI", "RPG skill tree screen", "talent tree UI", "ability tree screen"
    ],
    "quest": [
        "game quest log UI", "quest menu screen", "mission log UI", "journal quest screen"
    ],
    "shop": [
        "game shop UI", "item shop screen", "merchant menu UI", "in game store screen"
    ],
    "crafting": [
        "game crafting UI", "crafting menu screen", "item crafting interface", "recipe crafting screen"
    ],
    "pause_menu": [
        "game pause menu UI", "pause screen", "settings pause menu", "game menu overlay"
    ],
    "loading_screen": [
        "game loading screen UI", "loading screen tips", "loading screen game screenshot"
    ],
    "other": [
        "non ui gameplay screenshot", "game screenshot without UI", "cutscene screenshot"
    ],
    "battle_result": [
        "result", "results", "victory", "defeat", "mission complete", 
        "stage clear", "score", "reward", "rewards", "summary"
    ]
}

# Classes to avoid/de-prioritize
OVER_REPRESENTED_CLASSES = ["main_menu", "title_screen", "character_screen", "gameplay_hud", "dialogue"]

# --- Category to Genre Mapping (Requested by USER) ---
CAT_GENRES = {
    "skill_tree": [122, 2], # RPG, Strategy
    "crafting": [122, 126], # RPG, Simulation
    "lobby": [1],           # Action (Multiplayer)
    "equipment": [122],      # RPG
    "pause_menu": [1, 122],
    "shop": [122, 126],
    "quest": [122, 21],
    "loading_screen": [122, 1],
    "battle_result": [1, 122],
}

GENRE_MAP = {
    "RPG": 122,
    "Strategy": 2,
    "Simulation": 126,
    "Adventure": 21,
    "Action": 1,
}

def calculate_targeted_score(caption: str, title: str, genres: List[str], target_cat: str) -> Tuple[int, List[str]]:
    text = (caption + " " + title + " " + " ".join(genres)).lower()
    score = 0
    matched = []

    # Check against the specific target category keywords
    keywords = TARGET_CLASSES_KEYWORDS.get(target_cat, [])
    for kw in keywords:
        if kw.lower() in text:
            score += 10
            matched.append(f"match:{kw}")

    # Layout elements (general UI indicators)
    layout_elements = ["minimap", "progress bar", "chat", "tooltip", "popup", "skill bar", "health bar", "resource bar", "slot"]
    for el in layout_elements:
        if el in text:
            score += 2
            matched.append(f"ui:{el}")

    # Over-represented: -15
    for cls in OVER_REPRESENTED_CLASSES:
        if cls.replace("_", " ") in text:
            score -= 15
            matched.append(f"overrepresented:{cls}")

    # Negative: -30 (Strict non-UI filtering)
    negative_keywords = [
        "cutscene", "artwork", "environment only", "character render", 
        "landscape", "cinematic", "concept art", "box art", "promo", 
        "illustration", "drawing", "poster"
    ]
    if any(kw in text for kw in negative_keywords):
        score -= 30
        matched.append("negative_non_ui_content")

    return score, matched

def get_image_hash(image_content: bytes) -> str:
    return hashlib.md5(image_content).hexdigest()

def get_extension_from_url(url: str) -> str:
    ext = url.split("?")[0].split(".")[-1].lower()
    return ext if ext in {"jpg", "jpeg", "png", "webp"} else "jpg"

def load_metadata_safe() -> Tuple[List[Dict], Set[str], Set[str], Set[str]]:
    rows = []
    urls = set()
    hashes = set()
    filenames = set()
    if not os.path.exists(METADATA_FILE):
        return rows, urls, hashes, filenames
    with open(METADATA_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if row.get("screenshot_url"): urls.add(row["screenshot_url"])
            if row.get("image_hash"): hashes.add(row["image_hash"])
            if row.get("file_name"): filenames.add(row["file_name"])
    return rows, urls, hashes, filenames

def save_metadata(rows: List[Dict]) -> None:
    fieldnames = [
        "source_api", "moby_game_id", "game_title", "moby_url", "platform_id", "platform_name",
        "screenshot_caption", "screenshot_url", "file_name", "image_hash", "genre", "release_year",
        "collected_at", "schema_version", "is_game_ui", "ui_quality", "primary_screen_type",
        "secondary_screen_types", "visual_style_tags", "theme_tags", "layout_blocks",
        "layout_tokens", "components", "confidence", "needs_review", "review_status",
        "ui_score", "ui_score_reason", "source_target_screen_type", "notes",
    ]
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def download_image(url: str, game_id: str, platform_id: str, index: int) -> Tuple[Optional[str], Optional[str]]:
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if response.status_code != 200: return None, None
        content = response.content
        image_hash = get_image_hash(content)
        ext = get_extension_from_url(url)
        file_name = f"moby_{game_id}_{platform_id}_tgt_{index:03d}_{image_hash[:8]}.{ext}"
        save_path = os.path.join(IMAGE_DIR, file_name)
        os.makedirs(IMAGE_DIR, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(content)
        return file_name, image_hash
    except Exception as e:
        print(f"  [IMAGE ERROR] {url}: {e}")
        return None, None

def fetch_moby(endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
    if params is None: params = {}
    params = dict(params)
    params["api_key"] = API_KEY
    url = f"{BASE_URL}{endpoint}"
    try:
        time.sleep(REQUEST_DELAY_SECONDS)
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200: 
            return response.json()
        if response.status_code == 429:
            print("[WARN] Rate limit. Sleeping...")
            time.sleep(30)
            return fetch_moby(endpoint, params)
        return None
    except requests.exceptions.Timeout:
        print(f"  [API TIMEOUT] {endpoint}")
        return None
    except Exception as e:
        print(f"  [API ERROR] {endpoint}: {e}")
        return None

def main():
    if not API_KEY:
        print("[!] MOBYGAMES_API_KEY missing.")
        return

    # 1. Identify deficit classes from BALANCE_REPORT
    if not os.path.exists(BALANCE_REPORT):
        print(f"[!] {BALANCE_REPORT} not found. Please run labeling_tool.py and generate a report first.")
        return

    report_df = pd.read_csv(BALANCE_REPORT)
    # Filter for labeled counts below goal
    deficits = report_df[report_df['labeled'] < MIN_LABELED_GOAL].copy()
    deficits = deficits.sort_values(by='labeled', ascending=True)

    if deficits.empty:
        print("[DONE] All classes have reached the labeled goal of 30 samples.")
        return

    print("[STATUS] deficit classes detected (Target: 30 labeled):")
    target_plan = []
    for _, row in deficits.iterrows():
        cat = row['category']
        labeled = row['labeled']
        needed = MIN_LABELED_GOAL - labeled
        
        # Special handling for 'other': limit to 20 max total
        if cat == "other":
            if labeled >= 20: continue
            needed = 20 - labeled
            collect_target = max(5, needed)
        else:
            collect_target = int(min(MAX_COLLECT_PER_CLASS, max(MIN_COLLECT_PER_CLASS, needed * COLLECTION_MULTIPLIER)))
            
        target_plan.append({
            "category": cat,
            "labeled_before": labeled,
            "needed": needed,
            "collect_target": collect_target
        })
        print(f" - {cat:15}: Labeled {labeled:3} | Needed {needed:3} | Collection Goal {collect_target:3}")

    # 2. Setup collection
    all_rows, existing_urls, existing_hashes, existing_filenames = load_metadata_safe()
    collection_log = []
    
    pbar_total = sum(p['collect_target'] for p in target_plan)
    pbar = tqdm(total=pbar_total, desc="Overall Progress")

    try:
        for plan in target_plan:
            target_cat = plan['category']
            goal = plan['collect_target']
            collected_for_this = 0
            
            keywords = TARGET_CLASSES_KEYWORDS.get(target_cat, [target_cat])
            # Use class-specific genres instead of all genres
            target_genre_ids = CAT_GENRES.get(target_cat, [122])
            search_queries = keywords + [f"GENRE:{gid}" for gid in target_genre_ids]
            
            print(f"\n[TARGET START] Category: {target_cat} (Goal: {goal})")
            
            for query in search_queries:
                if collected_for_this >= goal: break
                
                print(f"  [SEARCH START] Query: {query}")
                query_found_count = 0
                
                if query.startswith("GENRE:"):
                    genre_id = query.split(":")[1]
                    # Use a random offset to find different games
                    import random
                    offset = random.randint(0, 200)
                    search_data = fetch_moby("/games", params={"genre": genre_id, "limit": 20, "offset": offset})
                else:
                    search_data = fetch_moby("/games", params={"title": query, "limit": 30})
                
                if not search_data:
                    print(f"  [SEARCH FAILED/TIMEOUT] Query: {query}")
                    continue
                
                games = search_data.get("games", [])
                for g_idx, game in enumerate(games):
                    if collected_for_this >= goal: break
                    
                    game_id = str(game.get("game_id", ""))
                    game_title = str(game.get("title", "Unknown"))
                    print(f"    [GAME {g_idx+1}/{len(games)}] Checking: {game_title} (ID: {game_id})")
                    
                    game_genres = [str(g.get("genre_name", "")) for g in game.get("genres", [])]
                    
                    platforms = game.get("platforms", []) or []
                    for platform in platforms:
                        if collected_for_this >= goal: break
                        
                        platform_id = str(platform.get("platform_id", ""))
                        platform_name = str(platform.get("platform_name", "Unknown"))
                        
                        shots_data = fetch_moby(f"/games/{game_id}/platforms/{platform_id}/screenshots")
                        if not shots_data: continue
                        
                        screenshots = shots_data.get("screenshots", [])
                        for i, shot in enumerate(screenshots):
                            if collected_for_this >= goal: break
                            
                            url = str(shot.get("image", ""))
                            if not url or url in existing_urls: continue
                            
                            caption = str(shot.get("caption", "") or "")
                            score, matched = calculate_targeted_score(caption, game_title, game_genres, target_cat)
                            
                            # Only accept if it has a positive score for this target
                            if score > 0:
                                file_name, image_hash = download_image(url, game_id, platform_id, i)
                                if not file_name: continue
                                if image_hash in existing_hashes:
                                    # cleanup
                                    if os.path.exists(os.path.join(IMAGE_DIR, file_name)):
                                        os.remove(os.path.join(IMAGE_DIR, file_name))
                                    continue
                                
                                row = {
                                    "source_api": "MobyGames",
                                    "moby_game_id": game_id,
                                    "game_title": game_title,
                                    "moby_url": str(game.get("moby_url", "")),
                                    "platform_id": platform_id,
                                    "platform_name": platform_name,
                                    "screenshot_caption": caption,
                                    "screenshot_url": url,
                                    "file_name": file_name,
                                    "image_hash": image_hash,
                                    "genre": ", ".join(game_genres),
                                    "release_year": "",
                                    "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "schema_version": SCHEMA_VERSION,
                                    "is_game_ui": "", 
                                    "ui_quality": "", 
                                    "primary_screen_type": "", 
                                    "secondary_screen_types": "[]", 
                                    "visual_style_tags": "[]", 
                                    "theme_tags": "[]",
                                    "layout_blocks": "[]", 
                                    "layout_tokens": "[]", 
                                    "components": "[]",
                                    "confidence": "", 
                                    "needs_review": "True", 
                                    "review_status": "unlabeled",
                                    "ui_score": str(score),
                                    "ui_score_reason": ", ".join(matched),
                                    "source_target_screen_type": target_cat,
                                    "notes": f"Targeted follow-up for deficit class {target_cat}",
                                }
                                
                                all_rows.append(row)
                                existing_urls.add(url)
                                existing_hashes.add(image_hash)
                                collected_for_this += 1
                                query_found_count += 1
                                pbar.update(1)
                                
                                collection_log.append({
                                    "target_class": target_cat,
                                    "labeled_before": plan['labeled_before'],
                                    "needed": plan['needed'],
                                    "newly_collected": collected_for_this,
                                    "remaining_estimate": max(0, plan['needed'] - collected_for_this),
                                    "file_name": file_name,
                                    "game_title": game_title,
                                    "screenshot_caption": caption,
                                    "screenshot_url": url,
                                    "ui_score": score,
                                    "ui_score_reason": ", ".join(matched)
                                })
                
                print(f"  [SEARCH DONE] Query: {query} | Newly collected: {query_found_count}")

            # Intermediate save after each class
            save_metadata(all_rows)
            if collection_log:
                pd.DataFrame(collection_log).to_csv(TARGET_REPORT, index=False, encoding="utf-8-sig")
            print(f"[TARGET DONE] Category: {target_cat} | Total collected: {collected_for_this}")

    except KeyboardInterrupt:
        print("\n[STOP] KeyboardInterrupt detected. Saving current progress...")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        save_metadata(all_rows)
        if collection_log:
            pd.DataFrame(collection_log).to_csv(TARGET_REPORT, index=False, encoding="utf-8-sig")
        pbar.close()
        print(f"\n[FINISH] Collected {len(collection_log)} new images in this session.")

if __name__ == "__main__":
    main()
