# -*- coding: utf-8 -*-
"""
MobyGames screenshot dataset builder for the new Game UI schema.

This script creates data/metadata.csv with the schema used by the
new labeling, training, vector DB, and search pipeline.

Main idea:
1. Collect screenshots and metadata from MobyGames API.
2. Do not assign UI labels here.
3. Initialize all AI labeling fields with empty JSON-safe values.
4. The labeling tool will later fill is_game_ui, ui_quality,
   primary_screen_type, style_tags, layout_blocks, and related fields.
"""

import csv
import hashlib
import json
import os
import time
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
SCHEMA_VERSION = "6.0"

# --- 수집 설정 (사용자 조정 가능) ---
TARGET_IMAGE_COUNT = 1200      # 최종 수집 목표 이미지 수
GAMES_PER_PAGE = 100           # 페이지당 게임 수
MAX_SCREENSHOTS_PER_GAME = 8   # 게임당 최대 스크린샷 수
REQUEST_DELAY_SECONDS = 1.6    # API 요청 간 기본 지연 시간 (초)
MAX_RATE_LIMIT_RETRIES = 5     # 연속 Rate Limit 발생 시 최대 재시도 횟수
MAX_BACKOFF_SECONDS = 300      # 최대 대기 시간 (초)
REQUEST_TIMEOUT = 30           # API 응답 대기 시간
# ------------------------------

# UI 가능성이 높은 키워드 (가중치 부여)
POSITIVE_KEYWORDS = {
    "menu", "main menu", "title", "options", "settings", "inventory", 
    "equipment", "character", "status", "map", "shop", "store", "skill", 
    "dialogue", "quest", "hud", "interface", "pause", "result", "save", "load",
    "battle", "combat", "score", "game over"
}

# 비-UI(일반 배경/영상) 키워드 (제외 대상)
NEGATIVE_KEYWORDS = {
    "cutscene", "intro", "ending", "movie", "fmv", "artwork", "cover", 
    "landscape", "character render", "cinematic", "credits", "scene", "room", "photo"
}

# 장르 밸런싱을 위한 카테고리 정의
GENRE_MAP = {
    "RPG": ["Role-Playing (RPG)", "RPG"],
    "Strategy": ["Strategy"],
    "Simulation": ["Simulation"],
    "Adventure": ["Adventure"],
    "Action/Shooter": ["Action", "Shooter"]
}
MAX_PER_GENRE = 250  # 특정 장르 과점 방지


def calculate_ui_score(caption: str) -> Tuple[int, List[str]]:
    if not caption:
        return 0, []
    text = caption.lower()
    score = 0
    matched = []
    for kw in POSITIVE_KEYWORDS:
        if kw in text:
            score += 5
            matched.append(f"+{kw}")
    for kw in NEGATIVE_KEYWORDS:
        if kw in text:
            score -= 10
            matched.append(f"-{kw}")
    return score, matched


def extract_release_year(platforms: List[Dict]) -> str:
    years = []
    for p in platforms:
        date = str(p.get("first_release_date", ""))
        if date and len(date) >= 4:
            try:
                year = int(date[:4])
                years.append(year)
            except Exception:
                pass
    return str(min(years)) if years else ""


def get_era_category(year_str: str) -> str:
    if not year_str:
        return "Unknown"
    try:
        year = int(year_str)
        if year >= 2020: return "2020s"
        if year >= 2010: return "2010s"
        if year >= 2000: return "2000s"
        return "Legacy"
    except Exception:
        return "Unknown"

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
    "notes",
]


def json_array(items: Optional[List] = None) -> str:
    return json.dumps(items or [], ensure_ascii=False)


def get_image_hash(image_content: bytes) -> str:
    return hashlib.md5(image_content).hexdigest()


def get_extension_from_url(url: str) -> str:
    ext = url.split("?")[0].split(".")[-1].lower()
    if ext not in {"jpg", "jpeg", "png", "webp"}:
        return "jpg"
    return ext


def make_composite_key(game_id: str, platform_id: str, screenshot_url: str) -> str:
    return f"{game_id}_{platform_id}_{screenshot_url}"


def extract_hash_from_file_name(file_name: str) -> str:
    if not file_name:
        return ""
    stem = os.path.splitext(os.path.basename(file_name))[0]
    maybe_hash = stem.split("_")[-1]
    return maybe_hash if len(maybe_hash) == 8 else ""


def normalize_existing_row(row: Dict[str, str]) -> Dict[str, str]:
    normalized = {key: str(row.get(key, "") or "") for key in FIELDNAMES}

    if not normalized["source_api"]:
        normalized["source_api"] = "MobyGames"
    if not normalized["schema_version"]:
        normalized["schema_version"] = SCHEMA_VERSION
    if not normalized["secondary_screen_types"]:
        normalized["secondary_screen_types"] = json_array()
    if not normalized["visual_style_tags"]:
        normalized["visual_style_tags"] = json_array()
    if not normalized["theme_tags"]:
        normalized["theme_tags"] = json_array()
    if not normalized["layout_blocks"]:
        normalized["layout_blocks"] = json_array()
    if not normalized["layout_tokens"]:
        normalized["layout_tokens"] = json_array()
    if not normalized["components"]:
        normalized["components"] = json_array()
    if not normalized["needs_review"]:
        normalized["needs_review"] = "true"
    if not normalized["review_status"]:
        normalized["review_status"] = "unlabeled"
    if not normalized["ui_score"]:
        normalized["ui_score"] = "0"
    if not normalized["ui_score_reason"]:
        normalized["ui_score_reason"] = ""
    if not normalized["release_year"]:
        normalized["release_year"] = ""
    if not normalized["image_hash"]:
        normalized["image_hash"] = extract_hash_from_file_name(normalized["file_name"])

    return normalized


def load_existing_metadata() -> Tuple[List[Dict[str, str]], Set[str], Set[str], Set[str]]:
    rows: List[Dict[str, str]] = []
    existing_urls: Set[str] = set()
    existing_composite_keys: Set[str] = set()
    existing_hashes: Set[str] = set()

    if not os.path.exists(METADATA_FILE):
        return rows, existing_urls, existing_composite_keys, existing_hashes

    with open(METADATA_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = normalize_existing_row(raw_row)
            rows.append(row)

            screenshot_url = row.get("screenshot_url", "")
            if screenshot_url:
                existing_urls.add(screenshot_url)

            game_id = row.get("moby_game_id", "")
            platform_id = row.get("platform_id", "")
            if game_id and platform_id and screenshot_url:
                existing_composite_keys.add(make_composite_key(game_id, platform_id, screenshot_url))

            image_hash = row.get("image_hash", "")
            if image_hash:
                existing_hashes.add(image_hash)

    return rows, existing_urls, existing_composite_keys, existing_hashes


def download_image(url: str, game_id: str, platform_id: str, index: int) -> Tuple[Optional[str], Optional[str], bool]:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return None, None, False

        content = response.content
        image_hash = get_image_hash(content)
        ext = get_extension_from_url(url)
        file_name = f"moby_{game_id}_{platform_id}_{index:03d}_{image_hash[:8]}.{ext}"
        save_path = os.path.join(IMAGE_DIR, file_name)

        if os.path.exists(save_path):
            return file_name, image_hash, True

        with open(save_path, "wb") as f:
            f.write(content)

        return file_name, image_hash, False
    except Exception as e:
        print(f"[!] Download error: {url} / {e}")
        return None, None, False


rate_limit_failure_count = 0

def fetch_moby(endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
    global rate_limit_failure_count
    
    if params is None:
        params = {}
    params = dict(params)
    params["api_key"] = API_KEY

    url = f"{BASE_URL}{endpoint}"

    try:
        # 기본 요청 지연 적용
        time.sleep(REQUEST_DELAY_SECONDS)
        
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            rate_limit_failure_count = 0  # 성공 시 실패 카운트 초기화
            return response.json()
            
        if response.status_code == 429:
            rate_limit_failure_count += 1
            if rate_limit_failure_count > MAX_RATE_LIMIT_RETRIES:
                print(f"\n[CRITICAL] 연속으로 {MAX_RATE_LIMIT_RETRIES}회 이상 Rate Limit이 발생했습니다. 안전을 위해 중단합니다.")
                return None
            
            # Exponential Backoff 적용: 60, 120, 240, 300...
            wait_time = min(60 * (2 ** (rate_limit_failure_count - 1)), MAX_BACKOFF_SECONDS)
            print(f"\n[WARN] Rate limit reached. Backoff sleeping for {wait_time} seconds. (Attempt {rate_limit_failure_count}/{MAX_RATE_LIMIT_RETRIES})")
            time.sleep(wait_time)
            return fetch_moby(endpoint, params)

        print(f"\n[!] API error {response.status_code}: {response.text}")
        return None
    except Exception as e:
        print(f"\n[!] Request error: {e}")
        return None


def make_new_row(
    *,
    game_id: str,
    game_title: str,
    moby_url: str,
    platform_id: str,
    platform_name: str,
    screenshot_caption: str,
    screenshot_url: str,
    file_name: str,
    image_hash: str,
    genres: List[str],
    release_year: str = "",
    ui_score: int = 0,
    ui_score_reason: str = "",
) -> Dict[str, str]:
    return {
        "source_api": "MobyGames",
        "moby_game_id": str(game_id),
        "game_title": str(game_title),
        "moby_url": str(moby_url or ""),
        "platform_id": str(platform_id),
        "platform_name": str(platform_name),
        "screenshot_caption": str(screenshot_caption or ""),
        "screenshot_url": str(screenshot_url),
        "file_name": str(file_name),
        "image_hash": str(image_hash or ""),
        "genre": ", ".join(genres),
        "release_year": str(release_year),
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
        "ui_score": str(ui_score),
        "ui_score_reason": ui_score_reason,
        "notes": "",
    }


def save_metadata(rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not API_KEY:
        print("[!] MOBYGAMES_API_KEY is missing in .env.")
        return

    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)

    all_rows, existing_urls, existing_composite_keys, existing_hashes = load_existing_metadata()
    collected_count = len(all_rows)

    print("=" * 60)
    print("MobyGames dataset builder")
    print(f"Target image count: {TARGET_IMAGE_COUNT}")
    print(f"Existing rows: {collected_count}")
    print(f"Schema version: {SCHEMA_VERSION}")
    print("=" * 60)

    # 장르 및 시대별 수집 현황 초기화
    genre_counts = {cat: 0 for cat in GENRE_MAP}
    era_counts = {"2020s": 0, "2010s": 0, "2000s": 0, "Legacy": 0, "Unknown": 0}
    
    for r in all_rows:
        # 장르 카운트
        r_genres = r.get("genre", "").lower()
        for cat, keywords in GENRE_MAP.items():
            if any(kw.lower() in r_genres for kw in keywords):
                genre_counts[cat] += 1
        
        # 시대 카운트
        era = get_era_category(r.get("release_year", ""))
        if era in era_counts:
            era_counts[era] += 1

    print(f"[STATUS] Current genre counts: {genre_counts}")
    print(f"[STATUS] Current era counts: {era_counts}")

    # 시대별 목표 비중 (2010년 이후를 70% 이상으로 설정)
    ERA_QUOTAS = {
        "2020s": int(TARGET_IMAGE_COUNT * 0.40),
        "2010s": int(TARGET_IMAGE_COUNT * 0.35),
        "2000s": int(TARGET_IMAGE_COUNT * 0.15),
        "Legacy": int(TARGET_IMAGE_COUNT * 0.10)
    }
    
    # 시대별 탐색을 위한 오프셋 포인트 (MobyGames ID/순서 특성 반영)
    # 0: 최상단(오래된 게임 위주), 높은 오프셋일수록 비교적 최신 게임 가능성 높음
    ERA_START_OFFSETS = [160000, 120000, 60000, 0]
    
    pbar = tqdm(total=TARGET_IMAGE_COUNT, desc="collecting")
    pbar.update(min(collected_count, TARGET_IMAGE_COUNT))

    skipped_count = 0
    
    try:
        for base_offset in ERA_START_OFFSETS:
            if collected_count >= TARGET_IMAGE_COUNT:
                break
                
            offset = base_offset
            print(f"\n[INFO] Starting collection from offset {offset}...")
            
            # 해당 오프셋 구간에서 최대 30페이지 정도만 탐색
            for page in range(30):
                if collected_count >= TARGET_IMAGE_COUNT:
                    break
                
                games_data = fetch_moby(
                    "/games",
                    params={
                        "limit": GAMES_PER_PAGE,
                        "offset": offset,
                        "format": "normal",
                    },
                )
                if not games_data:
                    # Rate limit 등으로 중단된 경우 루프 탈출
                    if rate_limit_failure_count > MAX_RATE_LIMIT_RETRIES:
                        return
                    break

                offset += GAMES_PER_PAGE
                games = games_data.get("games", [])
                if not games:
                    print(f"[!] No more games in this offset range ({offset}).")
                    break

                for game in games:
                    if collected_count >= TARGET_IMAGE_COUNT:
                        break

                    game_id = str(game.get("game_id", ""))
                    game_title = str(game.get("title", "Unknown"))
                    moby_url = str(game.get("moby_url", "") or "")
                    
                    # 출시 연도 확인
                    release_year = extract_release_year(game.get("platforms", []))
                    era = get_era_category(release_year)
                    
                    # 시대별 쿼터 확인
                    target_era = era if era != "Unknown" else "2010s"
                    if target_era in ERA_QUOTAS and era_counts[target_era] >= ERA_QUOTAS[target_era]:
                        if collected_count > 100:
                            continue

                    # 장르 확인 및 카테고리 매칭
                    game_genres = [str(g.get("genre_name", "")).strip() for g in game.get("genres", [])]
                    game_genres = [g for g in game_genres if g]
                    
                    target_cat = None
                    for cat, keywords in GENRE_MAP.items():
                        if any(any(kw.lower() in g.lower() for kw in keywords) for g in game_genres):
                            if genre_counts[cat] < MAX_PER_GENRE:
                                target_cat = cat
                                break
                    
                    if not target_cat and collected_count > 150:
                        continue

                    platforms = game.get("platforms", []) or []
                    shots_from_this_game = 0
                    
                    print(f"\n[INFO] Processing Game: {game_title} ({release_year})")
                    
                    for platform in platforms:
                        if collected_count >= TARGET_IMAGE_COUNT:
                            break
                        if shots_from_this_game >= MAX_SCREENSHOTS_PER_GAME:
                            break
                        
                        if target_cat and genre_counts[target_cat] >= MAX_PER_GENRE:
                            break
                        if target_era in ERA_QUOTAS and era_counts[target_era] >= ERA_QUOTAS[target_era] and collected_count > 150:
                            break

                        platform_id = str(platform.get("platform_id", ""))
                        platform_name = str(platform.get("platform_name", "Unknown"))
                        if not game_id or not platform_id:
                            continue

                        shots_data = fetch_moby(f"/games/{game_id}/platforms/{platform_id}/screenshots")
                        if not shots_data:
                            continue

                        screenshots = shots_data.get("screenshots", [])
                        if not screenshots:
                            continue

                        # UI 우선순위 정렬
                        scored_shots = []
                        for shot in screenshots:
                            caption = str(shot.get("caption", "") or "")
                            score, matched = calculate_ui_score(caption)
                            if score < -5:
                                continue
                            scored_shots.append((score, matched, shot))

                        scored_shots.sort(key=lambda x: x[0], reverse=True)

                        for shot_idx, (score, matched, shot) in enumerate(scored_shots):
                            if collected_count >= TARGET_IMAGE_COUNT:
                                break
                            if shots_from_this_game >= MAX_SCREENSHOTS_PER_GAME:
                                break

                            screenshot_url = str(shot.get("image", "") or "")
                            if not screenshot_url or screenshot_url in existing_urls:
                                skipped_count += 1
                                continue

                            composite_key = make_composite_key(game_id, platform_id, screenshot_url)
                            if composite_key in existing_composite_keys:
                                skipped_count += 1
                                continue

                            file_name, image_hash, is_duplicate_file = download_image(
                                screenshot_url,
                                game_id,
                                platform_id,
                                shot_idx,
                            )
                            if not file_name or not image_hash:
                                continue

                            if image_hash in existing_hashes:
                                skipped_count += 1
                                continue

                            if is_duplicate_file:
                                existing_hashes.add(image_hash)
                                existing_urls.add(screenshot_url)
                                existing_composite_keys.add(composite_key)
                                continue

                            row = make_new_row(
                                game_id=game_id,
                                game_title=game_title,
                                moby_url=moby_url,
                                platform_id=platform_id,
                                platform_name=platform_name,
                                screenshot_caption=str(shot.get("caption", "") or ""),
                                screenshot_url=screenshot_url,
                                file_name=file_name,
                                image_hash=image_hash,
                                genres=game_genres,
                                release_year=release_year,
                                ui_score=score,
                                ui_score_reason=", ".join(matched),
                            )

                            all_rows.append(row)
                            existing_hashes.add(image_hash)
                            existing_urls.add(screenshot_url)
                            existing_composite_keys.add(composite_key)
                            collected_count += 1
                            shots_from_this_game += 1
                            if target_cat:
                                genre_counts[target_cat] += 1
                            if target_era in era_counts:
                                era_counts[target_era] += 1
                            
                            pbar.update(1)
                            
                            # 이미지 하나 수집할 때마다 즉시 저장 (데이터 손실 방지)
                            save_metadata(all_rows)

                # 페이지 단위 저장
                save_metadata(all_rows)

    except KeyboardInterrupt:
        print("\n[INFO] 사용자에 의해 중단되었습니다. 진행 상황을 저장합니다...")
        save_metadata(all_rows)

    pbar.close()
    save_metadata(all_rows)

    # Calculate summary
    total_rows = len(all_rows)
    avg_score = 0
    pos_count = 0
    era_summary = {era: sum(1 for r in all_rows if get_era_category(r.get("release_year", "")) == era) for era in era_counts}
    
    if total_rows > 0:
        scores = [int(r.get("ui_score", 0) or 0) for r in all_rows]
        avg_score = sum(scores) / total_rows
        pos_count = sum(1 for s in scores if s > 0)

    print("=" * 60)
    print(f"Done. Total metadata rows: {total_rows}")
    print(f"Era Distribution: {era_summary}")
    print(f"Skipped duplicates: {skipped_count}")
    print(f"Average UI Score: {avg_score:.2f}")
    print(f"Images with positive score: {pos_count}")
    print(f"Metadata file: {METADATA_FILE}")
    print(f"Image directory: {IMAGE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
