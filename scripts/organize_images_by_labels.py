# -*- coding: utf-8 -*-
"""
=======================================================================
Game UI Image Organizer - Labels Based Classification
=======================================================================
This script organizes images from data/images into data/sorted_images
based on the labels provided in data/metadata.csv.

Features:
1. Copies (does NOT move) images into category folders.
2. Handles multi-tag style classification.
3. Collects unlabeled/unprocessed/rejected images into a review area.
4. Generates a summary report.
=======================================================================
"""

import os
import json
import shutil
import pandas as pd
from typing import Any, List, Dict
from collections import Counter

# ======================== Configuration ========================
IMAGE_DIR = "data/images"
METADATA_FILE = "data/metadata.csv"
OUTPUT_ROOT = "data/sorted_images"
MISSING_FILES_CSV = os.path.join(OUTPUT_ROOT, "missing_files.csv")

CLEAR_OUTPUT = True  # 기존 sorted_images 폴더 삭제 후 시작

# ======================== Helpers ========================

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
    """JSON 문자열 또는 콤마로 구분된 문자열을 리스트로 변환"""
    if is_blank(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    try:
        # JSON 파싱 시도
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, str):
            return [parsed]
        return []
    except Exception:
        # JSON이 아니면 콤마 분리 시도
        return [x.strip() for x in text.split(",") if x.strip()]

def parse_bool(value: Any, default: bool = False) -> bool:
    """다양한 형식의 boolean 값을 파싱"""
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

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def copy_image(src_name: str, sub_folder: str):
    """src_name 이미지를 OUTPUT_ROOT/sub_folder/src_name 으로 복사"""
    src_path = os.path.join(IMAGE_DIR, src_name)
    dest_dir = os.path.join(OUTPUT_ROOT, sub_folder)
    ensure_dir(dest_dir)
    dest_path = os.path.join(dest_dir, src_name)
    
    try:
        shutil.copy2(src_path, dest_path)
        return True
    except Exception as e:
        print(f"[!] 복사 실패 ({src_name} -> {sub_folder}): {e}")
        return False

# ======================== Main Logic ========================

def main():
    print(f"\n--- Game UI Image Organizer Start ---")

    # 1. 경로 체크
    if not os.path.exists(IMAGE_DIR):
        print(f"[ERROR] 원본 이미지 폴더가 없습니다: {IMAGE_DIR}")
        return

    if not os.path.exists(METADATA_FILE):
        print(f"[ERROR] 메타데이터 파일이 없습니다: {METADATA_FILE}")
        return

    # 2. 출력 폴더 초기화
    if CLEAR_OUTPUT and os.path.exists(OUTPUT_ROOT):
        print(f"[INFO] 기존 출력 폴더 삭제 중: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)
    
    ensure_dir(OUTPUT_ROOT)

    # 3. 데이터 로드
    try:
        df = pd.read_csv(METADATA_FILE, dtype=str, encoding="utf-8-sig").fillna("")
    except Exception as e:
        print(f"[ERROR] CSV 로드 실패: {e}")
        return

    total_rows = len(df)
    print(f"[INFO] 전체 메타데이터 행 수: {total_rows}")

    # 카운터 초기화
    counts = {
        "screen_type": Counter(),
        "style": Counter(),
        "unlabeled": Counter()
    }
    missing_files = []
    copied_to_screen_type_total = 0
    copied_to_style_total = 0
    copied_to_unlabeled_total = 0

    # 4. 행별 처리
    for idx, row in df.iterrows():
        file_name = row.get("file_name", "").strip()
        if not file_name:
            continue

        # 원본 이미지 존재 확인
        src_path = os.path.join(IMAGE_DIR, file_name)
        if not os.path.exists(src_path):
            missing_files.append({
                "file_name": file_name,
                "game_title": row.get("game_title", ""),
                "row_index": idx
            })
            continue

        # 라벨 추출 및 파싱
        review_status = row.get("review_status", "unlabeled").strip()
        is_game_ui = parse_bool(row.get("is_game_ui"), False)
        ui_quality = row.get("ui_quality", "").strip().lower()
        primary_screen_type = row.get("primary_screen_type", "").strip()
        needs_review = parse_bool(row.get("needs_review"), True)
        visual_style_tags = parse_json_list(row.get("visual_style_tags", "[]"))

        copy_happened = False

        # --- A. Screen Type 분류 ---
        # 조건: labeled, is_ui=True, quality in [keep, weak], primary_screen exists and not "other", needs_review=False
        if (review_status == "labeled" and 
            is_game_ui and 
            ui_quality in ["keep", "weak"] and 
            primary_screen_type and 
            primary_screen_type.lower() != "other" and 
            not needs_review):
            
            if copy_image(file_name, f"screen_type/{primary_screen_type}"):
                counts["screen_type"][primary_screen_type] += 1
                copied_to_screen_type_total += 1
                copy_happened = True

        # --- B. Style 분류 ---
        # 조건: labeled, is_ui=True, quality in [keep, weak], visual_style_tags exists, needs_review=False
        if (review_status == "labeled" and 
            is_game_ui and 
            ui_quality in ["keep", "weak"] and 
            visual_style_tags and 
            not needs_review):
            
            style_copy_count = 0
            for tag in visual_style_tags:
                tag_clean = str(tag).strip().lower().replace(" ", "_")
                if tag_clean:
                    if copy_image(file_name, f"style/{tag_clean}"):
                        counts["style"][tag_clean] += 1
                        style_copy_count += 1
            
            if style_copy_count > 0:
                copied_to_style_total += 1
                copy_happened = True

        # --- C. Unlabeled/Review/Other 분류 ---
        unlabeled_paths = []

        # review_status == "unlabeled" 또는 "retry_pending"
        if review_status in ["unlabeled", "retry_pending"]:
            unlabeled_paths.append("unlabeled/unlabeled_status")
        
        # needs_review == True
        if needs_review:
            unlabeled_paths.append("unlabeled/needs_review")
            
        # primary_screen_type 비어 있음
        if not primary_screen_type:
            unlabeled_paths.append("unlabeled/missing_screen_type")
            
        # primary_screen_type == "other"
        if primary_screen_type.lower() == "other":
            unlabeled_paths.append("unlabeled/other")
            
        # visual_style_tags 비어 있음 (단, UI로 채택된 경우만)
        if (not visual_style_tags and 
            review_status == "labeled" and 
            is_game_ui and 
            ui_quality in ["keep", "weak"]):
            unlabeled_paths.append("unlabeled/missing_style")
            
        # is_game_ui == False 또는 ui_quality == "reject" 또는 review_status == "skipped_permanent"
        if not is_game_ui or ui_quality == "reject" or review_status == "skipped_permanent":
            unlabeled_paths.append("unlabeled/rejected_or_non_ui")

        for p in unlabeled_paths:
            if copy_image(file_name, p):
                counts["unlabeled"][p.split("/")[-1]] += 1
                copy_happened = True
        
        if unlabeled_paths:
            copied_to_unlabeled_total += 1

    # 5. Missing Files 기록
    if missing_files:
        pd.DataFrame(missing_files).to_csv(MISSING_FILES_CSV, index=False, encoding="utf-8-sig")
        print(f"[WARN] 누락된 파일 정보가 저장되었습니다: {MISSING_FILES_CSV}")

    # 6. 결과 출력
    print(f"\n--- Organization Summary ---")
    print(f"Total metadata rows: {total_rows}")
    print(f"Copied to screen_type: {copied_to_screen_type_total}")
    print(f"Copied to style: {copied_to_style_total}")
    print(f"Copied to unlabeled (at least one category): {copied_to_unlabeled_total}")
    print(f"Missing image files: {len(missing_files)}")

    print(f"\n[Screen type counts]")
    for st, count in counts["screen_type"].most_common():
        print(f"  {st}: {count}")

    print(f"\n[Style counts]")
    for style, count in counts["style"].most_common():
        print(f"  {style}: {count}")

    print(f"\n[Unlabeled category counts]")
    for cat, count in counts["unlabeled"].items():
        print(f"  {cat}: {count}")

    print(f"\nOutput directory: {os.path.abspath(OUTPUT_ROOT)}")
    print(f"--- Done ---")

if __name__ == "__main__":
    main()
