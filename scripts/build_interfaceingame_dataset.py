# -*- coding: utf-8 -*-
import os
import time
import argparse
import urllib.robotparser
import requests
from bs4 import BeautifulSoup
import csv
import urllib.request

BASE_URL = "https://interfaceingame.com"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
USER_AGENT = "Antigravity/1.0 (Research Validation Bot; https://example.com/bot)"
OUTPUT_DIR = "data/interfaceingame_sample"
CSV_PATH = "metadata_interfaceingame_sample.csv"
SCHEMA_VERSION = "6.0"
MAX_TEST_IMAGES = 30

def check_robots_txt():
    print("[*] robots.txt 확인 중...")
    try:
        r = requests.get(ROBOTS_URL, timeout=5)
        if "Disallow:" in r.text and "User-agent: *" in r.text:
            # If Disallow is empty, it's allowed
            if "Disallow: /" in r.text and not "Disallow: \n" in r.text:
                print("[*] robots.txt 접근 권한 확인: 거부됨 (전체 차단)")
                return False
        print("[*] robots.txt 접근 권한 확인: 허용 (또는 제한 없음)")
        return True
    except Exception as e:
        print(f"[!] robots.txt 직접 확인 실패: {e}. 허용으로 간주합니다.")
        return True

def init_csv():
    headers = [
        "schema_version", "file_name", "game_title", "genres", "platforms", 
        "source_apis", "screenshot_urls", "screenshot_captions",
        "is_game_ui", "ui_quality", "primary_screen_types", "secondary_screen_types",
        "visual_style_tags", "style_tags", "theme_tags",
        "layout_blocks", "layout_tokens", "layout_positions", "layout_element_types",
        "layout_roles", "components", "confidence", "needs_review", "review_status", "notes"
    ]
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_to_csv(data_dict):
    headers = [
        "schema_version", "file_name", "game_title", "genres", "platforms", 
        "source_apis", "screenshot_urls", "screenshot_captions",
        "is_game_ui", "ui_quality", "primary_screen_types", "secondary_screen_types",
        "visual_style_tags", "style_tags", "theme_tags",
        "layout_blocks", "layout_tokens", "layout_positions", "layout_element_types",
        "layout_roles", "components", "confidence", "needs_review", "review_status", "notes"
    ]
    row = [data_dict.get(h, "") for h in headers]
    with open(CSV_PATH, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Interface In Game Scraper (PoC)")
    parser.add_argument("--test", action="store_true", help="10~30장 내외의 소규모 샘플 수집 및 저장 (안전 모드)")
    parser.add_argument("--dry-run", action="store_true", help="다운로드 없이 메타데이터 출력만 진행")
    args = parser.parse_args()

    # 기본은 dry-run 모드로 작동
    is_dry_run = True if not args.test else False
    if args.dry_run:
        is_dry_run = True

    if not check_robots_txt():
        print("[!] 로봇 배제 표준(robots.txt)에 의해 수집이 거부되었습니다. 안전을 위해 스크립트를 종료합니다.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not is_dry_run:
        init_csv()

    print(f"[*] 모드: {'DRY-RUN' if is_dry_run else 'TEST (MAX 30 IMAGES)'}")
    print("[*] 메인 페이지 파싱 시작...")
    
    headers = {'User-Agent': USER_AGENT}
    try:
        r = requests.get(f"{BASE_URL}/games/", headers=headers, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[!] 페이지 접근 실패: {e}")
        return

    soup = BeautifulSoup(r.text, 'html.parser')
    games = []
    for article in soup.find_all('article'):
        a = article.find('a')
        if a and 'href' in a.attrs:
            games.append(a['href'])
    
    print(f"[*] {len(games)}개의 게임 링크 발견.")
    
    collected_count = 0
    for game_url in games:
        if not game_url.startswith('http'):
            game_url = f"{BASE_URL}/{game_url.lstrip('/')}"
            
        if collected_count >= MAX_TEST_IMAGES and not is_dry_run:
            break
            
        time.sleep(2.0)  # 예의 바른 수집을 위한 필수 sleep
        print(f"  -> 게임 파싱 중: {game_url}")
        
        try:
            gr = requests.get(game_url, headers=headers, timeout=10)
            gr.raise_for_status()
        except Exception as e:
            print(f"     [!] 게임 페이지 접근 실패: {e}")
            continue
            
        gsoup = BeautifulSoup(gr.text, 'html.parser')
        
        game_title_h1 = gsoup.find('h1')
        game_title = game_title_h1.text.strip() if game_title_h1 else game_url.split('/')[-2]
        
        images = []
        for img in gsoup.find_all('img'):
            src = img.get('src', '')
            if 'uploads' in src and '-500x281' in src:
                # 고해상도 원본 URL 추정
                high_res_url = src.replace('-500x281', '')
                filename = high_res_url.split('/')[-1]
                caption = filename.replace('.jpg', '').replace('.png', '').replace('-', ' ').strip()
                images.append((high_res_url, filename, caption))
        
        for url, filename, caption in images:
            if collected_count >= MAX_TEST_IMAGES and not is_dry_run:
                break
                
            print(f"     [+] 발견: {filename} ({caption})")
            
            if not is_dry_run:
                # 다운로드 진행
                file_path = os.path.join(OUTPUT_DIR, filename)
                try:
                    time.sleep(1.0)
                    img_data = requests.get(url, headers=headers, timeout=10).content
                    with open(file_path, 'wb') as f:
                        f.write(img_data)
                    
                    # CSV 기록 (Schema 6.0 형식 맞춤, 나머지 필드는 빈 JSON/배열/문자열)
                    row_data = {
                        "schema_version": SCHEMA_VERSION,
                        "file_name": file_path,
                        "game_title": game_title,
                        "genres": "[]",
                        "platforms": "[]",
                        "source_apis": "InterfaceInGame",
                        "screenshot_urls": url,
                        "screenshot_captions": caption,
                        "is_game_ui": "True",
                        "ui_quality": "good",
                        "primary_screen_types": "[]",
                        "secondary_screen_types": "[]",
                        "visual_style_tags": "[]",
                        "style_tags": "[]",
                        "theme_tags": "[]",
                        "layout_blocks": "[]",
                        "layout_tokens": "[]",
                        "layout_positions": "[]",
                        "layout_element_types": "[]",
                        "layout_roles": "[]",
                        "components": "[]",
                        "confidence": "0",
                        "needs_review": "True",
                        "review_status": "pending",
                        "notes": "Sampled via build_interfaceingame_dataset.py"
                    }
                    append_to_csv(row_data)
                    collected_count += 1
                except Exception as e:
                    print(f"     [!] 다운로드 실패: {e}")

    print("-" * 50)
    if is_dry_run:
        print("[*] DRY-RUN 완료: 다운로드 없이 파싱 구조만 확인했습니다.")
    else:
        print(f"[*] 테스트 샘플 수집 완료: 총 {collected_count}장 저장됨.")
        print(f"[*] 저장 위치: {OUTPUT_DIR}")
        print(f"[*] 메타데이터: {CSV_PATH}")

if __name__ == "__main__":
    main()
