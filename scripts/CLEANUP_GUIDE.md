# Game UI Discovery Studio - File Cleanup Guide

이 문서는 새 스키마 6.0 기준으로 프로젝트 파일을 정리하기 위한 가이드다.

새 기준은 다음 구조를 사용한다.

```json
{
  "is_game_ui": true,
  "ui_quality": "keep",
  "primary_screen_type": "gameplay_hud",
  "secondary_screen_types": ["quest_panel"],
  "style_tags": ["dark_fantasy", "realistic"],
  "layout_blocks": [
    {
      "position": "top_left",
      "element_type": "health_bar",
      "role": "health"
    }
  ],
  "layout_tokens": ["top_left:health_bar:health"],
  "components": ["health_bar", "minimap"],
  "confidence": 0.89,
  "needs_review": false,
  "review_status": "labeled"
}
```

---

## 1. 최종 파이프라인에서 사용하는 파일

아래 파일들은 새 구조 기준으로 계속 사용한다.

| 파일 | 용도 |
|---|---|
| `build_mobygames_dataset.py` | MobyGames API에서 스크린샷과 메타데이터 수집 |
| `labeling_tool.py` | GPT-4o 기반 UI 여부, 화면 유형, 스타일, 레이아웃 라벨링 |
| `dataset.py` | PyTorch 학습용 Dataset |
| `model.py` | SigLIP2 backbone + 3-head classifier 모델 |
| `train.py` | primary_screen_type, style_tags, layout_tokens 학습 |
| `build_db.py` | 이미지 임베딩 DB 생성 |
| `search_engine.py` | 텍스트/이미지 검색 및 라벨 기반 재정렬 |
| `app_learning_compare_v2.py` | 최종 웹 앱 후보 |
| `prompt_builder.py` | 추후 UI 생성 기능용 프롬프트 생성 |
| `test_dataset.py` | 데이터셋 로드 테스트 |
| `test_integration.py` | Dataset + Model 통합 테스트 |
| `inspect_model.py` | 모델 구조 및 체크포인트 확인 |
| `requirements.txt` | 설치 패키지 목록 |
| `RUN_PIPELINE.md` | 전체 실행 순서 가이드 |

---

## 2. 보관만 해도 되는 파일

아래 파일들은 예전 실험 기록이나 비교용으로는 남겨도 되지만, 새 파이프라인에서는 메인이 아니다.

| 파일 | 이유 |
|---|---|
| `app.py` | 구버전 웹 앱. 새 구조에는 맞지 않음 |
| `app_learning_compare.py` | v2 이전 비교 웹 |
| `build_db_learning.py` | 구버전 학습 기반 DB 빌더 |
| `search_engine_learning.py` | 구버전 학습 검색 엔진 |
| `search_engine_learning_v2.py` | 기존 v2 검색 엔진. 새 `search_engine.py`로 통합 |
| `migrate_metadata.py` | 기존 CSV 마이그레이션용. 새로 시작하면 불필요 |
| `build_rawg_dataset.py` | RAWG 수집용. MobyGames로 갈아타면 보조/폐기 후보 |

보관하려면 `legacy/` 폴더를 만들어 옮기는 것을 추천한다.

예:

```bash
mkdir legacy
move app.py legacy\
move app_learning_compare.py legacy\
move build_db_learning.py legacy\
move search_engine_learning.py legacy\
move search_engine_learning_v2.py legacy\
move migrate_metadata.py legacy\
move build_rawg_dataset.py legacy\
```

Windows PowerShell에서는 다음처럼 해도 된다.

```powershell
mkdir legacy
Move-Item app.py legacy/
Move-Item app_learning_compare.py legacy/
Move-Item build_db_learning.py legacy/
Move-Item search_engine_learning.py legacy/
Move-Item search_engine_learning_v2.py legacy/
Move-Item migrate_metadata.py legacy/
Move-Item build_rawg_dataset.py legacy/
```

---

## 3. 삭제해도 되는 임시 파일

아래 파일들은 실행 중 자동 생성되거나 다시 만들 수 있다.

| 파일/폴더 | 설명 |
|---|---|
| `__pycache__/` | Python 캐시 |
| `.pytest_cache/` | pytest 캐시가 있다면 삭제 가능 |
| `data/embeddings.pt` | `build_db.py`로 다시 생성 가능 |
| `data/labeling_events.jsonl` | 라벨링 로그. 필요 없으면 삭제 가능 |
| `output/` | 학습 결과. 삭제 전 체크포인트 필요 여부 확인 |
| `output_run*/` | 이전 학습 결과. 비교용 아니면 삭제 가능 |

주의: `data/images/`와 `data/metadata.csv`는 삭제하면 안 된다.

---

## 4. 새 프로젝트 기준 최소 파일 목록

정말 깔끔하게 남기고 싶으면 아래만 남겨도 된다.

```text
build_mobygames_dataset.py
labeling_tool.py
dataset.py
model.py
train.py
build_db.py
search_engine.py
app_learning_compare_v2.py
prompt_builder.py
test_dataset.py
test_integration.py
inspect_model.py
requirements.txt
RUN_PIPELINE.md
.env
```

그리고 데이터 폴더는 아래 구조를 유지한다.

```text
data/
  images/
  metadata.csv
  embeddings.pt
```

---

## 5. 실행 순서 기준 최종 파일 흐름

```text
build_mobygames_dataset.py
-> data/metadata.csv 생성

labeling_tool.py
-> metadata.csv에 새 라벨 채우기

test_dataset.py
-> 데이터셋 로드 확인

inspect_model.py
-> 모델 구조 확인

test_integration.py
-> Dataset + Model 연결 확인

train.py
-> best_model.pth 생성

build_db.py
-> data/embeddings.pt 생성

app_learning_compare_v2.py
-> 최종 웹 실행
```

---

## 6. 지금 절대 섞으면 안 되는 구조

아래 구형 필드를 새 메인 구조에 다시 섞지 않는 것이 좋다.

| 구형 필드 | 새 필드 |
|---|---|
| `ui_label` | 제거 |
| `screen_type` | `primary_screen_type` |
| `layout_tags` | `layout_blocks`, `layout_tokens` |
| `components` 문자열 | `components` JSON 배열 |

구형 필드를 다시 섞으면 학습 코드와 검색 코드가 서로 다른 컬럼을 읽게 되어 버그가 생긴다.

---

## 7. 발표용 정리 문장

기존에는 단순 화면 유형과 스타일 태그 중심으로 게임 UI를 분류했지만, 최종 구조에서는 대표 화면 유형, 보조 화면 유형, 스타일, 위치 기반 레이아웃 블록, 세부 UI 컴포넌트를 분리하여 더 구체적인 검색과 추천이 가능하도록 데이터셋 구조를 재설계했다.
