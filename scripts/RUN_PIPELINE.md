# Game UI Dataset Pipeline 실행 순서

이 문서는 새 데이터셋 스키마 6.0 기준 실행 순서다.
기존 screen_type, ui_label 중심 구조는 사용하지 않는다.

## 0. 전체 목표

최종 데이터 흐름은 아래 순서다.

```text
MobyGames API 수집
-> metadata.csv 생성
-> GPT-4o 라벨링
-> SigLIP2 임베딩 DB 생성
-> PyTorch 학습
-> Gradio 웹에서 검색 및 추천
```

새 라벨 구조의 핵심 필드는 아래와 같다.

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
  "components": ["health_bar", "minimap", "skill_icons"],
  "confidence": 0.89,
  "needs_review": false,
  "review_status": "labeled"
}
```

## 1. 패키지 설치

```bash
pip install -r requirements.txt
```

CUDA 환경에서 PyTorch가 제대로 설치되어 있지 않으면, 본인 CUDA 버전에 맞는 PyTorch 설치 명령을 먼저 사용해야 한다.

## 2. .env 설정

프로젝트 루트에 `.env` 파일을 만들고 아래 값을 넣는다.

```env
MOBYGAMES_API_KEY=여기에_MobyGames_API_Key
OPENAI_API_KEY=여기에_OpenAI_API_Key
```

MobyGames 수집만 먼저 할 때는 `MOBYGAMES_API_KEY`만 있어도 된다.
GPT-4o 라벨링을 하려면 `OPENAI_API_KEY`가 필요하다.

## 3. MobyGames 데이터 수집

```bash
python build_mobygames_dataset.py
```

결과로 아래 파일과 폴더가 생성된다.

```text
data/images/
data/metadata.csv
```

`metadata.csv`는 처음부터 새 스키마 6.0 기준으로 생성된다.
수집 도중 중단돼도 중간 저장되도록 구성되어 있다.

## 4. GPT-4o 라벨링 실행

```bash
python labeling_tool.py
```

웹이 열리면 다음 순서로 진행한다.

1. 이미지를 확인한다.
2. `AI 분석 요청`을 누르거나 자동 라벨링을 시작한다.
3. GPT-4o가 아래 항목을 채운다.
   - is_game_ui
   - ui_quality
   - primary_screen_type
   - secondary_screen_types
   - style_tags
   - layout_blocks
   - layout_tokens
   - components
   - confidence
   - needs_review
4. 결과가 이상하면 수동 수정한다.
5. 저장 후 다음으로 넘어간다.

학습과 검색에 실제로 쓰이는 데이터 조건은 아래와 같다.

```text
review_status == labeled
is_game_ui == true
ui_quality == keep
primary_screen_type != ""
```

## 5. 데이터셋 로드 테스트

라벨링이 어느 정도 끝난 뒤 실행한다.

```bash
python test_dataset.py
```

확인할 내용은 아래와 같다.

```text
pixel_values
primary_screen_label
style_label
layout_label
```

이 테스트가 통과하면 `dataset.py`가 새 metadata 구조를 정상적으로 읽고 있다는 뜻이다.

## 6. 모델 구조 확인

```bash
python inspect_model.py
```

확인할 출력은 아래 3개다.

```text
logits_primary_screen_type
logits_style_tags
logits_layout_tokens
```

## 7. 통합 테스트

```bash
python test_integration.py
```

이 테스트는 데이터셋과 모델의 출력 차원이 서로 맞는지 확인한다.

## 8. 학습 실행

기본 실행:

```bash
python train.py --epochs 10 --batch_size 8 --output_dir output_new_schema --use_amp
```

조금 더 길게 학습:

```bash
python train.py --epochs 15 --batch_size 8 --output_dir output_new_schema_epoch15 --use_amp
```

학습 결과는 아래에 저장된다.

```text
output_new_schema/best_model.pth
output_new_schema/last_model.pth
output_new_schema/logs/
```

TensorBoard 확인:

```bash
tensorboard --logdir output_new_schema/logs
```

주요 그래프는 아래다.

```text
Loss/Train
Loss/Val
Metric/PrimaryScreen_Acc
Metric/Style_F1
Metric/Layout_F1
Metric/Selection_Score
```

## 9. 임베딩 DB 생성

학습 전에도 검색용으로 만들 수 있고, 학습 후에도 다시 만들 수 있다.

```bash
python build_db.py
```

결과:

```text
data/embeddings.pt
```

이 파일에는 이미지 임베딩뿐 아니라 아래 라벨도 함께 저장된다.

```text
primary_screen_types
secondary_screen_types
style_tags
layout_blocks
layout_tokens
components
```

## 10. 웹 실행

```bash
python app_learning_compare_v2.py
```

웹 기능은 크게 두 가지다.

1. 텍스트 입력 기반 추천
   - 예: `dark fantasy inventory screen with top status bar and center popup`
   - 입력 문장을 `primary_screen_type`, `style_tags`, `layout_tokens`로 해석한 뒤 추천한다.

2. 이미지 입력 기반 추천
   - 입력 이미지를 학습 모델로 분석한다.
   - `primary_screen_type`, `style_tags`, `layout_tokens` 예측을 기반으로 관련 UI를 추천한다.

## 11. 최종 발표용 핵심 설명

발표에서는 아래처럼 설명하면 된다.

```text
본 프로젝트는 MobyGames API를 통해 게임 스크린샷을 수집한 뒤,
GPT-4o를 이용해 게임 UI 여부, 화면 유형, 스타일, 레이아웃 블록을 구조화된 JSON 형태로 라벨링한다.
이후 SigLIP2 임베딩 검색과 PyTorch 학습 모델을 결합하여,
텍스트 또는 이미지 입력에 대해 유사한 게임 UI 레퍼런스를 추천한다.
```

## 12. 주의할 점

1. `metadata.csv`가 없으면 먼저 `build_mobygames_dataset.py`를 실행해야 한다.
2. 라벨링이 충분히 안 되어 있으면 학습이 안 된다.
3. `ui_quality`가 `keep`인 데이터만 학습과 검색에 들어간다.
4. `layout_blocks`는 사람이 보기 좋은 구조이고, `layout_tokens`는 학습과 검색용이다.
5. 기존 `ui_label`, `screen_type` 구조는 더 이상 사용하지 않는다.
