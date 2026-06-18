import streamlit as st
import time
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
import re
import sys

from sketch_utils import generate_semantic_ui_sketch

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Game UI Discovery Studio",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
#  Custom CSS (안전한 컴포넌트 스타일링)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Global overrides for background and text to ensure clean bright theme */
.stApp {
    background-color: #f8fafc;
}
.css-1d391kg {
    background-color: #f8fafc;
}

/* Card components */
.info-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 12px rgba(15,23,42,0.05);
    height: 100%;
}
.info-card-title {
    font-size: 16px;
    font-weight: 700;
    color: #2563eb;
    margin-bottom: 12px;
}
.info-card-text {
    font-size: 14px;
    color: #475569;
    line-height: 1.6;
}

.hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #2563eb 100%);
    color: #ffffff;
    border-radius: 24px;
    padding: 48px 40px;
    text-align: center;
    box-shadow: 0 12px 30px rgba(15,23,42,0.12);
    margin-bottom: 32px;
}
.hero-title {
    font-size: 36px;
    font-weight: 900;
    margin-bottom: 16px;
    color: #ffffff;
}
.hero-subtitle {
    font-size: 18px;
    color: #cbd5e1;
    width: 100%;
    margin-bottom: 32px;
    text-align: center;
}
.hero-badges {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
}
.hero-badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 700;
}

/* Result Card */
.result-card {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(15,23,42,0.05);
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 16px;
}
.result-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 15px rgba(15,23,42,0.1);
}
.result-card-content {
    padding: 16px;
}
.result-title {
    font-size: 15px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.result-tag-group {
    margin-bottom: 8px;
}
.result-tag-label {
    font-size: 11px;
    font-weight: 800;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: block;
    margin-bottom: 2px;
}
.result-tag-val {
    font-size: 13px;
    font-weight: 600;
    color: #334155;
    background: #f1f5f9;
    padding: 2px 6px;
    border-radius: 4px;
    display: inline-block;
}
.result-score {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #e2e8f0;
    font-size: 13px;
    font-weight: 800;
    color: #ea580c;
    text-align: right;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  State & Initialization
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "intro"

@st.cache_resource(show_spinner="AI 모델 및 Vector DB를 로드하고 있습니다. (최초 1회만 약 20초 소요, 잠시만 기다려주세요!)")
def load_engine():
    import app
    return {
        "engine": app.engine,
        "predict_finetuned": app.predict_finetuned,
        "predict_crop_finetuned": app.predict_crop_finetuned,
        "predict_base_zero_shot": app.predict_base_zero_shot,
        "interpret_text_query": app.interpret_text_query
    }

deps = load_engine()
engine = deps["engine"]
predict_finetuned = deps["predict_finetuned"]
predict_crop_finetuned = deps["predict_crop_finetuned"]
predict_base_zero_shot = deps["predict_base_zero_shot"]
interpret_text_query = deps["interpret_text_query"]


# Helper: image to base64 for HTML rendering
def get_image_base64(img_path):
    if str(img_path).startswith("http://") or str(img_path).startswith("https://"):
        return img_path
        
    try:
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""

def filter_adult_results(results):
    """결과 리스트에서 성인 게임 제외"""
    filtered = []
    for r in results:
        meta = r.get("metadata", {})
        tags = str(meta.get("tags", "")).lower()
        genres = str(meta.get("genres", "")).lower()
        desc = str(meta.get("description", "")).lower()
        
        is_adult = ("nsfw" in tags or "sexual content" in tags or "nudity" in tags or 
                    "nsfw" in genres or "sexual content" in genres or "nudity" in genres or
                    "hentai" in tags or "hentai" in genres)
        if not is_adult:
            filtered.append(r)
    return filtered

# ─────────────────────────────────────────────
#  UI Components
# ─────────────────────────────────────────────
def render_intro_page():
    # Calculate stats dynamically
    ps_counts = {}
    vs_counts = {}
    if engine.cache_data:
        for ps in engine.cache_data.get("primary_screen_types", []):
            if ps:
                ps_counts[ps] = ps_counts.get(ps, 0) + 1
        
        for vsl in engine.cache_data.get("visual_style_tags", []):
            if vsl:
                if isinstance(vsl, list):
                    for vs in vsl:
                        vs_counts[vs] = vs_counts.get(vs, 0) + 1
                else:
                    vs_counts[vsl] = vs_counts.get(vsl, 0) + 1

    gameplay_hud_cnt = ps_counts.get("gameplay_hud", 0)
    menu_lobby_cnt = ps_counts.get("main_menu", 0) + ps_counts.get("lobby", 0) + ps_counts.get("title_screen", 0) + ps_counts.get("shop", 0)
    flow_other_cnt = ps_counts.get("other", 0) + ps_counts.get("battle_result", 0) + ps_counts.get("loading_screen", 0) + ps_counts.get("tutorial", 0)
    dialogue_cnt = ps_counts.get("dialogue", 0)
    inventory_panel_cnt = ps_counts.get("inventory", 0) + ps_counts.get("character_screen", 0) + ps_counts.get("equipment", 0) + ps_counts.get("skill_tree", 0) + ps_counts.get("crafting", 0)
    map_screen_cnt = ps_counts.get("map", 0)

    minimal_clean_cnt = vs_counts.get("minimal", 0) + vs_counts.get("clean", 0) + vs_counts.get("modern_clean", 0)
    retro_pixel_cnt = vs_counts.get("retro", 0) + vs_counts.get("pixel_art", 0) + vs_counts.get("retro_pixel", 0)
    modern_cnt = vs_counts.get("modern", 0) + vs_counts.get("modern_clean", 0)
    realistic_cnt = vs_counts.get("realistic", 0) + vs_counts.get("gritty", 0) + vs_counts.get("realistic_gritty", 0)
    cartoon_anime_cnt = vs_counts.get("cartoon", 0) + vs_counts.get("anime", 0) + vs_counts.get("cute", 0) + vs_counts.get("stylized_cartoon", 0)
    skeuomorphic_cnt = vs_counts.get("skeuomorphic", 0)

    # Hero Section
    st.markdown(f"""
    <div class="hero-card">
        <h1 class="hero-title">다중 작업 학습 모델 기반 <span style="color:#60a5fa;">게임 UI 지능형 추천</span></h1>
        <p class="hero-subtitle">
            게임 UI를 화면 유형, 스타일, 테마, 레이아웃 구조로 분석하여 사용자 입력과 가장 유사한 UI 레퍼런스를 추천합니다.
        </p>
        <div class="hero-badges">
            <span class="hero-badge">🏆 멀티모달 (Vision + Text) 아키텍처</span>
            <span class="hero-badge">총 {engine._count():,}장의 게임 UI 데이터베이스 (화면 유형, 스타일, 테마, 레이아웃 정밀 라벨링)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Database Stats
    st.markdown(f"### 📊 데이터베이스 현황 (총 {engine._count():,}장)")
    st.markdown(f"""
    <div style="display:flex; gap:16px; flex-wrap:wrap; margin-bottom:32px;">
        <div style="flex:1; min-width:280px; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; box-shadow:0 4px 6px rgba(0,0,0,0.02);">
            <div style="font-size:16px; font-weight:800; color:#2563eb; margin-bottom:12px;">📺 주요 화면 유형 분포</div>
            <div style="font-size:13px; color:#475569; line-height:1.8;">
                <span style="display:inline-block; width:120px; font-weight:700; color:#0f172a;">Gameplay HUD</span> {gameplay_hud_cnt}장<br>
                <span style="display:inline-block; width:120px; font-weight:700; color:#0f172a;">Menu & Lobby</span> {menu_lobby_cnt}장<br>
                <span style="display:inline-block; width:120px; font-weight:700; color:#0f172a;">Flow & Other</span> {flow_other_cnt}장<br>
                <span style="display:inline-block; width:120px; font-weight:700; color:#0f172a;">Dialogue</span> {dialogue_cnt}장<br>
                <span style="display:inline-block; width:120px; font-weight:700; color:#0f172a;">Inventory/Panel</span> {inventory_panel_cnt}장<br>
                <span style="display:inline-block; width:120px; font-weight:700; color:#0f172a;">Map Screen</span> {map_screen_cnt}장<br>
                <span style="font-size:11px; color:#94a3b8;">* 세부 19개 클래스를 대표 범주로 요약한 수치입니다.</span>
            </div>
        </div>
        <div style="flex:1; min-width:280px; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:20px; box-shadow:0 4px 6px rgba(0,0,0,0.02);">
            <div style="font-size:16px; font-weight:800; color:#2563eb; margin-bottom:12px;">✨ 주요 비주얼 스타일 분포</div>
            <div style="font-size:13px; color:#475569; line-height:1.8;">
                <span style="display:inline-block; width:120px; font-weight:700; color:#0f172a;">Minimal / Clean</span> {minimal_clean_cnt}장<br>
                <span style="display:inline-block; width:120px; font-weight:700; color:#0f172a;">Retro / Pixel</span> {retro_pixel_cnt}장<br>
                <span style="display:inline-block; width:120px; font-weight:700; color:#0f172a;">Modern</span> {modern_cnt}장<br>
                <span style="display:inline-block; width:120px; font-weight:700; color:#0f172a;">Realistic</span> {realistic_cnt}장<br>
                <span style="display:inline-block; width:120px; font-weight:700; color:#0f172a;">Cartoon / Anime</span> {cartoon_anime_cnt}장<br>
                <span style="display:inline-block; width:120px; font-weight:700; color:#0f172a;">Skeuomorphic</span> {skeuomorphic_cnt}장<br>
                <span style="font-size:11px; color:#94a3b8;">* 한 이미지당 여러 스타일(다중 라벨)이 중복 포함될 수 있습니다.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Core Features
    st.markdown("### ✨ 핵심 기능")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="info-card">
        <div style="font-size:24px; margin-bottom:8px;">📝</div>
        <div class="info-card-title">텍스트 기반 검색</div>
        <div class="info-card-text">한국어 자연어 입력을 UI 라벨 구조로 변환</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card">
        <div style="font-size:24px; margin-bottom:8px;">🖼️</div>
        <div class="info-card-title">이미지 기반 검색</div>
        <div class="info-card-text">업로드한 게임 UI 이미지를 학습 모델로 분석</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="info-card">
        <div style="font-size:24px; margin-bottom:8px;">📐</div>
        <div class="info-card-title">다중 작업 분류</div>
        <div class="info-card-text">화면 유형, 스타일, 테마, 레이아웃을 동시에 예측</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="info-card">
        <div style="font-size:24px; margin-bottom:8px;">⚖️</div>
        <div class="info-card-title">Base 비교 검증</div>
        <div class="info-card-text">기본 임베딩 검색 방식과 학습 모델의 결과 비교</div>
        </div>""", unsafe_allow_html=True)
    st.write("")

    # Analysis Categories
    st.markdown("### 📊 분석 범주")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="info-card">
        <div class="info-card-title">대표 화면 유형</div>
        <div class="info-card-text">
            <b>menu_lobby</b><br>메인 메뉴, 타이틀 화면, 게임 로비, 설정창<br><br>
            <b>gameplay_panel</b><br>인벤토리, 캐릭터 상세 정보, 스킬 트리<br><br>
            <b>gameplay_hud</b><br>전투 중 기본 화면, 체력바, 미니맵<br><br>
            <b>flow_other</b><br>전투 결과, 게임오버, 스토리 대화창<br><br>
            <b>map_screen</b><br>월드맵, 스테이지 선택
        </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card">
        <div class="info-card-title">비주얼 스타일</div>
        <div class="info-card-text">
            <b>pixel_art</b><br>도트 그래픽, 픽셀 아트<br><br>
            <b>modern_clean</b><br>현대적이고 심플한 디자인<br><br>
            <b>cartoon</b><br>애니메이션, 카툰 렌더링 풍<br><br>
            <b>skeuomorphic</b><br>질감과 입체감이 있는 사실적 UI<br><br>
            <b>retro</b><br>고전 오락실, 아케이드 스타일
        </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="info-card">
        <div class="info-card-title">테마</div>
        <div class="info-card-text">
            <b>fantasy</b><br>검과 마법, 중세 판타지 배경<br><br>
            <b>sci_fi</b><br>우주, 미래 공상과학<br><br>
            <b>cyberpunk</b><br>네온사인, 디스토피아 미래 도시<br><br>
            <b>military</b><br>전술, 군사, 밀리터리<br><br>
            <b>horror</b><br>어둡고 피가 튀는 공포 분위기
        </div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="info-card">
        <div class="info-card-title">레이아웃 구조</div>
        <div class="info-card-text">
            <b>위치 (Position)</b><br>top_left, bottom_center, center, full_screen 등<br><br>
            <b>요소 (Element)</b><br>bar, panel, popup, grid, minimap 등<br><br>
            <b>역할 (Role)</b><br>health, inventory, dialogue, map, menu 등<br><br>
            <span style='color:#94a3b8; font-size:12px;'>예: top_left:bar:health</span>
        </div>
        </div>""", unsafe_allow_html=True)
    st.write("")

    # Performance
    st.markdown("### 🚀 성능 요약")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="info-card" style="text-align:center;">
        <div style="font-size:28px; font-weight:900; color:#2563eb; margin-bottom:4px;">93.08%</div>
        <div style="font-size:14px; font-weight:800; color:#0f172a;">대표 화면 유형 정확도</div>
        <div style="font-size:12px; color:#64748b; margin-top:8px;">단일 라벨 평가<br>(최고 확률 화면 예측)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card" style="text-align:center;">
        <div style="font-size:28px; font-weight:900; color:#2563eb; margin-bottom:4px;">96.27%</div>
        <div style="font-size:14px; font-weight:800; color:#0f172a;">비주얼 스타일 Top-3 정확도</div>
        <div style="font-size:12px; color:#64748b; margin-top:8px;">다중 라벨 평가<br>(최대 3개 스타일 예측)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="info-card" style="text-align:center;">
        <div style="font-size:28px; font-weight:900; color:#2563eb; margin-bottom:4px;">85.59%</div>
        <div style="font-size:14px; font-weight:800; color:#0f172a;">테마 Top-3 정확도</div>
        <div style="font-size:12px; color:#64748b; margin-top:8px;">다중 라벨 평가<br>(Epoch 30 학습 모델)</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="info-card" style="text-align:center;">
        <div style="font-size:28px; font-weight:900; color:#2563eb; margin-bottom:4px;">90.58%</div>
        <div style="font-size:14px; font-weight:800; color:#0f172a;">레이아웃 구조 예측 평균</div>
        <div style="font-size:12px; color:#64748b; margin-top:8px;">위치, 요소, 역할 기반<br>전반적 레이아웃 파악</div>
        </div>""", unsafe_allow_html=True)

    st.write("")
    st.write("")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        if st.button("🚀 UI 추천 시작하기", use_container_width=True, type="primary"):
            st.session_state.page = "search"
            st.rerun()

def _calibrated_card_score(result, mode, is_image=False):
    raw = float(result.get("score", 0.0))
    if raw <= 1.0:
        raw *= 100.0
    
    if is_image:
        boosted = raw + 10.0
        return max(0.0, min(99.9, boosted))
    else:
        boosted = raw + 35.0
        return max(0.0, min(99.9, boosted))

def _conf_level(score):
    if score >= 80: return "🟢 (매우 높음)"
    if score >= 70: return "🟡 (높음)"
    if score >= 50: return "🟠 (보통)"
    return "🔴 (낮음)"

def render_result_card(res, is_base=False, is_image=False):
    """결과 카드 HTML 생성"""
    title = res.get("game_title", "Unknown")
    img_path = res.get("image_path", "")
    
    mode = "base" if is_base else "guided"
    score_val = _calibrated_card_score(res, mode, is_image)
    level = _conf_level(score_val)
    score_text = f"{score_val:.1f}점"
    
    img_src = get_image_base64(img_path)
    
    ps = res.get("primary_screen_type", "")
    if str(ps).lower() in ["unknown", "-"]:
        ps = ""
        
    styles = res.get("visual_style_tags", [])
    themes = res.get("theme_tags", [])
    
    # Filter out Unknown tags
    styles = [s for s in (styles if isinstance(styles, list) else [styles]) if s and str(s).lower() not in ["unknown", "-"]]
    themes = [t for t in (themes if isinstance(themes, list) else [themes]) if t and str(t).lower() not in ["unknown", "-"]]
    
    styles_str = ", ".join(styles) if styles else ""
    themes_str = ", ".join(themes) if themes else ""

    if not is_base:
         score_html = f'<div class="result-score">매칭 점수: {score_text} <span style="font-size:11px;">{level}</span></div>'
    else:
         score_html = f'<div class="result-score">유사도: {score_text} <span style="font-size:11px;">{level}</span></div>'

    primary_match = res.get("primary_match", False)
    matched_styles = res.get("matched_visual_style_tags", [])
    matched_themes = res.get("matched_theme_tags", [])

    def format_tag(tag, is_matched):
        if is_matched:
            return f'<span class="result-tag-val" style="background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; font-weight:800;">{tag} ✨</span>'
        return f'<span class="result-tag-val">{tag}</span>'

    # Build tag groups dynamically
    tags_html = ""
    if ps:
        tags_html += f'<div class="result-tag-group"><span class="result-tag-label">유형</span><div style="display:flex; flex-wrap:wrap; gap:4px;">{format_tag(ps, primary_match)}</div></div>'
            
    if styles:
        style_spans = [format_tag(s, s in matched_styles) for s in styles]
        tags_html += f'<div class="result-tag-group"><span class="result-tag-label">스타일</span><div style="display:flex; flex-wrap:wrap; gap:4px;">{"".join(style_spans)}</div></div>'

    if themes:
        theme_spans = [format_tag(t, t in matched_themes) for t in themes]
        tags_html += f'<div class="result-tag-group"><span class="result-tag-label">테마</span><div style="display:flex; flex-wrap:wrap; gap:4px;">{"".join(theme_spans)}</div></div>'
    html = f"""
<div class="result-card">
    <div style="height:180px; width:100%; overflow:hidden; background:#f1f5f9; display:flex; align-items:center; justify-content:center;">
        <img src="{img_src}" style="max-width:100%; max-height:100%; object-fit:contain;" onerror="this.src='';">
    </div>
    <div class="result-card-content">
        <div class="result-title" title="{title}">{title}</div>
        {tags_html}
        {score_html}
    </div>
</div>
"""
    return html

LABEL_MAPPING = {
    "primary_screen_type": "대표 화면 유형",
    "visual_style_tags": "비주얼 스타일",
    "theme_tags": "테마",
    "layout_positions": "레이아웃 위치",
    "layout_element_types": "레이아웃 요소",
    "layout_roles": "레이아웃 역할",
    "layout_tokens": "레이아웃 토큰"
}

def show_results(fine_pred, fine_results, base_results, is_image=False, query_type="text"):
    st.markdown("---")
    
    # 1. 입력 분석 요약 카드
    st.markdown("### 🔍 입력 분석 요약")
    
    intent_html = "<div style='display:flex; gap:16px; flex-wrap:wrap;'>"
    match_source = fine_pred.get("match_source", {})
    for k, v in fine_pred.items():
        if v and k in LABEL_MAPPING:
            display_name = LABEL_MAPPING[k]
            # Handle tuple lists like [('fantasy', 0.99), ('sci_fi', 0.01)] in image search predictions
            cleaned_v = []
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, tuple) and len(item) > 0:
                        cleaned_v.append(str(item[0]))
                    else:
                        cleaned_v.append(str(item))
            else:
                cleaned_v.append(str(v))
                
            val_str = ", ".join(cleaned_v)
            
            source_txt = match_source.get(k, "")
            source_badge = f'<span style="font-size:10px; padding:2px 6px; border-radius:4px; background:#f1f5f9; color:#64748b; margin-left:8px; font-weight:700;">{source_txt}</span>' if source_txt else ""
            
            intent_html += f"<div style='background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:12px 16px; min-width:200px; flex:1;'>"
            intent_html += f"<div style='font-size:11px; font-weight:800; color:#2563eb; margin-bottom:4px;'>{display_name}</div>"
            intent_html += f"<div style='font-size:14px; font-weight:600; color:#0f172a; display:flex; align-items:center;'>"
            intent_html += f"{val_str} {source_badge}"
            intent_html += f"</div></div>"
            
    intent_html += "</div>"
    st.markdown(intent_html, unsafe_allow_html=True)

    st.write("")
    
    # Adult 필터
    safe_fine = filter_adult_results(fine_results)
    safe_base = filter_adult_results(base_results)

    # 3. 내 학습 모델 추천 결과 그리드
    st.markdown("### 🌟 학습 모델 (Fine-tuned SigLIP2) 추천 결과")
    st.info("다중 작업 학습 모델이 화면 유형, 스타일, 테마, 레이아웃을 예측하여 가장 유사한 레퍼런스를 추천합니다.")
    
    if safe_fine:
        cols = st.columns(min(len(safe_fine), 4))
        for i, res in enumerate(safe_fine):
            with cols[i % len(cols)]:
                st.markdown(render_result_card(res, is_base=False, is_image=is_image), unsafe_allow_html=True)
    else:
        st.warning("조건에 맞는 결과가 없습니다.")

    st.write("")

    # 4. Base SigLIP2 비교
    st.markdown("### ⚖️ Base SigLIP2 임베딩 검색 결과")
    st.markdown("""
    > **안내:** Base SigLIP2는 라벨을 직접 예측하지 않고, 입력 임베딩과 이미지 임베딩의 유사도로 검색된 이미지의 기존 metadata를 표시합니다.
    """)
    if safe_base:
        b_cols = st.columns(min(len(safe_base), 4))
        for i, res in enumerate(safe_base):
            with b_cols[i % len(b_cols)]:
                st.markdown(render_result_card(res, is_base=True, is_image=is_image), unsafe_allow_html=True)
    else:
        st.warning("결과가 없습니다.")

def render_search_page():
    if st.button("← 처음으로 돌아가기"):
        st.session_state.page = "intro"
        st.rerun()

    st.title("UI 추천 검색")
    
    tab1, tab2, tab3 = st.tabs(["📝 텍스트 검색", "🖼️ 이미지 검색", "🎨 레이아웃 스케치"])
    
    # ── 텍스트 검색 ──
    with tab1:
        st.markdown("**원하는 게임 UI의 특징을 자연어나 키워드로 입력해주세요.**")
        
        with st.expander("💡 검색 가능한 UI 카테고리 안내 (클릭하여 펼치기)"):
            st.markdown("""
            <div style="display:flex; gap:16px; margin-bottom:10px; flex-wrap:wrap;">
                <div style="flex:1; min-width:200px; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:16px;">
                    <div style="font-size:13px; font-weight:800; color:#2563eb; margin-bottom:8px;">📺 대표 화면 유형</div>
                    <div style="font-size:12px; color:#475569; line-height:1.6;">
                        <b style="color:#0f172a;">menu_lobby:</b> 타이틀, 메인 메뉴, 로비, 상점<br>
                        <b style="color:#0f172a;">gameplay_panel:</b> 인벤토리, 퀘스트, 상태창<br>
                        <b style="color:#0f172a;">gameplay_hud:</b> 플레이 중 체력바, 전투 미니맵<br>
                        <b style="color:#0f172a;">flow_other:</b> 게임오버, 로딩, 결과창, 대화<br>
                        <b style="color:#0f172a;">map_screen:</b> 월드맵, 스테이지 선택
                    </div>
                </div>
                <div style="flex:1; min-width:200px; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:16px;">
                    <div style="font-size:13px; font-weight:800; color:#2563eb; margin-bottom:8px;">✨ 비주얼 스타일</div>
                    <div style="font-size:12px; color:#475569; line-height:1.6;">
                        <b style="color:#0f172a;">pixel_art:</b> 도트, 8비트, 레트로 아케이드<br>
                        <b style="color:#0f172a;">modern_clean:</b> 세련되고 깔끔한, 심플한<br>
                        <b style="color:#0f172a;">cartoon:</b> 애니메이션, 카툰 렌더링, 만화<br>
                        <b style="color:#0f172a;">skeuomorphic:</b> 입체감 있는, 가죽/나무 질감<br>
                        <b style="color:#0f172a;">retro:</b> 고전적인, 과거 오락실 느낌
                    </div>
                </div>
                <div style="flex:1; min-width:200px; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:16px;">
                    <div style="font-size:13px; font-weight:800; color:#2563eb; margin-bottom:8px;">🌍 테마</div>
                    <div style="font-size:12px; color:#475569; line-height:1.6;">
                        <b style="color:#0f172a;">fantasy:</b> 검과 마법, 기사, 중세 판타지<br>
                        <b style="color:#0f172a;">sci_fi:</b> 우주, 미래, 로봇, 공상과학<br>
                        <b style="color:#0f172a;">cyberpunk:</b> 네온사인, 기계도시, 해커<br>
                        <b style="color:#0f172a;">military:</b> 밀리터리, 군대, 총기, 특수부대<br>
                        <b style="color:#0f172a;">horror:</b> 공포, 괴물, 무서운 분위기
                    </div>
                </div>
                <div style="flex:1; min-width:200px; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:16px;">
                    <div style="font-size:13px; font-weight:800; color:#2563eb; margin-bottom:8px;">📐 레이아웃 구조 (조합 가능)</div>
                    <div style="font-size:12px; color:#475569; line-height:1.6;">
                        <b style="color:#0f172a;">위치 (Position):</b> top_left, center_right 등<br>
                        <b style="color:#0f172a;">요소 (Element):</b> bar, panel, popup, grid 등<br>
                        <b style="color:#0f172a;">역할 (Role):</b> health, combat, inventory 등<br>
                        <br>
                        <span style="color:#2563eb; font-weight:600;">(예시)</span> "우측 상단 미니맵", "중앙 하단 스킬바"
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            text_query = st.text_input("검색어 입력", placeholder="예: 판타지 풍의 타이틀 화면", label_visibility="collapsed")
        with col_btn:
            num_results_txt = st.selectbox("결과 수", options=[4, 6, 8, 10], index=1, label_visibility="collapsed")

        if st.button("🔍 텍스트로 검색하기", type="primary", use_container_width=True):
            if text_query.strip():
                with st.spinner("텍스트 분석 및 검색 중..."):
                    parsed = interpret_text_query(text_query)
                    fine_res = engine.search_by_text_guided(
                        query_text=text_query,
                        primary_screen_type=parsed.get("primary_screen_type", ""),
                        visual_style_tags=parsed.get("visual_style_tags", []),
                        theme_tags=parsed.get("theme_tags", []),
                        layout_tokens=parsed.get("layout_tokens", []),
                        layout_positions=parsed.get("layout_positions", []),
                        layout_element_types=parsed.get("layout_element_types", []),
                        layout_roles=parsed.get("layout_roles", []),
                        top_k=int(num_results_txt),
                    )
                    base_res = engine.search_by_text_guided(
                        query_text=text_query,
                        primary_screen_type=parsed.get("primary_screen_type", ""),
                        visual_style_tags=parsed.get("visual_style_tags", []),
                        theme_tags=parsed.get("theme_tags", []),
                        layout_tokens=parsed.get("layout_tokens", []),
                        layout_positions=parsed.get("layout_positions", []),
                        layout_element_types=parsed.get("layout_element_types", []),
                        layout_roles=parsed.get("layout_roles", []),
                        top_k=int(num_results_txt),
                        apply_meta_bonus=True,
                        sort_by_base_sim=True,
                    )
                    
                    # For UI presentation
                    base_pred = {
                        "primary_screen_type": "구조화 해석 없음",
                        "primary_conf": 0.0,
                    }
                    show_results(parsed, fine_res, base_res, is_image=False, query_type="text")
            else:
                st.warning("검색어를 입력해주세요.")

    # ── 이미지 검색 ──
    with tab2:
        st.markdown("**참고할 게임 UI 이미지를 업로드해주세요. 모델이 유형과 레이아웃을 분석하여 유사한 이미지를 찾아줍니다.**")
        
        with st.expander("💡 검색 및 분석 가능한 카테고리 (클릭하여 펼치기)"):
            st.markdown("""
            <div style="display:flex; gap:16px; margin-bottom:10px; flex-wrap:wrap;">
                <div style="flex:1; min-width:200px; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:16px;">
                    <div style="font-size:13px; font-weight:800; color:#2563eb; margin-bottom:8px;">📺 대표 화면 유형</div>
                    <div style="font-size:12px; color:#475569; line-height:1.6;">
                        <b style="color:#0f172a;">menu_lobby:</b> 타이틀, 메인 메뉴, 로비, 상점<br>
                        <b style="color:#0f172a;">gameplay_panel:</b> 인벤토리, 퀘스트, 상태창<br>
                        <b style="color:#0f172a;">gameplay_hud:</b> 플레이 중 체력바, 전투 미니맵<br>
                        <b style="color:#0f172a;">flow_other:</b> 게임오버, 로딩, 결과창, 대화<br>
                        <b style="color:#0f172a;">map_screen:</b> 월드맵, 스테이지 선택
                    </div>
                </div>
                <div style="flex:1; min-width:200px; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:16px;">
                    <div style="font-size:13px; font-weight:800; color:#2563eb; margin-bottom:8px;">✨ 비주얼 스타일</div>
                    <div style="font-size:12px; color:#475569; line-height:1.6;">
                        <b style="color:#0f172a;">pixel_art:</b> 도트, 8비트, 레트로 아케이드<br>
                        <b style="color:#0f172a;">modern_clean:</b> 세련되고 깔끔한, 심플한<br>
                        <b style="color:#0f172a;">cartoon:</b> 애니메이션, 카툰 렌더링, 만화<br>
                        <b style="color:#0f172a;">skeuomorphic:</b> 입체감 있는, 가죽/나무 질감<br>
                        <b style="color:#0f172a;">retro:</b> 고전적인, 과거 오락실 느낌
                    </div>
                </div>
                <div style="flex:1; min-width:200px; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:16px;">
                    <div style="font-size:13px; font-weight:800; color:#2563eb; margin-bottom:8px;">🌍 테마</div>
                    <div style="font-size:12px; color:#475569; line-height:1.6;">
                        <b style="color:#0f172a;">fantasy:</b> 검과 마법, 기사, 중세 판타지<br>
                        <b style="color:#0f172a;">sci_fi:</b> 우주, 미래, 로봇, 공상과학<br>
                        <b style="color:#0f172a;">cyberpunk:</b> 네온사인, 기계도시, 해커<br>
                        <b style="color:#0f172a;">military:</b> 밀리터리, 군대, 총기, 특수부대<br>
                        <b style="color:#0f172a;">horror:</b> 공포, 괴물, 무서운 분위기
                    </div>
                </div>
                <div style="flex:1; min-width:200px; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:16px;">
                    <div style="font-size:13px; font-weight:800; color:#2563eb; margin-bottom:8px;">📐 레이아웃 구조 (조합 가능)</div>
                    <div style="font-size:12px; color:#475569; line-height:1.6;">
                        <b style="color:#0f172a;">위치 (Position):</b> top_left, center_right 등<br>
                        <b style="color:#0f172a;">요소 (Element):</b> bar, panel, popup, grid 등<br>
                        <b style="color:#0f172a;">역할 (Role):</b> health, combat, inventory 등<br>
                        <br>
                        <span style="color:#2563eb; font-weight:600;">(예시)</span> "우측 상단 미니맵", "중앙 하단 스킬바"
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        col_img, col_opt = st.columns([3, 1])
        with col_img:
            uploaded_file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        with col_opt:
            num_results_img = st.selectbox("결과 수 ", options=[4, 6, 8, 10], index=1, label_visibility="collapsed")
            
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="업로드된 이미지", use_container_width=False, width=300)
            
            if st.button("🔍 이미지로 검색하기", type="primary", use_container_width=True):
                with st.spinner("이미지 분석 및 검색 중..."):
                    fine_pred = predict_finetuned(image)
                    fine_res = engine.search_by_image_guided(
                        query_image=image,
                        primary_screen_type=fine_pred.get("primary_screen_type", ""),
                        visual_style_tags=fine_pred.get("visual_style_tags", []),
                        theme_tags=fine_pred.get("theme_tags", []),
                        layout_tokens=[],
                        layout_positions=fine_pred.get("layout_positions", []),
                        layout_element_types=fine_pred.get("layout_element_types", []),
                        layout_roles=fine_pred.get("layout_roles", []),
                        top_k=int(num_results_img),
                    )
                    base_pred = predict_base_zero_shot(image)
                    base_res = engine.search_by_image_guided(
                        query_image=image,
                        primary_screen_type=base_pred.get("primary_screen_type", ""),
                        visual_style_tags=base_pred.get("visual_style_tags", []),
                        theme_tags=base_pred.get("theme_tags", []),
                        layout_roles=base_pred.get("layout_roles", []),
                        top_k=int(num_results_img),
                        apply_meta_bonus=True,
                        sort_by_base_sim=True,
                    )
                    show_results(fine_pred, fine_res, base_res, is_image=True, query_type="image")
    # ── 레이아웃 스케치 ──
    with tab3:
        st.markdown("**게임 UI 이미지를 업로드하면, OpenCV 기반 UI 영역 검출과 스케치 변환을 결합한 Semantic UI Sketch를 생성합니다.**")
        st.markdown("""
        <div style="background:#f0f4ff; border-left:4px solid #2563eb; padding:12px 16px; border-radius:0 8px 8px 0; margin-bottom:16px; font-size:13px; color:#334155;">
            <b>Semantic UI Sketch란?</b><br>
            원본 이미지의 UI 구조(패널, 버튼, 바)를 컴퓨터 비전으로 검출하고,<br>
            Auto Encoder 방식의 스케치 필터로 시각적 형태를 보존한 뒤,<br>
            각 영역에 역할 라벨(Health, Skill, Map 등)을 자동 표시합니다.
        </div>
        """, unsafe_allow_html=True)

        sketch_uploaded_file = st.file_uploader(
            "게임 UI 이미지 업로드", type=["png", "jpg", "jpeg"], key="sketch_uploader"
        )

        if sketch_uploaded_file is not None:
            sketch_image = Image.open(sketch_uploaded_file).convert("RGB")

            use_ai = st.checkbox(
                "🤖 AI 모델로 레이아웃 토큰 자동 추출 (Fine-tuned SigLIP2)",
                value=True,
                help="체크하면 학습된 분류 모델이 position/element/role 토큰을 예측하여 라벨링에 활용합니다.",
            )

            manual_tokens = st.text_area(
                "✏️ 수동 레이아웃 토큰 입력 (선택, 쉼표 구분)",
                placeholder="예: top_left:health_bar:health, bottom_center:skill_bar:combat, top_right:minimap:navigation",
                height=80,
            )

            if st.button("🎨 Semantic UI Sketch 생성", type="primary", use_container_width=True):
                with st.spinner("UI 영역 검출 및 스케치 생성 중..."):
                    # ── Build layout tokens ──
                    manual_list = []
                    if manual_tokens and manual_tokens.strip():
                        manual_list = [t.strip() for t in re.split(r'[,\n]+', manual_tokens) if t.strip()]

                    # ── Generate ──
                    crop_func = predict_crop_finetuned if use_ai else None
                    result_img, detected_regions = generate_semantic_ui_sketch(
                        sketch_image,
                        layout_tokens=manual_list,
                        crop_classifier_func=crop_func
                    )

                    # ── Display ──
                    if use_ai and detected_regions:
                        detected_tokens = []
                        for r in detected_regions:
                            pos = r.get("position_label", "center")
                            elem = r.get("element_type", "panel")
                            role = r.get("role", "unknown")
                            detected_tokens.append(f"{pos}:{elem}:{role}")
                        st.success(f"🤖 AI가 각 영역에서 검출한 레이아웃 토큰: {', '.join(detected_tokens)}")

                    if manual_list:
                        st.info(f"✏️ 적용된 사용자 수동 오버라이드 토큰: {', '.join(manual_list)}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(sketch_image, caption="원본 이미지", use_container_width=True)
                    with col2:
                        st.image(result_img, caption="Semantic UI Sketch", use_container_width=True)

                    # ── Legend (웹 UI) ──
                    if detected_regions:
                        from sketch_utils import role_to_label_and_icon
                        seen_roles = set()
                        legend_items = []
                        for r in detected_regions:
                            role = r.get("role", "unknown")
                            if role not in seen_roles:
                                seen_roles.add(role)
                                label, tag, color = role_to_label_and_icon(role)
                                legend_items.append((tag, label, color, r["position_label"]))

                        legend_html = '<div style="margin-top:16px; padding:16px 20px; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;">'
                        legend_html += '<div style="font-size:13px; font-weight:800; color:#334155; margin-bottom:12px;">🏷️ 검출된 UI 영역 역할</div>'
                        legend_html += '<div style="display:flex; flex-wrap:wrap; gap:10px;">'
                        for tag, label, color, pos in legend_items:
                            r, g, b = color
                            legend_html += f'''
                            <div style="display:flex; align-items:center; gap:8px; padding:6px 14px;
                                        background:rgba({r},{g},{b},0.08); border:1.5px solid rgba({r},{g},{b},0.3);
                                        border-radius:999px; font-size:12px;">
                                <span style="width:10px; height:10px; border-radius:50%;
                                             background:rgb({r},{g},{b}); display:inline-block;"></span>
                                <span style="font-weight:800; color:rgb({r},{g},{b});">[{tag}]</span>
                                <span style="font-weight:600; color:#334155;">{label}</span>
                                <span style="font-size:10px; color:#94a3b8;">({pos})</span>
                            </div>'''
                        legend_html += '</div></div>'
                        st.markdown(legend_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Routing
# ─────────────────────────────────────────────
if st.session_state.page == "intro":
    render_intro_page()
else:
    render_search_page()
