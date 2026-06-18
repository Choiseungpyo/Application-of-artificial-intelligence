# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, TypedDict, Callable
import traceback

# LangGraph가 없는 환경에서도 import 에러가 나지 않도록 try-except 처리
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_INSTALLED = True
except ImportError:
    StateGraph = None
    START = None
    END = None
    LANGGRAPH_INSTALLED = False

class GameUILabelingState(TypedDict, total=False):
    """LangGraph 기반 게임 UI 라벨링 상태 정의"""
    image: Any
    row_info: Dict[str, Any]
    raw_result: Optional[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    attempts: int
    max_attempts: int
    workflow_log: List[str]

# ======================== Graph Nodes ========================

def prepare_input(state: GameUILabelingState) -> GameUILabelingState:
    """분석 시작 전 입력을 준비하고 정규화합니다."""
    print("[LANGGRAPH] Node: prepare_input")
    workflow_log = state.get("workflow_log", [])
    
    if state.get("image") is None:
        return {**state, "error": "Image is missing", "workflow_log": workflow_log + ["Error: Missing Image"]}
    
    row_info = state.get("row_info") or {}
    return {
        **state,
        "row_info": row_info,
        "attempts": state.get("attempts", 0),
        "max_attempts": state.get("max_attempts", 1),
        "workflow_log": workflow_log + ["Input prepared"]
    }

def analyze_with_gpt(state: GameUILabelingState, analyzer_func: Callable) -> GameUILabelingState:
    """주입된 분석 함수(GPT)를 사용하여 이미지를 분석합니다."""
    print("[LANGGRAPH] Node: analyze_with_gpt")
    if state.get("error"):
        return state

    try:
        # labeling_tool.py에서 전달받은 get_ai_analysis_direct 호출
        result = analyzer_func(state["image"], state["row_info"])
        
        if result is None:
            return {**state, "error": "GPT analysis returned None", "workflow_log": state.get("workflow_log", []) + ["GPT returned None"]}
        
        return {
            **state,
            "result": result,
            "raw_result": result,
            "workflow_log": state.get("workflow_log", []) + ["GPT analysis completed"]
        }
    except Exception as e:
        return {
            **state, 
            "error": f"GPT execution error: {str(e)}", 
            "workflow_log": state.get("workflow_log", []) + [f"Error: {str(e)}"]
        }

def validate_result(state: GameUILabelingState) -> GameUILabelingState:
    """분석 결과의 품질을 검증하고 필요한 경우 검토 플래그를 설정합니다."""
    print("[LANGGRAPH] Node: validate_result")
    result = state.get("result")
    if not result or state.get("error"):
        return state

    # 검증 로직 (AX 워크플로우의 핵심인 데이터 품질 관리)
    needs_review = result.get("needs_review", False)
    confidence = result.get("confidence", 0.0)
    primary = result.get("primary_screen_type", "other")
    components = result.get("components", [])

    reasons = []
    if confidence < 0.70:
        needs_review = True
        reasons.append("Low confidence")
    if primary == "other":
        needs_review = True
        reasons.append("Generic screen type (other)")
    if not components or len(components) == 0:
        needs_review = True
        reasons.append("Empty components")

    if reasons:
        result["needs_review"] = True
        print(f"[LANGGRAPH] Validation Flagged: {', '.join(reasons)}")
    
    return {
        **state,
        "result": result,
        "workflow_log": state.get("workflow_log", []) + [f"Validation finished. Needs review: {needs_review}"]
    }

def decide_review_status(state: GameUILabelingState) -> GameUILabelingState:
    """최종 상태를 결정합니다."""
    print("[LANGGRAPH] Node: decide_review_status")
    return {**state, "workflow_log": state.get("workflow_log", []) + ["Workflow finished"]}

# ======================== Graph Builder ========================

def build_labeling_graph(analyzer_func: Callable):
    """LangGraph 워크플로우를 구성합니다."""
    if not LANGGRAPH_INSTALLED:
        return None
        
    workflow = StateGraph(GameUILabelingState)

    # 노드 추가
    workflow.add_node("prepare_input", prepare_input)
    workflow.add_node("analyze_with_gpt", lambda x: analyze_with_gpt(x, analyzer_func))
    workflow.add_node("validate_result", validate_result)
    workflow.add_node("decide_review_status", decide_review_status)

    # 엣지 연결
    workflow.add_edge(START, "prepare_input")
    workflow.add_edge("prepare_input", "analyze_with_gpt")
    workflow.add_edge("analyze_with_gpt", "validate_result")
    workflow.add_edge("validate_result", "decide_review_status")
    workflow.add_edge("decide_review_status", END)

    return workflow.compile()

def run_langgraph_analysis(image, row_info, analyzer_func, normalizer_func=None):
    """외부(labeling_tool.py)에서 호출하는 진입점입니다."""
    if not LANGGRAPH_INSTALLED:
        print("[LANGGRAPH] Library not installed. Fallback to direct.")
        return None

    print("[LANGGRAPH] Game UI labeling workflow started")
    try:
        app = build_labeling_graph(analyzer_func)
        if app is None:
            return None
            
        initial_state = {
            "image": image,
            "row_info": row_info or {},
            "attempts": 0,
            "max_attempts": 1,
            "workflow_log": []
        }
        
        final_state = app.invoke(initial_state)
        print("[LANGGRAPH] Workflow finished successfully")
        return final_state.get("result")
    except Exception as e:
        print(f"[LANGGRAPH] Workflow execution failed: {e}")
        traceback.print_exc()
        return None
