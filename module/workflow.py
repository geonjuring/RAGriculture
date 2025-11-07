"""
워크플로우 구성 모듈
LangGraph 워크플로우 구성 및 분기 함수
"""
from langgraph.graph import END, StateGraph, START
from .models import GraphState
from .config import debug_print
from .nodes import (
    retrieval_node,
    augmentation_node,
    generation_node,
    quality_check_node,
    llm_judge_validation_node,
    route_question,
    retrieve,
    check_question_validity
)


def decide_validity(state: GraphState) -> str:
    """질문 유효성에 따른 라우팅 결정"""
    is_valid = state.get("question_valid", True)
    if is_valid:
        return "route_question"
    else:
        return END


def decide_route(state: GraphState) -> str:
    """라우팅 결정을 위한 조건부 엣지 함수"""
    route = state.get("route", "web_search")  # 원래 기본값 유지
    
    debug_print(f"Route decision: {route}")
    
    # 하이브리드 검색 제거 - 단순 라우팅만 수행
    if route == "vectorstore":
        return "retrieve"
    else:
        return "web_search"


def decide_quality(state: GraphState) -> str:
    """품질 점수에 따른 재처리 결정"""
    quality_scores = state.get("quality_scores", {})
    overall_score = quality_scores.get("overall_score", 0.0)
    retry_count = state.get("retry_count", 0)
    
    if overall_score < 0.7 and retry_count < 2:
        debug_print(f"Quality too low ({overall_score:.3f}), retrying...")
        return "transform_query_node"
    else:
        return END


def decide_final_quality(state: GraphState) -> str:
    """최종 품질 결정 - LLM 판단 직접 사용 (임계값 제거)"""
    llm_judge_scores = state.get("llm_judge_scores", {})
    retry_count = state.get("retry_count", 0)
    
    MAX_RETRIES = 3
    if retry_count >= MAX_RETRIES:
        debug_print(f"최대 재시도 횟수 도달 ({retry_count}/{MAX_RETRIES})")
        return END
    
    # LLM의 직접 판단 사용
    should_output = llm_judge_scores.get("should_output", False)
    needs_correction = llm_judge_scores.get("needs_correction", True)
    reasoning = llm_judge_scores.get("reasoning", "")
    
    debug_print(f"🤖 LLM 판단: should_output={should_output}, needs_correction={needs_correction}")
    if reasoning:
        debug_print(f"💭 판단 근거: {reasoning[:200]}...")
    
    # LLM이 출력 가능하다고 판단하면 종료
    if should_output and not needs_correction:
        debug_print("✅ LLM 판단: 출력 가능 - 워크플로우 종료")
        return END
    
    # LLM이 수정이 필요하다고 판단하면 재검색
    if needs_correction or not should_output:
        correction_suggestions = llm_judge_scores.get("correction_suggestions", "")
        if correction_suggestions:
            debug_print(f"📝 수정 제안: {correction_suggestions[:200]}...")
        debug_print("❌ LLM 판단: 수정 필요 - 재검색")
        return "transform_query_node"  # 재검색 루프로 이동
    
    # 기본값: 재검색
    debug_print("❌ LLM 판단: 출력 불가 - 재검색")
    return "transform_query_node"


def create_workflow(nodes_dict: dict):
    """워크플로우 그래프 생성"""
    workflow = StateGraph(GraphState)
    
    # 노드 추가
    if "check_validity" in nodes_dict:
        workflow.add_node("check_validity", nodes_dict["check_validity"])
    if "route_question" in nodes_dict:
        workflow.add_node("route_question", nodes_dict["route_question"])
    if "retrieve" in nodes_dict:
        workflow.add_node("retrieve", nodes_dict["retrieve"])
    if "web_search" in nodes_dict:
        workflow.add_node("web_search", nodes_dict["web_search"])
    if "retrieval_node" in nodes_dict:
        workflow.add_node("retrieval_node", nodes_dict["retrieval_node"])
    if "augmentation_node" in nodes_dict:
        workflow.add_node("augmentation_node", nodes_dict["augmentation_node"])
    if "generation_node" in nodes_dict:
        workflow.add_node("generation_node", nodes_dict["generation_node"])
    if "quality_check_node" in nodes_dict:
        workflow.add_node("quality_check_node", nodes_dict["quality_check_node"])
    if "llm_judge_validation" in nodes_dict:
        workflow.add_node("llm_judge_validation", nodes_dict["llm_judge_validation"])
    
    # 워크플로우 연결
    workflow.add_edge(START, "check_validity")
    
    workflow.add_conditional_edges(
        "check_validity",
        decide_validity,
        {
            "route_question": "route_question",
            END: END
        }
    )
    
    # decide_route 함수가 반환할 수 있는 값들만 포함
    route_mapping = {
        "retrieve": "retrieve"
    }
    
    # web_search 노드가 있는 경우에만 추가
    if "web_search" in nodes_dict:
        route_mapping["web_search"] = "web_search"
    
    # decide_route 함수를 래핑하여 존재하지 않는 노드에 대한 fallback 처리
    # route_mapping을 먼저 생성한 후 래핑 함수 내부에서 사용
    def safe_decide_route(state: GraphState) -> str:
        """decide_route를 래핑하여 존재하지 않는 노드는 retrieve로 fallback"""
        route = decide_route(state)
        # route_mapping에 없는 값인 경우 retrieve로 fallback
        if route not in route_mapping:
            debug_print(f"⚠️ {route} 노드가 없어 retrieve로 fallback")
            return "retrieve"
        return route
    
    workflow.add_conditional_edges(
        "route_question",
        safe_decide_route,  # 래핑된 함수 사용
        route_mapping
    )
    
    # 검색 → 복잡도 평가 → RAG 핵심 검색 (transform_query_node는 피드백 루프에서만 사용)
    if "assess_complexity_node" in nodes_dict:
        workflow.add_node("assess_complexity_node", nodes_dict["assess_complexity_node"])
        workflow.add_edge("retrieve", "assess_complexity_node")
        # 노드가 있는 경우에만 엣지 추가
        if "web_search" in nodes_dict:
            workflow.add_edge("web_search", "assess_complexity_node")
        
        # assess_complexity_node에서 retrieval_node로 직접 연결 (transform_query_node 건너뛰기)
        workflow.add_edge("assess_complexity_node", "retrieval_node")
        
        # transform_query_node는 피드백 루프에서만 사용 (원래 검색 경로로 재검색)
        if "transform_query_node" in nodes_dict:
            workflow.add_node("transform_query_node", nodes_dict["transform_query_node"])
            # transform_query_node 후 원래 검색 경로로 재검색
            workflow.add_conditional_edges(
                "transform_query_node",
                decide_route,  # 원래 검색 경로 확인
                {
                    "retrieve": "retrieve",
                    "web_search": "web_search" if "web_search" in nodes_dict else "retrieve"
                }
            )
    else:
        workflow.add_edge("retrieve", "retrieval_node")
        # 노드가 있는 경우에만 엣지 추가
        if "web_search" in nodes_dict:
            workflow.add_edge("web_search", "retrieval_node")
        
        # transform_query_node는 피드백 루프에서만 사용 (원래 검색 경로로 재검색)
        if "transform_query_node" in nodes_dict:
            workflow.add_node("transform_query_node", nodes_dict["transform_query_node"])
            # transform_query_node 후 원래 검색 경로로 재검색
            workflow.add_conditional_edges(
                "transform_query_node",
                decide_route,  # 원래 검색 경로 확인
                {
                    "retrieve": "retrieve",
                    "web_search": "web_search" if "web_search" in nodes_dict else "retrieve"
                }
            )
    
    # RAG 핵심 흐름
    workflow.add_edge("retrieval_node", "augmentation_node")
    workflow.add_edge("augmentation_node", "generation_node")
    workflow.add_edge("generation_node", "quality_check_node")
    workflow.add_edge("quality_check_node", "llm_judge_validation")
    
    # LLM Judge 검증에 따른 최종 분기
    workflow.add_conditional_edges(
        "llm_judge_validation",
        decide_final_quality,
        {
            "transform_query_node": "transform_query_node" if "transform_query_node" in nodes_dict else "retrieval_node",
            END: END
        }
    )
    
    return workflow.compile()

