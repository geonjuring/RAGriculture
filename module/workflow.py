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
    route_question,
    retrieve,
    check_question_validity,
    analyze_image,
)


def decide_validity(state: GraphState) -> str:
    """질문 유효성에 따른 라우팅 결정"""
    is_valid = state.get("question_valid", True)
    if is_valid:
        return "route_question"
    else:
        # 유효성 검사 실패 시 web_search로 라우팅
        debug_print("⚠️ 질문 유효성 검사 실패, web_search로 라우팅")
        return "web_search"


def has_image(state: GraphState) -> bool:
    """이미지 존재 여부 확인"""
    image = state.get("image", "")
    return image and image != "" and image is not None


def decide_start_route(state: GraphState) -> str:
    """시작 시 이미지 존재 여부에 따른 라우팅 결정"""
    if has_image(state):
        debug_print("이미지를 확인했습니다. analyze_image")
        return "analyze_image"
    else:
        debug_print("이미지가 없습니다. check_validity")
        return "check_validity"


def decide_route(state: GraphState) -> str:
    """라우팅 결정을 위한 조건부 엣지 함수"""
    route = state.get("route", "web_search")  # 원래 기본값 유지
    
    debug_print(f"Route decision: {route}")

    # 하이브리드 검색 제거 - 단순 라우팅만 수행
    if route == "vectorstore":
        return "retrieve"
    else:
        return "web_search"


def decide_document_availability(state: GraphState) -> str:
    """문서 존재 여부에 따른 라우팅 결정"""
    documents = state.get("retrieved_docs", state.get("documents", []))
    route = state.get("route", "vectorstore")
    
    # 문서가 없고 원래 경로가 vectorstore면 web_search로 전환
    if not documents and route == "vectorstore":
        debug_print("⚠️ 벡터스토어에서 문서를 찾지 못함, web_search로 전환")
        return "web_search"
    
    # 문서가 있으면 계속 진행
    return "generation_node"


def decide_retry_route(state: GraphState) -> str:
    """재검색 시 경로 결정 (재시도 횟수에 따라 경로 전환)"""
    retry_count = state.get("retry_count", 0)
    route = state.get("route", "vectorstore")
    
    # 재시도 2회 이상이고 벡터스토어였으면 web_search로 전환
    if retry_count >= 2 and route == "vectorstore":
        debug_print(f"⚠️ 재시도 {retry_count}회, 벡터스토어에서 웹검색으로 전환")
        return "web_search"
    
    if route == "vectorstore":
        return "retrieve"

    # 원래 경로 유지
    return route


def decide_final_quality(state: GraphState) -> str:
    """최종 품질 결정 - LLM 판단 직접 사용 (임계값 제거)"""
    llm_judge_scores = state.get("llm_judge_scores", {})
    retry_count = state.get("retry_count", 0)
    route = state.get("route", "vectorstore")
    documents = state.get("retrieved_docs", state.get("documents", []))
    
    MAX_RETRIES = 3
    if retry_count >= MAX_RETRIES:
        debug_print(f"최대 재시도 횟수 도달 ({retry_count}/{MAX_RETRIES})")
        return END
    
    # 문서가 없고 벡터스토어였으면 web_search로 전환
    if not documents and route == "vectorstore" and retry_count >= 1:
        debug_print("⚠️ 문서 없음 + 재시도 중, web_search로 전환")
        return "web_search"
    
    # LLM의 직접 판단 사용
    should_output = llm_judge_scores.get("should_output", False)
    needs_correction = llm_judge_scores.get("needs_correction", True)
    insufficient_information = llm_judge_scores.get("insufficient_information", False)
    reasoning = llm_judge_scores.get("reasoning", "")
    
    debug_print(f"🤖 LLM 판단: should_output={should_output}, needs_correction={needs_correction}, insufficient_info={insufficient_information}")
    if reasoning:
        debug_print(f"💭 판단 근거: {reasoning[:200]}...")
    
    # 정보 부족으로 인한 수정 필요 시 웹 검색으로 전환
    if insufficient_information:
        debug_print("⚠️ LLM 판단: 정보 부족 - 웹 검색으로 전환")
        return "web_search"

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
    if "assess_complexity_node" in nodes_dict:
        workflow.add_node("assess_complexity_node", nodes_dict["assess_complexity_node"])
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
    if "answer_refinement_node" in nodes_dict:
        workflow.add_node("answer_refinement_node", nodes_dict["answer_refinement_node"])
    if "llm_judge_node" in nodes_dict:
        workflow.add_node("llm_judge_node", nodes_dict["llm_judge_node"])
    if "transform_query_node" in nodes_dict:
        workflow.add_node("transform_query_node", nodes_dict["transform_query_node"])
    if "analyze_image" in nodes_dict:
        workflow.add_node("analyze_image", nodes_dict["analyze_image"])
    
    # 워크플로우 연결
    if "analyze_image" in nodes_dict:
        start_route_mapping = {
            "check_validity": "check_validity",
            "analyze_image": "analyze_image",
        }
        workflow.add_conditional_edges(
            START,
            decide_start_route,
            start_route_mapping
        )
        workflow.add_edge("analyze_image", "check_validity")
    else:
        workflow.add_edge(START, "check_validity")
    
    # check_validity 후 복잡도 평가 또는 라우팅으로 분기
    if "assess_complexity_node" in nodes_dict:
        # 복잡도 평가가 있는 경우: check_validity → assess_complexity → route_question
        def decide_validity_with_complexity(state: GraphState) -> str:
            """질문 유효성에 따른 라우팅 결정 (복잡도 평가 포함)"""
            is_valid = state.get("question_valid", True)
            if is_valid:
                return "assess_complexity_node"
            else:
                # 유효성 검사 실패 시 web_search로 라우팅
                debug_print("⚠️ 질문 유효성 검사 실패, web_search로 라우팅")
                return "web_search"
        
        workflow.add_conditional_edges(
            "check_validity",
            decide_validity_with_complexity,
            {
                "assess_complexity_node": "assess_complexity_node",
                "web_search": "web_search" if "web_search" in nodes_dict else "retrieval_node"
            }
        )
        # 복잡도 평가 후 라우팅으로
        workflow.add_edge("assess_complexity_node", "route_question")
    else:
        # 복잡도 평가가 없는 경우: check_validity → route_question
        workflow.add_conditional_edges(
            "check_validity",
            decide_validity,
            {
                "route_question": "route_question",
                "web_search": "web_search" if "web_search" in nodes_dict else "retrieval_node"
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

    
    # 검색 → RAG 핵심 검색 (transform_query_node는 피드백 루프에서만 사용)
    # 복잡도 평가는 이미 라우팅 전에 완료되었으므로, 검색 후 바로 retrieval_node로 연결
    workflow.add_edge("retrieve", "retrieval_node")
    # 노드가 있는 경우에만 엣지 추가
    if "web_search" in nodes_dict:
        workflow.add_edge("web_search", "retrieval_node")
    
    # transform_query_node는 피드백 루프에서만 사용 (재시도 횟수에 따라 경로 전환)
    if "transform_query_node" in nodes_dict:
        # transform_query_node 후 재시도 횟수에 따라 경로 결정
        workflow.add_conditional_edges(
            "transform_query_node",
            decide_retry_route,  # 재시도 횟수에 따라 경로 전환
            {
                "retrieve": "retrieve",
                "web_search": "web_search" if "web_search" in nodes_dict else "retrieve",
            }
        )
    
    # RAG 핵심 흐름
    workflow.add_edge("retrieval_node", "augmentation_node")
    
    # augmentation_node 후 문서 존재 여부 확인
    # 문서가 없고 원래 경로가 vectorstore면 web_search로 전환
    if "web_search" in nodes_dict:
        workflow.add_conditional_edges(
            "augmentation_node",
            decide_document_availability,
            {
                "generation_node": "generation_node",
                "web_search": "web_search"
            }
        )
    else:
        # web_search 노드가 없으면 그냥 진행
        workflow.add_edge("augmentation_node", "generation_node")
    
    # 답변 정리 노드 추가
    if "answer_refinement_node" in nodes_dict:
        workflow.add_edge("generation_node", "answer_refinement_node")
        
        # LLM Judge 노드가 있는 경우: answer_refinement_node → llm_judge_node → decide_final_quality
        if "llm_judge_node" in nodes_dict:
            workflow.add_edge("answer_refinement_node", "llm_judge_node")
            workflow.add_conditional_edges(
                "llm_judge_node",
                decide_final_quality,
                {
                    "transform_query_node": "transform_query_node" if "transform_query_node" in nodes_dict else "retrieval_node",
                    END: END
                }
            )
        else:
            # llm_judge_node가 없는 경우 answer_refinement_node에서 직접 종료
            workflow.add_edge("answer_refinement_node", END)
    else:
        # answer_refinement_node가 없는 경우 generation_node에서 직접 종료
        workflow.add_edge("generation_node", END)
    
    return workflow.compile()
