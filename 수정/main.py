"""
메인 실행 모듈
RAG 시스템 통합 및 실행 함수
"""
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote.messages import random_uuid
from .config import MODEL_NAME, debug_print
from .retrieval import setup_all_crop_retrievers, setup_reranker, setup_web_search_tool
from .metrics import RAGMetrics
from .prompts import setup_llm_and_prompts
from .nodes import initialize_nodes
from .workflow import create_workflow
from .location import get_location_context


def initialize_rag_system():
    """RAG 시스템 초기화"""
    debug_print("🚀 RAG 시스템 초기화 시작...")
    
    # 1. LLM 및 임베딩 모델 초기화
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    embedding_model = OpenAIEmbeddings()
    
    # 2. 프롬프트 시스템 초기화 (grader들을 위해 먼저 초기화)
    debug_print("📝 프롬프트 시스템 초기화 중...")
    prompts = setup_llm_and_prompts(llm)
    
    # 3. 벡터스토어 설정
    debug_print("📚 벡터스토어 설정 중...")
    crop_retrievers = setup_all_crop_retrievers()
    
    # 4. 리랭커 설정
    debug_print("🔧 리랭커 설정 중...")
    reranker_result = setup_reranker()
    if isinstance(reranker_result, tuple):
        reranker, free_reranker = reranker_result
    else:
        reranker = reranker_result
        free_reranker = None
    
    # 5. 웹 검색 도구 설정
    debug_print("🌐 웹 검색 도구 설정 중...")
    web_search_tool = setup_web_search_tool()
    
    # 6. RAG 파이프라인 설정 (하이브리드 검색 제거)
    # rag_pipeline을 None으로 설정하여 fallback 모드 사용 (crop_retrievers 직접 사용)
    rag_pipeline = None
    debug_print("📚 RAG 파이프라인: 기본 검색 모드 사용 (벡터스토어 직접 검색)")
    
    # 7. RAG 메트릭스 초기화 (LLM 기반 평가 사용)
    debug_print("📊 RAG 메트릭스 초기화 중...")
    rag_metrics = RAGMetrics(
        embedding_model=embedding_model,
        llm=llm,
        grade_documents_grader=prompts.get("grade_documents_grader"),
        hallucination_grader=prompts.get("hallucination_grader"),
        answer_grader=prompts.get("answer_grader")
    )
    
    # rag_pipeline이 None이므로 rag_metrics는 별도로 관리
    # (rag_pipeline이 있을 때만 업데이트)
    
    # 8. 노드 함수 초기화
    debug_print("🔧 노드 함수 초기화 중...")
    from . import nodes
    nodes.initialize_nodes(
        rag_pipeline, 
        llm, 
        crop_retrievers, 
        reranker, 
        free_reranker,
        question_router=prompts.get("question_router"),  # 프롬프트 전달
        question_validator=prompts.get("question_validator"),  # 프롬프트 전달
        web_search_tool=web_search_tool,  # 웹 검색 도구 전달
        rag_metrics=rag_metrics,  # RAG 메트릭스 전달
        image_route_router=prompts.get("image_route_router")  # 이미지 라우팅 라우터 전달
    )
    
    # 9. 워크플로우 구성
    debug_print("🔄 워크플로우 구성 중...")
    nodes_dict = {
        "check_validity": nodes.check_question_validity,
        "route_question": nodes.route_question,
        "retrieve": nodes.retrieve,
        "web_search": nodes.web_search,  # 웹 검색 노드 추가
        "assess_complexity_node": nodes.assess_complexity_node,  # 복잡도 평가 노드 추가
        "transform_query_node": nodes.transform_query_node,  # 질문 재작성 노드 추가
        "retrieval_node": nodes.retrieval_node,
        "augmentation_node": nodes.augmentation_node,
        "generation_node": nodes.generation_node,
        "quality_check_node": nodes.quality_check_node,
        "llm_judge_validation": nodes.llm_judge_validation_node,
        "analyze_image": nodes.analyze_image,
    }
    
    app = create_workflow(nodes_dict)
    
    debug_print("✅ RAG 시스템 초기화 완료!")
    
    return {
        "app": app,
        "llm": llm,
        "embedding_model": embedding_model,
        "rag_metrics": rag_metrics,
        "crop_retrievers": crop_retrievers,
        "reranker": reranker,
        "web_search_tool": web_search_tool,
        "rag_pipeline": rag_pipeline,
        "prompts": prompts
    }


def run_rag_system(question: str, image_path: str = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """RAG 시스템 실행 함수"""
    debug_print(f"\n🔍 질문: {question}")
    if image_path:
        debug_print(f"🖼️ 이미지: {image_path}")
    debug_print("=" * 80)
    
    # 시스템 초기화 (필요시)
    if config is None:
        system = initialize_rag_system()
        app = system["app"]
    else:
        app = config.get("app")
        if app is None:
            system = initialize_rag_system()
            app = system["app"]
    
    # 실행 설정
    run_config = RunnableConfig(
        recursion_limit=50,
        configurable={"thread_id": random_uuid()}
    )
    
    # 입력 설정
    inputs = {
        "question": question,
        "image": image_path if image_path else None,
        "retry_count": 0
    }
    
    # 실행
    result = app.invoke(inputs, run_config)
    
    # 결과 출력
    debug_print(f"\n💬 답변:")
    debug_print("-" * 50)
    debug_print(result.get("answer", result.get("generation", "답변을 생성할 수 없습니다.")))
    
    # 이미지 분석 결과 출력
    if image_path:
        image_result = result.get("image_result", "")
        if image_result:
            debug_print(f"\n🖼️ 이미지 분석 결과: {image_result}")
    
    # 처리 정보 출력
    if result.get("question_valid") is False:
        debug_print(f"\n❌ 질문 유효성: 무효")
    else:
        debug_print(f"\n✅ 질문 유효성: 유효")
    
    complexity_level = result.get("complexity_level", "unknown")
    debug_print(f"\n📊 복잡도 수준: {complexity_level}")
    
    # 품질 점수 출력
    quality_scores = result.get("quality_scores", {})
    if quality_scores:
        debug_print(f"\n📊 RAG 품질 점수:")
        debug_print(f"   검색 정확도: {quality_scores.get('retrieval_accuracy', 0):.3f}")
        debug_print(f"   답변 관련성: {quality_scores.get('answer_relevance', 0):.3f}")
        debug_print(f"   답변 정확도: {quality_scores.get('answer_correctness', 0):.3f}")
        debug_print(f"   할루시네이션 점수: {quality_scores.get('hallucination_score', 0):.3f}")
        debug_print(f"   전체 점수: {quality_scores.get('overall_score', 0):.3f}")
    
    # 검색된 문서 정보
    documents = result.get("retrieved_docs", result.get("documents", []))
    if documents:
        debug_print(f"\n📚 검색된 문서 ({len(documents)}개):")
        for i, doc in enumerate(documents[:5]):  # 상위 5개만 출력
            source = doc.metadata.get('source', 'Unknown')
            debug_print(f"   {i+1}. {source}")
    
    debug_print("=" * 80)
    
    return result

