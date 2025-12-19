"""
LangGraph 노드 함수 모듈
모든 워크플로우 노드 함수 정의
"""
from typing import Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from .models import GraphState, LLMJudgeScores
from .config import debug_print
from .utils import get_current_question, preserve_state_fields, get_current_image, format_docs
from .error_handler import robust_error_handling, ErrorType
from .location import get_location_context, get_farm_info
from .image import image_classification_model, image_classification_processor, all_classes
from langchain_core.output_parsers import StrOutputParser


# 노드 함수들은 전역 변수에 의존하므로 초기화 함수 필요
def initialize_nodes(rag_pipeline, llm, crop_retrievers, reranker, free_reranker_func=None, 
                     question_router=None, question_validator=None, web_search_tool=None,
                     weather_manager=None, pest_predictor=None, rag_prompt=None, judge_llm=None):
    """노드 함수들을 초기화하는 함수 (전역 변수 설정)"""
    global _rag_pipeline, _llm, _crop_retrievers, _reranker, _free_reranker
    global _question_router, _question_validator, _web_search_tool, _rag_prompt
    global _weather_manager, _pest_predictor, _judge_llm
    
    _rag_pipeline = rag_pipeline
    _llm = llm
    _crop_retrievers = crop_retrievers
    _reranker = reranker
    _free_reranker = free_reranker_func
    _question_router = question_router
    _question_validator = question_validator
    _web_search_tool = web_search_tool
    _weather_manager = weather_manager
    _pest_predictor = pest_predictor
    _rag_prompt = rag_prompt
    _judge_llm = judge_llm


# 전역 변수 (initialize_nodes로 설정됨)
_rag_pipeline = None
_llm = None
_crop_retrievers = None
_reranker = None
_free_reranker = None
_question_router = None
_question_validator = None
_web_search_tool = None
_weather_manager = None
_pest_predictor = None
_rag_prompt = None
_judge_llm = None

def retrieval_node(state: GraphState) -> GraphState:
    """검색 노드 - RAG의 핵심"""
    debug_print("==== [RETRIEVAL NODE] ====")
    question = state["question"]

    try:
        location_context = get_location_context()
        if location_context and "경작지 위치 정보가 설정되지 않았습니다" not in location_context:
            enhanced_question = f"{question}\n\n{location_context}"
        else:
            enhanced_question = question

        if _rag_pipeline is None:
            debug_print("==== [FALLBACK: USING CROP_RETRIEVERS] ====")
            all_docs = []
            for crop_name, retriever in _crop_retrievers.items():
                if retriever:
                    try:
                        docs = retriever.invoke(enhanced_question)
                        all_docs.extend(docs)
                    except Exception as e:
                        debug_print(f"⚠️ {crop_name} fallback 검색 실패: {e}")
            return {
                "retrieved_docs": all_docs,
                "status": "fallback_retrieved" if all_docs else "no_documents"
            }
        
        retrieved_docs = _rag_pipeline.enhanced_retrieval(enhanced_question)
        
        return {
            "retrieved_docs": retrieved_docs,
            "status": "retrieved" if retrieved_docs else "no_documents"
        }
        
    except Exception as e:
        debug_print(f"==== [RETRIEVAL ERROR: {e}] ====")
        return {
            "retrieved_docs": [],
            "status": "error",
            "error_message": f"Retrieval failed: {str(e)}"
        }


def augmentation_node(state: GraphState) -> GraphState:
    """증강 노드 - 검색된 문서를 컨텍스트로 변환"""
    debug_print("==== [AUGMENTATION NODE] ====")
    
    try:
        documents = state.get("retrieved_docs", state.get("documents", []))
        
        if not documents:
            return {
                "context": "관련 문서를 찾을 수 없습니다.",
                "status": "no_documents"
            }
        
        if _rag_pipeline is None:
            context = format_docs(documents)  # utils.format_docs 사용
            return {
                "context": context,
                "status": "fallback_augmented"
            }
        
        context = _rag_pipeline.format_docs(documents)
        return {
            "context": context,
            "status": "augmented"
        }
        
    except Exception as e:
        debug_print(f"==== [AUGMENTATION ERROR: {e}] ====")
        return {
            "context": "컨텍스트 생성 중 오류가 발생했습니다.",
            "status": "error",
            "error_message": f"Augmentation failed: {str(e)}"
        }


def generation_node(state: GraphState) -> GraphState:
    """생성 노드 - 컨텍스트를 바탕으로 답변 생성"""
    debug_print("==== [GENERATION NODE] ====")
    question = state["question"]
    context = state["context"]
    image_result = state.get("image_result", "")
    
    try:
        # 현재 날짜 정보 가져오기
        from datetime import datetime
        current_date = datetime.now()
        current_date_str = current_date.strftime("%Y년 %m월 %d일")
        current_month = current_date.month
        current_season = ""
        if current_month in [12, 1, 2]:
            current_season = "겨울"
        elif current_month in [3, 4, 5]:
            current_season = "봄"
        elif current_month in [6, 7, 8]:
            current_season = "여름"
        elif current_month in [9, 10, 11]:
            current_season = "가을"
        
        date_context = f"📅 현재 날짜: {current_date_str} ({current_season})"
        
        # 위치 컨텍스트
        location_context = get_location_context()
        
        # 기상 데이터 컨텍스트 추가
        weather_context = ""
        pest_forecast_context = ""
        
        farm_info = get_farm_info()
        if farm_info and _weather_manager:
            latitude = farm_info.get('latitude')
            longitude = farm_info.get('longitude')
            
            if latitude and longitude:
                # 기상 컨텍스트 (질문에서 날짜 추출하여 적절한 데이터 사용)
                weather_context = _weather_manager.get_weather_for_date(
                    latitude, longitude, question=question
                )
                
                # 병해충 예측 컨텍스트 (질문에서 작물 추출)
                if _pest_predictor:
                    # 질문에서 작물명 추출
                    question_lower = question.lower()
                    crop = None
                    growth_stage = "생육기"  # 기본값
                    
                    if "토마토" in question or "tomato" in question_lower:
                        crop = "토마토"
                    elif "딸기" in question or "strawberry" in question_lower:
                        crop = "딸기"
                    
                    # 생육 단계 추출 (선택사항)
                    if "개화" in question or "개화기" in question:
                        growth_stage = "개화기"
                    elif "착과" in question or "착과기" in question:
                        growth_stage = "착과기"
                    elif "수확" in question or "수확기" in question:
                        growth_stage = "수확기"
                    
                    if crop:
                        pest_forecast_context = _pest_predictor.get_pest_forecast_context(
                            latitude, longitude, crop, growth_stage
                        )
        
        # 컨텍스트 통합
        enhanced_context = context
        # 날짜 정보를 맨 앞에 추가
        enhanced_context = f"{date_context}\n\n{enhanced_context}"
        if location_context and "경작지 위치 정보가 설정되지 않았습니다" not in location_context:
            enhanced_context = f"{enhanced_context}\n\n{location_context}"
        if weather_context:
            enhanced_context = f"{enhanced_context}\n\n{weather_context}"
            debug_print(f"✅ 기상 컨텍스트 추가됨: {weather_context[:100]}...")
        if pest_forecast_context:
            enhanced_context = f"{enhanced_context}\n\n{pest_forecast_context}"
            debug_print(f"✅ 병해충 예측 컨텍스트 추가됨: {pest_forecast_context[:100]}...")
        
        # 디버그: 최종 컨텍스트 확인
        if weather_context or pest_forecast_context:
            debug_print(f"📊 통합된 컨텍스트 길이: {len(enhanced_context)} 문자")
            debug_print(f"📊 기상 정보 포함 여부: {'✅' if weather_context else '❌'}")
            debug_print(f"📊 병해충 예측 포함 여부: {'✅' if pest_forecast_context else '❌'}")
        debug_print(f"📅 현재 날짜 정보 추가됨: {current_date_str} ({current_season})")


        if _rag_pipeline is None or _rag_pipeline.rag_chain is None:
            if _llm is None:
                return {
                    "answer": "AI 모델이 초기화되지 않았습니다.",
                    "status": "error",
                    "error_message": "LLM not initialized"
                }
            
            fallback_prompt = _rag_prompt
            chain = fallback_prompt | _llm | StrOutputParser()

            answer = chain.invoke({"question": question, "context": enhanced_context, "farm_info": farm_info, "image_result": image_result})
            return {
                "answer": answer,
                "status": "fallback_generated"
            }
        
        answer = _rag_pipeline.rag_chain.invoke({"context": enhanced_context, "question": question})
        
        return {
            "answer": answer,
            "status": "generated"
        }
        
    except Exception as e:
        debug_print(f"==== [GENERATION ERROR: {e}] ====")
        return {
            "answer": "답변 생성 중 오류가 발생했습니다.",
            "status": "error",
            "error_message": f"Generation failed: {str(e)}"
        }


def answer_refinement_node(state: GraphState) -> GraphState:
    """하이브리드 답변 보강 노드 - RAG 답변 + LLM 자체 지식 보충"""
    debug_print("==== [HYBRID ANSWER REFINEMENT NODE] ====")
    question = state.get("question", "")
    rag_answer = state.get("answer", "")
    retry_count = state.get("retry_count", 0)
    route = state.get("route", "vectorstore")
    documents = state.get("retrieved_docs", state.get("documents", []))
    
    try:
        if not rag_answer or rag_answer.strip() == "":
            debug_print("⚠️ RAG 답변이 없어 정리 건너뜀")
            return state
        
        # ⭐ RAG 답변 출력
        debug_print("=" * 80)
        debug_print("📄 [RAG 시스템 답변]")
        debug_print("=" * 80)
        debug_print(rag_answer)
        debug_print("=" * 80)
        debug_print(f"📏 RAG 답변 길이: {len(rag_answer)} 문자")
        
        if _llm is None:
            debug_print("⚠️ LLM이 없어 정리 건너뜀")
            return state
        
        try:
            debug_print("🔧 하이브리드 답변 보강 중...")
            
            # 검색된 문서 정보 추가 (참고용)
            doc_summary = ""
            if documents:
                doc_summary = "\n\n=== 참조 문서 요약 ===\n"
                for i, doc in enumerate(documents[:5]):  # 상위 5개만
                    content = doc.page_content[:200] if hasattr(doc, 'page_content') else str(doc)[:200]
                    source = doc.metadata.get("source", "Unknown")
                    doc_summary += f"\n[문서 {i+1}] 출처: {source}\n{content}...\n"
            
            # 1단계: RAG 답변 분석 - 부족한 정보 식별
            debug_print("📊 1단계: RAG 답변 분석 중...")
            gap_analysis_prompt = f"""다음 RAG 답변을 분석하여 질문에 대해 부족한 정보가 있는지 확인하세요.

**질문:**
{question}

**RAG 답변:**
{rag_answer}

**참조 문서 요약:**
{doc_summary if doc_summary else "참조 문서 없음"}

다음 항목을 분석하세요:
1. 질문의 핵심 요구사항이 모두 충족되었는가?
2. RAG 답변에 누락된 중요한 정보가 있는가?
3. 구체적인 수치, 시기, 방법 등이 부족한가?
4. 질문의 의도와 답변이 완전히 일치하는가?

부족한 정보가 있다면 구체적으로 나열하세요. 없으면 "정보 충분"이라고만 답하세요."""
            
            gap_analysis = _llm.invoke(gap_analysis_prompt).content
            debug_print(f"📋 부족한 정보 분석 결과: {gap_analysis[:200]}...")
            
            # 2단계: LLM 자체 지식으로 보충 답변 생성 (부족한 정보만)
            llm_supplement = ""
            if "정보 충분" not in gap_analysis and len(gap_analysis.strip()) > 10:
                debug_print("📝 RAG 답변에 부족한 정보 감지, LLM 자체 지식으로 보충")
                
                llm_knowledge_prompt = f"""당신은 농업 전문가입니다. 다음 질문에 대해 일반적인 농업 지식을 바탕으로 답변하세요.

**질문:**
{question}

**RAG 답변에서 부족한 부분:**
{gap_analysis}

**중요 지침:**
- RAG 답변에 이미 포함된 정보는 반복하지 마세요
- 부족한 정보만 보충하세요
- 일반적으로 알려진 농업 지식을 사용하되, 확실하지 않은 정보는 명시하세요
- 수치나 구체적 정보는 "일반적으로", "보통", "일반적으로 알려진 바에 따르면" 등의 표현을 사용하세요
- 확실하지 않은 정보는 "확인 필요", "전문가 상담 권장" 등의 표현을 사용하세요
- RAG 답변과 충돌하는 정보는 제공하지 마세요

부족한 정보에 대한 보충 답변만 제공하세요. RAG 답변에 이미 충분한 정보가 있다면 "추가 보충 불필요"라고 답하세요."""
                
                llm_supplement = _llm.invoke(llm_knowledge_prompt).content
                debug_print(f"💡 LLM 보충 답변 생성 완료: {len(llm_supplement)} 문자")
            else:
                debug_print("✅ RAG 답변이 충분하여 LLM 보충 불필요")
            
            # 3단계: RAG 답변과 LLM 보충 답변 통합
            if llm_supplement and "추가 보충 불필요" not in llm_supplement:
                debug_print("🔗 RAG 답변과 LLM 보충 답변 통합 중...")
                
                integration_prompt = f"""다음 두 답변을 통합하여 완전한 답변을 생성하세요.

**원본 질문:**
{question}

**RAG 시스템 답변 (문서 기반, 우선순위 높음):**
{rag_answer}

**LLM 자체 지식 보충 (부족한 정보만):**
{llm_supplement}

**참조 문서 (참고용):**
{doc_summary if doc_summary else "참조 문서 없음"}

## 통합 지침

1. **RAG 답변을 기본 골격으로 유지**: RAG 답변의 모든 핵심 정보는 그대로 유지하세요
2. **LLM 보충 답변에서 RAG 답변에 없는 정보만 추가**: 중복을 피하고 새로운 정보만 통합하세요
3. **정보 출처 구분**:
   - RAG 답변 내용: 문서 기반 정보 (기본 정보)
   - LLM 보충 내용: 일반 농업 지식 (확인 필요 시 전문가 상담 권장)
4. **논리적 흐름과 구조 유지**: 자연스럽게 통합하여 읽기 쉽게 구성하세요
5. **표현 개선**:
   - 불필요한 중복 제거
   - 문장을 자연스럽게 다듬기
   - 마크다운 형식 활용 (제목, 목록, 강조 등)
   - 단락 구분 및 구조 정리
6. **정보 정확성 유지**:
   - RAG 답변의 수치, 시기, 방법 등 구체적 정보는 변경하지 않음
   - LLM 보충 정보는 일반적인 지식임을 명시

## 주의사항

- RAG 답변과 LLM 보충 답변이 충돌하면 RAG 답변을 우선하세요
- LLM 보충 정보는 "일반적으로", "보통" 등의 표현을 사용하여 불확실성을 표시하세요
- 질문과 무관한 정보를 추가하지 마세요

통합된 최종 답변을 생성하세요."""
                
                enhanced_answer = _llm.invoke(integration_prompt).content
                debug_print("✅ 하이브리드 답변 통합 완료")
            else:
                # LLM 보충이 필요 없으면 기존 RAG 답변만 구조화
                debug_print("📝 RAG 답변만 구조화 및 개선 중...")
                
                enhancement_prompt = f"""당신은 농업 전문가이자 답변 개선 전문가입니다.
다음 RAG 시스템이 검색한 문서 기반 답변을 보강하고 개선하세요.

**원본 질문:**
{question}

**RAG 시스템 답변 (검색된 문서 기반):**
{rag_answer}

**참조 문서 (참고용):**
{doc_summary if doc_summary else "참조 문서 없음"}

## 작업 지시사항

### ✅ 수행할 작업:
1. **내용 보강**: 
   - RAG 답변에 이미 포함된 정보를 더 명확하게 표현
   - 불완전한 설명을 보완 (단, 검색된 문서에 근거한 정보만)
   - 논리적 흐름 개선

2. **구조화 및 가독성 개선**:
   - 불필요한 중복 제거
   - 문장을 자연스럽게 다듬기
   - 마크다운 형식 활용 (제목, 목록, 강조 등)
   - 단락 구분 및 구조 정리

3. **정보 정확성 유지**:
   - RAG 답변의 모든 핵심 정보는 그대로 유지
   - 수치, 시기, 방법 등 구체적 정보는 변경하지 않음
   - 검색된 문서에 나온 정보만 사용

### ❌ 금지 사항:
1. **새로운 정보 추가 금지**: 
   - 검색된 문서에 없는 정보를 추가하지 마세요
   - 추측이나 일반적인 농업 지식을 추가하지 마세요

2. **내용 변경 금지**:
   - RAG 답변의 핵심 내용을 변경하지 마세요
   - 수치, 시기, 방법 등 구체적 정보는 그대로 유지하세요
   - 문서에 나온 정보와 다른 내용으로 변경하지 마세요

3. **모순 정보 추가 금지**:
   - RAG 답변과 모순되는 정보를 추가하지 마세요
   - 질문과 무관한 정보를 추가하지 마세요

## 보강 원칙

- **RAG 답변 기본 유지**: RAG 답변의 모든 핵심 정보는 그대로 유지
- **표현 개선만**: 같은 내용을 더 명확하고 읽기 쉽게 표현
- **구조화만**: 정보를 더 잘 구조화하여 가독성 향상
- **문서 기반만**: 검색된 문서에 나온 정보만 사용

위 RAG 답변을 보강하여 개선된 답변을 생성하세요. 새로운 정보를 추가하지 말고, 기존 RAG 답변의 내용을 더 명확하고 읽기 쉽게 개선하세요.
"""
                
                enhanced_answer = _llm.invoke(enhancement_prompt).content
                debug_print("✅ RAG 답변 구조화 완료")
            
            debug_print(f"📏 최종 보강된 답변 길이: {len(enhanced_answer)} 문자")
            
            # ⭐ 보강된 답변 출력
            debug_print("=" * 80)
            debug_print("✨ [하이브리드 보강된 답변]")
            debug_print("=" * 80)
            debug_print(enhanced_answer)
            debug_print("=" * 80)
            
            return {
                **state,
                "answer": enhanced_answer,
                "original_answer": rag_answer,
                "gap_analysis": gap_analysis,  # LLM Judge에서 활용하기 위해 저장
                "llm_supplement": llm_supplement if llm_supplement and "추가 보충 불필요" not in llm_supplement else None,
                "status": "hybrid_enhanced" if llm_supplement and "추가 보충 불필요" not in llm_supplement else "enhanced"
            }
            
        except Exception as e:
            debug_print(f"⚠️ 하이브리드 답변 보강 실패: {e}")
            import traceback
            debug_print(traceback.format_exc())
            # 보강 실패 시 기존 RAG 답변 사용
            debug_print("⚠️ 보강 실패, 원본 RAG 답변 사용")
            return {
                **state,
                "answer": rag_answer,
                "original_answer": rag_answer,
                "status": "enhancement_failed"
            }
        
    except Exception as e:
        debug_print(f"❌ 답변 보강 실패: {e}")
        import traceback
        debug_print(traceback.format_exc())
        return {
            **state,
            "answer": rag_answer,
            "original_answer": rag_answer,
            "status": "refinement_failed",
            "error_message": f"Answer refinement failed: {str(e)}"
        }


@robust_error_handling(ErrorType.VALIDATION_ERROR)
def llm_judge_node(state: GraphState) -> GraphState:
    """LLM Judge 노드 - 답변 품질 평가 및 검증만 수행"""
    debug_print("==== [LLM JUDGE NODE] ====")
    question = state.get("question", "")
    refined_answer = state.get("answer", "")  # answer_refinement_node에서 개선된 답변
    documents = state.get("retrieved_docs", state.get("documents", []))
    retry_count = state.get("retry_count", 0)
    route = state.get("route", "vectorstore")
    
    try:
        # 답변이 없으면 평가 스킵
        if not refined_answer or refined_answer.strip() == "":
            debug_print("⚠️ 답변이 없어 평가 건너뜀")
            return preserve_state_fields(state, {
                "llm_judge_scores": {
                    "accuracy": 0,
                    "completeness": 0,
                    "logical_consistency": 0,
                    "usefulness": 0,
                    "hallucination": 0,
                    "overall_score": 0,
                    "should_output": False,
                    "needs_correction": True,
                    "correction_suggestions": "",
                    "reasoning": "답변이 없어 평가 불가",
                    "is_valid": False,
                },
                "status": "judge_skipped"
            })
        
        # LLM 존재 여부 확인
        if _llm is None and not (_rag_pipeline and getattr(_rag_pipeline, "llm", None)):
            debug_print("⚠️ LLM이 없어 평가 건너뜀")
            return preserve_state_fields(state, {
                "llm_judge_scores": {
                    "accuracy": 0,
                    "completeness": 0,
                    "logical_consistency": 0,
                    "usefulness": 0,
                    "hallucination": 0,
                    "overall_score": 0,
                    "should_output": False,
                    "needs_correction": True,
                    "correction_suggestions": "",
                    "reasoning": "LLM이 없어 평가 불가",
                    "is_valid": False,
                },
                "status": "judge_skipped"
            })
        
        # 참조 문서 요약 (출처 강조)
        doc_summary = ""
        if documents:
            doc_summary = "\n\n=== 참조 문서 요약 ===\n"
            for i, doc in enumerate(documents[:10]):
                content = getattr(doc, "page_content", str(doc))[:300]
                source = getattr(doc, "metadata", {}).get("source", "Unknown")
                
                # 출처 신뢰도 힌트 추가
                trust_hint = ""
                if "rda.go.kr" in source or "nongsaro.go.kr" in source:
                    trust_hint = "[신뢰도 높음: 공식 농업 기관]"
                elif ".go.kr" in source or ".edu" in source or ".ac.kr" in source:
                    trust_hint = "[신뢰도 높음: 정부/학술 기관]"
                
                doc_summary += f"\n[문서 {i+1}] 출처: {source} {trust_hint}\n{content}...\n"
        
        # answer_refinement_node에서 수행한 gap_analysis 가져오기 (중복 평가 방지)
        gap_analysis = state.get("gap_analysis", "")
        gap_analysis_section = ""
        if gap_analysis and gap_analysis.strip() and "정보 충분" not in gap_analysis:
            gap_analysis_section = f"""
● 이전 정보 부족 분석 결과 (참고용)
answer_refinement_node에서 이미 수행한 분석 결과입니다. 이를 참고하여 중복 평가를 피하세요.
{gap_analysis}

**중요**: 위 분석 결과를 참고하되, 최종 답변(보강 후)의 품질을 기준으로 평가하세요.
보강 과정에서 부족한 정보가 보충되었는지 확인하세요.
"""
        
        # LLM Judge 프롬프트 (LLMJudgeScores 스키마에 맞춘 최종 버전)
        judge_prompt = f"""
당신은 농업 전문가이자 고급 LLM 품질 심사관(LLM-as-a-Judge)입니다.
당신의 역할은 "질문, 보강된 RAG 답변, 참조 문서"를 기반으로
LLMJudgeScores 스키마에 맞추어 답변의 품질을 정량/정성적으로 평가하는 것입니다.

[입력 정보]

● 질문 (Question)
{question}

● 보강된 RAG 답변 (Refined Answer)
{refined_answer}

● 참조 문서 요약 (Context)
{doc_summary if doc_summary else "참조 문서 없음"}
{gap_analysis_section}
● 검색 경로 (Route): {route}  (vectorstore 또는 web_search)
● 재시도 횟수: {retry_count}/3

============================================================
[평가 항목 및 기준]

아래 6개 항목 각각에 대해 0~100점(정수)을 부여하세요.

1) accuracy (정확성과 전문성, 0~100)
- 100점: 완벽하게 정확하고 전문적인 정보
- 80-99점: 대체로 정확하고 전문적이나 일부 부정확한 부분 있음
- 60-79점: 일반적으로 정확하나 전문 용어 사용이나 전문 지식 깊이 부족
- 40-59점: 일부 정확하나 전문성 부족
- 0-39점: 대부분 부정확하거나 비전문적

평가 기준:
- 농약 정보, 병해충, 방제법, 재배법 등이 실제 농업 지식과 일치하는가?
- 농업 전문 용어를 정확하게 사용했는가?
- 수치, 시기, 조건 등이 정확한가?

2) completeness (완전성과 상세도, 0~100)
- 100점: 질문의 모든 부분에 대해 완전하고 상세하게 답변
- 80-99점: 대부분의 질문에 답변하나 일부 상세 정보 누락
- 60-79점: 핵심 질문에 답변하나 상세 정보 부족
- 40-59점: 일부만 답변하나 많은 정보 누락
- 0-39점: 질문에 거의 답변하지 못함

평가 기준:
- 질문의 모든 부분에 답변했는가?
- 중요한 정보가 누락되지 않았는가?
- 구체적인 수치, 시기, 방법 등 충분한 상세 정보를 제공하는가?

3) logical_consistency (논리적 일관성과 구조화, 0~100)
- 100점: 논리적으로 완벽하게 구조화된 답변
- 80-99점: 대체로 논리적이나 일부 불일치
- 60-79점: 일반적으로 논리적이나 구조 부족
- 40-59점: 일부 논리적이나 많은 불일치
- 0-39점: 논리적으로 일관성 없음

평가 기준:
- 답변 내부에 논리적 모순이 없는가?
- 문단/단계 간 흐름이 자연스럽고 구조화되어 있는가?
- 참조 문서 내용과 논리적으로 일관되는가?

4) usefulness (실용성, 0~100)
- 100점: 즉시 적용 가능한 실용적인 정보
- 80-99점: 대체로 유용하나 일부 실용성 부족
- 60-79점: 일반적으로 유용하나 구체성 부족
- 40-59점: 일부 유용하나 대부분 추상적
- 0-39점: 거의 유용하지 않음

평가 기준:
- 실제 농업 현장에서 바로 활용 가능한 정보인가?
- 실행 가능한 단계, 조건, 주의사항 등이 포함되어 있는가?
- 사용자가 의사결정을 내리는 데 도움을 주는가?

5) hallucination (사실 기반 정도, 0~100)
- 100점: 모든 주장이 참조 문서에 완벽하게 근거하고 있음
- 80-99점: 대부분의 주장이 문서에 근거하나 일부 불확실한 부분 있음
- 60-79점: 주요 주장은 문서에 근거하나 일부 추측이나 문서에 없는 정보 포함
- 40-59점: 많은 주장이 문서에 근거하지 않거나 추측에 의존
- 0-39점: 대부분의 정보가 문서에 근거하지 않거나 거짓 정보 포함

평가 기준:
- 답변의 각 주장이 참조 문서 또는 일반적으로 알려진 농업 지식에 근거하는가?
- 문서에 없는 내용을 과도하게 추측하지 않았는가?
- 문서 내용과 모순되는 부분이 없는가?
- 거짓 정보나 명백히 잘못된 정보를 포함하지 않았는가?

6) intent_alignment (질문 의도 부합성, 0~100)
- 100점: 사용자의 핵심 의도를 정확히 파악하고 그에 맞는 답변을 제공함
- 80-99점: 의도에 대체로 부합하나 약간의 초점 이탈 있음
- 60-79점: 관련 정보는 제공하나 사용자가 진짜 궁금해하는 핵심을 놓침
- 40-59점: 질문의 주제와 관련은 있으나 엉뚱한 측면을 설명함 (예: 방제약을 물었는데 증상만 설명)
- 0-39점: 질문의 의도와 완전히 다른 동문서답

평가 기준:
- 답변이 질문의 핵심 의도(Key Intent)를 정확히 타격하고 있는가?
- 질문은 A를 묻는데 B를 답하고 있지 않은가?
- 사용자가 이 답변을 보고 "내가 궁금한 건 이게 아닌데"라고 할 가능성이 없는가?

============================================================
[출처 신뢰도 가중치 (Source Credibility)]

평가 시 참조 문서의 '출처(Source)'를 반드시 확인하고 가중치를 두세요.

1. **공식/학술 기관 우선**:
   - 농촌진흥청(rda.go.kr), 농사로(nongsaro.go.kr), 대학(.ac.kr), 정부(.go.kr) 등 신뢰할 수 있는 도메인의 정보를 최우선으로 신뢰하세요.
   - 이러한 출처의 정보와 일반 블로그/뉴스 정보가 충돌하면, **공식 기관의 정보를 정답으로 간주**하세요.

2. **일반 웹 검색 결과 주의**:
   - 출처가 불분명한 블로그, 카페, 커뮤니티 글은 신뢰도를 낮게 평가하세요.
   - 특히 농약 희석 배수나 안전 사용 기준 등 민감한 정보는 공식 출처가 아니면 감점 요인이 될 수 있습니다.

============================================================
[평가 예시 (Few-Shot Examples)]

다음은 평가의 논리를 보여주는 예시입니다. 특정 작물이 아니라 "평가 논리"를 참고하세요.

Case 1: Hallucination (문서에 없는 내용 날조)
- 상황: 문서는 "A 약제 사용"만 언급했는데, 답변에서 "A 약제를 500배 희석하여 3일 간격 살포"라고 구체적 수치를 날조함.
- 평가:
  - hallucination: 20점 (문서에 없는 수치를 지어냄, 매우 위험)
  - accuracy: 40점 (약제 이름은 맞았으나 사용법이 틀림)
  - should_output: False
  - needs_correction: True
  - reasoning: "문서에는 희석 배수와 살포 간격이 없는데 답변에서 이를 임의로 생성했습니다. 이는 농작물에 피해를 줄 수 있는 심각한 할루시네이션입니다."

Case 2: Intent Mismatch (질문 의도 불일치)
- 상황: 질문은 "탄저병 방제약"을 물었는데, 답변은 "탄저병의 증상과 원인"만 길게 설명하고 약제 정보는 없음.
- 평가:
  - usefulness: 30점 (사용자가 원하는 약제 정보가 없음)
  - completeness: 40점 (핵심 질문에 답하지 않음)
  - should_output: False
  - needs_correction: True
  - reasoning: "답변 내용은 정확하지만, 사용자가 질문한 '방제약' 정보가 누락되어 있습니다. 질문의 핵심 의도를 충족하지 못했습니다."

Case 3: Good Answer (이상적인 답변)
- 상황: 질문에 대해 문서에 있는 내용을 기반으로 답변하고, 문서에 없는 내용은 "문서에 관련 정보가 없습니다"라고 솔직하게 명시함.
- 평가:
  - hallucination: 100점 (문서에 있는 내용만 말하고, 없는 건 없다고 함)
  - accuracy: 100점
  - usefulness: 90점
  - should_output: True
  - needs_correction: False

============================================================
[추가 판단 항목]

아래 항목들도 반드시 함께 판단하세요.

6) overall_score (종합 점수, 0~100)
- 위 6개 항목(accuracy, completeness, logical_consistency, usefulness, hallucination, intent_alignment)을 종합 평가한 점수입니다.
- 정확성(accuracy), 사실 기반(hallucination), 의도 부합성(intent_alignment)을 가장 중요하게 반영하세요.
- 한두 항목이 매우 낮다면 overall_score도 낮아야 합니다.

7) is_valid (bool)
- 이 답변이 "농업 도메인에서 의미 있고 유효한 답변"인지 판단하세요.
- 심각한 사실 오류, 문서와의 모순, 위험한 조언이 있으면 False로 판단하세요.

8) should_output (bool)
- 이 답변을 사용자에게 그대로 출력해도 되는지 판단하세요.
- 충분히 정확하고, 사실에 근거하고, 실용적이며, 위험하지 않다면 True입니다.
- 확신이 없거나 위험 가능성이 있으면 False로 판단하세요.

9) needs_correction (bool)
- 이 답변이 수정/보완이 필요한지 판단하세요.
- 중요한 정보 누락, 문서와의 모순, 명백한 오류가 있으면 True입니다.
- 사소한 표현 문제만 있는 경우에는 False일 수 있습니다.

10) correction_suggestions (string)
- needs_correction이 True인 경우, 어떤 부분을 어떻게 수정해야 하는지 구체적으로 제안하세요.
- 잘못된 정보, 누락된 정보, 보완해야 할 점을 항목별로 설명하세요.
- needs_correction이 False인 경우 빈 문자열을 사용하세요.

11) reasoning (string)
- 위의 점수와 판단을 내린 근거를 상세히 서술하세요.
- 각 점수에 영향을 준 핵심 요소를 설명하고,
  답변의 강점과 약점, 할루시네이션 위험 여부를 구체적으로 기술하세요.

============================================================
[출력 형식]

- 당신은 LLMJudgeScores 스키마에 맞춰 structured output을 생성해야 합니다.
- 각 필드는 다음 형식을 지켜야 합니다:
  - accuracy, completeness, logical_consistency, usefulness, hallucination, overall_score: 0~100 정수
  - is_valid, should_output, needs_correction: True 또는 False
  - correction_suggestions, reasoning: 문자열

농업 정보의 정확성과 안전성이 매우 중요하므로,
조금이라도 확신이 없다면 점수를 보수적으로 주고,
should_output을 False로 판단하는 것을 우선하세요.

12) insufficient_information (bool)
- 참조 문서에 답변에 필요한 핵심 정보가 부족한지 판단하세요.
- 문서에 정보가 없어서 답변이 부실하거나 추측해야 한다면 True로 설정하세요.
- 이 값이 True이면 시스템은 웹 검색을 시도할 것입니다.
"""
        
        # LLM 선택 (Judge LLM 우선, 그 다음 파이프라인 LLM, 마지막으로 기본 LLM)
        llm_to_use = None
        if _judge_llm:
            llm_to_use = _judge_llm
            debug_print(f"⚖️ Judge LLM 사용: {getattr(_judge_llm, 'model_name', 'Unknown')}")
        elif _rag_pipeline and hasattr(_rag_pipeline, "llm") and _rag_pipeline.llm:
            llm_to_use = _rag_pipeline.llm
        elif _llm:
            llm_to_use = _llm
        
        if llm_to_use:
            try:
                # 구조화된 LLM 호출로 품질 평가
                structured_llm = llm_to_use.with_structured_output(LLMJudgeScores)
                judge_result = structured_llm.invoke(judge_prompt)
                result_dict = judge_result.model_dump()
                
                # 방어적 기본값 보정 (혹시라도 누락된 필드가 있을 경우)
                result_dict.setdefault("accuracy", 0)
                result_dict.setdefault("completeness", 0)
                result_dict.setdefault("logical_consistency", 0)
                result_dict.setdefault("usefulness", 0)
                result_dict.setdefault("hallucination", 0)
                result_dict.setdefault("overall_score", 0)
                result_dict.setdefault("should_output", False)
                result_dict.setdefault("needs_correction", True)
                result_dict.setdefault("correction_suggestions", "")
                result_dict.setdefault("reasoning", "")
                result_dict.setdefault("is_valid", result_dict.get("should_output", False))
                result_dict.setdefault("insufficient_information", False)
                
                debug_print(
                    f"✅ 품질 평가 완료: "
                    f"accuracy={result_dict.get('accuracy')}, "
                    f"completeness={result_dict.get('completeness')}, "
                    f"logical_consistency={result_dict.get('logical_consistency')}, "
                    f"usefulness={result_dict.get('usefulness')}, "
                    f"hallucination={result_dict.get('hallucination')}, "
                    f"overall={result_dict.get('overall_score')}"
                )
                debug_print(
                    f"✅ 최종 판단: should_output={result_dict.get('should_output')}, "
                    f"needs_correction={result_dict.get('needs_correction')}, "
                    f"is_valid={result_dict.get('is_valid')}, "
                    f"insufficient_information={result_dict.get('insufficient_information')}"
                )
                
                # quality_scores 생성 (하위 호환성)
                quality_scores = {
                    "retrieval_accuracy": 0.8 if documents else 0.2,
                    "answer_relevance": result_dict.get("usefulness", 0) / 100.0,
                    "answer_correctness": result_dict.get("accuracy", 0) / 100.0,
                    "hallucination_score": result_dict.get("hallucination", 0) / 100.0,
                    "overall_score": result_dict.get("overall_score", 0) / 100.0,
                    "evaluation_method": "LLM",
                }
                
                return preserve_state_fields(state, {
                    "quality_scores": quality_scores,
                    "llm_judge_scores": {
                        "accuracy": result_dict.get("accuracy", 0),
                        "completeness": result_dict.get("completeness", 0),
                        "logical_consistency": result_dict.get("logical_consistency", 0),
                        "usefulness": result_dict.get("usefulness", 0),
                        "hallucination": result_dict.get("hallucination", 0),
                        "intent_alignment": result_dict.get("intent_alignment", 0),
                        "overall_score": result_dict.get("overall_score", 0),
                        "should_output": result_dict.get("should_output", False),
                        "needs_correction": result_dict.get("needs_correction", True),
                        "correction_suggestions": result_dict.get("correction_suggestions", ""),
                        "reasoning": result_dict.get("reasoning", ""),
                        "is_valid": result_dict.get("is_valid", False),
                        "insufficient_information": result_dict.get("insufficient_information", False),
                    },
                    "status": "judged",
                })
            
            except Exception as e:
                debug_print(f"⚠️ 구조화된 출력 실패: {e}, 기본값 사용")
                # Fallback: 기본값 설정
                return preserve_state_fields(state, {
                    "quality_scores": {
                        "retrieval_accuracy": 0.5,
                        "answer_relevance": 0.5,
                        "answer_correctness": 0.5,
                        "hallucination_score": 0.5,
                        "overall_score": 0.5,
                        "evaluation_method": "Fallback",
                    },
                    "llm_judge_scores": {
                        "accuracy": 50,
                        "completeness": 50,
                        "logical_consistency": 50,
                        "usefulness": 50,
                        "hallucination": 50,
                        "intent_alignment": 50,
                        "overall_score": 50,
                        "should_output": True,
                        "needs_correction": False,
                        "correction_suggestions": "",
                        "reasoning": f"구조화된 출력 실패: {str(e)}",
                        "is_valid": True,
                        "insufficient_information": False,
                    },
                    "status": "judge_fallback",
                })
        
        # LLM이 없는 경우 기본값 반환 (이전에 걸리면 여기까지 안 옴)
        return preserve_state_fields(state, {
            "quality_scores": {
                "retrieval_accuracy": 0.3,
                "answer_relevance": 0.3,
                "answer_correctness": 0.3,
                "hallucination_score": 0.5,
                "overall_score": 0.35,
                "evaluation_method": "None",
            },
            "llm_judge_scores": {
                "accuracy": 0,
                "completeness": 0,
                "logical_consistency": 0,
                "usefulness": 0,
                "hallucination": 0,
                "intent_alignment": 0,
                "overall_score": 0,
                "should_output": False,
                "needs_correction": True,
                "correction_suggestions": "",
                "reasoning": "LLM이 없어 평가 불가",
                "is_valid": False,
                "insufficient_information": False,
            },
            "status": "judge_skipped",
        })
    
    except Exception as e:
        debug_print(f"==== [LLM JUDGE ERROR: {e}] ====")
        return preserve_state_fields(state, {
            "quality_scores": {
                "retrieval_accuracy": 0.3,
                "answer_relevance": 0.3,
                "answer_correctness": 0.3,
                "hallucination_score": 0.5,
                "overall_score": 0.35,
                "evaluation_method": "Error",
            },
            "llm_judge_scores": {
                "accuracy": 0,
                "completeness": 0,
                "logical_consistency": 0,
                "usefulness": 0,
                "hallucination": 0,
                "intent_alignment": 0,
                "overall_score": 0,
                "should_output": False,
                "needs_correction": True,
                "correction_suggestions": "",
                "reasoning": f"오류 발생: {str(e)}",
                "is_valid": False,
                "insufficient_information": False,
            },
            "status": "judge_failed",
            "error_message": f"LLM Judge failed: {str(e)}",
        })



# 라우팅 및 검색 노드들
@robust_error_handling(ErrorType.API_ERROR)
def route_question(state):
    """질문을 적절한 데이터 소스로 라우팅하는 노드"""
    debug_print("==== [ROUTE QUESTION] ====")
    question = get_current_question(state)
    
    try:
        if _question_router:
            source = _question_router.invoke({"question": question})
            
            result = {}
            
            if source.datasource == "web_search":
                debug_print("==== [ROUTE QUESTION TO WEB SEARCH] ====")
                result["route"] = "web_search"
            elif source.datasource == "vectorstore":
                debug_print("==== [ROUTE QUESTION TO VECTORSTORE] ====")
                result["route"] = "vectorstore"
            else:
                debug_print("==== [ROUTE QUESTION TO WEB SEARCH (DEFAULT)] ====")
                result["route"] = "web_search"
            
            return preserve_state_fields(state, result)
        else:
            # Fallback: 간단한 키워드 기반 라우팅
            question_lower = question.lower()
            if "딸기" in question_lower or "토마토" in question_lower or "strawberry" in question_lower or "tomato" in question_lower:
                return preserve_state_fields(state, {"route": "vectorstore", "status": "routed"})
            else:
                return preserve_state_fields(state, {"route": "web_search", "status": "routed"})
    except Exception as e:
        debug_print(f"==== [ROUTE QUESTION ERROR: {e}] ====")
        # 기본값으로 웹 검색 라우팅
        return preserve_state_fields(state, {
            "route": "web_search",
            "status": "error",
            "error_message": f"Routing failed: {str(e)}"
        })


def analyze_image(state):
    debug_print("==== [IMAGE ANALYSIS] ====")
    import torch
    from PIL import Image

    question = state["question"]
    image_data = state["image"]

    # 이미지 분류 모델이 없는 경우 처리
    if image_classification_model is None or image_classification_processor is None:
        debug_print("⚠️ 이미지 분류 모델이 초기화되지 않았습니다.")
        return {
            "question": question,
            "image_result": "이미지 분류 모델을 사용할 수 없습니다.",
            "documents": []
        }

    try:
        image = Image.open(image_data).convert('RGB')
        
        # 예측
        extended_inputs = image_classification_processor(image, return_tensors="pt")
        
        with torch.no_grad():
            extended_pred = torch.nn.functional.softmax(image_classification_model(**extended_inputs).logits[0], dim=-1)
        
        top_idx = extended_pred.argmax()
        
        if top_idx.item() in all_classes:
            class_name = all_classes[top_idx.item()]
        else:
            class_name = f"Unknown ({top_idx.item()})"
        enhanced_question = f"{question}\n\n이미지 분석 결과: {class_name}"

        return {"question": enhanced_question, "image_result": class_name, "documents": []}
    except Exception as e:
        debug_print(f"⚠️ 이미지 분석 실패: {e}")
        return {
            "question": question,
            "image_result": f"이미지 분석 중 오류 발생: {str(e)}",
            "documents": []
        }


@robust_error_handling(ErrorType.PROCESSING_ERROR)
def retrieve(state):
    """벡터스토어에서 문서를 검색하는 노드"""
    debug_print("==== [RETRIEVE] ====")
    question = get_current_question(state)

    try:
        crop = None
        question_lower = question.lower()
        if "딸기" in question_lower or "strawberry" in question_lower:
            crop = "strawberry"
        elif "토마토" in question_lower or "tomato" in question_lower:
            crop = "tomato"
        
        if crop and crop in _crop_retrievers:
            retriever = _crop_retrievers[crop]
            documents = retriever.invoke(question)
        else:
            all_documents = []
            for crop_name, retriever in _crop_retrievers.items():
                if retriever:
                    try:
                        docs = retriever.invoke(question)
                        all_documents.extend(docs)
                    except Exception as e:
                        debug_print(f"⚠️ {crop_name} 검색 실패: {e}")
            documents = all_documents
        
        if _reranker:
            if _reranker == "free_cross_encoder" and _free_reranker:
                documents = _free_reranker(documents, question, top_n=5)
            else:
                documents = _reranker.compress_documents(documents, query=question)
        else:
            documents = documents[:5]
        
        return preserve_state_fields(state, {"documents": documents})
        
    except Exception as e:
        debug_print(f"==== [RETRIEVE ERROR: {e}] ====")
        return preserve_state_fields(state, {
            "documents": [],
            "status": "error",
            "error_message": f"Retrieval failed: {str(e)}"
        })


@robust_error_handling(ErrorType.VALIDATION_ERROR)
def check_question_validity(state):
    """질문의 유효성을 검사하고 부적합한 경우 ChatGPT로 보강하는 노드"""
    debug_print("==== [CHECK QUESTION VALIDITY] ====")
    question = state.get("question", "")
    
    try:
        if _question_validator:
            validation_result = _question_validator.invoke({"question": question})
            
            is_valid = validation_result.validity.lower() == "yes"
            rewritten_question = validation_result.rewritten_question.strip() if hasattr(validation_result, 'rewritten_question') else question
            
            # 재작성된 질문이 있고 원본과 다르면 사용
            if rewritten_question and rewritten_question != question and rewritten_question != "":
                debug_print(f"📝 질문 재작성: '{question}' → '{rewritten_question}'")
                result = {
                    "question_valid": is_valid,
                    "stop_reason": validation_result.reasoning if not is_valid else "",
                    "question": rewritten_question,  # 재작성된 질문 사용
                    "current_question": rewritten_question,  # 재작성된 질문 사용
                    "original_question": question,  # 원본 질문 저장
                    "question_was_rewritten": True
                }
            else:
                # 재작성되지 않았거나 원본과 같으면 원본 사용
                result = {
                    "question_valid": is_valid,
                    "stop_reason": validation_result.reasoning if not is_valid else "",
                    "question": question,
                    "current_question": question,
                    "original_question": question,
                    "question_was_rewritten": False
                }
            
            return preserve_state_fields(state, result)
        else:
            # Fallback: 간단한 유효성 검사
            if not question or len(question.strip()) < 3:
                return preserve_state_fields(state, {
                    "question_valid": False,
                    "current_question": question,
                    "original_question": question,
                    "stop_reason": "질문이 너무 짧습니다.",
                    "question_was_rewritten": False,
                    "status": "validated"
                })
            else:
                return preserve_state_fields(state, {
                    "question_valid": True,
                    "current_question": question,
                    "original_question": question,
                    "question_was_rewritten": False,
                    "status": "validated"
                })
    except Exception as e:
        debug_print(f"==== [VALIDATION ERROR: {e}] ====")
        # 기본값으로 유효한 질문으로 처리
        return preserve_state_fields(state, {
            "question_valid": True,
            "current_question": question,
            "original_question": question,
            "question_was_rewritten": False,
            "status": "error",
            "error_message": f"Validation failed: {str(e)}"
        })


# 웹 검색 노드
@robust_error_handling(ErrorType.NETWORK_ERROR)
def web_search(state: GraphState) -> GraphState:
    """웹에서 정보를 검색하는 노드 (에러 처리 강화)"""
    debug_print("==== [WEB SEARCH] ====")
    question = get_current_question(state)

    try:
        if _web_search_tool is None:
            debug_print("⚠️ 웹 검색 도구가 설정되지 않았습니다.")
            return preserve_state_fields(state, {
                "documents": [],
                "retrieved_docs": [],
                "status": "error",
                "error_message": "Web search tool not configured"
            })
        
        web_results = _web_search_tool.invoke({"query": question})
        web_results_docs = [
            Document(
                page_content=web_result["content"],
                metadata={"source": web_result["url"]},
            )
            for web_result in web_results
        ]
        
        debug_print(f"웹 검색 결과 수: {len(web_results_docs)}")

        result = {
            "documents": web_results_docs,
            "retrieved_docs": web_results_docs
        }
        return preserve_state_fields(state, result)
        
    except Exception as e:
        debug_print(f"==== [WEB SEARCH ERROR: {e}] ====")
        return preserve_state_fields(state, {
            "documents": [],
            "retrieved_docs": [],
            "status": "error",
            "error_message": f"Web search failed: {str(e)}"
        })


# 복잡도 평가 노드
@robust_error_handling(ErrorType.PROCESSING_ERROR)
def assess_complexity_node(state: GraphState) -> GraphState:
    """질문의 복잡도를 평가하는 노드 (에러 처리 강화)"""
    debug_print("==== [ASSESS COMPLEXITY] ====")
    question = get_current_question(state)
    
    try:
        # 간단한 복잡도 평가
        if "그리고" in question or "또한" in question or "또는" in question:
            complexity_level = "multi_question"
            question_count = 2
            should_process_secondary = True
        elif len(question) > 100 or "분류" in question or "분석" in question:
            complexity_level = "complex"
            question_count = 1
            should_process_secondary = False
        else:
            complexity_level = "simple"
            question_count = 1
            should_process_secondary = False
        
        result = {
            "complexity_level": complexity_level,
            "question_count": question_count,
            "should_process_secondary": should_process_secondary
        }
        
        return preserve_state_fields(state, result)
        
    except Exception as e:
        debug_print(f"==== [COMPLEXITY ASSESSMENT ERROR: {e}] ====")
        # 기본값으로 단순 질문으로 처리
        return preserve_state_fields(state, {
            "complexity_level": "simple",
            "question_count": 1,
            "should_process_secondary": False,
            "status": "error",
            "error_message": f"Complexity assessment failed: {str(e)}"
        })


# 질문 재작성 노드
@robust_error_handling(ErrorType.PROCESSING_ERROR)
def transform_query_node(state: GraphState) -> GraphState:
    """질문을 검색에 최적화된 형태로 재작성하는 노드 (웹검색/벡터스토어 모두 고려)"""
    debug_print("==== [TRANSFORM QUERY] ====")
    question = get_current_question(state)
    retry_count = state.get("retry_count", 0)
    original_route = state.get("route", "vectorstore")  # 원래 검색 경로 유지
    llm_judge_scores = state.get("llm_judge_scores", {})
    correction_suggestions = llm_judge_scores.get("correction_suggestions", "")
    
    try:
        if _llm is None:
            debug_print("⚠️ LLM이 없어 기본 재작성만 수행")
            return preserve_state_fields(state, {
                "question": question,
                "current_question": question,
                "route": original_route  # 원래 경로 유지
            })
        
        # LLM Judge의 수정 제안이 있으면 활용
        if correction_suggestions:
            rewrite_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""당신은 농업 질문을 검색에 최적화된 형태로 재작성하는 전문가입니다.

원래 검색 경로: {original_route} (vectorstore 또는 web_search)

다음 수정 제안을 바탕으로 질문을 개선하세요:
{correction_suggestions}

재작성 시 고려사항:
1. 웹검색인 경우: 최신 정보 검색에 적합한 키워드 사용
2. 벡터스토어인 경우: 문서 내 전문 용어와 일치하는 키워드 사용
3. 핵심 키워드를 명확히 하고 불필요한 단어 제거
4. 농업 전문 용어를 정확하게 사용"""),
                ("human", "다음 질문을 수정 제안에 따라 재작성해주세요: {question}")
            ])
        else:
            # 재시도 횟수에 따라 다른 전략 적용 (검색 경로 고려)
            if retry_count == 0:
                # 1차: 기본 재작성 (검색 경로 고려)
                if original_route == "web_search":
                    rewrite_prompt = ChatPromptTemplate.from_messages([
                        ("system", """당신은 농업 질문을 웹 검색에 최적화된 형태로 재작성하는 전문가입니다.
다음 단계를 따라 질문을 재작성하세요:
1. 최신 정보 검색에 적합한 키워드 추출
2. 검색 엔진에 최적화된 형태로 변환
3. 불필요한 단어 제거
4. 핵심 키워드를 명확히 표현"""),
                        ("human", "다음 질문을 웹 검색에 최적화된 형태로 재작성해주세요: {question}")
                    ])
                else:  # vectorstore
                    rewrite_prompt = ChatPromptTemplate.from_messages([
                        ("system", """당신은 농업 질문을 벡터스토어 검색에 최적화된 형태로 재작성하는 전문가입니다.
다음 단계를 따라 질문을 재작성하세요:
1. 핵심 키워드를 추출하고 유지
2. 문서 내 전문 용어와 일치하는 키워드 사용
3. 불필요한 단어 제거
4. 농업 관련 용어를 정확한 전문 용어로 변환"""),
                        ("human", "다음 질문을 벡터스토어 검색에 최적화된 형태로 재작성해주세요: {question}")
                    ])
            elif retry_count == 1:
                # 2차: 확장된 쿼리 (동의어 추가, 검색 경로 고려)
                if original_route == "web_search":
                    rewrite_prompt = ChatPromptTemplate.from_messages([
                        ("system", """당신은 농업 질문을 확장된 웹 검색 쿼리로 재작성하는 전문가입니다.
다음 단계를 따라 질문을 재작성하세요:
1. 핵심 키워드의 동의어 및 관련 용어 추가
2. 검색 범위를 넓혀 다양한 출처에서 정보 검색
3. 최신 정보 검색에 적합한 키워드 조합"""),
                        ("human", "다음 질문을 확장된 웹 검색 쿼리로 재작성해주세요: {question}")
                    ])
                else:  # vectorstore
                    rewrite_prompt = ChatPromptTemplate.from_messages([
                        ("system", """당신은 농업 질문을 확장된 벡터스토어 쿼리로 재작성하는 전문가입니다.
다음 단계를 따라 질문을 재작성하세요:
1. 핵심 키워드의 동의어 추가
2. 관련 용어 확장
3. 검색 범위 넓히기"""),
                        ("human", "다음 질문을 확장된 벡터스토어 쿼리로 재작성해주세요: {question}")
                    ])
            else:
                # 3차 이상: 간소화된 쿼리 (검색 경로 고려)
                rewrite_prompt = ChatPromptTemplate.from_messages([
                    ("system", """당신은 농업 질문을 간소화된 쿼리로 재작성하는 전문가입니다.
다음 단계를 따라 질문을 재작성하세요:
1. 핵심 키워드만 추출
2. 불필요한 설명 제거
3. 가장 중요한 검색어만 남기기"""),
                    ("human", "다음 질문을 간소화된 쿼리로 재작성해주세요 (핵심 키워드만): {question}")
                ])
        
        # LLM 호출
        chain = rewrite_prompt | _llm
        response = chain.invoke({"question": question})
        
        better_question = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        
        # 재작성된 질문이 원본과 같으면 변경 없음
        if better_question == question or not better_question:
            debug_print("==== [TRANSFORM QUERY RESULT: NO MEANINGFUL CHANGE] ====")
            return preserve_state_fields(state, {
                "stop_reason": "no_rewrite",
                "question": question,
                "current_question": question,
                "route": original_route  # 원래 경로 유지
            })
        
        debug_print(f"원본 질문: {question}")
        debug_print(f"재작성된 질문: {better_question} (재시도 {retry_count}, 경로: {original_route})")
        
        return preserve_state_fields(state, {
            "question": better_question,
            "current_question": better_question,
            "route": original_route,  # 원래 검색 경로 유지
            "retry_count": retry_count + 1  # 재시도 횟수 증가
        })
        
    except Exception as e:
        debug_print(f"==== [TRANSFORM QUERY ERROR: {e}] ====")
        # 에러 발생 시 원본 질문 반환
        return preserve_state_fields(state, {
            "question": question,
            "current_question": question,
            "route": original_route  # 원래 경로 유지
        })
