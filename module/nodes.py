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
from .location import get_location_context
from .image import image_classification_model, image_classification_processor, all_classes


# 노드 함수들은 전역 변수에 의존하므로 초기화 함수 필요
def initialize_nodes(rag_pipeline, llm, crop_retrievers, reranker, free_reranker_func=None, 
                     question_router=None, question_validator=None, web_search_tool=None,
                     image_route_router=None):
    """노드 함수들을 초기화하는 함수 (전역 변수 설정)"""
    global _rag_pipeline, _llm, _crop_retrievers, _reranker, _free_reranker
    global _question_router, _question_validator, _web_search_tool, _image_route_router
    
    _rag_pipeline = rag_pipeline
    _llm = llm
    _crop_retrievers = crop_retrievers
    _reranker = reranker
    _free_reranker = free_reranker_func
    _question_router = question_router
    _question_validator = question_validator
    _web_search_tool = web_search_tool
    _image_route_router = image_route_router


# 전역 변수 (initialize_nodes로 설정됨)
_rag_pipeline = None
_llm = None
_crop_retrievers = None
_reranker = None
_free_reranker = None
_question_router = None
_question_validator = None
_web_search_tool = None
_image_route_router = None

def retrieval_node(state: GraphState) -> GraphState:
    """검색 노드 - RAG의 핵심"""
    debug_print("==== [RETRIEVAL NODE] ====")
    question = state["question"]
    image_result = state.get("image_result", "")

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
    
    try:
        location_context = get_location_context()
        if location_context and "경작지 위치 정보가 설정되지 않았습니다" not in location_context:
            enhanced_context = f"{context}\n\n{location_context}"
        else:
            enhanced_context = context


        if _rag_pipeline is None or _rag_pipeline.rag_chain is None:
            if _llm is None:
                return {
                    "answer": "AI 모델이 초기화되지 않았습니다.",
                    "status": "error",
                    "error_message": "LLM not initialized"
                }
            
            fallback_prompt = f"""
다음 컨텍스트를 바탕으로 질문에 답변해주세요:

컨텍스트: {enhanced_context}

질문: {question}

답변:
"""
            answer = _llm.invoke(fallback_prompt).content
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
    """답변 정리 노드 - RAG 답변만 보강 (ChatGPT 자체 지식 사용 안 함)"""
    debug_print("==== [ANSWER REFINEMENT NODE] ====")
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
        
        # RAG 답변만 받아서 보강/개선 (ChatGPT 자체 지식 사용 안 함)
        try:
            debug_print("🔧 RAG 답변 보강 중...")
            
            # 검색된 문서 정보 추가 (참고용)
            doc_summary = ""
            if documents:
                doc_summary = "\n\n=== 참조 문서 요약 ===\n"
                for i, doc in enumerate(documents[:5]):  # 상위 5개만
                    content = doc.page_content[:200] if hasattr(doc, 'page_content') else str(doc)[:200]
                    source = doc.metadata.get("source", "Unknown")
                    doc_summary += f"\n[문서 {i+1}] 출처: {source}\n{content}...\n"
            
            # RAG 답변 보강 프롬프트
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
   - ChatGPT의 자체 지식을 사용하지 마세요
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
            debug_print("✅ RAG 답변 보강 완료")
            debug_print(f"📏 보강된 답변 길이: {len(enhanced_answer)} 문자")
            
            # ⭐ 보강된 답변 출력
            debug_print("=" * 80)
            debug_print("✨ [보강된 RAG 답변]")
            debug_print("=" * 80)
            debug_print(enhanced_answer)
            debug_print("=" * 80)
            
            return {
                **state,
                "answer": enhanced_answer,
                "original_answer": rag_answer,
                "status": "enhanced"
            }
            
        except Exception as e:
            debug_print(f"⚠️ RAG 답변 보강 실패: {e}")
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
        if not refined_answer or refined_answer.strip() == "":
            debug_print("⚠️ 답변이 없어 평가 건너뜀")
            return preserve_state_fields(state, {
                "llm_judge_scores": {
                    "should_output": False,
                    "needs_correction": True,
                    "reasoning": "답변이 없어 평가 불가",
                    "is_valid": False,
                    "overall_score": 0
                },
                "status": "judge_skipped"
            })
        
        if _llm is None:
            debug_print("⚠️ LLM이 없어 평가 건너뜀")
            return preserve_state_fields(state, {
                "llm_judge_scores": {
                    "should_output": False,
                    "needs_correction": True,
                    "reasoning": "LLM이 없어 평가 불가",
                    "is_valid": False,
                    "overall_score": 0
                },
                "status": "judge_skipped"
            })
        
        # 참조 문서 요약
        doc_summary = ""
        if documents:
            doc_summary = "\n\n=== 참조 문서 요약 ===\n"
            for i, doc in enumerate(documents[:10]):
                content = doc.page_content[:300] if hasattr(doc, 'page_content') else str(doc)[:300]
                source = doc.metadata.get("source", "Unknown")
                doc_summary += f"\n[문서 {i+1}] 출처: {source}\n{content}...\n"
        
        # LLM Judge 프롬프트
                judge_prompt = f"""
당신은 농업 전문가이자 답변 품질 검증자입니다. 다음 보강된 RAG 답변을 종합적으로 검증하고 판단하세요.

**질문:** {question}

**보강된 RAG 답변:** {refined_answer}

**참조 문서:**
{doc_summary if doc_summary else "참조 문서 없음"}

**검색 경로:** {route} (vectorstore 또는 web_search)
**재시도 횟수:** {retry_count}/3

## 통합 검증 과제

다음 관점에서 답변을 종합적으로 검증하세요:

### 1. 사실성 검증 (Fact-Checking)
- 답변의 각 주장이 참조 문서에 근거하는가?
- 참조 문서와 모순되는 정보가 있는가?
- 명확한 사실 오류가 있는가?
- 웹검색 결과인 경우 출처 신뢰성도 고려

### 2. 완전성 검증 (Completeness Check)
- 질문의 모든 부분에 답했는가?
- 중요한 정보가 누락되었는가?
- 사용자가 기대하는 정보를 모두 제공했는가?

### 3. 정확성 검증 (Accuracy Check)
- 농약 정보, 방제법, 재배법 등이 정확한가?
- 농업 전문 용어가 올바르게 사용되었는가?
- 수치나 시기 정보가 정확한가?

### 4. 일관성 검증 (Consistency Check)
- 답변 내부에 논리적 모순이 있는가?
- 참조 문서 간 모순이 있는가?
- 웹검색과 벡터스토어 결과 간 모순이 있는가?

### 5. 유용성 검증 (Usefulness Check)
- 실제 농업 현장에서 적용 가능한가?
- 구체적이고 실용적인 정보인가?
- 사용자에게 도움이 되는가?

### 6. 검색 결과 품질 검증
- 벡터스토어 결과: 관련성과 정확성
- 웹검색 결과: 출처 신뢰성, 관련성, 일관성
- 검색 결과가 질문에 충분히 답할 수 있는가?

## 최종 판단

위 모든 검증을 바탕으로 다음을 판단하세요:

1. **should_output**: 이 답변을 사용자에게 출력해도 되는가?
   - 모든 검증을 통과하고 충분히 정확하고 완전하면 True
   - 명확한 오류나 누락이 있으면 False

2. **needs_correction**: 답변이 수정이 필요한가?
   - 명확한 오류나 누락이 있으면 True
   - 충분히 정확하고 완전하면 False

3. **correction_suggestions**: 수정이 필요한 경우 구체적인 개선 방안 제시

농업 정보의 정확성이 매우 중요하므로, 확신이 없으면 should_output을 False로 판단하세요.
"""
                
        # LLM 사용 가능 여부 확인
        llm_to_use = None
        if _rag_pipeline and hasattr(_rag_pipeline, 'llm') and _rag_pipeline.llm:
            llm_to_use = _rag_pipeline.llm
        elif _llm:
            llm_to_use = _llm
        
        if llm_to_use:
            try:
                # 구조화된 LLM 호출로 품질 평가 및 판단
                structured_llm = llm_to_use.with_structured_output(LLMJudgeScores)
                judge_result = structured_llm.invoke(judge_prompt)
                result_dict = judge_result.model_dump()
                
                # overall_score는 LLM이 자동 계산하도록 함 (임계값 없이)
                if "overall_score" not in result_dict:
                    debug_print("⚠️ LLM이 overall_score를 계산하지 않음")
                    result_dict["overall_score"] = 0  # 기본값만 설정
                
                # is_valid는 should_output과 동일하게 설정
                if "is_valid" not in result_dict:
                    result_dict["is_valid"] = result_dict.get("should_output", False)
                
                # should_output과 needs_correction이 없으면 기본값 설정
                if "should_output" not in result_dict:
                    result_dict["should_output"] = result_dict.get("is_valid", False)
                if "needs_correction" not in result_dict:
                    result_dict["needs_correction"] = not result_dict.get("should_output", False)
                if "correction_suggestions" not in result_dict:
                    result_dict["correction_suggestions"] = ""
                if "reasoning" not in result_dict:
                    result_dict["reasoning"] = ""
                
                debug_print(f"✅ 품질 평가 완료: accuracy={result_dict.get('accuracy')}, completeness={result_dict.get('completeness')}, overall={result_dict.get('overall_score')}")
                debug_print(f"✅ 최종 판단: should_output={result_dict.get('should_output')}, needs_correction={result_dict.get('needs_correction')}")
                
                # quality_scores 생성 (하위 호환성)
                quality_scores = {
                    "retrieval_accuracy": 0.8 if documents else 0.2,
                    "answer_relevance": result_dict.get("usefulness", 0) / 100.0,
                    "answer_correctness": result_dict.get("accuracy", 0) / 100.0,
                    "hallucination_score": 1.0 if result_dict.get("accuracy", 0) >= 70 else 0.5,
                    "overall_score": result_dict.get("overall_score", 0) / 100.0,
                    "evaluation_method": "LLM"
                }
                
                return preserve_state_fields(state, {
                    "quality_scores": quality_scores,
                    "llm_judge_scores": {
                        "accuracy": result_dict.get("accuracy", 0),
                        "completeness": result_dict.get("completeness", 0),
                        "logical_consistency": result_dict.get("logical_consistency", 0),
                        "usefulness": result_dict.get("usefulness", 0),
                        "overall_score": result_dict.get("overall_score", 0),
                        "should_output": result_dict.get("should_output", False),
                        "needs_correction": result_dict.get("needs_correction", True),
                        "correction_suggestions": result_dict.get("correction_suggestions", ""),
                        "reasoning": result_dict.get("reasoning", ""),
                        "is_valid": result_dict.get("is_valid", False)
                    },
                    "status": "judged"
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
                        "evaluation_method": "Fallback"
                    },
                        "llm_judge_scores": {
                            "should_output": True,
                            "needs_correction": False,
                        "reasoning": f"구조화된 출력 실패: {str(e)}",
                            "is_valid": True,
                        "overall_score": 50
                    },
                    "status": "judge_fallback"
                })
        
        # LLM이 없는 경우 기본값 반환
        return preserve_state_fields(state, {
            "quality_scores": {
                "retrieval_accuracy": 0.3,
                "answer_relevance": 0.3,
                "answer_correctness": 0.3,
                "hallucination_score": 0.5,
                "overall_score": 0.35,
                "evaluation_method": "None"
            },
            "llm_judge_scores": {
                "should_output": False,
                "needs_correction": True,
                "reasoning": "LLM이 없어 평가 불가",
                "is_valid": False,
                "overall_score": 0
            },
            "status": "judge_skipped"
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
                "evaluation_method": "Error"
            },
            "llm_judge_scores": {
                "should_output": False,
                "needs_correction": True,
                "reasoning": f"오류 발생: {str(e)}",
                "is_valid": False,
                "overall_score": 0
            },
            "status": "judge_failed",
            "error_message": f"LLM Judge failed: {str(e)}"
        })


# 라우팅 및 검색 노드들
@robust_error_handling(ErrorType.API_ERROR)
def route_question(state):
    """질문을 적절한 데이터 소스로 라우팅하는 노드"""
    debug_print("==== [ROUTE QUESTION] ====")
    question = get_current_question(state)
    image = get_current_image(state)
    
    try:
        if image and image != "":
            debug_print("==== [ROUTE QUESTION TO IMAGE ANALYSIS] ====")
            return preserve_state_fields(state, {"route": "analyze_image", "status": "routed"})

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



def decide_image_route(state):
    """이미지 분석 후 라우팅 결정 함수 (conditional edge용)"""
    debug_print("==== [DECIDE IMAGE ROUTE] ====")
    
    question = state["question"]
    image_result = state.get("image_result", "")

    if not image_result:
        debug_print("⚠️ 이미지 분석 결과가 없습니다. web_search로 라우팅합니다.")
        return "web_search"

    if _image_route_router is None:
        debug_print("⚠️ 이미지 라우팅 라우터가 초기화되지 않았습니다.")
        return "web_search"
    
    image_source = _image_route_router.invoke({"question": question, "image_result": image_result})

    if image_source.datasource == "web_search":
        debug_print("==== [IMAGE ROUTE QUESTION TO WEB SEARCH] ====")
        return "web_search"
    elif image_source.datasource == "vectorstore":
        debug_print("==== [IMAGE ROUTE QUESTION TO VECTORSTORE] ====")
        return "retrieve"  # workflow에서 "retrieve"로 매핑



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


