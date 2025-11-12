"""
LangGraph 노드 함수 모듈
모든 워크플로우 노드 함수 정의
"""
from typing import Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from .models import GraphState, LLMJudgeScores
from .config import debug_print
from .utils import get_current_question, preserve_state_fields, get_current_image
from .error_handler import robust_error_handling, ErrorType
from .location import get_location_context
from .image import image_classification_model, image_classification_processor, all_classes


# 노드 함수들은 전역 변수에 의존하므로 초기화 함수 필요
def initialize_nodes(rag_pipeline, llm, crop_retrievers, reranker, free_reranker_func=None, 
                     question_router=None, question_validator=None, web_search_tool=None,
                     rag_metrics=None, image_route_router=None):
    """노드 함수들을 초기화하는 함수 (전역 변수 설정)"""
    global _rag_pipeline, _llm, _crop_retrievers, _reranker, _free_reranker
    global _question_router, _question_validator, _web_search_tool, _rag_metrics, _image_route_router
    
    _rag_pipeline = rag_pipeline
    _llm = llm
    _crop_retrievers = crop_retrievers
    _reranker = reranker
    _free_reranker = free_reranker_func
    _question_router = question_router
    _question_validator = question_validator
    _web_search_tool = web_search_tool
    _rag_metrics = rag_metrics
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
_rag_metrics = None
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
            context = "\n\n".join([
                f"문서 {i+1}: {doc.page_content}\n출처: {doc.metadata.get('source', 'Unknown')}"
                for i, doc in enumerate(documents)
            ])
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


def quality_check_node(state: GraphState) -> GraphState:
    """품질 검사 노드 - RAG 품질 평가"""
    debug_print("==== [QUALITY CHECK NODE] ====")
    question = state["question"]
    documents = state.get("retrieved_docs", state.get("documents", []))
    answer = state["answer"]
    
    try:
        if not documents:
            return {
                "quality_scores": {
                    "retrieval_accuracy": 0.2,
                    "answer_relevance": 0.3,
                    "answer_correctness": 0.2,
                    "hallucination_score": 0.5,
                    "overall_score": 0.3
                },
                "status": "quality_checked_no_docs"
            }
        
        # rag_metrics를 직접 사용 (rag_pipeline이 None이어도 사용 가능)
        if _rag_metrics:
            try:
                quality_scores = _rag_metrics.evaluate(
                    question=question,
                    answer=answer,
                    documents=documents
                )
                return {
                    "quality_scores": quality_scores,
                    "status": "quality_checked"
                }
            except Exception as e:
                debug_print(f"품질 평가 실패: {e}")
        
        # rag_pipeline에 rag_metrics가 있는 경우 (하위 호환성)
        if _rag_pipeline and hasattr(_rag_pipeline, 'rag_metrics'):
            try:
                quality_scores = _rag_pipeline.rag_metrics.evaluate(
                    question=question,
                    documents=documents,
                    answer=answer
                )
                return {
                    "quality_scores": quality_scores,
                    "status": "quality_checked"
                }
            except Exception as e:
                debug_print(f"품질 평가 실패: {e}")
        
        # Fallback 평가
        doc_quality = min(1.0, len(documents) / 3.0)
        answer_quality = min(1.0, len(answer) / 100.0)
        
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        keyword_match = len(question_words.intersection(answer_words)) / max(len(question_words), 1)
        
        overall_score = (doc_quality * 0.3 + answer_quality * 0.4 + keyword_match * 0.3)
        
        quality_scores = {
            "retrieval_accuracy": doc_quality,
            "answer_relevance": answer_quality,
            "answer_correctness": keyword_match,
            "hallucination_score": 0.7,
            "overall_score": overall_score
        }
        
        return {
            "quality_scores": quality_scores,
            "status": "fallback_quality_checked"
        }
        
    except Exception as e:
        debug_print(f"==== [QUALITY CHECK ERROR: {e}] ====")
        return {
            "quality_scores": {
                "retrieval_accuracy": 0.3,
                "answer_relevance": 0.3,
                "answer_correctness": 0.3,
                "hallucination_score": 0.5,
                "overall_score": 0.35
            },
            "status": "error",
            "error_message": f"Quality check failed: {str(e)}"
        }


def llm_judge_validation_node(state: GraphState) -> GraphState:
    """LLM-as-Judge 통합 검증 노드 - 모든 검증을 LLM이 직접 판단"""
    debug_print("==== [LLM JUDGE VALIDATION] ====")
    question = state.get("question", "")
    answer = state.get("answer", "")
    documents = state.get("retrieved_docs", state.get("documents", []))
    retry_count = state.get("retry_count", 0)
    route = state.get("route", "vectorstore")
    
    try:
        if not question or not answer:
            state["llm_judge_scores"] = {
                "should_output": False,
                "needs_correction": True,
                "reasoning": "답변 또는 질문이 없어 검증 불가",
                "is_valid": False,
                "overall_score": 0
            }
            state["status"] = "llm_judge_skipped"
            return state
        
        # 참조 문서 요약 (웹검색과 벡터스토어 구분)
        source_summary = ""
        web_docs = []
        vector_docs = []
        
        if documents:
            for i, doc in enumerate(documents[:10]):
                content = doc.page_content[:500] if hasattr(doc, 'page_content') else str(doc)[:500]
                source = doc.metadata.get("source", "Unknown")
                
                # 웹검색과 벡터스토어 결과 분리
                if source.startswith("http"):
                    web_docs.append(f"웹 검색 결과 {len(web_docs)+1} (출처: {source}):\n{content}...")
                else:
                    vector_docs.append(f"벡터스토어 결과 {len(vector_docs)+1} (출처: {source}):\n{content}...")
            
            if web_docs:
                source_summary += "=== 웹 검색 결과 ===\n" + "\n\n".join(web_docs) + "\n\n"
            if vector_docs:
                source_summary += "=== 벡터스토어 결과 ===\n" + "\n\n".join(vector_docs)
        
        # 통합 검증 프롬프트
        judge_prompt = f"""
당신은 농업 전문가이자 답변 품질 검증자입니다. 다음 답변을 종합적으로 검증하고 판단하세요.

**질문:** {question}

**생성된 답변:** {answer}

**참조 문서:**
{source_summary if source_summary else "참조 문서 없음"}

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
                structured_llm = llm_to_use.with_structured_output(LLMJudgeScores)
                result = structured_llm.invoke(judge_prompt)
                result_dict = result.model_dump()
                
                # overall_score 자동 계산 (없는 경우)
                if "overall_score" not in result_dict or result_dict["overall_score"] == 0:
                    scores = [
                        result_dict.get("accuracy", 0),
                        result_dict.get("completeness", 0),
                        result_dict.get("logical_consistency", 0),
                        result_dict.get("usefulness", 0)
                    ]
                    result_dict["overall_score"] = int(sum(scores) / len(scores))
                
                # is_valid는 should_output과 동일하게 설정 (하위 호환성)
                if "is_valid" not in result_dict:
                    result_dict["is_valid"] = result_dict.get("should_output", False)
                
                # should_output과 needs_correction이 없으면 기본값 설정
                if "should_output" not in result_dict:
                    result_dict["should_output"] = result_dict.get("is_valid", False)
                if "needs_correction" not in result_dict:
                    result_dict["needs_correction"] = not result_dict.get("should_output", False)
                if "correction_suggestions" not in result_dict:
                    result_dict["correction_suggestions"] = ""
                
                debug_print(f"✅ LLM 통합 검증 판단: should_output={result_dict.get('should_output')}, needs_correction={result_dict.get('needs_correction')}")
                debug_print(f"📊 점수: accuracy={result_dict.get('accuracy')}, completeness={result_dict.get('completeness')}, overall={result_dict.get('overall_score')}")
                debug_print(f"💭 판단 근거: {result_dict.get('reasoning', '')[:200]}...")
                
                state["llm_judge_scores"] = result_dict
                state["status"] = "llm_judge_completed"
                return state
                
            except Exception as e:
                debug_print(f"⚠️ LLM 통합 검증 구조화된 출력 실패: {e}")
                # Fallback: 일반 LLM 호출
                try:
                    fallback_response = llm_to_use.invoke(judge_prompt)
                    response_text = fallback_response.content.lower()
                    
                    should_output = "출력 가능" in response_text or "true" in response_text or "yes" in response_text
                    needs_correction = "수정 필요" in response_text or "false" in response_text or "no" in response_text
                    
                    state["llm_judge_scores"] = {
                        "should_output": should_output,
                        "needs_correction": needs_correction,
                        "reasoning": fallback_response.content,
                        "is_valid": should_output,
                        "overall_score": 70 if should_output else 50,
                        "correction_suggestions": ""
                    }
                    state["status"] = "llm_judge_fallback"
                    return state
                except Exception as e2:
                    debug_print(f"⚠️ LLM 통합 검증 Fallback 실패: {e2}")
        
        # LLM이 없는 경우
        state["llm_judge_scores"] = {
            "should_output": False,
            "needs_correction": True,
            "reasoning": "LLM이 없어 검증 불가",
            "is_valid": False,
            "overall_score": 0,
            "correction_suggestions": ""
        }
        state["status"] = "llm_judge_skipped"
        return state
            
    except Exception as e:
        debug_print(f"==== [LLM JUDGE ERROR: {e}] ====")
        state["llm_judge_scores"] = {
            "should_output": False,
            "needs_correction": True,
            "reasoning": f"LLM 통합 검증 실패: {str(e)}",
            "is_valid": False,
            "overall_score": 0,
            "correction_suggestions": ""
        }
        state["status"] = "llm_judge_error"
        state["error_message"] = f"LLM Judge failed: {str(e)}"
        return state


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
    """질문의 유효성을 검사하는 노드 (재작성 기능 제거)"""
    debug_print("==== [CHECK QUESTION VALIDITY] ====")
    question = state.get("question", "")
    
    try:
        if _question_validator:
            validation_result = _question_validator.invoke({"question": question})
            
            is_valid = validation_result.validity.lower() == "yes"
            
            result = {
                "question_valid": is_valid,
                "stop_reason": validation_result.reasoning if not is_valid else "",
                "question": question,  # 원본 질문 유지 (재작성 제거)
                "current_question": question  # 원본 질문 유지
            }
            
            # 재작성 기능 제거 - 원본 질문 그대로 유지
            
            return preserve_state_fields(state, result)
        else:
            # Fallback: 간단한 유효성 검사
            if not question or len(question.strip()) < 3:
                return preserve_state_fields(state, {
                    "question_valid": False,
                    "current_question": question,
                    "stop_reason": "질문이 너무 짧습니다.",
                    "status": "validated"
                })
            else:
                return preserve_state_fields(state, {
                    "question_valid": True,
                    "current_question": question,
                    "status": "validated"
                })
    except Exception as e:
        debug_print(f"==== [VALIDATION ERROR: {e}] ====")
        # 기본값으로 유효한 질문으로 처리
        return preserve_state_fields(state, {
            "question_valid": True,
            "current_question": question,
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

