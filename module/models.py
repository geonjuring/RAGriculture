"""
데이터 모델 정의 모듈
Pydantic 모델 및 타입 정의
"""
from typing import List, Literal, Annotated, Dict, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class RouteQuery(BaseModel):
    """사용자 쿼리를 가장 관련성 높은 데이터 소스로 라우팅하는 데이터 모델"""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="""농업 질문을 분석하여 적절한 데이터 소스로 라우팅하세요.
        
**vectorstore 선택 조건:**
- 토마토나 딸기 관련 질문 (재배법, 병해충, 농약, 관리 방법 등)
- 벡터스토어에 저장된 문서로 답변 가능한 질문
- 구체적이고 실용적인 농업 정보가 필요한 질문

**web_search 선택 조건:**
- 토마토/딸기 외의 작물 관련 질문 (참외, 수박, 멜론, 고추, 상추 등)
- 최신 농업 기술, 시장 정보, 신품종 등 최신 정보가 필요한 질문
- 농업과 무관한 일반적인 질문
- 딸기나 토마토 벡터스토어에 정보가 부족한 질문

반드시 "vectorstore" 또는 "web_search" 중 하나만 선택하세요.""",
    )


class QuestionValidity(BaseModel):
    """질문 유효성 검사 결과"""
    validity: str = Field(
        description="""질문이 유효한 농업 질문인지 판단하세요.
        
**"yes" (유효한 질문):**
- 실제 존재하는 작물, 병해충, 농약, 농업 기술에 대한 질문
- 재배법, 방제법, 환경조건, 수확시기, 토양관리 등 농업 관련 주제
- 다중 질문이 하나로 묶인 경우

**"no" (무효한 질문):**
- 존재하지 않는 개념 ("우주고추", "마법농약" 등)
- 논리적 모순 ("겨울철 수박 재배" 등)
- 농업과 무관한 질문 ("자동차 정비법", "요리 레시피" 등)
- 의미 없는 표현 ("농사가 좋아요" 등)

반드시 "yes" 또는 "no"로만 답변하세요.""",
    )
    reasoning: str = Field(
        description="""유효성 판단의 구체적인 근거를 설명하세요.
- 왜 유효한지 또는 무효한지 명확히 설명
- 질문의 내용을 분석한 결과를 구체적으로 서술
- 판단 근거를 명확하게 제시하세요.""",
    )
    rewritten_question: str = Field(
        description="""유효한 질문인 경우, 검색에 최적화된 형태로 재작성하세요.
        
**재작성 원칙:**
1. 모호한 표현을 구체적으로 개선 (예: "병해" → "탄저병 방제법")
2. 검색 키워드 강화: 중요한 농업 용어를 포함
3. 전문성 향상: 정확한 농업 용어 사용
4. 맥락 정보 추가: 재배 환경, 시기, 증상, 지역 특성 등

무효한 질문인 경우 원본 질문을 그대로 반환하세요.""",
    )


class LLMJudgeScores(BaseModel):
    """LLM Judge 평가 점수"""
    accuracy: int = Field(
        description="""답변이 실제 농업 정보와 일치하는 정도 (0-100점)
- 100점: 완벽하게 정확한 정보
- 80-99점: 대체로 정확하나 일부 부정확한 부분 있음
- 60-79점: 일반적으로 정확하나 일부 오류 있음
- 40-59점: 일부 정확하나 많은 오류 있음
- 0-39점: 대부분 부정확함

정확한 농약 정보, 방제법, 재배법 등이 실제 농업 지식과 일치하는지 평가하세요.""",
        ge=0, le=100
    )
    completeness: int = Field(
        description="""질문에 대한 답변이 완전한 정도 (0-100점)
- 100점: 질문의 모든 부분에 대해 완전히 답변
- 80-99점: 대부분의 질문에 답변하나 일부 누락
- 60-79점: 핵심 질문에 답변하나 상세 정보 부족
- 40-59점: 일부만 답변하나 많은 정보 누락
- 0-39점: 질문에 거의 답변하지 못함

질문이 요구하는 모든 정보를 제공했는지 평가하세요.""",
        ge=0, le=100
    )
    logical_consistency: int = Field(
        description="""답변의 논리적 일관성과 구조화 정도 (0-100점)
- 100점: 논리적으로 완벽하게 구조화된 답변
- 80-99점: 대체로 논리적이나 일부 불일치
- 60-79점: 일반적으로 논리적이나 구조 부족
- 40-59점: 일부 논리적이나 많은 불일치
- 0-39점: 논리적으로 일관성 없음

답변의 논리적 흐름, 구조, 일관성을 평가하세요.""",
        ge=0, le=100
    )
    usefulness: int = Field(
        description="""답변이 실제 농업 현장에서 유용한 정도 (0-100점)
- 100점: 즉시 적용 가능한 실용적인 정보
- 80-99점: 대체로 유용하나 일부 실용성 부족
- 60-79점: 일반적으로 유용하나 구체성 부족
- 40-59점: 일부 유용하나 대부분 추상적
- 0-39점: 거의 유용하지 않음

실제 농업 현장에서 적용 가능한 구체적이고 실용적인 정보인지 평가하세요.""",
        ge=0, le=100
    )
    overall_score: int = Field(
        description="""종합 평가 점수 (0-100점)
- 위 4개 항목(accuracy, completeness, logical_consistency, usefulness)을 종합적으로 평가
- 각 항목의 중요도를 고려하여 계산하세요
- 정확성과 완전성을 더 중요하게 평가하세요

**⚠️ 중요**: 임계값 없이 답변의 품질을 종합적으로 판단하여 점수를 부여하세요.""",
        ge=0, le=100
    )
    is_valid: bool = Field(
        description="""답변이 유효한지 여부
- 답변의 품질을 종합적으로 판단하여 True/False 결정
- 농업 정보의 정확성과 신뢰성을 고려하여 판단하세요

**⚠️ 중요**: 임계값 없이 답변의 전체적인 품질을 평가하여 판단하세요.""",
    )
    reasoning: str = Field(
        description="""평가 근거를 상세히 설명하세요.
- 각 점수를 부여한 이유
- 답변의 강점과 약점
- 개선이 필요한 부분
- 최종 판단 근거

구체적이고 객관적인 평가 근거를 제시하세요.""",
    )
    
    # 직접 판단 필드 추가
    should_output: bool = Field(
        description="""이 답변을 사용자에게 출력해도 되는지 판단
- True: 답변이 충분히 정확하고 완전하여 사용자에게 출력 가능
- False: 답변이 부정확하거나 불완전하여 재처리 필요

판단 기준:
1. 참조 문서에 충분히 근거하고 있는가?
2. 논리적으로 일관되고 유용한가?
3. 질문에 대한 답변이 완전한가?
4. 실제 농업 현장에서 적용 가능한가?

**⚠️ 중요**: 임계값 없이 답변의 전체적인 품질을 종합적으로 판단하여 결정하세요.
농업 정보의 정확성이 매우 중요하므로, 확신이 없으면 False를 선택하세요.""",
    )
    
    needs_correction: bool = Field(
        description="""답변이 수정이 필요한지 판단
- True: 답변에 명확한 오류나 누락이 있어 수정 필요
- False: 답변이 충분히 정확하고 완전함

수정이 필요한 경우:
1. 참조 문서와 모순되는 정보
2. 명확한 사실 오류
3. 중요한 정보 누락
4. 논리적 모순""",
    )
    
    correction_suggestions: str = Field(
        description="""수정이 필요한 경우 개선 방안 제시
- needs_correction이 True인 경우 구체적인 수정 방안 제시
- needs_correction이 False인 경우 빈 문자열""",
        default=""
    )


class MultiQuestionResult(BaseModel):
    """다중질의 분리 결과"""
    primary_question: str = Field(description="첫 번째 질문")
    secondary_question: str = Field(description="두 번째 질문")
    question_count: int = Field(description="질문 개수")
    should_process_secondary: bool = Field(description="두 번째 질문 처리 여부")


class QuestionComplexity(BaseModel):
    """질문 복잡도 평가 결과"""
    complexity_level: Literal["simple", "multi_question", "complex"] = Field(
        description="질문의 복잡도 수준"
    )
    question_count: int = Field(description="질문 개수")
    should_process_secondary: bool = Field(description="두 번째 질문 처리 여부")


class GradeDocuments(BaseModel):
    """문서 관련성 평가 결과
    
    질문과 검색된 문서의 관련성을 평가합니다.
    검색 정확도(retrieval_accuracy)와 답변 정확도(answer_correctness) 평가에 사용됩니다.
    """
    binary_score: str = Field(
        description="문서가 질문에 직접적으로 관련되어 있고, 질문에 답할 수 있는 정보를 포함하고 있는지 평가합니다. "
                   "평가 기준: 1) 직접적 관련성 - 문서가 질문에 직접적으로 관련되어 있는가? "
                   "2) 정보 충족도 - 문서가 질문에 답할 수 있는 정보를 포함하고 있는가? "
                   "3) 농업 관련성 - 문서가 농업 분야와 관련된 내용인가? "
                   "문서가 질문에 직접적으로 관련되어 있으면 'yes', 그렇지 않으면 'no'를 반환하세요."
    )


class GradeHallucinations(BaseModel):
    """할루시네이션 평가 결과
    
    답변이 주어진 문서에 근거하고 있는지 평가합니다.
    할루시네이션 점수(hallucination_score)와 답변 정확도(answer_correctness) 평가에 사용됩니다.
    """
    binary_score: str = Field(
        description="답변이 주어진 문서의 사실에 기반하고 있는지 평가합니다. "
                   "평가 기준: 1) 사실 기반 - 답변이 문서의 사실에 기반하는가? "
                   "2) 추측 금지 - 문서에 없는 정보로 추측하지 않았는가? "
                   "3) 일관성 - 답변이 문서 내용과 일치하는가? "
                   "4) 할루시네이션 - 거짓 정보나 잘못된 정보를 포함하지 않았는가? "
                   "답변이 문서의 사실에 기반하고 있으면 'yes', 그렇지 않으면 'no'를 반환하세요."
    )


class GradeAnswer(BaseModel):
    """답변 관련성 평가 결과
    
    답변이 사용자 질문을 해결하는지 평가합니다.
    답변 관련성(answer_relevance) 평가에 사용됩니다.
    """
    binary_score: str = Field(
        description="답변이 사용자 질문을 직접적으로 해결하는지 평가합니다. "
                   "평가 기준: 1) 질문 해결 - 답변이 질문을 직접적으로 해결하는가? "
                   "2) 완전성 - 답변이 질문에 대한 완전한 정보를 제공하는가? "
                   "3) 유용성 - 답변이 사용자에게 유용한 정보를 제공하는가? "
                   "4) 정확성 - 답변이 정확하고 신뢰할 수 있는가? "
                   "답변이 질문을 직접적으로 해결하면 'yes', 그렇지 않으면 'no'를 반환하세요."
    )

    

class GraphState(TypedDict):
    """LangGraph 워크플로우의 상태를 정의하는 클래스"""
    # 기본 질문 관련 필드들
    question: Annotated[str, "User question"]
    current_question: Annotated[str, "Current question being processed"]
    generation: Annotated[str, "Generated answer"]
    documents: Annotated[List[Document], "Retrieved documents"]
    
    # 질문 유효성 및 라우팅
    question_valid: Annotated[bool, "Whether question is valid"]
    route: Annotated[str, "Route to take (vectorstore or web_search)"]
    stop_reason: Annotated[str, "Reason for stopping"]
    retry_count: Annotated[int, "Number of retries"]
    
    # 전략 및 소스 타입
    strategy: Annotated[str, "Strategy to use"]
    source_type: Annotated[str, "Type of source"]
    
    # 다중질의 처리 관련
    question_count: Annotated[int, "Number of questions"]
    should_process_secondary: Annotated[bool, "Whether to process secondary question"]
    question_index: Annotated[int, "Current question index"]
    primary_question: Annotated[str, "Primary question"]
    secondary_question: Annotated[str, "Secondary question"]
    
    # 복잡도 및 질문 타입
    complexity_level: Annotated[str, "Complexity level"]
    question_type: Annotated[str, "Question type"]
    needs_decomposition: Annotated[bool, "Whether question needs decomposition"]
    first_answer: Annotated[str, "Answer to the first question"]
    
    # RAG 관련 필드들
    retrieved_docs: Annotated[List[Document], "Retrieved documents"]
    context: Annotated[str, "Formatted context from documents"]
    answer: Annotated[str, "Generated answer"]
    original_answer: Annotated[str, "Original RAG answer before refinement"]
    quality_scores: Annotated[Dict[str, float], "RAG quality scores"]
    status: Annotated[str, "Processing status"]
    llm_judge_scores: Annotated[Dict[str, Any], "LLM Judge scores"]

    # 이미지 분석 관련
    image: Annotated[str, "Image file"]
    image_result: Annotated[str, "Image analysis result"]
