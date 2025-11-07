"""
유틸리티 함수 모듈
공통 유틸리티 함수들
"""
from typing import Dict, Any, List
from langchain_core.documents import Document


def get_current_question(state: Dict[str, Any]) -> str:
    """현재 처리 중인 질문을 가져오는 함수"""
    return state.get("current_question", state.get("question", ""))


def preserve_state_fields(state: Dict[str, Any], result: Dict[str, Any], exclude_fields: List[str] = None) -> Dict[str, Any]:
    """상태 필드를 보존하면서 새로운 결과를 추가하는 함수"""
    if exclude_fields is None:
        exclude_fields = []
    
    # 기존 상태에서 제외할 필드들을 제외하고 복사
    preserved = {k: v for k, v in state.items() if k not in exclude_fields}
    
    # 새로운 결과 추가
    preserved.update(result)
    return preserved


def format_docs(docs: List[Document]) -> str:
    """검색된 문서를 컨텍스트로 포맷팅"""
    return "\n\n".join([
        f"문서 {i+1}: {doc.page_content}\n출처: {doc.metadata.get('source', 'Unknown')}"
        for i, doc in enumerate(docs)
    ])

