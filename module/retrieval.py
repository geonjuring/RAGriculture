"""
검색 시스템 초기화 모듈
벡터스토어 및 리랭커 설정
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_core.documents import Document
from langchain.retrievers.document_compressors import CohereRerank
from langchain_teddynote.tools.tavily import TavilySearch
from .config import (
    debug_print,
    WEB_SEARCH_INCLUDE_DOMAINS,
    WEB_SEARCH_EXCLUDE_DOMAINS,
    WEB_SEARCH_MAX_RESULTS
)

# PDFRetrievalChain import 처리 (상위 디렉토리의 rag.pdf 모듈)
try:
    # 방법 1: 직접 import 시도
    from rag.pdf import PDFRetrievalChain
except ImportError:
    try:
        # 방법 2: 상위 디렉토리에서 찾기
        current_file = Path(__file__).resolve()
        # RAGsystem/rag/retrieval.py -> calendar format re copy/rag/pdf.py
        parent_dir = current_file.parent.parent.parent  # calendar format re copy
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        from rag.pdf import PDFRetrievalChain
    except ImportError:
        # 방법 3: importlib를 사용하여 직접 로드
        import importlib.util
        current_file = Path(__file__).resolve()
        rag_dir = current_file.parent.parent.parent / "rag"
        parent_dir = rag_dir.parent  # calendar format re copy
        
        # sys.path에 부모 디렉토리 추가 (rag.base, rag.pdf를 찾기 위해)
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        # base.py를 먼저 로드하고 sys.modules에 등록
        base_path = rag_dir / "base.py"
        if base_path.exists():
            base_spec = importlib.util.spec_from_file_location("rag.base", base_path)
            base_module = importlib.util.module_from_spec(base_spec)
            # sys.modules에 등록하여 pdf.py에서 import할 수 있게 함
            sys.modules["rag.base"] = base_module
            base_spec.loader.exec_module(base_module)
            debug_print(f"✅ rag.base 모듈 로드 완료: {base_path}")
        
        # pdf.py를 로드
        pdf_path = rag_dir / "pdf.py"
        if pdf_path.exists():
            pdf_spec = importlib.util.spec_from_file_location("rag.pdf", pdf_path)
            pdf_module = importlib.util.module_from_spec(pdf_spec)
            # sys.modules에 등록하여 다른 곳에서 import할 수 있게 함
            sys.modules["rag.pdf"] = pdf_module
            pdf_spec.loader.exec_module(pdf_module)
            PDFRetrievalChain = pdf_module.PDFRetrievalChain
            debug_print(f"✅ rag.pdf 모듈 로드 완료: {pdf_path}")
        else:
            raise ImportError(f"PDFRetrievalChain을 찾을 수 없습니다. {pdf_path} 파일을 확인하세요.")


def setup_crop_specific_retriever(crop_name: str):
    """작물별 전용 벡터스토어 설정"""
    try:
        # 현재 파일 위치를 기준으로 절대 경로 계산
        current_file = Path(__file__).resolve()
        # module/retrieval.py -> RAGsystem module/
        base_dir = current_file.parent.parent
        
        # 작물별 데이터 경로 설정 (절대 경로 사용)
        if crop_name == "strawberry":
            data_dir = base_dir / "data" / "strawberry"
            data_paths = [
                {"path": str(data_dir / "딸기 재배 일정 통일 및 상세화.pdf"), "crop": "strawberry"},
                {"path": str(data_dir / "딸기 재배 기술 정론(최신 교정).pdf"), "crop": "strawberry"},
                {"path": str(data_dir / "딸기 재배 기술 총람 (최신 교정).pdf"), "crop": "strawberry"},
                {"path": str(data_dir / "딸기 길라잡이.PDF"), "crop": "strawberry"},  # 대문자 확장자
                {"path": str(data_dir / "딸기 길라잡이.pdf"), "crop": "strawberry"},  # 소문자 확장자도 시도
                {"path": str(data_dir / "딸기 농작업일정.pdf"), "crop": "strawberry"},
                {"path": str(data_dir / "딸기 병해충 및 비료.pdf"), "crop": "strawberry"},
                {"path": str(data_dir / "딸기 재배메뉴얼.pdf"), "crop": "strawberry"},
                {"path": str(data_dir / "딸기 재배법.pdf"), "crop": "strawberry"},
                {"path": str(data_dir / "딸기 재배일정표.pdf"), "crop": "strawberry"}  # ⭐ 새로 추가
            ]
            persist_dir = str(base_dir / "db" / "strawberry_vector")
        elif crop_name == "tomato":
            data_dir = base_dir / "data" / "tomato"
            data_paths = [
                {"path": str(data_dir / "토마토 반촉성재배 상세 일정 생성.pdf"), "crop": "tomato"},
                {"path": str(data_dir / "토마토 반촉성재배 기술 정론 (교정).pdf"), "crop": "tomato"},
                {"path": str(data_dir / "토마토 상업 재배 기술 총람 (교정).pdf"), "crop": "tomato"},
                {"path": str(data_dir / "토마토 농작업일정.pdf"), "crop": "tomato"},
                {"path": str(data_dir / "토마토 백과사전.pdf"), "crop": "tomato"},
                {"path": str(data_dir / "토마토 병해충 및 비료.pdf"), "crop": "tomato"},
                {"path": str(data_dir / "토마토 재배법.pdf"), "crop": "tomato"}
            ]
            persist_dir = str(base_dir / "db" / "tomato_vector")
        else:
            raise ValueError(f"지원되지 않는 작물: {crop_name}")

        # 파일 존재 확인 (중복 제거)
        valid_paths = []
        seen_paths = set()
        for item in data_paths:
            path = item["path"]
            if os.path.exists(path) and path not in seen_paths:
                valid_paths.append(item)
                seen_paths.add(path)
        
        if not valid_paths:
            debug_print(f"⚠️ {crop_name} 데이터 디렉토리: {data_dir}")
            debug_print(f"⚠️ 찾을 수 없는 파일 목록:")
            for item in data_paths:
                if not os.path.exists(item["path"]):
                    debug_print(f"   - {item['path']}")
            raise FileNotFoundError(f"⚠️ {crop_name} 데이터 파일을 찾을 수 없습니다. {data_dir} 디렉토리를 확인하세요.")

        debug_print(f"📚 {crop_name} 전용 벡터스토어 생성 중...")
        debug_print(f"   - 데이터 파일: {len(valid_paths)}개")
        debug_print(f"   - 저장 위치: {persist_dir}")

        crop_data = PDFRetrievalChain(
            valid_paths,
            persist_dir=persist_dir,
            force_rebuild=False
        ).create_chain()

        debug_print(f"✅ {crop_name} 전용 벡터스토어 사용")
        return crop_data.retriever

    except Exception as e:
        debug_print(f"❌ {crop_name} 벡터스토어 생성 실패: {e}")
        raise e


def setup_all_crop_retrievers() -> Dict[str, Any]:
    """모든 작물의 벡터스토어 설정"""
    retrievers = {}
    
    for crop in ["strawberry", "tomato"]:
        try:
            retrievers[crop] = setup_crop_specific_retriever(crop)
        except Exception as e:
            debug_print(f"❌ {crop} 리트리버 설정 실패: {e}")
            retrievers[crop] = None
    
    return retrievers


def setup_reranker():
    """리랭커 설정 (Cohere 우선, 실패 시 Cross Encoder)"""
    try:
        # 1. Cohere 리랭커 시도
        reranker = CohereRerank(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            top_n=5,
            model="rerank-multilingual-v3.0"  # 한국어 지원 모델
        )
        
        # API 테스트 (실제 사용 가능한지 확인)
        try:
            test_docs = [Document(page_content="test", metadata={})]
            test_result = reranker.compress_documents(test_docs, query="test")
            debug_print("✅ Cohere 리랭커 설정 및 API 테스트 완료")
            return reranker
        except Exception as api_error:
            debug_print(f"⚠️ Cohere API 테스트 실패: {api_error}")
            raise api_error  # Cross Encoder로 전환하기 위해 예외 발생
    
    except Exception as e:
        debug_print(f"⚠️ Cohere 리랭커 설정 실패: {e}")
        
        # 2. 대체 방안: Sentence Transformers Cross Encoder
        try:
            from sentence_transformers import CrossEncoder
            
            # Cross Encoder 모델 로드 (한국어 지원)
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            def free_reranker(documents, query, top_n=5):
                """무료 Cross Encoder를 사용한 reranker"""
                if len(documents) <= top_n:
                    return documents
                    
                # 점수 계산
                pairs = [(query, doc.page_content) for doc in documents]
                scores = cross_encoder.predict(pairs)
                
                # 점수 기준으로 정렬
                doc_scores = list(zip(documents, scores))
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                
                return [doc for doc, score in doc_scores[:top_n]]
            
            debug_print("✅ 무료 Cross Encoder reranker 설정 완료 (Cohere 대체)")
            return "free_cross_encoder", free_reranker
            
        except Exception as e2:
            debug_print(f"⚠️ Cross Encoder 설정도 실패: {e2}")
            debug_print("⚠️ 리랭킹 기능 없이 진행합니다")
            return None, None


def setup_web_search_tool(
    max_results: Optional[int] = None,
    include_domains: Optional[list] = None,
    exclude_domains: Optional[list] = None
):
    """
    웹 검색 도구 설정
    
    Args:
        max_results: 최대 검색 결과 수 (None이면 config.py의 기본값 사용)
        include_domains: 검색에 포함할 도메인 목록 (None이면 config.py의 기본값 사용)
        exclude_domains: 검색에서 제외할 도메인 목록 (None이면 config.py의 기본값 사용)
    
    Returns:
        TavilySearch 도구 객체
    
    Note:
        모든 파라미터는 config.py에서 중앙 관리됩니다.
        함수 호출 시 파라미터를 지정하면 config.py의 기본값을 덮어씁니다.
    """
    # config.py의 기본값 사용 (파라미터가 제공되지 않은 경우)
    if max_results is None:
        max_results = WEB_SEARCH_MAX_RESULTS
    
    if include_domains is None:
        include_domains = WEB_SEARCH_INCLUDE_DOMAINS
    
    if exclude_domains is None:
        exclude_domains = WEB_SEARCH_EXCLUDE_DOMAINS
    
    # 기본 파라미터 설정
    search_params = {
        "max_results": max_results
    }
    
    # 도메인 필터링 옵션 추가
    if include_domains:
        search_params["include_domains"] = include_domains
        debug_print(f"✅ 검색 포함 도메인: {include_domains}")
    
    if exclude_domains:
        search_params["exclude_domains"] = exclude_domains
        debug_print(f"✅ 검색 제외 도메인: {exclude_domains}")
    
    web_search_tool = TavilySearch(**search_params)
    web_search_tool.name = "web_search"
    web_search_tool.description = (
        "딸기, 토마토, 망고에 대한 작물 재배법, 병해충, 농약 및 비료(퇴비) 관련 정보가 문서에 없거나 부족할 경우, 웹에서 검색을 수행합니다. "
        "딸기, 토마토, 망고에 대한 질문이 아닌 다른 작물에 대한 질문이 들어오면 웹검색을 사용하세요. "
        "최신 정보가 필요한 경우에도 사용하세요. "
        "웹 검색 결과를 활용한 응답일 경우, 반드시 '🔎 웹 검색 결과를 기반으로 제공된 정보입니다.' 문구를 출력해야 합니다."
    )
    return web_search_tool

