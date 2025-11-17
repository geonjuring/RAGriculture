"""
환경 설정 모듈
환경 변수 로드 및 기본 설정
"""
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_teddynote.models import get_model_name, LLMs

# 환경 변수 로드
load_dotenv()
logging.langsmith("True-RAG-System")

# 디버깅 로그 제어 설정
DEBUG_MODE = True

# 최신 LLM 모델 이름 가져오기
MODEL_NAME = get_model_name(LLMs.GPT4)


def debug_print(*args, **kwargs):
    """디버깅 로그를 조건부로 출력하는 함수"""
    if DEBUG_MODE:
        print(*args, **kwargs)

