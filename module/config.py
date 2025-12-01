"""
환경 설정 모듈
환경 변수 로드 및 기본 설정
"""
from pathlib import Path
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_teddynote.models import get_model_name, LLMs

# 환경 변수 로드 (상위 디렉토리 우선)
current_dir = Path(__file__).parent  # module/
parent_env = current_dir.parent.parent / ".env"  # langchain-kr/.env
local_env = current_dir.parent / ".env"  # modelupdate/.env

# 디버깅: 경로 확인
import os
if os.getenv("DEBUG_ENV_LOAD"):
    print(f"🔍 .env 파일 검색:")
    print(f"   상위 디렉토리: {parent_env} (존재: {parent_env.exists()})")
    print(f"   로컬 디렉토리: {local_env} (존재: {local_env.exists()})")

if parent_env.exists():
    load_dotenv(dotenv_path=parent_env, override=True)
    if os.getenv("DEBUG_ENV_LOAD"):
        print(f"✅ 상위 디렉토리 .env 로드: {parent_env}")
elif local_env.exists():
    load_dotenv(dotenv_path=local_env, override=True)
    if os.getenv("DEBUG_ENV_LOAD"):
        print(f"✅ 로컬 디렉토리 .env 로드: {local_env}")
else:
    load_dotenv(override=True)  # 기본 경로에서 시도
    if os.getenv("DEBUG_ENV_LOAD"):
        print(f"⚠️ 기본 경로에서 .env 로드 시도")

logging.langsmith("True-RAG-System")

# 디버깅 로그 제어 설정
DEBUG_MODE = True

# 최신 LLM 모델 이름 가져오기
MODEL_NAME = get_model_name(LLMs.GPT4)
JUDGE_MODEL_NAME = "gemini-1.5-pro"

# ============================================================================
# Web Search 도메인 필터링 설정
# ============================================================================

# 검색에 포함할 도메인 목록 (None이면 모든 도메인 검색)
WEB_SEARCH_INCLUDE_DOMAINS = [
    'rda.go.kr',              # 농촌진흥청 - 가장 신뢰도 높음
    'nongsaro.go.kr',         # 농사로 - 농촌진흥청 운영
    'mafra.go.kr',            # 농림축산식품부
        # 지역별 농업기술원 (시도별)
    'seoul.at.kr',            # 서울특별시농업기술센터
    'busan.at.kr',            # 부산광역시농업기술센터
    'daegu.at.kr',            # 대구광역시농업기술센터
    'incheon.at.kr',          # 인천광역시농업기술센터
    'gwangju.at.kr',         # 광주광역시농업기술센터
    'daejeon.at.kr',         # 대전광역시농업기술센터
    'ulsan.at.kr',           # 울산광역시농업기술센터
    'sejong.at.kr',          # 세종특별자치시농업기술센터
    'gyeonggi.at.go.kr',     # 경기도농업기술원
    'gangwon.at.go.kr',      # 강원도농업기술원
    'chungbuk.at.go.kr',     # 충청북도농업기술원
    'chungnam.at.go.kr',     # 충청남도농업기술원
    'jeonbuk.at.go.kr',      # 전라북도농업기술원
    'jeonnam.at.go.kr',      # 전라남도농업기술원
    'gyeongbuk.at.go.kr',    # 경상북도농업기술원
    'gyeongnam.at.go.kr',    # 경상남도농업기술원
    'jeju.at.go.kr',         # 제주특별자치도농업기술원
]

# 검색에서 제외할 도메인 목록
WEB_SEARCH_EXCLUDE_DOMAINS = [
    'blog.naver.com',         # 개인 블로그
    'cafe.naver.com',         # 커뮤니티
    'blog.daum.net',          # 개인 블로그
    'gmarket.co.kr',          # 쇼핑몰
    'auction.co.kr',          # 쇼핑몰
    'coupang.com',            # 쇼핑몰
]

# 최대 검색 결과 수
WEB_SEARCH_MAX_RESULTS = 10


def debug_print(*args, **kwargs):
    """디버깅 로그를 조건부로 출력하는 함수"""
    if DEBUG_MODE:
        print(*args, **kwargs)

