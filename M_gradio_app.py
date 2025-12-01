"""
RAG 시스템 Gradio 인터페이스
HTML 파일 없이 Gradio 컴포넌트만 사용

업데이트: modelupdate/module의 모든 기능 통합
- 이미지 업로드 및 작물 분류
- 지오코딩을 통한 위치 정보 개선
- 에러 처리 개선
- RAG 메트릭스 표시
"""

import sys
import os
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path.cwd()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 상위 디렉토리를 Python 경로에 추가
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import pandas as pd
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

# 모듈화된 기능 import
from module.main import initialize_rag_system, run_rag_system
from module.location import (
    setup_farm_location, 
    get_location_context, 
    get_farm_info,
    get_geo_manager,
    set_farm_info
)
from module.weather_forecast import WeatherForecastManager
from module.pest_forecast import PestForecastPredictor
from module.error_handler import robust_error_handling, ErrorType
from module.config import debug_print
from module.search_history import SearchHistoryManager


# LangChain 관련
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import List

# 전역 변수
rag_system = None
rag_app = None
llm = None
weather_manager = None
pest_predictor = None
# web_search_tool과 schedule_llm은 더 이상 사용하지 않음 (고정된 일정표 사용)
# web_search_tool = None
# schedule_llm = None

def initialize_systems():
    """RAG 시스템 초기화"""
    global rag_system, rag_app, llm, weather_manager, pest_predictor, search_history_manager
    
    if rag_system is None:
        print("🚀 RAG 시스템 초기화 중...")
        rag_system = initialize_rag_system()
        
        # 초기화된 컴포넌트 추출
        llm = rag_system['llm']
        rag_app = rag_system['app']
        
        # 기상 및 병해충 예측 모듈 추출 (일정표 검색용)
        try:
            from module.retrieval import setup_all_crop_retrievers
            crop_retrievers = setup_all_crop_retrievers()
            weather_manager = WeatherForecastManager()
            pest_predictor = PestForecastPredictor(weather_manager, crop_retrievers)
            print("✅ 기상/병해충 모듈 초기화 완료 (일정표 검색용)")
        except Exception as e:
            debug_print(f"⚠️ 기상/병해충 모듈 초기화 실패: {e}")
        
        # 검색 기록 관리자 초기화
        try:
            search_history_manager = SearchHistoryManager(retention_days=30)
            print("✅ 검색 기록 관리자 초기화 완료")
        except Exception as e:
            debug_print(f"⚠️ 검색 기록 관리자 초기화 실패: {e}")
        
        print("✅ RAG 시스템 초기화 완료!")
    
    # 일정표 생성 시스템은 더 이상 필요 없음 (고정된 일정표 사용)
    # if web_search_tool is None:
    #     print("🔍 Web Search 도구 초기화 중...")
    #     web_search_tool = setup_web_search_tool()
    #     schedule_llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    #     print("✅ 일정표 생성 시스템 초기화 완료!")
    
    return rag_system, rag_app, llm

# 시스템 초기화
initialize_systems()

# ============================================================================
# 유틸리티 함수
# ============================================================================

def fix_markdown_strikethrough(text: str) -> str:
    """
    마크다운 취소선 문제 해결: 안전망 역할만 수행
    
    주의: 이 함수는 안전망으로만 사용됩니다.
    마크다운 형식은 주로 프롬프트에서 올바르게 생성되도록 지시합니다.
    이 함수는 LLM이 실수로 틸드(~)를 사용한 경우를 대비한 최소한의 보정만 수행합니다.
    """
    import re
    
    # 1단계: 취소선 제거 (~~텍스트~~ → 텍스트)
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    
    # 2단계: 틸드(~)를 하이픈(-)으로 변환 (안전망)
    # 숫자 범위 표현에서 틸드(~)가 사용된 경우만 변환
    text = re.sub(r'(\d+)~(\d+)', r'\1-\2', text)
    
    return text

# ============================================================================
# 1. RAG 시스템 - 일반 질문 답변 (이미지 지원)
# ============================================================================

@robust_error_handling(ErrorType.PROCESSING_ERROR)
def rag_query(question: str, image: Optional[Any] = None) -> str:
    """RAG 시스템으로 질문 답변 (이미지 지원)"""
    if not question or not question.strip():
        return "질문을 입력해주세요."
    
    try:
        # 이미지 경로 처리
        image_path = None
        if image is not None:
            # Gradio Image 컴포넌트에서 파일 경로 추출
            if isinstance(image, str):
                image_path = image
            elif isinstance(image, dict) and 'path' in image:
                image_path = image['path']
            elif hasattr(image, 'name'):  # tempfile.NamedTemporaryFile
                image_path = image.name
        
        # 일반 RAG 워크플로우 실행
        result = run_rag_system(
            question=question,
            image_path=image_path,
            config={"app": rag_app}
        )
        
        answer = result.get("answer", result.get("generation", "답변을 생성할 수 없습니다."))
        
        # 이미지 분석 결과 추가
        image_result = result.get("image_result")
        if image_result:
            answer = f"**🖼️ 이미지 분석 결과:** {image_result}\n\n" + answer
        
        # 마크다운 취소선 문제 해결 (예: 1015℃ → 10~15℃, 7080% → 70~80%)
        answer = fix_markdown_strikethrough(answer)
        
        # 검색 기록 저장
        global search_history_manager
        if search_history_manager:
            try:
                search_history_manager.add_search(
                    question=question,
                    answer=answer,
                    search_type="general",
                    metadata={"has_image": image_path is not None}
                )
            except Exception as e:
                debug_print(f"⚠️ 검색 기록 저장 실패: {e}")
        
        return answer
    
    except Exception as e:
        debug_print(f"❌ RAG 쿼리 오류: {e}")
        return f"오류가 발생했습니다: {str(e)}\n\n다시 시도해주세요."

# ============================================================================
# 2. 일정표 생성 시스템 (Web Search + ChatGPT 독립 검색)
# ============================================================================

# 일정표 항목 모델
class ScheduleItem(BaseModel):
    period: str = Field(description="재배 기간 (예: 3월~5월, 6월)")
    task: str = Field(description="재배 작업명 (예: 육묘기, 정식기, 수확기)")

class ScheduleList(BaseModel):
    schedules: List[ScheduleItem] = Field(description="재배 일정 목록")

# 고정된 일정표 데이터
FIXED_SCHEDULES = {
    "토마토": [
        {"period": "10월 중순~11월 초", "task": "파종", "details": "종자 소독, 파종, 발아 관리"},
        {"period": "10월 말~1월 초 (70~80일)", "task": "육묘", "details": "온·광·관수·도장 억제, 1·2차 이식"},
        {"period": "1월 말~2월 초", "task": "정식", "details": "본포 준비, 정식, 활착 관리"},
        {"period": "2~3월", "task": "생육 관리", "details": "유인, 적엽, 적과, 온도·수분·양액 관리"},
        {"period": "3~5월", "task": "착과·비대", "details": "수정 관리(진동수정), 병해 방제"},
        {"period": "4~8월", "task": "수확", "details": "1~10단 수확"},
        {"period": "7~8월", "task": "재배 종료", "details": "마지막 수확 후 제거, 소독"},
    ],
    "딸기": [
        {"period": "3~4월", "task": "모주 정식", "details": "모주 관리, 정식"},
        {"period": "5~7월", "task": "런너 유인 및 자묘 생산", "details": "자묘받기, 분리"},
        {"period": "7~8월", "task": "자묘 육묘", "details": "묘 키우기, 활착 관리"},
        {"period": "9월 상~중순", "task": "본포 정식", "details": "본식, 활착 관리"},
        {"period": "9월 하~10월 상", "task": "화아분화 유도", "details": "저온·단일 관리"},
        {"period": "10~12월", "task": "개화·착과", "details": "수정, 꽃 관리"},
        {"period": "11~5월", "task": "수확", "details": "장기 수확"},
        {"period": "5월말", "task": "재배 종료", "details": "제거, 소독"},
    ]
}

@robust_error_handling(ErrorType.PROCESSING_ERROR)
def generate_schedule_web_chatgpt(crop: str, location: Optional[str] = None) -> tuple:
    """고정된 일정표 반환 (촉성재배 고정)
    
    Args:
        crop: 작물명 (사용자 입력)
        location: 경작지 위치 (선택사항)
    """
    if not crop or not crop.strip():
        return pd.DataFrame(), "", "작물명을 입력해주세요."
    
    crop = crop.strip()
    
    # 고정된 일정표 확인
    if crop not in FIXED_SCHEDULES:
        return pd.DataFrame(), "", f"'{crop}'에 대한 일정표가 준비되지 않았습니다. 현재 지원되는 작물: {', '.join(FIXED_SCHEDULES.keys())}"
    
    try:
        # 고정된 일정표 데이터 가져오기
        schedules = FIXED_SCHEDULES[crop]
        
        debug_print("=" * 80)
        debug_print(f"📅 일정표 조회: {crop} (촉성재배)")
        debug_print("=" * 80)
        debug_print(f"✅ 고정된 일정표 사용: {len(schedules)}개 단계")
        debug_print("=" * 80)
        
        # DataFrame 생성
        schedule_data = [
            {"단계": item["task"], "기간": item["period"], "주요 작업": item["details"]}
            for item in schedules
        ]
        main_df = pd.DataFrame(schedule_data)
        
        # JSON 형식으로 변환
        schedule_json = json.dumps({
            "crop": crop,
            "cultivation_method": "촉성재배",
            "location": location if location and location.strip() else None,
            "schedules": schedules,
            "generated_at": datetime.now().isoformat(),
            "source": "고정된 일정표"
        }, ensure_ascii=False, indent=2)
        
        debug_print(f"📊 일정표 항목 수: {len(schedules)}개")
        debug_print("=" * 80)
        
        return main_df, schedule_json, f"{crop} 일정표가 성공적으로 조회되었습니다."
    
    except Exception as e:
        debug_print(f"❌ 일정표 조회 오류: {e}")
        import traceback
        debug_print(traceback.format_exc())
        return pd.DataFrame(), "", f"일정표 조회 중 오류가 발생했습니다: {str(e)}"

# ============================================================================
# 2. Gradio 인터페이스
# ============================================================================

def create_gradio_interface():
    """Gradio 인터페이스 생성 (심플한 농업 AI 스타일)"""
    
    # 심플한 연두색 계열 테마
    custom_theme = gr.themes.Soft(
        primary_hue="green",
        secondary_hue="lime",
        neutral_hue="gray",
    ).set(
        body_background_fill="#f6f8f4",
        body_background_fill_dark="#111827",
        button_primary_background_fill="#4caf50",
        button_primary_background_fill_hover="#43a047",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#e8f5e9",
        button_secondary_background_fill_hover="#c8e6c9",
        button_secondary_text_color="#2e7d32",
        border_color_primary="#c5e1a5",
    )

    
    # 최대한 단순한 카드 기반 레이아웃
    custom_css = """
    .gradio-container {
        background: #f6f8f4 !important;
        font-family: 'Pretendard', 'Malgun Gothic', '맑은 고딕', system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* 전체 폭 약간 줄여서 깔끔하게 */
    .gradio-container > div {
        max-width: 1080px;
        margin: 0 auto;
    }

    /* 헤더 카드 */
    .agri-header {
        background: #ffffff;
        border-radius: 18px;
        padding: 18px 22px;
        border: 1px solid #dde7d2;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
    }
    .agri-header-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .agri-logo-circle {
        width: 44px;
        height: 44px;
        border-radius: 999px;
        background: #e8f5e9;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
    }
    .agri-title-main {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1b5e20;
    }
    .agri-title-sub {
        font-size: 0.9rem;
        color: #5f6f55;
        margin-top: 4px;
    }
    .agri-badge {
        font-size: 0.78rem;
        padding: 6px 10px;
        border-radius: 999px;
        background: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #c5e1a5;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
    }

    /* 공통 카드 스타일 */
    .agri-card {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #dde7d2;
        padding: 14px 16px;
    }
    .agri-card + .agri-card {
        margin-top: 10px;
    }

    /* 섹션 타이틀 */
    .agri-section-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1b5e20;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 6px;
    }
    .agri-section-desc {
        font-size: 0.82rem;
        color: #5f6f55;
        margin-bottom: 8px;
    }

    /* 입력 영역 */
    .agri-input textarea, .agri-input input {
        border-radius: 10px !important;
        border: 1px solid #c5d6bd !important;
        background: #fbfdf9 !important;
        font-size: 0.92rem !important;
    }
    .agri-input textarea:focus, .agri-input input:focus {
        border-color: #81c784 !important;
        box-shadow: 0 0 0 2px rgba(129, 199, 132, 0.25) !important;
    }

    /* 이미지 업로드 박스 */
    .agri-image-box {
        border-radius: 12px !important;
        border: 1px dashed #a5d6a7 !important;
        background: #f4faf4 !important;
    }

    /* 답변 / 팁 카드 */
    .agri-answer {
        max-height: 1200px;
        overflow: auto;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* 데이터프레임(검색 기록, 일정표) */
    .dataframe {
        font-size: 0.82rem;
    }
    
    /* 사이드바 열기 버튼 (닫혔을 때 표시) */
    #history_sidebar_open_btn, #schedule_sidebar_open_btn {
        position: fixed !important;
        left: 0 !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        z-index: 1000 !important;
        border-radius: 0 8px 8px 0 !important;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1) !important;
        padding: 8px 6px !important;
        min-width: 32px !important;
        max-width: 32px !important;
        font-size: 0.75rem !important;
    }


    /* 탭 스타일 최소만 수정 */
    .tabs {
        margin-top: 18px;
    }

    /* 푸터 */
    .agri-footer {
        font-size: 0.78rem;
        color: #7b8b6e;
        text-align: right;
        margin-top: 22px;
        padding-bottom: 10px;
    }
    """

    with gr.Blocks(title="🌾 RAGriculture - 농업 AI 도우미", theme=custom_theme, css=custom_css) as demo:
        demo.load(
            fn=None,
            js="""
            () => {
                setTimeout(() => {
                    document.querySelectorAll('.agri-input textarea, .agri-input input[type="text"]').forEach(el => {
                        el.setAttribute('spellcheck', 'false');
                    });
                }, 100);
            }
            """
        )
        # 헤더
        with gr.Row():
            gr.Markdown(
                """
                <div class="agri-header">
                  <div class="agri-header-left">
                    <div class="agri-logo-circle">🌾</div>
                    <div>
                      <div class="agri-title-main">RAGriculture · 농업 AI 도우미</div>
                      <div class="agri-title-sub">
                        작물 진단, 재배 정보, 일정 관리까지 한 번에 확인하세요.
                      </div>
                    </div>
                  </div>
                  <div>
                    <div class="agri-badge">
                      <span>🤖 AI 기반 농업 정보</span>
                    </div>
                  </div>
                </div>
                """
            )

        with gr.Tabs():
            # 탭 1: RAG 질문 / 이미지
            with gr.Tab("💬 농업 Q&A"):
                with gr.Row():
                    # 왼쪽 사이드바 (검색 기록)
                    with gr.Column(scale=1, min_width=250, visible=True) as history_sidebar:
                        with gr.Row():
                            with gr.Column(scale=3):
                                gr.Markdown(
                                    """
                                    <div class="agri-card" style="margin-bottom: 0;">
                                      <div class="agri-section-title">📚 최근 검색 기록</div>
                                    </div>
                                    """
                                )
                            history_sidebar_toggle_btn = gr.Button("◀", variant="secondary", size="sm", scale=1, min_width=35)
                        with gr.Row():
                            history_refresh_btn = gr.Button("🔄 새로고침", variant="secondary", size="sm", scale=1)
                            history_delete_btn = gr.Button("🗑️ 선택 삭제", variant="stop", size="sm", scale=1)
                        rag_history = gr.Dataframe(
                            label="최근 질문",
                            headers=["시간", "질문"],
                            interactive=False,
                            wrap=True,
                        )
                        # 선택된 행 인덱스를 저장할 숨겨진 컴포넌트
                        history_selected_row = gr.Number(value=-1, visible=False)
                    
                    # 사이드바가 닫혔을 때 표시되는 열기 버튼
                    history_sidebar_open_btn = gr.Button("▶", variant="secondary", size="sm", visible=False, elem_id="history_sidebar_open_btn", min_width=32)
                    
                    # 메인 컨텐츠 영역
                    with gr.Column(scale=3):
                        with gr.Row():
                            with gr.Column(scale=2):
                                q_card = gr.Markdown(
                                    value="""
                                    <div class="agri-card">
                                      <div class="agri-section-title">🌱 농업 관련 질문</div>
                                      <div class="agri-section-desc">
                                        병해, 생육 관리, 양분, 환경 관리 등 궁금한 점을 자유롭게 물어보세요.<br/>
                                        작물 사진을 함께 올리면 이미지 기반 분석도 수행합니다.
                                      </div>
                                    </div>
                                    """,
                                )

                                rag_question = gr.Textbox(
                                    label="질문",
                                    placeholder="예) 딸기 탄저병이 의심되는데 방제 방법과 약제 살포 시 주의사항을 알려줘.",
                                    lines=3,
                                    elem_classes=["agri-input", "agri-card"],
                                )

                                rag_image = gr.Image(
                                    label="작물 이미지 (선택)",
                                    type="filepath",
                                    sources=["upload"],
                                    elem_classes=["agri-image-box", "agri-card"],
                                )

                                rag_submit_btn = gr.Button("질문하기", variant="primary")

                            with gr.Column(scale=2):
                                rag_answer = gr.Markdown(
                                    label="답변",
                                    value="질문을 입력하고 `질문하기` 버튼을 눌러주세요.",
                                    elem_classes=["agri-card", "agri-answer"],
                                )

                        gr.Markdown(
                            """
                            <div class="agri-card">
                              <div class="agri-section-title">📝 사용 팁</div>
                              <ul style="padding-left:18px; margin:0; font-size:0.82rem; color:#5f6f55;">
                                <li>가능하면 <strong>작물명 + 증상 + 재배 환경</strong>을 함께 적어주세요.</li>
                                <li>예) "시설 토마토, 낮에는 30℃ 이상, 잎에 갈색 반점이 생기며 퍼지는 증상"</li>
                                <li>이미지를 올리면 작물을 자동 분류하고, 해당 작물 기준으로 답변합니다.</li>
                              </ul>
                            </div>
                            """
                        )

                # ---- 검색 기록 로직 (기존 코드 유지) ----
                def load_search_history():
                    """검색 기록 로드"""
                    global search_history_manager
                    if not search_history_manager:
                        return pd.DataFrame()
                    
                    try:
                        records = search_history_manager.get_recent_searches(limit=20, search_type="general")
                        if not records:
                            return pd.DataFrame(columns=["시간", "질문"])
                        
                        history_data = []
                        for record in records:
                            timestamp = record.get('timestamp', '')
                            try:
                                dt = datetime.fromisoformat(timestamp)
                                time_str = dt.strftime('%m/%d %H:%M')
                            except:
                                time_str = timestamp[:16]
                            
                            question = record.get('question', '')
                            history_data.append({
                                "시간": time_str,
                                "질문": question[:50] + "..." if len(question) > 50 else question
                            })
                        
                        return pd.DataFrame(history_data)
                    except Exception as e:
                        debug_print(f"⚠️ 검색 기록 로드 실패: {e}")
                        return pd.DataFrame(columns=["시간", "질문"])
                
                def use_history_record(evt: gr.SelectData):
                    """검색 기록 선택 시 질문과 답변에 적용"""
                    if evt.index[0] is not None:
                        global search_history_manager
                        if search_history_manager:
                            try:
                                records = search_history_manager.get_recent_searches(limit=20, search_type="general")
                                if evt.index[0] < len(records):
                                    selected_record = records[evt.index[0]]
                                    question = selected_record.get('question', '')
                                    answer = selected_record.get('answer_full') or selected_record.get('answer', '')
                                    return question, answer, evt.index[0]
                            except Exception as e:
                                debug_print(f"⚠️ 검색 기록 사용 실패: {e}")
                        return "", "", -1
                    return "", "", -1
                
                def delete_selected_history(selected_idx):
                    """선택된 검색 기록 삭제"""
                    global search_history_manager
                    if not search_history_manager or selected_idx < 0:
                        return load_search_history()
                    
                    try:
                        records = search_history_manager.get_recent_searches(limit=20, search_type="general")
                        if 0 <= selected_idx < len(records):
                            record = records[selected_idx]
                            search_id = record.get('id')
                            if search_id:
                                if search_history_manager.delete_search(search_id):
                                    debug_print(f"✅ 검색 기록 삭제 성공: {search_id}")
                                    return load_search_history()
                    except Exception as e:
                        debug_print(f"⚠️ 검색 기록 삭제 실패: {e}")
                    return load_search_history()
                
                # 사이드바 상태를 저장할 전역 변수
                history_sidebar_visible_state = [True]
                
                def toggle_history_sidebar():
                    """검색 기록 사이드바 여닫기"""
                    history_sidebar_visible_state[0] = not history_sidebar_visible_state[0]
                    btn_text = "◀" if history_sidebar_visible_state[0] else "▶"
                    open_btn_visible = not history_sidebar_visible_state[0]
                    return (
                        gr.update(visible=history_sidebar_visible_state[0]),
                        gr.update(value=btn_text),
                        gr.update(visible=open_btn_visible)
                    )
                
                history_sidebar_toggle_btn.click(
                    fn=toggle_history_sidebar,
                    outputs=[history_sidebar, history_sidebar_toggle_btn, history_sidebar_open_btn]
                )
                
                history_sidebar_open_btn.click(
                    fn=toggle_history_sidebar,
                    outputs=[history_sidebar, history_sidebar_toggle_btn, history_sidebar_open_btn]
                )
                
                history_refresh_btn.click(
                    fn=load_search_history,
                    outputs=[rag_history]
                )
                
                history_delete_btn.click(
                    fn=delete_selected_history,
                    inputs=[history_selected_row],
                    outputs=[rag_history]
                )
                
                rag_history.select(
                    fn=use_history_record,
                    outputs=[rag_question, rag_answer, history_selected_row]
                )
                
                # 페이지 로드 시 검색 기록 로드
                demo.load(
                    fn=load_search_history,
                    outputs=[rag_history]
                )
                
                # 질문 제출 후 검색 기록 새로고침
                rag_submit_btn.click(
                    fn=rag_query,
                    inputs=[rag_question, rag_image],
                    outputs=[rag_answer]
                ).then(
                    fn=load_search_history,
                    outputs=[rag_history]
                )
                
                rag_question.submit(
                    fn=rag_query,
                    inputs=[rag_question, rag_image],
                    outputs=[rag_answer]
                ).then(
                    fn=load_search_history,
                    outputs=[rag_history]
                )

            # 탭 2: 재배 일정표
            with gr.Tab("📅 재배 일정표"):
                with gr.Row():
                    # 왼쪽 사이드바 (검색 기록)
                    with gr.Column(scale=1, min_width=250, visible=True) as schedule_history_sidebar:
                        with gr.Row():
                            with gr.Column(scale=3):
                                gr.Markdown(
                                    """
                                    <div class="agri-card" style="margin-bottom: 0;">
                                      <div class="agri-section-title">📚 최근 검색 기록</div>
                                    </div>
                                    """
                                )
                            schedule_sidebar_toggle_btn = gr.Button("◀", variant="secondary", size="sm", scale=1, min_width=35)
                        with gr.Row():
                            schedule_history_refresh_btn = gr.Button("🔄 새로고침", variant="secondary", size="sm", scale=1)
                            schedule_history_delete_btn = gr.Button("🗑️ 선택 삭제", variant="stop", size="sm", scale=1)
                        schedule_history = gr.Dataframe(
                            label="최근 질문",
                            headers=["시간", "작물", "질문"],
                            interactive=False,
                            wrap=True,
                        )
                        # 선택된 행 인덱스를 저장할 숨겨진 컴포넌트
                        schedule_history_selected_row = gr.Number(value=-1, visible=False)
                    
                    # 사이드바가 닫혔을 때 표시되는 열기 버튼
                    schedule_sidebar_open_btn = gr.Button("▶", variant="secondary", size="sm", visible=False, elem_id="schedule_sidebar_open_btn", min_width=32)
                    
                    # 메인 컨텐츠 영역
                    with gr.Column(scale=3):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown(
                                    """
                                    <div class="agri-card">
                                      <div class="agri-section-title">📅 작물별 재배 일정표</div>
                                      <div class="agri-section-desc">
                                        현재는 <strong>딸기, 토마토</strong>에 대해 고정된 촉성재배 일정표를 제공합니다.
                                      </div>
                                    </div>
                                    """
                                )
                                schedule_crop = gr.Dropdown(
                                    label="작물 선택",
                                    choices=["딸기", "토마토"],
                                    value="딸기",
                                    info="재배할 작물을 선택하세요.",
                                )
                                schedule_location = gr.Textbox(
                                    label="경작지 위치",
                                    placeholder="예) 전라남도 순천시 해룡면 신대리",
                                    info="기상·병해충 정보를 위해 위치 정보가 필요합니다.",
                                    lines=2,
                                    elem_classes=["agri-input"],
                                )
                                schedule_generate_btn = gr.Button("일정표 불러오기", variant="primary")
                                schedule_weather_info = gr.Markdown(
                                    label="🌤️ 현재 기상 정보",
                                    value="",
                                    visible=False,
                                    elem_classes=["agri-card"],
                                )
                                schedule_status = gr.Markdown(value="", elem_classes=["agri-card"], visible=False)

                            with gr.Column(scale=2):
                                schedule_table = gr.Dataframe(
                                    label="재배 일정표",
                                    headers=["단계", "기간", "주요 작업"],
                                    interactive=False,
                                    wrap=True,
                                )
                                schedule_json = gr.JSON(
                                    label="일정표 데이터 (JSON)",
                                    visible=False
                                )

                        # 상세 작업 검색
                        gr.Markdown(
                            """
                            <div class="agri-card" style="margin-top:10px;">
                              <div class="agri-section-title">🔍 일정 내 세부 작업 검색</div>
                              <div class="agri-section-desc">
                                일정표에서 궁금한 작업(정식, 수확, 환기 관리 등)에 대해 더 자세히 물어보세요.<br/>
                                위치를 입력하면 해당 지역 기상과 병해충 위험을 반영해 답변합니다.
                              </div>
                            </div>
                            """
                        )

                        schedule_task_search = gr.Textbox(
                            label="검색할 작업/키워드",
                            placeholder="예) 정식 방법, 수확 시 주의사항, 환기 관리",
                            lines=2,
                            elem_classes=["agri-input"],
                        )
                        schedule_search_btn = gr.Button("상세 정보 검색", variant="primary")
                        schedule_search_result = gr.Markdown(
                            label="검색 결과",
                            value="검색 결과가 여기에 표시됩니다.",
                            elem_classes=["agri-card", "agri-answer"],
                        )

                # 일정 검색 기록 로직
                def load_schedule_history():
                    """일정표 검색 기록 로드"""
                    global search_history_manager
                    if not search_history_manager:
                        return pd.DataFrame()
                    
                    try:
                        records = search_history_manager.get_recent_searches(limit=20, search_type="schedule")
                        if not records:
                            return pd.DataFrame(columns=["시간", "작물", "질문"])
                        
                        history_data = []
                        for record in records:
                            timestamp = record.get('timestamp', '')
                            try:
                                dt = datetime.fromisoformat(timestamp)
                                time_str = dt.strftime('%m/%d %H:%M')
                            except:
                                time_str = timestamp[:16]
                            
                            crop = record.get('crop', '미지정')
                            question = record.get('question', '')
                            if crop and crop in question:
                                question = question.replace(crop, '').strip()
                            
                            history_data.append({
                                "시간": time_str,
                                "작물": crop,
                                "질문": question[:50] + "..." if len(question) > 50 else question
                            })
                        
                        return pd.DataFrame(history_data)
                    except Exception as e:
                        debug_print(f"⚠️ 일정표 검색 기록 로드 실패: {e}")
                        return pd.DataFrame(columns=["시간", "작물", "질문"])
                
                def use_schedule_history_record(evt: gr.SelectData):
                    """일정표 검색 기록 선택 시 검색어와 답변에 적용"""
                    if evt.index[0] is not None:
                        global search_history_manager
                        if search_history_manager:
                            try:
                                records = search_history_manager.get_recent_searches(limit=20, search_type="schedule")
                                if evt.index[0] < len(records):
                                    selected_record = records[evt.index[0]]
                                    task_query = selected_record.get('metadata', {}).get('task_query', '')
                                    if not task_query:
                                        question = selected_record.get('question', '')
                                        crop = selected_record.get('crop', '')
                                        if crop and crop in question:
                                            task_query = question.replace(crop, '').strip()
                                        else:
                                            task_query = question
                                    answer = selected_record.get('answer_full') or selected_record.get('answer', '')
                                    return task_query, answer, evt.index[0]
                            except Exception as e:
                                debug_print(f"⚠️ 일정표 검색 기록 사용 실패: {e}")
                        return "", "", -1
                    return "", "", -1
                
                def delete_selected_schedule_history(selected_idx):
                    """선택된 일정표 검색 기록 삭제"""
                    global search_history_manager
                    if not search_history_manager or selected_idx < 0:
                        return load_schedule_history()
                    
                    try:
                        records = search_history_manager.get_recent_searches(limit=20, search_type="schedule")
                        if 0 <= selected_idx < len(records):
                            record = records[selected_idx]
                            search_id = record.get('id')
                            if search_id:
                                if search_history_manager.delete_search(search_id):
                                    debug_print(f"✅ 일정표 검색 기록 삭제 성공: {search_id}")
                                    return load_schedule_history()
                    except Exception as e:
                        debug_print(f"⚠️ 일정표 검색 기록 삭제 실패: {e}")
                    return load_schedule_history()
                
                # 사이드바 상태를 저장할 전역 변수
                schedule_sidebar_visible_state = [True]
                
                def toggle_schedule_sidebar():
                    """일정표 검색 기록 사이드바 여닫기"""
                    schedule_sidebar_visible_state[0] = not schedule_sidebar_visible_state[0]
                    btn_text = "◀" if schedule_sidebar_visible_state[0] else "▶"
                    open_btn_visible = not schedule_sidebar_visible_state[0]
                    return (
                        gr.update(visible=schedule_sidebar_visible_state[0]),
                        gr.update(value=btn_text),
                        gr.update(visible=open_btn_visible)
                    )
                
                schedule_sidebar_toggle_btn.click(
                    fn=toggle_schedule_sidebar,
                    outputs=[schedule_history_sidebar, schedule_sidebar_toggle_btn, schedule_sidebar_open_btn]
                )
                
                schedule_sidebar_open_btn.click(
                    fn=toggle_schedule_sidebar,
                    outputs=[schedule_history_sidebar, schedule_sidebar_toggle_btn, schedule_sidebar_open_btn]
                )
                
                schedule_history_refresh_btn.click(
                    fn=load_schedule_history,
                    outputs=[schedule_history]
                )
                
                schedule_history_delete_btn.click(
                    fn=delete_selected_schedule_history,
                    inputs=[schedule_history_selected_row],
                    outputs=[schedule_history]
                )
                
                schedule_history.select(
                    fn=use_schedule_history_record,
                    outputs=[schedule_task_search, schedule_search_result, schedule_history_selected_row]
                )
                
                
                # 페이지 로드 시 일정표 검색 기록 로드
                demo.load(
                    fn=load_schedule_history,
                    outputs=[schedule_history]
                )

                def update_schedule(crop, location):
                    """일정표 생성 및 출력 업데이트 (기상 데이터 포함)"""
                    df, json_data, status_msg = generate_schedule_web_chatgpt(crop, location)
                    
                    # 기상 데이터 수집
                    weather_display = ""
                    if location and location.strip():
                        try:
                            geo_manager = get_geo_manager()
                            if geo_manager:
                                result = geo_manager.get_final_address(location.strip(), verbose=False)
                                if result:
                                    farm_info_dict = {
                                        'longitude': result['coordinates']['longitude'],
                                        'latitude': result['coordinates']['latitude'],
                                        'road_address': result['final_address']['road_address'],
                                        'legal_address': result['final_address']['legal_address'],
                                        'full_address': result['final_address']['full_address'],
                                        'user_input': location.strip()
                                    }
                                    
                                    # 기상 데이터 수집
                                    if farm_info_dict and weather_manager:
                                        latitude = farm_info_dict.get('latitude')
                                        longitude = farm_info_dict.get('longitude')
                                        
                                        debug_print(f"🌤️ 일정표 생성 시 기상 데이터 수집 시작: lat={latitude}, lon={longitude}")
                                        
                                        if latitude and longitude:
                                            try:
                                                ultra_short = weather_manager.get_ultra_short_forecast(latitude, longitude)
                                                short = weather_manager.get_short_forecast(latitude, longitude)
                                                
                                                debug_print(f"📊 초단기 예보: {len(ultra_short) if ultra_short else 0}개, 단기 예보: {len(short) if short else 0}개")
                                                
                                                current_forecast = None
                                                if ultra_short or short:
                                                    for fcst in (ultra_short or []):
                                                        if fcst.get("temp") is not None:
                                                            current_forecast = fcst
                                                            break
                                                    if not current_forecast and short:
                                                        for fcst in short:
                                                            if fcst.get("temp") is not None:
                                                                current_forecast = fcst
                                                                break
                                                
                                                if current_forecast:
                                                    location_name = farm_info_dict.get('road_address') or farm_info_dict.get('legal_address') or "해당 위치"
                                                    
                                                    debug_print(f"✅ 현재 기상 예보 데이터 발견: {current_forecast}")
                                                    
                                                    # 기상 데이터 표시용 포맷팅
                                                    weather_display = f"""### 🌤️ 현재 기상 정보
**위치**: {location_name}

"""
                                                    if current_forecast.get("temp") is not None:
                                                        weather_display += f"- **현재 기온**: {current_forecast.get('temp')}℃\n"
                                                    if current_forecast.get("temp_max") and current_forecast.get("temp_min"):
                                                        weather_display += f"- **예상 기온 범위**: {current_forecast['temp_min']}~{current_forecast['temp_max']}℃\n"
                                                    if current_forecast.get("rh"):
                                                        weather_display += f"- **현재 습도**: {current_forecast.get('rh')}%\n"
                                                    if current_forecast.get("wind_speed"):
                                                        weather_display += f"- **현재 풍속**: {current_forecast.get('wind_speed')}m/s\n"
                                                    if current_forecast.get("precipitation"):
                                                        precip = current_forecast.get("precipitation")
                                                        if precip and precip != "0" and "없음" not in str(precip):
                                                            weather_display += f"- **강수량**: {precip}\n"
                                                    
                                                    debug_print(f"📝 기상 데이터 표시 문자열 생성 완료: {len(weather_display)}자")
                                                    
                                                    # 병해충 예측 수행
                                                    if pest_predictor and crop:
                                                        try:
                                                            debug_print(f"🐛 병해충 예측 시작: 작물={crop}, 위치={location_name}")
                                                            pest_prediction = pest_predictor.predict_pest_risk(
                                                                latitude=latitude,
                                                                longitude=longitude,
                                                                crop=crop,
                                                                growth_stage="생육기",  # 기본값, 필요시 수정 가능
                                                                reference_time=None  # 현재 시간 사용
                                                            )
                                                            
                                                            debug_print(f"✅ 병해충 예측 완료: 전체 위험도={pest_prediction.get('overall_risk', 'N/A')}")
                                                            
                                                            # 병해충 정보를 기상데이터 표시에 추가
                                                            weather_display += "\n---\n\n"
                                                            weather_display += f"### 🐛 병해충 발생 위험도\n"
                                                            weather_display += f"**전체 위험도**: {pest_prediction.get('overall_risk', '낮음')}\n\n"
                                                            
                                                            # 위험한 병해충 목록
                                                            high_risk_pests = [pf for pf in pest_prediction.get('pest_forecasts', []) 
                                                                              if pf.get('risk_level') in ['경계', '심각']]
                                                            
                                                            if high_risk_pests:
                                                                weather_display += "**현재 위험한 병해충**:\n"
                                                                for pest in high_risk_pests:
                                                                    pest_name = pest.get('pest_name', '알 수 없음')
                                                                    risk_level = pest.get('risk_level', '알 수 없음')
                                                                    forecast_period = pest.get('forecast_period', '')
                                                                    weather_display += f"- **{pest_name}**: {risk_level}"
                                                                    if forecast_period:
                                                                        weather_display += f" ({forecast_period})"
                                                                    weather_display += "\n"
                                                            else:
                                                                weather_display += "**현재 위험한 병해충**: 없음\n"
                                                            
                                                            debug_print(f"📝 병해충 정보 추가 완료: {len(high_risk_pests)}개 위험 병해충")
                                                            
                                                        except Exception as e:
                                                            debug_print(f"⚠️ 병해충 예측 실패: {e}")
                                                            import traceback
                                                            debug_print(traceback.format_exc())
                                                    
                                                else:
                                                    debug_print("⚠️ 기상 예보 데이터에서 현재 기온 정보를 찾을 수 없습니다.")
                                            except Exception as e:
                                                debug_print(f"⚠️ 기상 정보 수집 실패: {e}")
                                                import traceback
                                                debug_print(traceback.format_exc())
                        except Exception as e:
                            debug_print(f"⚠️ 지오코딩 실패: {e}")
                    
                    # 상태 메시지가 있을 때만 표시
                    status_update = gr.update(value=status_msg, visible=True) if status_msg and status_msg.strip() else gr.update(value="", visible=False)
                    
                    # 기상 데이터가 있으면 표시, 없으면 숨김
                    if weather_display and weather_display.strip():
                        debug_print(f"✅ 기상 데이터 표시: {len(weather_display)}자")
                        weather_update = gr.update(value=weather_display, visible=True)
                    else:
                        debug_print("⚠️ 기상 데이터가 없어 표시하지 않습니다.")
                        weather_update = gr.update(value="", visible=False)
                    
                    return df, json_data, status_update, weather_update
                
                def search_schedule_task(crop, location, task_query, schedule_df):
                    """일정표 작업 상세 검색 (기상 데이터 및 병해충 예측 포함)"""
                    if not task_query or not task_query.strip():
                        return "검색어를 입력해주세요."
                    
                    if schedule_df is None or schedule_df.empty:
                        return "일정표를 먼저 생성해주세요."
                    
                    # 위치 정보 필수 체크
                    if not location or not location.strip():
                        return "위치 정보를 입력해주세요."
                    
                    try:
                        enhanced_question = f"{crop} {task_query.strip()}"
                        
                        # 위치 정보 설정
                        farm_info_dict = None
                        if location and location.strip():
                            try:
                                geo_manager = get_geo_manager()
                                if geo_manager:
                                    result = geo_manager.get_final_address(location.strip(), verbose=False)
                                    if result:
                                        road_addr = result['final_address']['road_address']
                                        legal_addr = result['final_address']['legal_address']
                                        display_address = road_addr or legal_addr or location.strip()
                                        
                                        farm_info_dict = {
                                            'longitude': result['coordinates']['longitude'],
                                            'latitude': result['coordinates']['latitude'],
                                            'road_address': road_addr,
                                            'legal_address': legal_addr,
                                            'full_address': result['final_address']['full_address'],
                                            'user_input': location.strip()
                                        }
                                        set_farm_info(farm_info_dict)
                            except Exception as e:
                                debug_print(f"⚠️ 지오코딩 실패: {e}")
                        
                        if not farm_info_dict:
                            return "위치 정보를 확인할 수 없습니다.", gr.update(value="", visible=False)
                        
                        # 기상 데이터 표시용 변수 초기화
                        weather_display = ""
                        
                        # 기상 데이터 수집 및 질문에 포함
                        current_forecast = None
                        location_name = None
                        if farm_info_dict and weather_manager:
                            latitude = farm_info_dict.get('latitude')
                            longitude = farm_info_dict.get('longitude')
                            
                            debug_print(f"🌤️ 기상 데이터 수집 시작: lat={latitude}, lon={longitude}")
                            
                            if latitude and longitude:
                                try:
                                    ultra_short = weather_manager.get_ultra_short_forecast(latitude, longitude)
                                    short = weather_manager.get_short_forecast(latitude, longitude)
                                    
                                    debug_print(f"📊 초단기 예보: {len(ultra_short) if ultra_short else 0}개, 단기 예보: {len(short) if short else 0}개")
                                    
                                    if ultra_short or short:
                                        for fcst in (ultra_short or []):
                                            if fcst.get("temp") is not None:
                                                current_forecast = fcst
                                                break
                                        if not current_forecast and short:
                                            for fcst in short:
                                                if fcst.get("temp") is not None:
                                                    current_forecast = fcst
                                                    break
                                        
                                        if current_forecast:
                                            location_name = farm_info_dict.get('road_address') or farm_info_dict.get('legal_address') or "해당 위치"
                                            
                                            debug_print(f"✅ 현재 기상 예보 데이터 발견: {current_forecast}")
                                            
                                            # 기상 데이터 표시용 포맷팅
                                            weather_display = f"""### 🌤️ 현재 기상 정보
**위치**: {location_name}

"""
                                            if current_forecast.get("temp") is not None:
                                                weather_display += f"- **현재 기온**: {current_forecast.get('temp')}℃\n"
                                            if current_forecast.get("temp_max") and current_forecast.get("temp_min"):
                                                weather_display += f"- **예상 기온 범위**: {current_forecast['temp_min']}~{current_forecast['temp_max']}℃\n"
                                            if current_forecast.get("rh"):
                                                weather_display += f"- **현재 습도**: {current_forecast.get('rh')}%\n"
                                            if current_forecast.get("wind_speed"):
                                                weather_display += f"- **현재 풍속**: {current_forecast.get('wind_speed')}m/s\n"
                                            if current_forecast.get("precipitation"):
                                                precip = current_forecast.get("precipitation")
                                                if precip and precip != "0" and "없음" not in str(precip):
                                                    weather_display += f"- **강수량**: {precip}\n"
                                            
                                            debug_print(f"📝 기상 데이터 표시 문자열 생성 완료: {len(weather_display)}자")
                                            
                                            # RAG 질문에 포함할 기상 컨텍스트 (기존 방식 유지)
                                            weather_context = "\n\n[중요: 아래 현재 기상 조건을 반드시 고려하여 해당 지역에 맞는 맞춤형 조언을 제공해주세요]\n"
                                            weather_context += f"위치: {location_name}\n"
                                            if current_forecast.get("temp") is not None:
                                                weather_context += f"현재 기온: {current_forecast.get('temp')}℃\n"
                                            if current_forecast.get("temp_max") and current_forecast.get("temp_min"):
                                                weather_context += f"예상 기온 범위: {current_forecast['temp_min']}~{current_forecast['temp_max']}℃\n"
                                            if current_forecast.get("rh"):
                                                weather_context += f"현재 습도: {current_forecast.get('rh')}%\n"
                                            if current_forecast.get("wind_speed"):
                                                weather_context += f"현재 풍속: {current_forecast.get('wind_speed')}m/s\n"
                                            if current_forecast.get("precipitation"):
                                                precip = current_forecast.get("precipitation")
                                                if precip and precip != "0" and "없음" not in str(precip):
                                                    weather_context += f"강수량: {precip}\n"
                                            weather_context += "\n위 기상 조건을 반드시 고려하여 현재 시점과 지역에 맞는 구체적이고 실용적인 조언을 제공해주세요."
                                            enhanced_question = f"{enhanced_question}{weather_context}"
                                        else:
                                            weather_display = ""
                                            debug_print("⚠️ 기상 예보 데이터에서 현재 기온 정보를 찾을 수 없습니다.")
                                    else:
                                        debug_print("⚠️ 초단기/단기 예보 데이터가 없습니다.")
                                except Exception as e:
                                    debug_print(f"⚠️ 기상 정보 수집 실패: {e}")
                                    import traceback
                                    debug_print(traceback.format_exc())
                                    weather_display = ""
                            else:
                                debug_print("⚠️ 위도/경도 정보가 없습니다.")
                        else:
                            if not farm_info_dict:
                                debug_print("⚠️ 경작지 정보가 없습니다.")
                            if not weather_manager:
                                debug_print("⚠️ 기상 예보 관리자가 초기화되지 않았습니다.")
                        
                        # 일정표 컨텍스트 추가 (핵심 변경 사항)
                        if schedule_df is not None and not schedule_df.empty:
                            schedule_context = "\n\n[참고: 현재 작물의 재배 일정표]\n"
                            # DataFrame을 문자열로 변환 (기간: 작업 - 상세)
                            for _, row in schedule_df.iterrows():
                                period = row.get('기간', '')        # ✅
                                task = row.get('단계', '')         # ✅
                                details = row.get('주요 작업', '')  # ✅
                                schedule_context += f"- {period}: {task} ({details})\n"
                            
                            schedule_context += "\n[지침: 답변 시 반드시 위 일정표를 언급하세요]\n"
                            schedule_context += "1. 질문한 시기나 작업이 위 일정표의 어느 단계(Period, Task)에 해당하는지 명시하세요.\n"
                            schedule_context += "2. 해당 단계의 '상세 내용(Details)'을 인용하여 답변을 구성하세요.\n"
                            schedule_context += "3. 예시: '현재 일정표상 [10월 중순]은 [파종] 단계이므로...'\n"
                            
                            enhanced_question += schedule_context
                            debug_print(f"📅 일정표 컨텍스트 강력 추가됨 ({len(schedule_df)}개 항목)")
                        
                        # RAG 검색 수행
                        result = run_rag_system(
                            question=enhanced_question,
                            image_path=None,
                            config={"app": rag_app}
                        )
                        
                        answer = result.get("answer", result.get("generation", "답변을 생성할 수 없습니다."))
                        final_answer = fix_markdown_strikethrough(answer)
                        
                        # 검색 기록 저장
                        global search_history_manager
                        if search_history_manager:
                            try:
                                location_name_for_history = farm_info_dict.get('road_address') or farm_info_dict.get('legal_address') or location
                                search_history_manager.add_search(
                                    question=f"{crop} {task_query.strip()}",
                                    answer=final_answer,
                                    search_type="schedule",
                                    location=location_name_for_history,
                                    crop=crop,
                                    metadata={"task_query": task_query.strip()}
                                )
                            except Exception as e:
                                debug_print(f"⚠️ 검색 기록 저장 실패: {e}")
                        
                        return final_answer
                        
                    except Exception as e:
                        debug_print(f"❌ 상세 작업 검색 오류: {e}")
                        import traceback
                        debug_print(traceback.format_exc())
                        return f"검색 중 오류가 발생했습니다: {str(e)}"
                schedule_generate_btn.click(
                    fn=update_schedule,
                    inputs=[schedule_crop, schedule_location],
                    outputs=[schedule_table, schedule_json, schedule_status, schedule_weather_info]
                )
                
                # 일정표 검색 후 검색 기록 새로고침
                schedule_search_btn.click(
                    fn=search_schedule_task,
                    inputs=[schedule_crop, schedule_location, schedule_task_search, schedule_table],
                    outputs=[schedule_search_result]
                ).then(
                    fn=load_schedule_history,
                    outputs=[schedule_history]
                )
                
                schedule_task_search.submit(
                    fn=search_schedule_task,
                    inputs=[schedule_crop, schedule_location, schedule_task_search, schedule_table],
                    outputs=[schedule_search_result]
                ).then(
                    fn=load_schedule_history,
                    outputs=[schedule_history]
                )

        gr.Markdown(
            """
            <div class="agri-footer">
              🌾 RAGriculture · 농업 데이터를 바탕으로 한 지능형 상담 도우미
            </div>
            """
        )
    
    return demo


# ============================================================================
# 5. 실행
# ============================================================================

if __name__ == "__main__":
    # 시스템 초기화 확인
    if rag_app is None:
        initialize_systems()
    
    # Gradio 인터페이스 생성 및 실행
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )


