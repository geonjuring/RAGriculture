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
    global rag_system, rag_app, llm, weather_manager, pest_predictor
    
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
    """마크다운 취소선 문제 해결: 숫자 범위를 올바르게 표시"""
    import re
    
    # 잘못된 취소선 제거: ~~숫자~~ 패턴을 숫자로 변환 (예: ~~510~~ → 510)
    # 숫자만 있는 취소선은 잘못된 것으로 간주하고 제거
    text = re.sub(r'~~(\d+(?:\.\d+)?)~~', r'\1', text)
    
    # 소수점과 숫자가 섞인 취소선 패턴 제거 (예: ~~9.510~~ → 9.510)
    text = re.sub(r'~~(\d+\.\d+)~~', r'\1', text)
    
    # 4자리 숫자+특수문자 → 2자리~2자리+특수문자 (예: 1015℃ → 10~15℃)
    text = re.sub(r'(\d{2})(\d{2})([℃%])', r'\1~\2\3', text)
    
    # 4자리 숫자+공백+특수문자 → 2자리~2자리+특수문자 (예: 7080 % → 70~80%)
    text = re.sub(r'(\d{2})(\d{2})\s*([%℃])', r'\1~\2\3', text)
    
    # 소수점 포함 숫자 패턴 수정 (예: 9.510일 → 9.5~10일, 단 실제로는 9월 5~10일 의미일 수 있음)
    # 하지만 이 패턴은 너무 공격적이므로 주석 처리
    # text = re.sub(r'(\d+)\.(\d{2})(\d{2})(일|월|년|℃|%)', r'\1.\2~\3\4', text)
    
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
        
        # 출처 정보 추가
        documents = result.get("retrieved_docs", result.get("documents", []))
        if documents:
            sources = []
            for doc in documents[:3]:  # 상위 3개만 표시
                source = doc.metadata.get('source', 'Unknown')
                # 파일명만 표시 (경로 제거)
                if os.path.sep in source:
                    source = os.path.basename(source)
                sources.append(f"- {source}")
            
            if sources:
                answer += "\n\n**📚 출처:**\n" + "\n".join(sources)
        
        # 품질 점수 표시 (선택사항)
        quality_scores = result.get("quality_scores", {})
        if quality_scores and quality_scores.get("overall_score", 0) > 0:
            overall_score = quality_scores.get("overall_score", 0)
            answer += f"\n\n**📊 답변 품질 점수:** {overall_score:.2f}/1.00"
        
        # 마크다운 취소선 문제 해결 (예: 1015℃ → 10~15℃, 7080% → 70~80%)
        answer = fix_markdown_strikethrough(answer)
        
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
    """Gradio 인터페이스 생성"""
    
    with gr.Blocks(title="RAG 시스템", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🌾 농업 정보 검색 시스템")
        gr.Markdown("농업 관련 질문에 대한 전문적인 답변 제공")
        
        with gr.Tabs():
            # 탭 1: RAG 질문 답변 (이미지 지원)
            with gr.Tab("🔍 RAG 질문 답변"):
                gr.Markdown("### 농업 관련 질문을 입력하세요 (이미지 업로드 가능)")
                gr.Markdown("💡 **새로운 기능**: 작물 이미지를 업로드하면 자동으로 작물을 분류하고 관련 정보를 제공합니다.")
                
                rag_question = gr.Textbox(
                    label="질문",
                    placeholder="예: 딸기 탄저병 방제법은?",
                    lines=3
                )
                rag_image = gr.Image(
                    label="작물 이미지 (선택사항)",
                    type="filepath",
                    sources=["upload"],
                    visible=True
                )
                rag_submit_btn = gr.Button("검색", variant="primary")
                
                rag_answer = gr.Markdown(label="답변")
                
                rag_submit_btn.click(
                    fn=rag_query,
                    inputs=[rag_question, rag_image],
                    outputs=[rag_answer]
                )
                
                rag_question.submit(
                    fn=rag_query,
                    inputs=[rag_question, rag_image],
                    outputs=[rag_answer]
                )
            
            # 탭 2: 일정표 생성
            with gr.Tab("📅 재배 일정표 생성"):
                gr.Markdown("### 작물별 재배 일정표 생성")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        schedule_crop = gr.Dropdown(
                            label="작물 선택",
                            choices=["딸기", "토마토"],
                            value="딸기",
                            info="재배할 작물을 선택하세요"
                        )
                        schedule_location = gr.Textbox(
                            label="경작지 위치 (선택사항)",
                            placeholder="예: 전라남도 순천시 해룡면 신대리",
                            info="지오코딩을 통해 정확한 위치 정보로 변환됩니다",
                            lines=2
                        )
                        schedule_generate_btn = gr.Button("일정표 생성", variant="primary")
                        schedule_status = gr.Markdown(value="")
                    
                    with gr.Column(scale=2):
                        schedule_table = gr.Dataframe(
                            label="재배 일정표",
                            headers=["단계", "기간", "주요 작업"],
                            interactive=False,
                            wrap=True
                        )
                        schedule_json = gr.JSON(
                            label="일정표 데이터 (JSON)",
                            visible=False
                        )
                
                # 상세 작업 검색 섹션
                gr.Markdown("---")
                gr.Markdown("### 🔍 상세 작업 검색")
                gr.Markdown("일정표의 특정 작업에 대한 상세 정보를 검색할 수 있습니다.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        schedule_task_search = gr.Textbox(
                            label="작업명 또는 검색어",
                            placeholder="예: 정식, 수확, 환기 관리, 병해충 방제",
                            info="일정표의 작업명을 입력하거나 자유롭게 검색어를 입력하세요",
                            lines=2
                        )
                        schedule_search_btn = gr.Button("상세 검색", variant="primary")
                        schedule_search_info = gr.Markdown(
                            value="💡 일정표를 먼저 생성한 후 검색하세요.",
                            visible=True
                        )
                    
                    with gr.Column(scale=2):
                        schedule_search_result = gr.Markdown(
                            label="검색 결과",
                            value="검색 결과가 여기에 표시됩니다."
                        )
                
                def update_schedule(crop, location):
                    """일정표 생성 및 출력 업데이트"""
                    df, json_data, status_msg = generate_schedule_web_chatgpt(crop, location)
                    return df, json_data, status_msg
                
                def search_schedule_task(crop, location, task_query, schedule_df):
                    """일정표 작업 상세 검색 (기상 데이터 및 병해충 예측 포함)"""
                    if not task_query or not task_query.strip():
                        return "검색어를 입력해주세요.", "💡 검색어를 입력하고 검색 버튼을 클릭하세요."
                    
                    if schedule_df is None or schedule_df.empty:
                        return "일정표를 먼저 생성해주세요.", "⚠️ 일정표를 먼저 생성한 후 검색하세요."
                    
                    try:
                        # 작물명과 작업명을 결합하여 질문 생성
                        enhanced_question = f"{crop} {task_query.strip()}"
                        
                        # 위치 정보 설정 (일정표 생성 시 사용한 위치 재사용)
                        farm_info_dict = None
                        if location and location.strip():
                            try:
                                geo_manager = get_geo_manager()
                                if geo_manager:
                                    result = geo_manager.get_final_address(location.strip(), verbose=False)
                                    if result:
                                        # road_address가 None이면 legal_address 또는 user_input 사용
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
                                        debug_print(f"✅ 위치 정보 설정 완료: {display_address}")
                            except Exception as geo_error:
                                debug_print(f"⚠️ 지오코딩 실패: {geo_error}")
                        
                        # 기상 데이터 및 병해충 예측 정보 수집
                        weather_info = ""
                        pest_info = ""
                        
                        if farm_info_dict and weather_manager:
                            latitude = farm_info_dict.get('latitude')
                            longitude = farm_info_dict.get('longitude')
                            
                            if latitude and longitude:
                                # 현재 기상 상태
                                try:
                                    # 직접 좌표를 사용하여 기상 데이터 가져오기
                                    ultra_short = weather_manager.get_ultra_short_forecast(latitude, longitude)
                                    short = weather_manager.get_short_forecast(latitude, longitude)
                                    
                                    if ultra_short or short:
                                        # 기온 데이터가 있는 첫 번째 예보 찾기
                                        current_forecast = None
                                        
                                        # 초단기예보에서 기온이 있는 첫 번째 항목 찾기
                                        if ultra_short:
                                            for fcst in ultra_short:
                                                if fcst.get("temp") is not None:
                                                    current_forecast = fcst
                                                    break
                                            # 기온이 없으면 첫 번째 항목 사용
                                            if current_forecast is None and ultra_short:
                                                current_forecast = ultra_short[0]
                                        
                                        # 초단기예보에 기온이 없으면 단기예보에서 찾기
                                        if current_forecast is None or current_forecast.get("temp") is None:
                                            if short:
                                                for fcst in short:
                                                    if fcst.get("temp") is not None:
                                                        current_forecast = fcst
                                                        break
                                                # 기온이 없으면 첫 번째 항목 사용
                                                if current_forecast is None and short:
                                                    current_forecast = short[0]
                                        
                                        if current_forecast:
                                            # 위치 정보 추출
                                            location_name = farm_info_dict.get('road_address') or farm_info_dict.get('legal_address') or "해당 위치"
                                            
                                            # 기상 정보 문자열 생성
                                            weather_info = "## 🌤️ 현재 기상 상태\n\n"
                                            weather_info += f"📍 **위치**: {location_name}\n\n"
                                            weather_info += f"🌤️ 현재 기상 조건 ({location_name} 기준):\n"
                                            
                                            # 기온 정보를 강조하여 표시
                                            if current_forecast.get("temp") is not None:
                                                weather_info += f"🌡️ **현재 기온: {current_forecast['temp']}℃**\n"
                                            
                                            if current_forecast.get("temp_max") is not None and current_forecast.get("temp_min") is not None:
                                                weather_info += f"🌡️ **예상 기온 범위: {current_forecast['temp_min']}~{current_forecast['temp_max']}℃**\n"
                                            
                                            if current_forecast.get("rh"):
                                                weather_info += f"- 습도: {current_forecast['rh']}%\n"
                                            
                                            if current_forecast.get("precipitation"):
                                                weather_info += f"- 강수량: {current_forecast['precipitation']}\n"
                                            
                                            if current_forecast.get("wind_speed"):
                                                weather_info += f"- 풍속: {current_forecast['wind_speed']}m/s\n"
                                            
                                            if current_forecast.get("sky_condition"):
                                                weather_info += f"- 하늘 상태: {current_forecast['sky_condition']}\n"
                                            
                                            weather_info += "\n"
                                            
                                            # 단기 예보 추가 (오늘~3일)
                                            if short:
                                                weather_info += "### 📅 단기 예보 (오늘~3일)\n\n"
                                                # 날짜별로 그룹화
                                                from collections import defaultdict
                                                from datetime import datetime
                                                daily_forecasts = defaultdict(list)
                                                
                                                for fcst in short[:15]:  # 상위 15개만
                                                    fcst_date = fcst.get("fcst_datetime", "")
                                                    if fcst_date:
                                                        try:
                                                            date_obj = datetime.strptime(fcst_date, "%Y%m%d%H%M")
                                                            date_key = date_obj.strftime("%Y-%m-%d")
                                                            daily_forecasts[date_key].append(fcst)
                                                        except:
                                                            pass
                                                
                                                for date_key in sorted(daily_forecasts.keys())[:3]:  # 최대 3일
                                                    day_forecasts = daily_forecasts[date_key]
                                                    if day_forecasts:
                                                        temps = [f.get("temp") for f in day_forecasts if f.get("temp") and f.get("temp") is not None]
                                                        rh_values = [f.get("rh") for f in day_forecasts if f.get("rh") and f.get("rh") is not None]
                                                        precip = [f.get("precipitation") for f in day_forecasts if f.get("precipitation")]
                                                        
                                                        weather_info += f"**{date_key}**: "
                                                        if temps:
                                                            weather_info += f"온도 {min(temps)}~{max(temps)}℃, "
                                                        if rh_values:
                                                            weather_info += f"습도 평균 {sum(rh_values)/len(rh_values):.0f}%, "
                                                        if any(p and p != "0" and "없음" not in str(p) for p in precip):
                                                            weather_info += "강수 예상, "
                                                        weather_info = weather_info.rstrip(", ") + "\n"
                                                
                                                weather_info += "\n"
                                        else:
                                            # current_forecast가 없어도 단기 예보만이라도 표시
                                            if short:
                                                weather_info = "## 🌤️ 현재 기상 상태\n\n"
                                                weather_info += "### 📅 단기 예보 (오늘~3일)\n\n"
                                                from collections import defaultdict
                                                from datetime import datetime
                                                daily_forecasts = defaultdict(list)
                                                
                                                for fcst in short[:15]:
                                                    fcst_date = fcst.get("fcst_datetime", "")
                                                    if fcst_date:
                                                        try:
                                                            date_obj = datetime.strptime(fcst_date, "%Y%m%d%H%M")
                                                            date_key = date_obj.strftime("%Y-%m-%d")
                                                            daily_forecasts[date_key].append(fcst)
                                                        except:
                                                            pass
                                                
                                                for date_key in sorted(daily_forecasts.keys())[:3]:
                                                    day_forecasts = daily_forecasts[date_key]
                                                    if day_forecasts:
                                                        temps = [f.get("temp") for f in day_forecasts if f.get("temp") and f.get("temp") is not None]
                                                        rh_values = [f.get("rh") for f in day_forecasts if f.get("rh") and f.get("rh") is not None]
                                                        precip = [f.get("precipitation") for f in day_forecasts if f.get("precipitation")]
                                                        
                                                        weather_info += f"**{date_key}**: "
                                                        if temps:
                                                            weather_info += f"온도 {min(temps)}~{max(temps)}℃, "
                                                        if rh_values:
                                                            weather_info += f"습도 평균 {sum(rh_values)/len(rh_values):.0f}%, "
                                                        if any(p and p != "0" and "없음" not in str(p) for p in precip):
                                                            weather_info += "강수 예상, "
                                                        weather_info = weather_info.rstrip(", ") + "\n"
                                                
                                                weather_info += "\n"
                                except Exception as e:
                                    debug_print(f"⚠️ 기상 정보 수집 실패: {e}")
                                
                                # 병해충 예측 정보 (작물명이 포함된 경우)
                                if pest_predictor:
                                    try:
                                        # 질문에서 생육 단계 추출
                                        growth_stage = "생육기"  # 기본값
                                        task_lower = task_query.lower()
                                        if "개화" in task_query or "개화기" in task_query:
                                            growth_stage = "개화기"
                                        elif "착과" in task_query or "착과기" in task_query:
                                            growth_stage = "착과기"
                                        elif "수확" in task_query or "수확기" in task_query:
                                            growth_stage = "수확기"
                                        
                                        pest_context = pest_predictor.get_pest_forecast_context(
                                            latitude, longitude, crop, growth_stage
                                        )
                                        
                                        if pest_context:
                                            pest_info = f"## ⚠️ 병해충 예측 정보\n\n{pest_context}\n\n"
                                            
                                            # 추가 상세 정보 제공
                                            forecast = pest_predictor.predict_pest_risk(
                                                latitude, longitude, crop, growth_stage
                                            )
                                            
                                            if forecast and forecast.get("pest_forecasts"):
                                                pest_info += "### 📊 상세 예측 정보\n\n"
                                                
                                                for pf in forecast.get("pest_forecasts", []):
                                                    if pf["risk_level"] in ["주의", "경계", "심각"]:
                                                        pest_info += f"**{pf['pest_name']}** ({pf['forecast_period']})\n"
                                                        pest_info += f"- 위험도: {pf['risk_level']} (점수: {pf['risk_score']}/3)\n"
                                                        pest_info += f"- 예상 조건:\n"
                                                        pest_info += f"  • 연속 고습 시간: {pf['conditions']['max_continuous_humid_hours']}시간\n"
                                                        pest_info += f"  • 총 강수량: {pf['conditions']['total_rain']}mm\n"
                                                        if pf['conditions'].get('risk_periods'):
                                                            pest_info += f"  • 위험 시간대: {pf['conditions']['risk_periods'][0]['start']}\n"
                                                        pest_info += f"- 권장 조치: {pf['recommendation']}\n\n"
                                                
                                                pest_info += "\n"
                                    except Exception as e:
                                        debug_print(f"⚠️ 병해충 예측 정보 수집 실패: {e}")
                        
                        # RAG 검색 수행
                        result = run_rag_system(
                            question=enhanced_question,
                            image_path=None,
                            config={"app": rag_app}
                        )
                        
                        answer = result.get("answer", result.get("generation", "답변을 생성할 수 없습니다."))
                        
                        # 기상 정보 및 병해충 예측 정보를 답변 상단에 추가
                        final_answer = ""
                        
                        # 기상 정보가 있는 경우 표시
                        if weather_info:
                            final_answer += weather_info
                        elif farm_info_dict and weather_manager:
                            # 기상 정보를 가져오려고 했지만 실패한 경우 안내
                            final_answer += "## ⚠️ 기상 정보\n\n"
                            final_answer += "기상 데이터를 가져올 수 없습니다. 다음을 확인해주세요:\n"
                            final_answer += "- API 키 설정 확인 (`.env` 파일의 `WEATHER_API_KEY`)\n"
                            final_answer += "- 네트워크 연결 확인\n"
                            final_answer += "- 기상청 API 서비스 상태 확인\n\n"
                        
                        # 병해충 예측 정보가 있는 경우 표시
                        if pest_info:
                            final_answer += pest_info
                        elif farm_info_dict and pest_predictor and (crop in ["토마토", "딸기"]):
                            # 병해충 예측을 시도했지만 결과가 없는 경우
                            # (위험도가 낮은 경우이므로 별도 안내 불필요)
                            pass
                        
                        final_answer += "## 💡 작업 상세 정보\n\n" + answer
                        
                        # 출처 정보 추가
                        documents = result.get("retrieved_docs", result.get("documents", []))
                        if documents:
                            sources = []
                            for doc in documents[:3]:  # 상위 3개만 표시
                                source = doc.metadata.get('source', 'Unknown')
                                if os.path.sep in source:
                                    source = os.path.basename(source)
                                sources.append(f"- {source}")
                            
                            if sources:
                                final_answer += "\n\n**📚 출처:**\n" + "\n".join(sources)
                        
                        # 마크다운 취소선 문제 해결
                        final_answer = fix_markdown_strikethrough(final_answer)
                        
                        info_msg = f"✅ '{task_query.strip()}'에 대한 검색 완료"
                        if weather_info:
                            info_msg += " (기상 정보 포함)"
                        if pest_info:
                            info_msg += " (병해충 예측 포함)"
                        
                        return final_answer, info_msg
                        
                    except Exception as e:
                        debug_print(f"❌ 상세 작업 검색 오류: {e}")
                        import traceback
                        debug_print(traceback.format_exc())
                        return f"검색 중 오류가 발생했습니다: {str(e)}", "❌ 검색 실패"
                
                schedule_generate_btn.click(
                    fn=update_schedule,
                    inputs=[schedule_crop, schedule_location],
                    outputs=[schedule_table, schedule_json, schedule_status]
                )
                
                schedule_search_btn.click(
                    fn=search_schedule_task,
                    inputs=[schedule_crop, schedule_location, schedule_task_search, schedule_table],
                    outputs=[schedule_search_result, schedule_search_info]
                )
                
                schedule_task_search.submit(
                    fn=search_schedule_task,
                    inputs=[schedule_crop, schedule_location, schedule_task_search, schedule_table],
                    outputs=[schedule_search_result, schedule_search_info]
                )
            
        # 푸터
        gr.Markdown("---")
        gr.Markdown("### 💡 사용 팁")
        gr.Markdown("""
        - **RAG 질문 답변**: 
          - 농업 관련 모든 질문에 답변할 수 있습니다
          - 작물 이미지를 업로드하면 자동으로 작물을 분류하고 관련 정보를 제공합니다
          - 경작지 위치를 입력하면 해당 지역에 맞는 정보를 제공합니다
          - 질문에 작물명(토마토/딸기)이 포함되면 병해충 위험도 예측 정보도 함께 제공됩니다
        - **재배 일정표 생성**: 
          - 딸기와 토마토의 재배 일정표를 조회할 수 있습니다
          - 일정표 생성 후 "상세 작업 검색" 섹션에서 특정 작업의 상세 정보를 검색할 수 있습니다
          - 예: 일정표에서 "정식" 작업을 확인한 후, 검색창에 "정식" 또는 "정식 방법"을 입력하여 상세 정보 검색
          - 경작지 위치를 입력하면 기상 조건을 고려한 맞춤형 조언을 받을 수 있습니다
        """)
    
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

