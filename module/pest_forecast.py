"""
예보 기반 병해충 예측 모듈
예보 데이터를 활용한 병해충 발생 위험도 예측
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import os
from .weather_forecast import WeatherForecastManager
from .config import debug_print, MODEL_NAME
try:
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
    from typing import List as TypingList
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    debug_print("⚠️ LLM 관련 모듈을 import할 수 없습니다. 병해충 목록/규칙 자동 추출 기능이 제한됩니다.")


class PestForecastPredictor:
    """예보 기반 병해충 예측 클래스"""
    
    # 예보 간격(시간) 가정값
    # 단기/초단기 예보가 1시간 간격이라고 가정 (3시간 간격이면 3.0으로 변경)
    FCST_STEP_HOURS: float = 1.0
    
    # 캐시 파일 경로
    CACHE_DIR = Path(__file__).parent.parent / "cache"
    CACHE_FILE_PATH = CACHE_DIR / "pest_prediction_rules.json"
    
    # 작물별 병해충 예측 조건 (기본값 - fallback용)
    DEFAULT_PEST_PREDICTION_RULES = {
        "토마토": {
            "탄저병": {
                "temp_range": (20, 30),
                "rh_threshold": 85,
                "continuous_humid_hours": 12,
                "rain_threshold": 10,
                "forecast_horizon": 72,  # 72시간 예보 확인
                # "sensitive_stages": ["개화기", "수확기"],  # 필요시 병해충별로 지정 가능
            },
            "흰가루병": {
                "temp_range": (15, 25),
                "rh_threshold": 70,
                "continuous_humid_hours": 8,
                "forecast_horizon": 48,
            },
            "역병": {
                "temp_range": (18, 25),
                "rh_threshold": 90,
                "rain_threshold": 15,
                "forecast_horizon": 24,
            },
            "세균성점무늬병": {
                "temp_range": (20, 28),
                "rh_threshold": 80,
                "continuous_humid_hours": 10,
                "rain_threshold": 5,
                "forecast_horizon": 48,
            },
        },
        "딸기": {
            "탄저병": {
                "temp_range": (18, 28),
                "rh_threshold": 90,
                "continuous_humid_hours": 12,
                "rain_threshold": 10,
                "forecast_horizon": 72,
            },
            "잿빛곰팡이병": {
                "temp_range": (15, 25),
                "rh_threshold": 85,
                "continuous_humid_hours": 10,
                "forecast_horizon": 48,
            },
            "흰가루병": {
                "temp_range": (18, 25),
                "rh_threshold": 70,
                "continuous_humid_hours": 8,
                "forecast_horizon": 48,
            },
            "곰팡이성병": {
                "temp_range": (15, 25),
                "rh_threshold": 90,
                "continuous_humid_hours": 12,
                "rain_threshold": 10,
                "forecast_horizon": 72,
            },
        },
    }
    
    def __init__(
        self, 
        forecast_manager: Optional[WeatherForecastManager] = None,
        crop_retrievers: Optional[Dict[str, Any]] = None,
        auto_load_rules: bool = True
    ):
        """
        Args:
            forecast_manager: WeatherForecastManager 인스턴스 (없으면 자동 생성)
            crop_retrievers: 작물별 벡터스토어 리트리버 딕셔너리 (병해충 목록 추출용)
            auto_load_rules: 벡터스토어에서 예측 규칙 자동 로드 여부
        """
        self.forecast_manager = forecast_manager or WeatherForecastManager()
        self.crop_retrievers = crop_retrievers or {}
        self.extracted_pest_lists: Dict[str, List[str]] = {}   # 추출된 병해충 목록 캐시
        self.extracted_rules_cache: Dict[str, Dict[str, Any]] = {}  # 추출된 예측 규칙 캐시
        
        # 예측 규칙 초기화 (기본값으로 시작)
        self.PEST_PREDICTION_RULES = self._deep_copy_rules(self.DEFAULT_PEST_PREDICTION_RULES)
        
        # 캐시 파일에서 먼저 로드 시도
        cached_rules = self._load_rules_from_cache()
        if cached_rules:
            debug_print(f"📂 캐시 파일에서 예측 규칙 로드 중...")
            for crop, pests in cached_rules.items():
                if crop not in self.PEST_PREDICTION_RULES:
                    self.PEST_PREDICTION_RULES[crop] = {}
                self.PEST_PREDICTION_RULES[crop].update(pests)
                # 캐시에도 저장
                if crop not in self.extracted_rules_cache:
                    self.extracted_rules_cache[crop] = {}
                self.extracted_rules_cache[crop].update(pests)
            debug_print(f"✅ 캐시 파일에서 {sum(len(p) for p in cached_rules.values())}개 규칙 로드 완료")
        
        # 벡터스토어에서 예측 규칙 자동 로드 (캐시에 없는 것만)
        if auto_load_rules and self.crop_retrievers:
            try:
                self._load_prediction_rules_from_vectorstore()
                # 추출 완료 후 캐시 파일에 저장 (새로 추출된 것이 있으면)
                if self.extracted_rules_cache:
                    self._save_rules_to_cache()
            except Exception as e:
                debug_print(f"⚠️ 벡터스토어에서 예측 규칙 자동 로드 실패 (기본값 사용): {e}")
    
    def _deep_copy_rules(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        """규칙 딕셔너리 깊은 복사"""
        import copy
        return copy.deepcopy(rules)
    
    def _load_rules_from_cache(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """캐시 파일에서 규칙 로드"""
        if not self.CACHE_FILE_PATH.exists():
            debug_print(f"📂 캐시 파일이 없습니다: {self.CACHE_FILE_PATH}")
            return None
        
        try:
            with open(self.CACHE_FILE_PATH, 'r', encoding='utf-8') as f:
                cached_rules = json.load(f)
            return cached_rules
        except Exception as e:
            debug_print(f"⚠️ 캐시 파일 로드 실패: {e}")
            return None
    
    def _save_rules_to_cache(self):
        """규칙을 캐시 파일에 저장"""
        if not self.extracted_rules_cache:
            debug_print("⚠️ 저장할 규칙이 없습니다.")
            return
        
        try:
            # 캐시 디렉토리 생성
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            # 기존 캐시 파일이 있으면 먼저 로드하여 병합
            existing_rules = self._load_rules_from_cache() or {}
            for crop, pests in self.extracted_rules_cache.items():
                if crop not in existing_rules:
                    existing_rules[crop] = {}
                existing_rules[crop].update(pests)
            
            # 병합된 규칙 저장
            with open(self.CACHE_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(existing_rules, f, ensure_ascii=False, indent=2)
            
            total_count = sum(len(p) for p in existing_rules.values())
            debug_print(f"💾 예측 규칙을 캐시 파일에 저장 완료 ({total_count}개 규칙)")
        except Exception as e:
            debug_print(f"⚠️ 캐시 파일 저장 실패: {e}")
    
    def predict_pest_risk(
        self,
        latitude: float,
        longitude: float,
        crop: str,
        growth_stage: str = "생육기",
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        예보 기반 병해충 위험도 예측
        
        Args:
            latitude: 위도
            longitude: 경도
            crop: 작물명 ("토마토" 또는 "딸기")
            growth_stage: 생육 단계 (예: "파종기", "생육기", "개화기", "착과기", "과실비대기", "수확기")
            reference_time: 기준 시각 (None이면 현재 시각 사용, 테스트/백테스트용)
            
        Returns:
            {
                "overall_risk": "낮음" | "주의" | "경계" | "심각",
                "pest_forecasts": [...],
                "summary": "요약 텍스트",
                "forecast_date": "예보 기준일",
                "crop": 작물명,
                "growth_stage": 생육단계
            }
        """
        # 기준 시각 설정
        ref_time = reference_time or datetime.now()

        # 모든 예보 데이터 조회
        forecasts = self.forecast_manager.get_all_forecasts(latitude, longitude)
        
        # 작물명 정규화
        crop_normalized = self._normalize_crop_name(crop)
        
        # 작물별 병해충 예측 규칙 가져오기 (동적으로 로드된 규칙 사용)
        crop_rules = self.PEST_PREDICTION_RULES.get(crop_normalized, {})
        
        if not crop_rules:
            debug_print(f"⚠️ '{crop}'에 대한 병해충 예측 규칙이 없습니다.")
            return {
                "overall_risk": "낮음",
                "pest_forecasts": [],
                "summary": f"'{crop}'에 대한 예측 규칙이 없습니다.",
                "forecast_date": ref_time.strftime("%Y-%m-%d %H:%M"),
                "crop": crop_normalized,
                "growth_stage": growth_stage,
            }
        
        pest_forecasts: List[Dict[str, Any]] = []
        
        for pest_name, conditions in crop_rules.items():
            prediction = self._predict_single_pest(
                forecasts,
                pest_name,
                conditions,
                crop_normalized,
                growth_stage,
                reference_time=ref_time,
            )
            if prediction:
                pest_forecasts.append(prediction)
        
        # 전체 위험도 결정 (가장 높은 점수 기준)
        max_risk_score = max([p.get("risk_score", 0) for p in pest_forecasts], default=0)
        risk_level_map = {
            0: "낮음",
            1: "주의",
            2: "경계",
            3: "심각"
        }
        overall_risk = risk_level_map.get(max_risk_score, "낮음")
        
        # 요약 생성
        summary_parts = [f"전체 위험도: {overall_risk}"]
        high_risk_pests = [pf for pf in pest_forecasts if pf["risk_level"] in ["경계", "심각"]]
        if high_risk_pests:
            for pf in high_risk_pests:
                summary_parts.append(
                    f"{pf['pest_name']}: {pf['risk_level']} ({pf['forecast_period']})"
                )
        else:
            summary_parts.append("현재 위험한 병해충 없음")
        
        return {
            "overall_risk": overall_risk,
            "pest_forecasts": pest_forecasts,
            "summary": " | ".join(summary_parts),
            "forecast_date": ref_time.strftime("%Y-%m-%d %H:%M"),
            "crop": crop_normalized,
            "growth_stage": growth_stage
        }
    
    def _normalize_crop_name(self, crop: str) -> str:
        """작물명 정규화"""
        crop_lower = crop.lower()
        if "토마토" in crop or "tomato" in crop_lower:
            return "토마토"
        elif "딸기" in crop or "strawberry" in crop_lower:
            return "딸기"
        else:
            return crop
    
    def _predict_single_pest(
        self,
        forecasts: Dict[str, Any],
        pest_name: str,
        conditions: Dict[str, Any],
        crop: str,
        growth_stage: str,
        reference_time: datetime,
    ) -> Optional[Dict[str, Any]]:
        """단일 병해충 예측"""
        forecast_horizon = conditions.get("forecast_horizon", 72)
        temp_range = conditions.get("temp_range")
        rh_threshold = conditions.get("rh_threshold")
        continuous_hours = conditions.get("continuous_humid_hours", 0)
        rain_threshold = conditions.get("rain_threshold", 0)
        
        # 예보 데이터에서 해당 기간 데이터 추출
        forecast_data: List[Dict[str, Any]] = []
        
        # 초단기 + 단기 예보 통합
        for fcst_type in ["ultra_short", "short"]:
            fcst_list = forecasts.get(fcst_type, [])
            for fcst in fcst_list:
                fcst_datetime_str = fcst.get("fcst_datetime", "")
                if fcst_datetime_str:
                    try:
                        # YYYYMMDDHHMM 형식 파싱
                        fcst_time = datetime.strptime(fcst_datetime_str, "%Y%m%d%H%M")
                        hours_ahead = (fcst_time - reference_time).total_seconds() / 3600
                        
                        if 0 <= hours_ahead <= forecast_horizon:
                            forecast_data.append({
                                "hours_ahead": hours_ahead,
                                "temp": fcst.get("temp"),
                                "rh": fcst.get("rh"),
                                "precipitation": self._parse_precipitation(fcst.get("precipitation", 0)),
                                "fcst_time": fcst_time,
                                "fcst_datetime": fcst_datetime_str
                            })
                    except (ValueError, TypeError) as e:
                        debug_print(f"⚠️ 예보 시간 파싱 오류: {e}")
                        continue
        
        # 시간순 정렬
        forecast_data.sort(key=lambda x: x["hours_ahead"])
        
        if not forecast_data:
            debug_print(f"⚠️ {pest_name} 예측: 예보 데이터가 없습니다.")
            return None
        
        # 위험 조건 확인
        risk_score = 0
        continuous_humid_count = 0  # 연속으로 조건 만족한 step 개수
        max_continuous_count = 0
        total_rain = 0.0
        risk_periods: List[Dict[str, Any]] = []
        
        for data in forecast_data:
            temp = data.get("temp")
            rh = data.get("rh")
            rain = data.get("precipitation", 0)
            
            if temp is not None and rh is not None and temp_range and rh_threshold is not None:
                in_temp_range = temp_range[0] <= temp <= temp_range[1]
                is_high_humid = rh >= rh_threshold
                
                if in_temp_range and is_high_humid:
                    continuous_humid_count += 1
                    max_continuous_count = max(max_continuous_count, continuous_humid_count)

                    # 연속 고습 시간 임계값이 설정된 경우에만 위험 구간 기록
                    if continuous_hours and continuous_hours > 0:
                        prev_hours = (continuous_humid_count - 1) * self.FCST_STEP_HOURS
                        curr_hours = continuous_humid_count * self.FCST_STEP_HOURS
                        # 이번 step에서 임계값을 처음 넘는 시점만 기록
                        if prev_hours < continuous_hours <= curr_hours:
                            risk_periods.append({
                                # threshold를 처음 만족한 시점(끝 시점)을 기록
                                "start": data["fcst_time"].strftime("%Y-%m-%d %H:%M"),
                                "hours_ahead": round(data["hours_ahead"], 1)
                            })
                else:
                    continuous_humid_count = 0
            
            total_rain += rain
        
        # 연속 고습 시간(시간 단위)로 변환
        max_continuous_humid_hours = max_continuous_count * self.FCST_STEP_HOURS
        
        # 위험도 점수 계산
        # 연속 고습 시간이 설정되어 있고, 실제로 그 이상 유지된 경우에만 +2
        if continuous_hours and continuous_hours > 0 and max_continuous_humid_hours >= continuous_hours:
            risk_score += 2
        
        if rain_threshold and rain_threshold > 0 and total_rain >= rain_threshold:
            risk_score += 1
        
        # 생육 단계별 가중치 (민감한 단계는 위험도 증가)
        # 개별 병해충 규칙에 sensitive_stages가 있으면 우선 사용, 없으면 기본값 사용
        default_sensitive_stages = ["개화기", "착과기", "과실비대기", "수확기"]
        sensitive_stages = conditions.get("sensitive_stages") or default_sensitive_stages
        if growth_stage in sensitive_stages:
            risk_score += 1
        
        # 점수 범위 제한
        risk_score = min(risk_score, 3)
        
        # 위험도 레벨 결정
        risk_level_map = {
            0: "낮음",
            1: "주의",
            2: "경계",
            3: "심각"
        }
        risk_level = risk_level_map.get(risk_score, "낮음")
        
        # 예보 기간 텍스트
        if forecast_horizon <= 24:
            forecast_period = "24시간 내"
        elif forecast_horizon <= 48:
            forecast_period = "48시간 내"
        else:
            forecast_period = "72시간 내"
        
        # 권장 조치
        recommendations: List[str] = []
        if risk_score >= 3:
            recommendations.append("즉시 사전 방제 약제 살포 권장")
            recommendations.append("환기 강화로 습도 관리 필수")
            recommendations.append("병 발생 시 즉시 감염 부위 제거")
        elif risk_score >= 2:
            recommendations.append("사전 방제 약제 살포 권장")
            recommendations.append("환기 강화로 습도 관리")
            recommendations.append("병 발생 모니터링 강화")
        elif risk_score == 1:
            recommendations.append("환기 강화 및 습도 모니터링")
            recommendations.append("예방적 관리 권장")
        else:
            recommendations.append("정상 관리 유지")
        
        return {
            "pest_name": pest_name,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "forecast_period": forecast_period,
            "conditions": {
                "max_continuous_humid_hours": max_continuous_humid_hours,
                "total_rain": round(total_rain, 1),
                "risk_periods": risk_periods[:3]  # 최대 3개만
            },
            "recommendation": " | ".join(recommendations) if recommendations else "정상 관리"
        }
    
    def _parse_precipitation(self, precip_value: Any) -> float:
        """강수량 파싱 (문자열 또는 숫자 처리)"""
        if precip_value is None:
            return 0.0
        
        if isinstance(precip_value, (int, float)):
            return float(precip_value)
        
        if isinstance(precip_value, str):
            # "강수없음", "0", "1.5mm" 등의 형식 처리
            if "없음" in precip_value or precip_value == "0":
                return 0.0
            
            # 숫자만 추출
            import re
            numbers = re.findall(r'\d+\.?\d*', precip_value)
            if numbers:
                return float(numbers[0])
        
        return 0.0
    
    def get_pest_forecast_context(
        self,
        latitude: float,
        longitude: float,
        crop: str,
        growth_stage: str = "생육기"
    ) -> str:
        """
        병해충 예측 정보를 컨텍스트 문자열로 반환
        
        Returns:
            병해충 예측 컨텍스트 문자열
        """
        try:
            forecast = self.predict_pest_risk(latitude, longitude, crop, growth_stage)
            
            if not forecast or not forecast.get("pest_forecasts"):
                return ""
            
            context_parts: List[str] = [f"⚠️ 병해충 예측 정보 ({forecast.get('forecast_date', '')}):"]
            context_parts.append(f"전체 위험도: {forecast.get('overall_risk', '낮음')}")
            context_parts.append("")
            
            # 위험도가 높은 병해충만 표시 (주의 이상)
            high_risk_pests = [
                pf for pf in forecast.get("pest_forecasts", [])
                if pf["risk_level"] in ["주의", "경계", "심각"]
            ]
            
            if high_risk_pests:
                for pf in high_risk_pests:
                    context_parts.append(f"🔴 {pf['pest_name']}: {pf['risk_level']} ({pf['forecast_period']})")
                    context_parts.append(f"   예상 조건: 연속 고습 {pf['conditions']['max_continuous_humid_hours']}시간, 강수량 {pf['conditions']['total_rain']}mm")
                    context_parts.append(f"   권장 조치: {pf['recommendation']}")
                    context_parts.append("")
            else:
                context_parts.append("✅ 현재 위험한 병해충 없음")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            debug_print(f"⚠️ 병해충 예측 컨텍스트 생성 실패: {e}")
            return ""
    
    def extract_pest_list_from_vectorstore(self, crop: str, use_cache: bool = True) -> List[str]:
        """
        벡터스토어에서 작물별 병해충 목록 자동 추출
        
        Args:
            crop: 작물명 ("토마토" 또는 "딸기")
            use_cache: 캐시된 결과 사용 여부
            
        Returns:
            병해충 목록 리스트
        """
        # 캐시 확인
        crop_normalized = self._normalize_crop_name(crop)
        if use_cache and crop_normalized in self.extracted_pest_lists:
            debug_print(f"✅ '{crop_normalized}' 병해충 목록 캐시 사용")
            return self.extracted_pest_lists[crop_normalized]
        
        # 리트리버 확인
        crop_key = "tomato" if crop_normalized == "토마토" else "strawberry" if crop_normalized == "딸기" else None
        if not crop_key or crop_key not in self.crop_retrievers:
            debug_print(f"⚠️ '{crop}'에 대한 벡터스토어 리트리버가 없습니다.")
            return []
        
        retriever = self.crop_retrievers[crop_key]
        if not retriever:
            debug_print(f"⚠️ '{crop}' 리트리버가 None입니다.")
            return []
        
        try:
            # 벡터스토어에서 병해충 관련 문서 검색
            query = f"{crop_normalized}에 발생하는 모든 병해충 목록"
            debug_print(f"🔍 벡터스토어에서 '{crop_normalized}' 병해충 목록 추출 중...")
            
            documents = retriever.invoke(query)
            
            if not documents:
                debug_print(f"⚠️ '{crop_normalized}' 병해충 관련 문서를 찾을 수 없습니다.")
                return []
            
            # 문서 내용 결합
            context = "\n\n".join([doc.page_content for doc in documents[:10]])  # 상위 10개만
            
            # LLM을 사용하여 병해충 목록 추출
            if not LLM_AVAILABLE:
                debug_print("⚠️ LLM을 사용할 수 없어 병해충 목록 추출을 건너뜁니다.")
                return []
            
            # 구조화된 출력을 위한 모델 정의
            class PestList(BaseModel):
                pests: TypingList[str] = Field(description="병해충명 목록")
            
            llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
            structured_llm = llm.with_structured_output(PestList)
            
            extraction_prompt = f"""
다음 문서에서 {crop_normalized}에 발생하는 모든 병해충(병, 해충, 바이러스 등)의 이름을 추출하세요.

문서 내용:
{context[:3000]}  # 너무 길면 잘라냄

요구사항:
1. 병해충명만 추출 (설명 제외)
2. 중복 제거
3. 정확한 병해충명만 포함 (예: "탄저병", "흰가루병", "점박이응애" 등)
4. 일반적인 설명이나 문구는 제외

병해충 목록을 JSON 형식으로 반환하세요.
"""
            
            result = structured_llm.invoke(extraction_prompt)
            pest_list = result.pests if hasattr(result, 'pests') else []
            
            # 결과 저장
            self.extracted_pest_lists[crop_normalized] = pest_list
            
            debug_print(f"✅ '{crop_normalized}' 병해충 목록 추출 완료: {len(pest_list)}개")
            debug_print(f"   추출된 병해충: {', '.join(pest_list[:10])}{'...' if len(pest_list) > 10 else ''}")
            
            return pest_list
            
        except Exception as e:
            debug_print(f"❌ '{crop_normalized}' 병해충 목록 추출 실패: {e}")
            import traceback
            debug_print(traceback.format_exc())
            return []
    
    def get_all_pests_for_crop(self, crop: str) -> Dict[str, Any]:
        """
        작물별 모든 병해충 정보 반환 (예보 기반 예측 가능 + 벡터스토어에서 추출)
        
        Args:
            crop: 작물명
            
        Returns:
            {
                "forecast_predictable": [...],  # 예보 기반 예측 가능한 병해충
                "vectorstore_pests": [...],      # 벡터스토어에서 추출한 모든 병해충
                "missing_predictions": [...]     # 벡터스토어에는 있지만 예측 규칙이 없는 병해충
            }
        """
        crop_normalized = self._normalize_crop_name(crop)
        
        # 예보 기반 예측 가능한 병해충
        forecast_predictable = list(self.PEST_PREDICTION_RULES.get(crop_normalized, {}).keys())
        
        # 벡터스토어에서 추출한 병해충 목록
        vectorstore_pests = self.extract_pest_list_from_vectorstore(crop)
        
        # 예측 규칙이 없는 병해충
        missing_predictions = [
            pest for pest in vectorstore_pests 
            if pest not in forecast_predictable
        ]
        
        return {
            "forecast_predictable": forecast_predictable,
            "vectorstore_pests": vectorstore_pests,
            "missing_predictions": missing_predictions,
            "total_forecast_pests": len(forecast_predictable),
            "total_vectorstore_pests": len(vectorstore_pests),
            "missing_count": len(missing_predictions)
        }
    
    def _load_prediction_rules_from_vectorstore(self):
        """벡터스토어에서 예측 규칙 자동 로드"""
        debug_print("🔍 벡터스토어에서 병해충 예측 규칙 추출 중...")
        
        total_extracted = 0
        for crop in ["토마토", "딸기"]:
            crop_normalized = self._normalize_crop_name(crop)
            
            # 벡터스토어에서 병해충 목록 추출
            pest_list = self.extract_pest_list_from_vectorstore(crop, use_cache=True)
            
            if not pest_list:
                debug_print(f"⚠️ '{crop}'의 병해충 목록을 찾을 수 없습니다.")
                continue
            
            debug_print(f"📋 '{crop}' 병해충 {len(pest_list)}개 발견, 예측 규칙 추출 중...")
            
            # 각 병해충에 대해 예측 규칙 추출
            extracted_count = 0
            skipped_count = 0
            for pest_name in pest_list:
                # 이미 추출된 규칙이 있으면 스킵 (기본값이 아닌 추출된 규칙)
                if crop_normalized in self.extracted_rules_cache and \
                   pest_name in self.extracted_rules_cache[crop_normalized]:
                    skipped_count += 1
                    continue
                
                # 기본값에 있는지 확인 (기본값이면 추출 시도 안 함)
                is_default = (crop_normalized in self.DEFAULT_PEST_PREDICTION_RULES and 
                             pest_name in self.DEFAULT_PEST_PREDICTION_RULES[crop_normalized])
                
                # 기본값이 아닌 경우에만 추출 시도 (기본값은 이미 있으므로)
                if not is_default:
                    try:
                        rule = self.auto_extract_prediction_rules(crop, pest_name)
                        if rule:
                            # 규칙 저장
                            if crop_normalized not in self.PEST_PREDICTION_RULES:
                                self.PEST_PREDICTION_RULES[crop_normalized] = {}
                            
                            self.PEST_PREDICTION_RULES[crop_normalized][pest_name] = rule
                            
                            # 캐시에 저장
                            if crop_normalized not in self.extracted_rules_cache:
                                self.extracted_rules_cache[crop_normalized] = {}
                            self.extracted_rules_cache[crop_normalized][pest_name] = rule
                            
                            extracted_count += 1
                            debug_print(f"✅ '{crop}' {pest_name} 예측 규칙 추출 완료")
                        else:
                            debug_print(f"⚠️ '{crop}' {pest_name} 예측 규칙 추출 실패 (기본값 사용 또는 미적용)")
                    except Exception as e:
                        debug_print(f"⚠️ '{crop}' {pest_name} 예측 규칙 추출 중 오류: {e}")
            
            if extracted_count > 0 or skipped_count > 0:
                debug_print(f"📊 '{crop}': {extracted_count}개 규칙 추출, {skipped_count}개 스킵 (캐시/기본값)")
            
            total_extracted += extracted_count
        
        if total_extracted > 0:
            debug_print(f"✅ 예측 규칙 로드 완료 (새로 추출: {total_extracted}개)")
        else:
            debug_print(f"✅ 예측 규칙 로드 완료 (모두 캐시/기본값 사용)")
    
    def auto_extract_prediction_rules(
        self,
        crop: str,
        pest_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        벡터스토어에서 병해충 예측 규칙 자동 추출
        
        Args:
            crop: 작물명 ("토마토" 또는 "딸기")
            pest_name: 병해충명
            
        Returns:
            예측 규칙 딕셔너리 또는 None (추출 실패 시)
        """
        crop_normalized = self._normalize_crop_name(crop)
        
        # 리트리버 키 매핑 (tomato, strawberry)
        crop_key = "tomato" if crop_normalized == "토마토" else "strawberry" if crop_normalized == "딸기" else None
        if not crop_key or crop_key not in self.crop_retrievers:
            return None
        
        retriever = self.crop_retrievers[crop_key]
        if not retriever:
            return None
        
        if not LLM_AVAILABLE:
            debug_print("⚠️ LLM을 사용할 수 없어 예측 규칙 추출을 건너뜁니다.")
            return None
        
        try:
            # 병해충 발생 조건 검색
            queries = [
                f"{crop_normalized} {pest_name} 발생 조건",
                f"{crop_normalized} {pest_name} 온도 습도",
                f"{crop_normalized} {pest_name} 예방 조건",
                f"{crop_normalized} {pest_name} 발생 온도 습도 강수량",
            ]
            
            all_docs = []
            for query in queries:
                try:
                    docs = retriever.invoke(query)
                    all_docs.extend(docs[:3])  # 상위 3개만
                except Exception as e:
                    debug_print(f"⚠️ 검색 쿼리 '{query}' 실패: {e}")
            
            if not all_docs:
                debug_print(f"⚠️ '{pest_name}' 관련 문서를 찾을 수 없습니다.")
                return None
            
            # 문서 내용 결합 (중복 제거)
            seen_content = set()
            unique_docs = []
            for doc in all_docs:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
            
            context = "\n\n".join([doc.page_content for doc in unique_docs[:5]])
            
            if len(context) < 100:  # 내용이 너무 짧으면 스킵
                debug_print(f"⚠️ '{pest_name}' 관련 문서 내용이 부족합니다.")
                return None
            
            # LLM으로 구조화된 규칙 추출
            class PestRule(BaseModel):
                temp_min: int = Field(description="최저 발생 온도 (℃)")
                temp_max: int = Field(description="최고 발생 온도 (℃)")
                rh_threshold: int = Field(description="발생 습도 임계값 (%)")
                continuous_humid_hours: Optional[int] = Field(
                    description="연속 고습 시간 (시간, 정보가 없으면 None)"
                )
                rain_threshold: Optional[int] = Field(
                    description="강수량 임계값 (mm, 정보가 없으면 None)"
                )
            
            extraction_prompt = f"""
다음 문서에서 {crop_normalized}의 {pest_name} 발생 조건을 추출하세요.

문서 내용:
{context[:4000]}

요구사항:
1. 온도 범위: 최저 온도와 최고 온도를 추출 (℃ 단위)
2. 습도 임계값: 병해충이 발생하기 시작하는 습도 (%)
3. 연속 고습 시간: 습도가 높게 유지되어야 하는 시간 (정보가 없으면 None)
4. 강수량 임계값: 강수량이 영향을 미치는 경우 임계값 (mm, 정보가 없으면 None)

중요:
- 문서에 명시된 정확한 수치만 추출하세요
- 추측하거나 일반적인 값을 사용하지 마세요
- 정보가 없으면 None을 반환하세요
- 온도는 정수로, 습도와 강수량도 정수로 반환하세요
"""
            
            llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
            structured_llm = llm.with_structured_output(PestRule)
            result = structured_llm.invoke(extraction_prompt)
            
            # PEST_PREDICTION_RULES 형식으로 변환
            rule: Dict[str, Any] = {
                "temp_range": (result.temp_min, result.temp_max),
                "rh_threshold": result.rh_threshold,
                "forecast_horizon": 72,  # 기본값 (필요시 후에 조정 가능)
            }
            
            # 선택적 필드 추가
            if result.continuous_humid_hours is not None:
                rule["continuous_humid_hours"] = result.continuous_humid_hours
            else:
                # None이면 연속 고습 조건 미사용 → 0으로 두고, 점수 계산에서 무시
                rule["continuous_humid_hours"] = 0
            
            if result.rain_threshold is not None:
                rule["rain_threshold"] = result.rain_threshold
            else:
                rule["rain_threshold"] = 0
            
            return rule
            
        except Exception as e:
            debug_print(f"⚠️ {pest_name} 예측 규칙 추출 실패: {e}")
            return None

