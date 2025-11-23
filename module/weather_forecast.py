"""
기상 예보 데이터 모듈
초단기/단기/중기 예보 조회
"""
import os
import requests
import math
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .config import debug_print
from .location import get_farm_info


class WeatherForecastManager:
    """기상 예보 데이터 관리 클래스"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        medium_api_key: Optional[str] = None
    ):
        """
        Args:
            api_key: 기상청 API 인증키 (단기 + 중기 모두 사용 가능)
            medium_api_key: 중기예보 API 키 (선택사항, 없으면 api_key 사용)
        """
        self.api_key = api_key or os.getenv("WEATHER_API_KEY")
        # 중기예보 키가 별도로 없으면 단기예보 키 사용
        self.medium_api_key = medium_api_key or os.getenv("WEATHER_MEDIUM_API_KEY") or self.api_key
        
        if not self.api_key:
            debug_print("⚠️ 기상청 API 키가 설정되지 않았습니다.")
    
    def _convert_to_grid(self, latitude: float, longitude: float) -> tuple:
        """위경도를 기상청 격자 좌표로 변환"""
        RE = 6371.00877
        GRID = 5.0
        SLAT1 = 30.0
        SLAT2 = 60.0
        OLON = 126.0
        OLAT = 38.0
        XO = 43
        YO = 136
        
        deg_rad = math.pi / 180.0
        re = RE / GRID
        slat1 = SLAT1 * deg_rad
        slat2 = SLAT2 * deg_rad
        olon = OLON * deg_rad
        olat = OLAT * deg_rad
        
        sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
        sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
        sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
        sf = math.pow(sf, sn) * math.cos(slat1) / sn
        ro = math.tan(math.pi * 0.25 + olat * 0.5)
        ro = re * sf / math.pow(ro, sn)
        
        ra = math.tan(math.pi * 0.25 + (latitude) * deg_rad * 0.5)
        ra = re * sf / math.pow(ra, sn)
        theta = longitude * deg_rad - olon
        if theta > math.pi:
            theta -= 2.0 * math.pi
        if theta < -math.pi:
            theta += 2.0 * math.pi
        theta *= sn
        
        grid_x = int(ra * math.sin(theta) + XO + 0.5)
        grid_y = int(ro - ra * math.cos(theta) + YO + 0.5)
        
        return (grid_x, grid_y)
    
    def _get_base_time(self) -> str:
        """현재 시간 기준 base_time 계산 (초단기예보용)"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # 초단기예보는 매시간 정각에 발표 (00, 30분 기준)
        if minute < 30:
            base_hour = hour - 1 if hour > 0 else 23
        else:
            base_hour = hour
        
        # 새벽 시간대(00시~02시) 처리 개선
        # 새벽 시간대에는 데이터 생성이 지연될 수 있으므로 이전 시간대 사용
        if hour < 2:
            # 00시~01시: 전날 23시 사용
            # 01시~02시: 00시 또는 전날 23시 시도
            if hour == 0:
                base_hour = 23
            elif hour == 1:
                # 01시 30분 이전이면 전날 23시, 이후면 00시
                base_hour = 23 if minute < 30 else 0
        
        return f"{base_hour:02d}00"
    
    def get_ultra_short_forecast(
        self,
        latitude: float,
        longitude: float
    ) -> List[Dict[str, Any]]:
        """
        초단기예보 조회 (0~6시간)
        
        Returns:
            시간별 예보 데이터 리스트
        """
        if not self.api_key:
            debug_print("❌ API 키가 없어 초단기예보를 조회할 수 없습니다.")
            return []
        
        grid_x, grid_y = self._convert_to_grid(latitude, longitude)
        
        url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
        base_date = datetime.now().strftime("%Y%m%d")
        base_time = self._get_base_time()
        
        # serviceKey는 URL 인코딩 필요 (특수문자 처리)
        params = {
            "serviceKey": self.api_key,  # requests.get이 자동으로 인코딩하지만 명시적으로 처리
            "pageNo": 1,
            "numOfRows": 100,
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": grid_x,
            "ny": grid_y
        }
        
        try:
            # 공공데이터포털 API 호출 (최대 2회 재시도)
            max_retries = 2
            for retry in range(max_retries):
                # 방법 1: params 사용 (일반적인 방법)
                response = requests.get(url, params=params, timeout=10)
                
                # 403 오류 시 방법 2: serviceKey를 직접 URL에 포함 (인코딩된 키 사용)
                if response.status_code == 403:
                    debug_print("⚠️ 403 오류 발생, serviceKey를 URL에 직접 포함하여 재시도...")
                    from urllib.parse import urlencode, quote
                    # serviceKey를 제외한 파라미터
                    other_params = {k: v for k, v in params.items() if k != "serviceKey"}
                    # serviceKey를 URL 인코딩하여 직접 포함
                    encoded_key = quote(self.api_key, safe='')
                    query_string = urlencode(other_params)
                    url_with_key = f"{url}?serviceKey={encoded_key}&{query_string}"
                    response = requests.get(url_with_key, timeout=10)
                
                response.raise_for_status()
                
                # 응답 확인
                if response.status_code == 200:
                    data = response.json()
                    # API 응답 에러 체크
                    result_code = data.get("response", {}).get("header", {}).get("resultCode", "")
                    result_msg = data.get("response", {}).get("header", {}).get("resultMsg", "")
                    
                    # NO_DATA 오류(03)인 경우 이전 시간대 시도
                    if result_code == "03" and retry < max_retries - 1:
                        debug_print(f"⚠️ API 응답 오류: {result_code} - {result_msg}")
                        debug_print(f"   이전 시간대 데이터 시도 중... (재시도 {retry + 1}/{max_retries - 1})")
                        
                        # 이전 시간대 base_time 계산
                        current_hour = int(base_time[:2])
                        prev_hour = (current_hour - 1) % 24
                        base_time = f"{prev_hour:02d}00"
                        
                        # 전날로 넘어가는 경우 base_date도 조정
                        if prev_hour == 23:
                            base_date_obj = datetime.strptime(base_date, "%Y%m%d")
                            base_date_obj = base_date_obj - timedelta(days=1)
                            base_date = base_date_obj.strftime("%Y%m%d")
                        
                        params["base_date"] = base_date
                        params["base_time"] = base_time
                        continue
                    elif result_code != "00":
                        debug_print(f"⚠️ API 응답 오류: {result_code} - {result_msg}")
                        return []
                    
                    # 성공적으로 데이터를 받은 경우
                    return self._parse_forecast_data(data, "초단기")
                else:
                    debug_print(f"❌ HTTP 오류: {response.status_code}")
                    debug_print(f"   응답 내용: {response.text[:500]}")
                    return []
        except requests.exceptions.HTTPError as e:
            debug_print(f"❌ 초단기예보 조회 실패 (HTTP): {e}")
            if hasattr(e, 'response') and e.response is not None:
                debug_print(f"   상태 코드: {e.response.status_code}")
                debug_print(f"   응답 내용: {e.response.text[:500]}")
            return []
        except Exception as e:
            debug_print(f"❌ 초단기예보 조회 실패: {e}")
            return []
    
    def get_short_forecast(
        self,
        latitude: float,
        longitude: float
    ) -> List[Dict[str, Any]]:
        """
        단기예보 조회 (1~3일, 최대 10일)
        
        Returns:
            시간별 예보 데이터 리스트
        """
        if not self.api_key:
            debug_print("❌ API 키가 없어 단기예보를 조회할 수 없습니다.")
            return []
        
        grid_x, grid_y = self._convert_to_grid(latitude, longitude)
        
        url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
        now = datetime.now()
        base_date = now.strftime("%Y%m%d")
        base_time = "0500"  # 05시 기준
        
        # 새벽 시간대(00시~05시)에는 전날 0500 데이터 사용
        if now.hour < 5:
            base_date_obj = now - timedelta(days=1)
            base_date = base_date_obj.strftime("%Y%m%d")
            debug_print(f"📅 새벽 시간대 감지: 전날 데이터 사용 ({base_date} {base_time})")
        
        # serviceKey는 URL 인코딩 필요
        params = {
            "serviceKey": self.api_key,
            "pageNo": 1,
            "numOfRows": 1000,
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": grid_x,
            "ny": grid_y
        }
        
        try:
            # 공공데이터포털 API 호출 (최대 2회 재시도)
            max_retries = 2
            for retry in range(max_retries):
                response = requests.get(url, params=params, timeout=10)
                
                # 403 오류 시 serviceKey를 URL에 직접 포함하여 재시도
                if response.status_code == 403:
                    debug_print("⚠️ 403 오류 발생, serviceKey를 URL에 직접 포함하여 재시도...")
                    from urllib.parse import urlencode, quote
                    other_params = {k: v for k, v in params.items() if k != "serviceKey"}
                    encoded_key = quote(self.api_key, safe='')
                    query_string = urlencode(other_params)
                    url_with_key = f"{url}?serviceKey={encoded_key}&{query_string}"
                    response = requests.get(url_with_key, timeout=10)
                
                response.raise_for_status()
                
                if response.status_code == 200:
                    data = response.json()
                    result_code = data.get("response", {}).get("header", {}).get("resultCode", "")
                    result_msg = data.get("response", {}).get("header", {}).get("resultMsg", "")
                    
                    # NO_DATA 오류(03)인 경우 이전 날짜 시도
                    if result_code == "03" and retry < max_retries - 1:
                        debug_print(f"⚠️ API 응답 오류: {result_code} - {result_msg}")
                        debug_print(f"   이전 날짜 데이터 시도 중... (재시도 {retry + 1}/{max_retries - 1})")
                        
                        # 이전 날짜의 0500 데이터 시도
                        base_date_obj = datetime.strptime(base_date, "%Y%m%d")
                        base_date_obj = base_date_obj - timedelta(days=1)
                        base_date = base_date_obj.strftime("%Y%m%d")
                        
                        params["base_date"] = base_date
                        continue
                    elif result_code != "00":
                        debug_print(f"⚠️ API 응답 오류: {result_code} - {result_msg}")
                        return []
                    
                    # 성공적으로 데이터를 받은 경우
                    return self._parse_forecast_data(data, "단기")
                else:
                    debug_print(f"❌ HTTP 오류: {response.status_code}")
                    debug_print(f"   응답 내용: {response.text[:500]}")
                    return []
        except requests.exceptions.HTTPError as e:
            debug_print(f"❌ 단기예보 조회 실패 (HTTP): {e}")
            if hasattr(e, 'response') and e.response is not None:
                debug_print(f"   상태 코드: {e.response.status_code}")
                debug_print(f"   응답 내용: {e.response.text[:500]}")
            return []
        except Exception as e:
            debug_print(f"❌ 단기예보 조회 실패: {e}")
            return []
    
    def get_medium_forecast(
        self,
        latitude: float,
        longitude: float
    ) -> Dict[str, Any]:
        """
        중기예보 조회 (4~10일 이후)
        
        Returns:
            중기예보 데이터 딕셔너리
        """
        if not self.medium_api_key:
            debug_print("❌ 중기예보 API 키가 없어 중기예보를 조회할 수 없습니다.")
            return {}
        
        # 중기예보는 지역 코드(regId)를 사용
        # 위경도 기반으로 지역 코드 추정
        reg_id = self._get_region_code(latitude, longitude)
        
        url = "http://apis.data.go.kr/1360000/MidFcstInfoService/getMidLandFcst"
        base_date = datetime.now().strftime("%Y%m%d")
        # 중기예보는 06시 또는 18시에 발표
        base_time = "0600" if datetime.now().hour < 18 else "1800"
        
        params = {
            "serviceKey": self.medium_api_key,
            "pageNo": 1,
            "numOfRows": 10,
            "dataType": "JSON",
            "regId": reg_id,
            "tmFc": base_date + base_time
        }
        
        try:
            # 공공데이터포털 API 호출
            response = requests.get(url, params=params, timeout=10)
            
            # 403 오류 시 serviceKey를 URL에 직접 포함하여 재시도
            if response.status_code == 403:
                debug_print("⚠️ 403 오류 발생, serviceKey를 URL에 직접 포함하여 재시도...")
                from urllib.parse import urlencode, quote
                other_params = {k: v for k, v in params.items() if k != "serviceKey"}
                encoded_key = quote(self.medium_api_key, safe='')
                query_string = urlencode(other_params)
                url_with_key = f"{url}?serviceKey={encoded_key}&{query_string}"
                response = requests.get(url_with_key, timeout=10)
            
            response.raise_for_status()
            
            if response.status_code == 200:
                data = response.json()
                result_code = data.get("response", {}).get("header", {}).get("resultCode", "")
                if result_code != "00":
                    result_msg = data.get("response", {}).get("header", {}).get("resultMsg", "")
                    debug_print(f"⚠️ API 응답 오류: {result_code} - {result_msg}")
                    return {}
                return self._parse_medium_forecast_data(data)
            else:
                debug_print(f"❌ HTTP 오류: {response.status_code}")
                debug_print(f"   응답 내용: {response.text[:500]}")
                return {}
        except requests.exceptions.HTTPError as e:
            debug_print(f"❌ 중기예보 조회 실패 (HTTP): {e}")
            if hasattr(e, 'response') and e.response is not None:
                debug_print(f"   상태 코드: {e.response.status_code}")
                debug_print(f"   응답 내용: {e.response.text[:500]}")
            return {}
        except Exception as e:
            debug_print(f"❌ 중기예보 조회 실패: {e}")
            return {}
    
    def _get_region_code(self, latitude: float, longitude: float) -> str:
        """
        위경도 기반으로 중기예보 지역 코드 추정
        
        Returns:
            지역 코드 (regId)
        """
        # 간단한 지역 코드 매핑 (위도 기반)
        # 실제로는 더 정교한 매핑이 필요할 수 있음
        if latitude > 38.0:
            # 강원도, 경기도 북부
            if longitude > 127.5:
                return "11D10000"  # 경기
            else:
                return "11D20000"  # 강원
        elif latitude > 37.0:
            # 서울, 인천, 경기 남부
            if longitude > 127.0:
                return "11B00000"  # 서울
            else:
                return "11B00000"  # 서울 (기본값)
        elif latitude > 36.0:
            # 충청도
            if longitude > 127.0:
                return "11C20000"  # 충남
            else:
                return "11C10000"  # 충북
        elif latitude > 35.0:
            # 전라도
            if longitude > 127.0:
                return "11F20000"  # 전남
            else:
                return "11F10000"  # 전북
        else:
            # 경상도
            if longitude > 128.5:
                return "11H20000"  # 경남
            else:
                return "11H10000"  # 경북
        
        # 기본값: 서울
        return "11B00000"
    
    def _parse_medium_forecast_data(self, api_response: Dict) -> Dict[str, Any]:
        """중기예보 데이터 파싱"""
        try:
            items = api_response.get("response", {}).get("body", {}).get("items", {}).get("item", [])
            
            if not items:
                debug_print("⚠️ 중기예보 데이터가 없습니다.")
                return {}
            
            # 첫 번째 항목 사용
            item = items[0] if isinstance(items, list) else items
            
            medium_data = {
                "forecast_type": "중기",
                "region": item.get("regId", ""),
                "forecast_date": item.get("tmFc", ""),
                "temp_high_low": item.get("taMin3", ""),  # 3일 후 최저기온
                "temp_low_low": item.get("taMax3", ""),   # 3일 후 최고기온
                "precipitation_trend": item.get("rnSt3Am", ""),  # 강수 경향
                "weather_summary": item.get("wf3Am", ""),  # 날씨 요약
            }
            
            debug_print("✅ 중기예보 데이터 파싱 완료")
            return medium_data
            
        except Exception as e:
            debug_print(f"⚠️ 중기예보 데이터 파싱 오류: {e}")
            return {}
    
    def _parse_forecast_data(
        self,
        api_response: Dict,
        forecast_type: str
    ) -> List[Dict[str, Any]]:
        """예보 데이터 파싱 (초단기/단기)"""
        forecast_list = []
        
        try:
            items = api_response.get("response", {}).get("body", {}).get("items", {}).get("item", [])
            
            if not items:
                debug_print(f"⚠️ {forecast_type}예보 데이터가 없습니다.")
                return []
            
            # 시간별로 그룹화
            time_groups = {}
            for item in items:
                fcst_date = item.get("fcstDate", "")
                fcst_time = item.get("fcstTime", "")
                fcst_time_key = fcst_date + fcst_time
                
                if fcst_time_key not in time_groups:
                    time_groups[fcst_time_key] = {
                        "fcst_date": fcst_date,
                        "fcst_time": fcst_time,
                        "fcst_datetime": fcst_date + fcst_time
                    }
                
                category = item.get("category")
                fcst_value = item.get("fcstValue")
                
                if category == "TMP":  # 기온
                    time_groups[fcst_time_key]["temp"] = float(fcst_value)
                elif category == "TMX":  # 최고기온
                    time_groups[fcst_time_key]["temp_max"] = float(fcst_value)
                elif category == "TMN":  # 최저기온
                    time_groups[fcst_time_key]["temp_min"] = float(fcst_value)
                elif category == "REH":  # 습도
                    time_groups[fcst_time_key]["rh"] = float(fcst_value)
                elif category == "PCP":  # 강수량
                    if fcst_value and fcst_value != "0" and fcst_value != "강수없음":
                        time_groups[fcst_time_key]["precipitation"] = fcst_value
                elif category == "WSD":  # 풍속
                    time_groups[fcst_time_key]["wind_speed"] = float(fcst_value)
                elif category == "VEC":  # 풍향
                    time_groups[fcst_time_key]["wind_dir"] = float(fcst_value)
                elif category == "SKY":  # 하늘상태
                    sky_code = int(fcst_value)
                    sky_map = {1: "맑음", 3: "구름많음", 4: "흐림"}
                    time_groups[fcst_time_key]["sky_condition"] = sky_map.get(sky_code, "알 수 없음")
            
            # 리스트로 변환
            for fcst_time_key, data in time_groups.items():
                forecast_list.append({
                    "forecast_type": forecast_type,
                    **data
                })
            
            # 시간순 정렬
            forecast_list.sort(key=lambda x: x["fcst_datetime"])
            
            debug_print(f"✅ {forecast_type}예보 {len(forecast_list)}개 데이터 파싱 완료")
            
        except Exception as e:
            debug_print(f"⚠️ 예보 데이터 파싱 오류: {e}")
        
        return forecast_list
    
    def get_all_forecasts(
        self,
        latitude: float,
        longitude: float
    ) -> Dict[str, Any]:
        """
        모든 예보 데이터 통합 조회
        
        Returns:
            {
                "ultra_short": [...],  # 0~6시간
                "short": [...],        # 1~3일
                "medium": {...}        # 4~10일
            }
        """
        return {
            "ultra_short": self.get_ultra_short_forecast(latitude, longitude),
            "short": self.get_short_forecast(latitude, longitude),
            "medium": self.get_medium_forecast(latitude, longitude)
        }
    
    def get_current_weather_context(self) -> str:
        """
        현재 경작지의 기상 데이터를 컨텍스트 문자열로 반환
        
        Returns:
            기상 데이터 컨텍스트 문자열
        """
        farm_info = get_farm_info()
        if not farm_info:
            return ""
        
        try:
            latitude = farm_info.get('latitude')
            longitude = farm_info.get('longitude')
            
            if not latitude or not longitude:
                return ""
            
            # 위치 정보 추출
            location_name = farm_info.get('road_address') or farm_info.get('legal_address') or "해당 위치"
            
            # 초단기 + 단기 예보 조회
            ultra_short = self.get_ultra_short_forecast(latitude, longitude)
            short = self.get_short_forecast(latitude, longitude)
            
            if not ultra_short and not short:
                return ""
            
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
            
            if not current_forecast:
                return ""
            
            # 컨텍스트 문자열 생성 (위치 정보 포함, 예보 데이터임을 명시)
            context_parts = [f"🌤️ 현재 기상 조건 ({location_name} 기준, 기상청 예보 데이터):"]
            
            # 기온 정보를 가장 먼저 표시하고 강조
            if current_forecast.get("temp") is not None:
                context_parts.append(f"🌡️ **현재 기온: {current_forecast['temp']}℃**")
            
            if current_forecast.get("temp_max") is not None and current_forecast.get("temp_min") is not None:
                context_parts.append(f"🌡️ **예상 기온 범위: {current_forecast['temp_min']}~{current_forecast['temp_max']}℃**")
            
            if current_forecast.get("rh") is not None:
                context_parts.append(f"- 습도: {current_forecast['rh']}%")
            
            if current_forecast.get("precipitation"):
                context_parts.append(f"- 강수량: {current_forecast['precipitation']}")
            
            if current_forecast.get("wind_speed") is not None:
                context_parts.append(f"- 풍속: {current_forecast['wind_speed']}m/s")
            
            if current_forecast.get("sky_condition"):
                context_parts.append(f"- 하늘 상태: {current_forecast['sky_condition']}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            debug_print(f"⚠️ 기상 데이터 컨텍스트 생성 실패: {e}")
            return ""
    
    def extract_date_from_question(self, question: str) -> Optional[datetime]:
        """
        질문에서 날짜 정보 추출
        과거/미래를 자동으로 판단
        
        Args:
            question: 사용자 질문
            
        Returns:
            추출된 날짜 또는 None (현재 날짜 의미)
        """
        now = datetime.now()
        question_lower = question.lower()
        
        # "3월", "4월" 같은 월 정보 추출
        month_match = re.search(r'(\d+)월', question)
        if month_match:
            month = int(month_match.group(1))
            if 1 <= month <= 12:
                # 현재 월과 비교하여 과거/미래 판단
                if month < now.month:
                    # 현재보다 이전 월이면 내년으로 간주 (미래)
                    # 예: 현재 11월, 질문 "3월" → 내년 3월 (미래)
                    target_date = datetime(now.year + 1, month, 15)
                    debug_print(f"📅 월 정보 추출: '{month}월' → {target_date.strftime('%Y-%m-%d')} (미래, 내년)")
                    return target_date
                else:
                    # 현재보다 이후 월이면 올해로 간주
                    # 예: 현재 11월, 질문 "12월" → 올해 12월
                    target_date = datetime(now.year, month, 15)
                    # 올해인데 현재보다 미래면 그대로, 과거면 이미 지난 날짜
                    if target_date < now:
                        # 이미 지난 날짜면 내년으로 처리 (사용자가 보통 내년을 의미)
                        target_date = datetime(now.year + 1, month, 15)
                        debug_print(f"📅 월 정보 추출: '{month}월' → {target_date.strftime('%Y-%m-%d')} (이미 지난 날짜, 내년으로 처리)")
                    else:
                        debug_print(f"📅 월 정보 추출: '{month}월' → {target_date.strftime('%Y-%m-%d')} (올해)")
                    return target_date
        
        # "작년 3월", "작년 4월" 같은 과거 월 정보
        last_year_match = re.search(r'작년\s*(\d+)월', question)
        if last_year_match:
            month = int(last_year_match.group(1))
            if 1 <= month <= 12:
                target_date = datetime(now.year - 1, month, 15)
                debug_print(f"📅 작년 월 정보 추출: '작년 {month}월' → {target_date.strftime('%Y-%m-%d')} (과거)")
                return target_date
        
        # "내년 2월" 같은 미래 월 정보
        next_year_match = re.search(r'내년\s*(\d+)월', question)
        if next_year_match:
            month = int(next_year_match.group(1))
            if 1 <= month <= 12:
                target_date = datetime(now.year + 1, month, 15)
                debug_print(f"📅 내년 월 정보 추출: '내년 {month}월' → {target_date.strftime('%Y-%m-%d')} (미래)")
                return target_date
        
        # "봄", "여름", "가을", "겨울" 같은 계절 정보
        # 계절도 현재 시점과 비교하여 과거/미래 판단
        if "봄" in question or "춘계" in question:
            target_date = datetime(now.year, 4, 15)
            if target_date < now:
                # 이미 지난 봄이면 내년 봄으로 처리
                target_date = datetime(now.year + 1, 4, 15)
                debug_print(f"📅 계절 정보 추출: '봄' → {target_date.strftime('%Y-%m-%d')} (이미 지난 봄, 내년으로 처리)")
            else:
                debug_print(f"📅 계절 정보 추출: '봄' → {target_date.strftime('%Y-%m-%d')} (올해)")
            return target_date
        elif "여름" in question or "하계" in question:
            target_date = datetime(now.year, 7, 15)
            if target_date < now:
                target_date = datetime(now.year + 1, 7, 15)
                debug_print(f"📅 계절 정보 추출: '여름' → {target_date.strftime('%Y-%m-%d')} (이미 지난 여름, 내년으로 처리)")
            else:
                debug_print(f"📅 계절 정보 추출: '여름' → {target_date.strftime('%Y-%m-%d')} (올해)")
            return target_date
        elif "가을" in question or "추계" in question:
            target_date = datetime(now.year, 10, 15)
            if target_date < now:
                target_date = datetime(now.year + 1, 10, 15)
                debug_print(f"📅 계절 정보 추출: '가을' → {target_date.strftime('%Y-%m-%d')} (이미 지난 가을, 내년으로 처리)")
            else:
                debug_print(f"📅 계절 정보 추출: '가을' → {target_date.strftime('%Y-%m-%d')} (올해)")
            return target_date
        elif "겨울" in question or "동계" in question:
            target_date = datetime(now.year, 1, 15)
            # 1월은 연도 경계 처리 필요
            if now.month >= 2:  # 2월 이후면 올해 1월은 이미 지남
                target_date = datetime(now.year + 1, 1, 15)
                debug_print(f"📅 계절 정보 추출: '겨울' → {target_date.strftime('%Y-%m-%d')} (이미 지난 겨울, 내년으로 처리)")
            else:
                debug_print(f"📅 계절 정보 추출: '겨울' → {target_date.strftime('%Y-%m-%d')} (올해)")
            return target_date
        
        return None
    
    def get_weather_for_date(
        self,
        latitude: float,
        longitude: float,
        target_date: Optional[datetime] = None,
        question: Optional[str] = None
    ) -> str:
        """
        특정 날짜의 기상 데이터 조회 (예보 또는 관측)
        과거/미래를 자동으로 판단하여 적절한 데이터 소스 사용
        
        Args:
            latitude: 위도
            longitude: 경도
            target_date: 조회할 날짜 (None이면 현재 날짜)
            question: 사용자 질문 (날짜 추출용)
            
        Returns:
            기상 데이터 컨텍스트 문자열
        """
        # 날짜가 지정되지 않았으면 질문에서 추출 시도
        if target_date is None and question:
            target_date = self.extract_date_from_question(question)
        
        # 날짜가 여전히 None이면 현재 날짜 사용
        if target_date is None:
            target_date = datetime.now()
        
        now = datetime.now()
        days_diff = (target_date - now).days
        
        # 날짜 판단 및 로깅
        if days_diff > 0:
            debug_print(f"📅 날짜 판단: {target_date.strftime('%Y-%m-%d')} (미래, {days_diff}일 후)")
        elif days_diff < 0:
            debug_print(f"📅 날짜 판단: {target_date.strftime('%Y-%m-%d')} (과거, {abs(days_diff)}일 전)")
        else:
            debug_print(f"📅 날짜 판단: {target_date.strftime('%Y-%m-%d')} (오늘)")
        
        # 모든 날짜에 대해 예보 데이터 사용 (현재 시점 기준)
        # 과거/미래 날짜는 예보 범위를 벗어나므로 현재 예보 데이터 사용
        if days_diff != 0:
            debug_print(f"⚠️ {target_date.strftime('%Y-%m-%d')} 날짜는 예보 범위를 벗어나므로 현재 예보 데이터를 사용합니다.")
        
        return self.get_current_weather_context()

