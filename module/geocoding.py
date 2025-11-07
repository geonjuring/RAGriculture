"""
지오코딩 모듈
네이버 지오코딩 API를 사용한 주소-좌표 변환 기능
"""
import requests
import json
import os
from dotenv import load_dotenv
from .config import debug_print


class GeocodingManager:
    """
    네이버 지오코딩 API를 사용한 주소-좌표 변환 클래스
    
    주요 기능:
    - 주소 → 좌표 변환 (지오코딩)
    - 좌표 → 주소 변환 (리버스 지오코딩)
    - 통합 워크플로우 (주소 → 좌표 → 정확한 주소)
    """
    
    def __init__(self):
        """클래스 초기화"""
        self.client_id = None
        self.client_secret = None
        self._load_api_keys()
    
    def _load_api_keys(self):
        """API 키 로드"""
        load_dotenv()
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        
        if not self.client_id or not self.client_secret:
            raise ValueError("NAVER_CLIENT_ID 또는 NAVER_CLIENT_SECRET이 환경변수에 설정되지 않았습니다.")
    
    def geocode(self, address):
        """
        주소를 좌표로 변환하는 메서드 (지오코딩)
        
        Args:
            address (str): 변환할 주소
        
        Returns:
            dict: 좌표 정보가 담긴 딕셔너리
        """
        # API 요청 설정
        url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": self.client_id,
            "X-NCP-APIGW-API-KEY": self.client_secret,
        }
        params = {
            "query": address,
            "output": "json"
        }
        
        try:
            # API 요청
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            # JSON 파싱
            result = response.json()
            
            # 응답 타입 확인
            if not isinstance(result, dict):
                raise Exception(f"예상치 못한 응답 형식: {type(result)}")
            
            # API 응답 상태 확인
            if result.get('status') != 'OK':
                error_msg = result.get('errorMessage', '알 수 없는 오류')
                raise Exception(f"API 오류: {error_msg}")
            
            # 주소 정보 추출
            addresses = result.get('addresses', [])
            if not addresses:
                raise Exception(f"'{address}'에 대한 좌표 정보를 찾을 수 없습니다.")
            
            # 첫 번째 결과 사용
            first_address = addresses[0]
            if not isinstance(first_address, dict):
                raise Exception(f"주소 정보 형식 오류: {type(first_address)}")
            
            coordinate_info = {
                'longitude': float(first_address.get('x', 0)),
                'latitude': float(first_address.get('y', 0)),
                'road_address': first_address.get('roadAddress', ''),
                'jibun_address': first_address.get('jibunAddress', ''),
                'raw_response': result
            }
            
            return coordinate_info
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API 요청 실패: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"응답 JSON 파싱 실패: {e}")
        except Exception as e:
            raise Exception(f"주소 변환 실패: {e}")
    
    def reverse_geocode(self, longitude, latitude):
        """
        좌표를 주소로 변환하는 메서드
        
        Args:
            longitude (float): 경도
            latitude (float): 위도
        
        Returns:
            dict: 주소 정보가 담긴 딕셔너리
        """
        # 좌표 설정
        coords = f"{longitude},{latitude}"
        
        # API 요청 설정
        url = "https://maps.apigw.ntruss.com/map-reversegeocode/v2/gc"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": self.client_id,
            "X-NCP-APIGW-API-KEY": self.client_secret,
        }
        params = {
            "coords": coords,
            "output": "json",
            "orders": "roadaddr,legalcode"
        }
        
        try:
            # API 요청
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            # JSON 파싱
            result = response.json()
            
            # API 응답 상태 확인
            if result.get('status', {}).get('code') != 0:
                error_msg = result.get('status', {}).get('message', '알 수 없는 오류')
                raise Exception(f"API 오류: {error_msg}")
            
            # 주소 정보 추출
            results = result.get('results', [])
            if not results:
                raise Exception("해당 좌표에 대한 주소 정보를 찾을 수 없습니다.")
            
            address_info = {
                'road_address': None,
                'legal_address': None,
                'full_address': None,
                'raw_response': result
            }
            
            # 도로명 주소 정보 추출
            for item in results:
                if item.get('name') == 'roadaddr':
                    road_info = item
                    region = road_info.get('region', {})
                    land = road_info.get('land', {})
                    
                    # 주소 조합
                    address_parts = []
                    if region.get('area1', {}).get('name'):
                        address_parts.append(region['area1']['name'])
                    if region.get('area2', {}).get('name'):
                        address_parts.append(region['area2']['name'])
                    if land.get('name'):
                        address_parts.append(land['name'])
                    if land.get('number1'):
                        address_parts.append(land['number1'])
                    if land.get('number2'):
                        address_parts.append(land['number2'])
                    
                    address_info['road_address'] = ' '.join(address_parts)
                    break
            
            # 법정동 정보 추출
            for item in results:
                if item.get('name') == 'legalcode':
                    legal_info = item
                    region = legal_info.get('region', {})
                    
                    # 법정동 주소 조합
                    legal_parts = []
                    if region.get('area1', {}).get('name'):
                        legal_parts.append(region['area1']['name'])
                    if region.get('area2', {}).get('name'):
                        legal_parts.append(region['area2']['name'])
                    if region.get('area3', {}).get('name'):
                        legal_parts.append(region['area3']['name'])
                    
                    address_info['legal_address'] = ' '.join(legal_parts)
                    break
            
            # 전체 주소 정보 생성
            if address_info['road_address'] and address_info['legal_address']:
                address_info['full_address'] = f"도로명: {address_info['road_address']}\n법정동: {address_info['legal_address']}"
            elif address_info['road_address']:
                address_info['full_address'] = f"도로명: {address_info['road_address']}"
            elif address_info['legal_address']:
                address_info['full_address'] = f"법정동: {address_info['legal_address']}"
            
            return address_info
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API 요청 실패: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"응답 JSON 파싱 실패: {e}")
        except Exception as e:
            raise Exception(f"주소 변환 실패: {e}")
    
    def get_final_address(self, user_address, verbose=True):
        """
        주소 입력 → 좌표 추출 → 리버스 지오코딩 → 최종 주소
        
        Args:
            user_address (str): 사용자가 입력한 주소
            verbose (bool): 상세 출력 여부
        
        Returns:
            dict: 최종 주소 정보
        """
        if verbose:
            debug_print(f"🔍 입력 주소: {user_address}")
            debug_print("=" * 60)
        
        try:
            # 1단계: 지오코딩 (주소 → 좌표)
            if verbose:
                debug_print("📍 1단계: 주소를 좌표로 변환 중...")
            
            geocode_result = self.geocode(user_address)
            longitude = geocode_result['longitude']
            latitude = geocode_result['latitude']
            
            if verbose:
                debug_print(f"✅ 좌표 추출 완료: {longitude}, {latitude}")
            
            # 2단계: 리버스 지오코딩 (좌표 → 정확한 주소)
            if verbose:
                debug_print("\n📍 2단계: 좌표를 정확한 주소로 변환 중...")
            
            reverse_result = self.reverse_geocode(longitude, latitude)
            
            # 최종 결과 구성
            final_result = {
                'input_address': user_address,
                'coordinates': {
                    'longitude': longitude,
                    'latitude': latitude
                },
                'final_address': {
                    'road_address': reverse_result['road_address'],
                    'legal_address': reverse_result['legal_address'],
                    'full_address': reverse_result['full_address']
                },
                'raw_data': {
                    'geocode_info': geocode_result,
                    'reverse_geocode_info': reverse_result
                }
            }
            
            # 결과 출력
            if verbose:
                debug_print("\n🎉 최종 주소 변환 완료!")
                debug_print("=" * 60)
                debug_print("📋 변환 결과:")
                debug_print(f"   입력 주소: {user_address}")
                debug_print(f"   추출 좌표: {longitude}, {latitude}")
                debug_print(f"   최종 도로명: {reverse_result['road_address']}")
                debug_print(f"   최종 법정동: {reverse_result['legal_address']}")
                debug_print("=" * 60)
            
            return final_result
            
        except Exception as e:
            if verbose:
                debug_print(f"❌ 주소 변환 실패: {e}")
            return None

