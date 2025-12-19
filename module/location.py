"""
경작지 위치 관리 모듈
"""
from typing import Dict, Any, Optional
from .config import debug_print

# GeocodingManager를 선택적으로 import
try:
    from .geocoding import GeocodingManager
    _GEOCODING_AVAILABLE = True
except ImportError:
    GeocodingManager = None
    _GEOCODING_AVAILABLE = False
    debug_print("⚠️ GeocodingManager를 사용할 수 없습니다. 지오코딩 기능이 비활성화됩니다.")

# 경작지 위치 관리를 위한 전역 변수 (실제 사용 시에는 더 나은 방법 사용 권장)
_USER_FARM_INFO: Optional[Dict[str, Any]] = None
_geo_manager: Optional[Any] = None


def get_farm_location() -> Optional[Dict[str, Any]]:
    """현재 설정된 경작지 위치 정보를 반환하는 함수"""
    return _USER_FARM_INFO


def get_farm_info() -> Optional[Dict[str, Any]]:
    """현재 설정된 경작지 정보를 반환하는 함수"""
    global _USER_FARM_INFO
    if _USER_FARM_INFO is None:
        debug_print("⚠️ 경작지 정보가 설정되지 않았습니다.")
    return _USER_FARM_INFO


def set_farm_info(farm_data: Dict[str, Any]) -> None:
    """경작지 정보를 직접 설정하는 함수"""
    global _USER_FARM_INFO
    _USER_FARM_INFO = farm_data
    # road_address가 None이면 legal_address 또는 user_input 사용 도로명 또는 법정동 주소
    display_address = farm_data.get('road_address') or farm_data.get('legal_address') or farm_data.get('user_input', 'N/A')
    debug_print(f"✅ 경작지 정보가 설정되었습니다: {display_address}")


def clear_farm_info() -> None:
    """경작지 정보를 초기화하는 함수"""
    global _USER_FARM_INFO
    _USER_FARM_INFO = None
    debug_print("✅ 경작지 정보가 초기화되었습니다.")


def get_location_context() -> str:
    """현재 경작지 위치 정보를 AI 모델에서 사용할 수 있는 구조화된 텍스트로 변환"""
    farm_info = get_farm_info()
    
    if not farm_info:
        return "경작지 위치 정보가 설정되지 않았습니다."
    
    context = f"""
📍 경작지 위치 정보:
- 주소: {farm_info.get('road_address', 'N/A')}
- 법정동: {farm_info.get('legal_address', 'N/A')}
- 좌표: ({farm_info.get('longitude')}, {farm_info.get('latitude')})
"""
    return context.strip()


def get_geo_manager():
    """GeocodingManager 인스턴스를 가져오거나 생성하는 함수"""
    global _geo_manager
    
    if not _GEOCODING_AVAILABLE:
        print("❌ GeocodingManager 모듈을 import할 수 없습니다.")
        print("💡 module/geocoding.py 파일이 존재하는지 확인하세요.")
        return None
    
    if _geo_manager is None:
        try:
            _geo_manager = GeocodingManager()
            debug_print("✅ GeocodingManager 인스턴스 생성 완료")
        except ValueError as e:
            # API 키 관련 에러 (항상 출력)
            print(f"❌ GeocodingManager 초기화 실패: {e}")
            print("💡 NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET 환경변수가 .env 파일에 설정되어 있는지 확인하세요.")
            print("💡 .env 파일 위치: RAGsystem module/.env (또는 프로젝트 루트/.env)")
            return None
        except Exception as e:
            # 기타 에러 (항상 출력)
            print(f"❌ GeocodingManager 초기화 실패: {e}")
            print("💡 지오코딩 모듈 초기화 중 예상치 못한 오류가 발생했습니다.")
            return None
    
    return _geo_manager


def setup_farm_location(geo_manager=None) -> Optional[Dict[str, Any]]:
    """사용자가 직접 주소를 입력하여 경작지 위치를 설정하는 함수"""
    global _USER_FARM_INFO
    
    # geo_manager가 제공되지 않으면 자동으로 가져오기
    if geo_manager is None:
        geo_manager = get_geo_manager()
    
    if not geo_manager:
        # get_geo_manager()에서 이미 상세한 에러 메시지를 출력했으므로 여기서는 간단히 처리
        # 에러 메시지는 get_geo_manager()에서 이미 출력됨
        return None
    
    debug_print("🌱 경작지 위치 설정")
    debug_print("=" * 50)
    
    try:
        user_address = input("\n📍 경작지 주소를 입력해주세요: ").strip()
        
        if not user_address:
            debug_print("❌ 주소를 입력해주세요.")
            return None
        
        debug_print(f"🔍 입력된 주소: {user_address}")
        debug_print("📍 주소를 좌표로 변환하고 정확한 주소를 확인하는 중...")
        
        # 지오코딩 + 리버스 지오코딩 통합 처리
        result = geo_manager.get_final_address(user_address, verbose=False)
        
        if result:
            debug_print("✅ 주소 변환 완료!")
            _USER_FARM_INFO = {
                'longitude': result['coordinates']['longitude'],
                'latitude': result['coordinates']['latitude'],
                'road_address': result['final_address']['road_address'],
                'legal_address': result['final_address']['legal_address'],
                'full_address': result['final_address']['full_address'],
                'user_input': user_address,
                'raw_data': result
            }
            
            debug_print(f"🎉 경작지 위치 설정 완료!")
            debug_print(f"📍 설정된 위치: {_USER_FARM_INFO['road_address']}")
            return _USER_FARM_INFO
        else:
            debug_print("❌ 주소를 찾을 수 없습니다.")
            return None
            
    except KeyboardInterrupt:
        debug_print("\n\n❌ 사용자가 취소했습니다.")
        return None
    except Exception as e:
        debug_print(f"❌ 오류 발생: {e}")
        return None

