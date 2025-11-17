"""
에러 처리 시스템 모듈
"""
import time
import functools
from typing import Callable, Any, Dict
from enum import Enum
from .config import debug_print


class ErrorType(Enum):
    """에러 타입 정의"""
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    SYSTEM_ERROR = "system_error"


class ErrorHandler:
    """에러 처리 핸들러 클래스"""
    
    def __init__(self):
        self.error_counts = {}
        self.max_retries = 3
        self.base_delay = 1.0
        self.max_delay = 60.0
        
    def get_retry_delay(self, error_type: ErrorType, attempt: int) -> float:
        """지수 백오프 지연 시간 계산"""
        if error_type == ErrorType.NETWORK_ERROR:
            return min(self.base_delay * (2 ** attempt), self.max_delay)
        elif error_type == ErrorType.API_ERROR:
            return min(self.base_delay * (1.5 ** attempt), self.max_delay / 2)
        else:
            return min(self.base_delay * (1.2 ** attempt), self.max_delay / 4)
    
    def should_retry(self, error_type: ErrorType, attempt: int) -> bool:
        """재시도 여부 결정"""
        if attempt >= self.max_retries:
            return False
        
        if error_type == ErrorType.SYSTEM_ERROR:
            return attempt < 1  # 시스템 에러는 1회만 재시도
        
        return True
    
    def handle_error(self, error: Exception, error_type: ErrorType, attempt: int) -> Dict[str, Any]:
        """에러 처리 및 복구 정보 반환"""
        error_key = f"{error_type.value}_{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        retry_delay = self.get_retry_delay(error_type, attempt)
        should_retry = self.should_retry(error_type, attempt)
        
        return {
            "error": str(error),
            "error_type": error_type.value,
            "attempt": attempt,
            "retry_delay": retry_delay,
            "should_retry": should_retry,
            "error_count": self.error_counts[error_key]
        }


# 전역 에러 핸들러
error_handler = ErrorHandler()


def robust_error_handling(error_type: ErrorType = ErrorType.PROCESSING_ERROR):
    """강화된 에러 처리 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            max_attempts = 3
            
            while attempt < max_attempts:
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        debug_print(f"✅ {func.__name__} 재시도 성공 (시도 {attempt + 1})")
                    return result
                    
                except Exception as e:
                    attempt += 1
                    error_info = error_handler.handle_error(e, error_type, attempt)
                    
                    debug_print(f"❌ {func.__name__} 에러 발생 (시도 {attempt}): {error_info['error']}")
                    debug_print(f"   에러 타입: {error_info['error_type']}")
                    debug_print(f"   재시도 가능: {error_info['should_retry']}")
                    
                    if not error_info['should_retry']:
                        debug_print(f"🚫 {func.__name__} 최대 재시도 횟수 초과")
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": error_type.value,
                            "attempts": attempt
                        }
                    
                    if attempt < max_attempts:
                        delay = error_info['retry_delay']
                        debug_print(f"⏳ {delay:.2f}초 후 재시도...")
                        time.sleep(delay)
                    else:
                        debug_print(f"🚫 {func.__name__} 모든 재시도 실패")
                        return {
                            "status": "error",
                            "error": str(e),
                            "error_type": error_type.value,
                            "attempts": attempt
                        }
            
            return {
                "status": "error",
                "error": "모든 재시도 실패",
                "error_type": error_type.value,
                "attempts": attempt
            }
        
        return wrapper
    return decorator


def retry_with_backoff(func: Callable, max_retries: int = 3, base_delay: float = 1.0) -> Callable:
    """지수 백오프를 사용한 재시도 함수"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    debug_print(f"🚫 {func.__name__} 최대 재시도 횟수 초과: {e}")
                    raise e
                
                delay = base_delay * (2 ** attempt)
                debug_print(f"⏳ {func.__name__} 재시도 {attempt + 1}/{max_retries} ({delay:.2f}초 후)")
                time.sleep(delay)
        
        return None
    return wrapper


def system_recovery(state: Dict[str, Any]) -> Dict[str, Any]:
    """시스템 복구 메커니즘"""
    debug_print("🔄 시스템 복구 시작...")
    
    # 1. 상태 검증 및 복구
    if not state.get("question"):
        debug_print("⚠️ 질문이 없습니다. 기본 질문으로 복구")
        state["question"] = "농업 관련 질문을 입력해주세요"
    
    # 2. 재시도 카운터 초기화
    if "retry_count" not in state:
        state["retry_count"] = 0
    
    # 3. 복구 완료
    debug_print("✅ 시스템 복구 완료")
    return {
        "status": "recovered",
        "state": state,
        "recovery_attempted": True
    }


def handle_specific_errors(func: Callable) -> Callable:
    """구체적인 에러 핸들러"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError as e:
            debug_print(f"🔑 키 에러: {e}")
            return {"status": "error", "error": f"필수 키 누락: {e}"}
        except ValueError as e:
            debug_print(f"📊 값 에러: {e}")
            return {"status": "error", "error": f"잘못된 값: {e}"}
        except ConnectionError as e:
            debug_print(f"🌐 연결 에러: {e}")
            return {"status": "error", "error": f"네트워크 연결 실패: {e}"}
        except Exception as e:
            debug_print(f"❌ 예상치 못한 에러: {e}")
            return {"status": "error", "error": f"시스템 에러: {e}"}
    
    return wrapper

