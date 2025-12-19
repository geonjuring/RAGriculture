"""
검색 기록 관리 모듈
검색 기록 저장, 조회, 삭제 기능
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .config import debug_print


class SearchHistoryManager:
    """검색 기록 관리 클래스"""
    
    def __init__(self, history_file: Optional[str] = None, retention_days: int = 30):
        """
        Args:
            history_file: 검색 기록 파일 경로 (None이면 기본 경로 사용)
            retention_days: 기록 보관 기간 (일)
        """
        if history_file is None:
            # 기본 경로: module 디렉토리의 상위 디렉토리에 cache 폴더 생성
            base_dir = Path(__file__).parent.parent
            cache_dir = base_dir / "cache"
            cache_dir.mkdir(exist_ok=True)
            history_file = str(cache_dir / "search_history.json")
        
        self.history_file = history_file
        self.retention_days = retention_days
        self._ensure_history_file()
    
    def _ensure_history_file(self):
        """검색 기록 파일이 없으면 생성"""
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """검색 기록 로드"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            debug_print(f"⚠️ 검색 기록 로드 실패: {e}")
            return []
    
    def _save_history(self, history: List[Dict[str, Any]]):
        """검색 기록 저장"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            debug_print(f"⚠️ 검색 기록 저장 실패: {e}")
    
    def _cleanup_old_records(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """오래된 기록 삭제"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        filtered = []
        
        for record in history:
            try:
                record_date = datetime.fromisoformat(record.get('timestamp', ''))
                if record_date >= cutoff_date:
                    filtered.append(record)
            except (ValueError, TypeError):
                # 잘못된 날짜 형식이면 유지 (에러 방지)
                filtered.append(record)
        
        return filtered
    
    def add_search(
        self,
        question: str,
        answer: str,
        search_type: str = "general",
        location: Optional[str] = None,
        crop: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        검색 기록 추가
        
        Args:
            question: 검색 질문
            answer: 검색 답변
            search_type: 검색 타입 ("general" 또는 "schedule")
            location: 위치 정보 (일정표 검색의 경우)
            crop: 작물명 (일정표 검색의 경우)
            metadata: 추가 메타데이터
        
        Returns:
            추가된 검색 기록
        """
        history = self._load_history()
        
        # 답변 요약 (너무 길면 잘라서 저장)
        answer_summary = answer[:500] + "..." if len(answer) > 500 else answer
        
        record = {
            "id": f"search_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer_summary,
            "answer_full": answer,  # 전체 답변도 저장
            "search_type": search_type,
            "location": location,
            "crop": crop,
            "metadata": metadata or {}
        }
        
        history.insert(0, record)  # 최신 기록을 맨 앞에 추가
        
        # 오래된 기록 정리
        history = self._cleanup_old_records(history)
        
        # 최대 기록 수 제한 (메모리 관리)
        max_records = 1000
        if len(history) > max_records:
            history = history[:max_records]
        
        self._save_history(history)
        debug_print(f"✅ 검색 기록 저장: {question[:50]}...")
        
        return record
    
    def get_recent_searches(
        self,
        limit: int = 20,
        search_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        최근 검색 기록 조회
        
        Args:
            limit: 조회할 최대 개수
            search_type: 검색 타입 필터 ("general" 또는 "schedule", None이면 전체)
        
        Returns:
            검색 기록 리스트
        """
        history = self._load_history()
        
        # 오래된 기록 정리
        history = self._cleanup_old_records(history)
        self._save_history(history)  # 정리된 기록 저장
        
        # 검색 타입 필터링
        if search_type:
            history = [h for h in history if h.get('search_type') == search_type]
        
        return history[:limit]
    
    def get_search_by_id(self, search_id: str) -> Optional[Dict[str, Any]]:
        """ID로 검색 기록 조회"""
        history = self._load_history()
        for record in history:
            if record.get('id') == search_id:
                return record
        return None
    
    def delete_search(self, search_id: str) -> bool:
        """검색 기록 삭제"""
        history = self._load_history()
        original_count = len(history)
        history = [h for h in history if h.get('id') != search_id]
        
        if len(history) < original_count:
            self._save_history(history)
            debug_print(f"✅ 검색 기록 삭제: {search_id}")
            return True
        return False
    
    def clear_all_history(self):
        """모든 검색 기록 삭제"""
        self._save_history([])
        debug_print("✅ 모든 검색 기록 삭제 완료")
    
    def get_statistics(self) -> Dict[str, Any]:
        """검색 기록 통계"""
        history = self._load_history()
        history = self._cleanup_old_records(history)
        
        total = len(history)
        general_count = sum(1 for h in history if h.get('search_type') == 'general')
        schedule_count = sum(1 for h in history if h.get('search_type') == 'schedule')
        
        # 날짜별 통계
        date_counts = {}
        for record in history:
            try:
                record_date = datetime.fromisoformat(record.get('timestamp', ''))
                date_key = record_date.strftime('%Y-%m-%d')
                date_counts[date_key] = date_counts.get(date_key, 0) + 1
            except (ValueError, TypeError):
                pass
        
        return {
            "total": total,
            "general": general_count,
            "schedule": schedule_count,
            "date_counts": date_counts,
            "retention_days": self.retention_days
        }

