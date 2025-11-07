# schedule_html_exporter.py
"""
작물 재배 일정표 HTML 및 JSON 내보내기 모듈
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

# 모듈 레벨 변수 (외부에서 설정 가능)
USER_FARM_INFO = None

def get_crop_display_name(crop_code):
    """작물 코드를 표시명으로 변환 (하드코딩 제거)"""
    # result 데이터에서 crop_name이 있으면 사용, 없으면 crop_code 그대로 사용
    return crop_code

def get_main_schedule_order(schedules):
    """메인 일정표에서 작업 순서 추출 (하드코딩 제거)"""
    order = {}
    for i, schedule in enumerate(schedules):
        task = schedule.get('task', '')
        if task:
            order[task] = i + 1
    return order

def extract_month_number(month_str):
    """월 문자열에서 숫자 추출 (범용 처리)"""
    import re
    numbers = re.findall(r'\d+', str(month_str))
    if numbers:
        return int(numbers[0])
    return 999

def extract_keywords_auto(task, details):
    """자동 키워드 추출 (하드코딩 제거)"""
    keywords = set()
    
    # 작업명에서 키워드 추출
    if task:
        keywords.add(task)
    
    # 상세내용에서 주요 단어 추출
    if details:
        import re
        words = re.findall(r'[\w가-힣]+', details)
        keywords.update([word for word in words if len(word) > 1])
    
    return list(keywords)

def get_annual_cycle_info(schedules):
    """메인 일정표에서 연간 순서 정보 추출"""
    # 메인 일정표의 기간을 분석하여 연간 순서 파악
    cycle_info = {
        'has_cross_year': False,
        'start_month': None,
        'end_month': None
    }
    
    for schedule in schedules:
        period = schedule.get('period', '')
        if '~' in period and ('12월' in period or '1월' in period or '2월' in period):
            cycle_info['has_cross_year'] = True
            # 연간 순서가 있는 경우 처리
            break
    
    return cycle_info

class ScheduleExporter:
    """일정표를 HTML 및 JSON 형식으로 내보내는 클래스"""
    
    def __init__(self):
        self.html_template = self._get_html_template()
    
    def export_to_html(self, result: Dict[str, Any], filename: str = None) -> str:
        """일정표를 HTML 형식으로 내보내기"""
        crop = result.get("crop", "unknown")
        crop_name = result.get("crop_name", crop)  # 하드코딩 제거
        
        # 위치 정보 가져오기
        farm_info = self._get_farm_info()
        location_info = self._create_location_html(farm_info)
        
        # 테이블 HTML 생성
        main_table_html = self._create_main_table_html(result)
        detailed_table_html = self._create_detailed_table_html(result)  # 추가
        overlap_table_html = self._create_overlap_table_html(result)
        
        # 월별 캘린더 생성
        monthly_calendar = self._create_monthly_calendar_html(result)
        
        # HTML 내용 생성
        html_content = self.html_template.format(
            crop_name=crop_name,
            location_info=location_info,
            status=result.get('status', 'unknown'),
            documents_count=len(result.get('documents', [])),
            schedules_count=len(result.get('schedules', [])),
            detailed_tasks_count=len(result.get('detailed_tasks', [])),  # 추가
            overlaps_count=len(result.get('overlaps', [])),
            current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            main_table_html=main_table_html,
            detailed_table_html=detailed_table_html,  # 추가
            overlap_table_html=overlap_table_html,
            monthly_calendar=monthly_calendar,
            current_time_korean=datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')
        )
        
        # 파일명 생성
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{crop_name}_schedule_{timestamp}.html"
        
        # HTML 파일 저장
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML 일정표가 저장되었습니다: {filename}")
        return filename
    
    def export_to_json(self, result: Dict[str, Any], filename: str = None) -> str:
        """일정표를 JSON 형식으로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            crop_name = get_crop_display_name(result.get("crop", "unknown"))
            filename = f"{crop_name}_schedule_{timestamp}.json"
        
        # 기본 정보
        crop = result.get("crop", "unknown")
        crop_name = get_crop_display_name(crop)
        
        # 위치 정보
        location_info = self._get_farm_info()
        
        # 처리 정보
        processing_info = {
            "documents_found": len(result.get("documents", [])),
            "schedules_extracted": len(result.get("schedules", [])),
            "detailed_tasks_extracted": len(result.get("detailed_tasks", [])),  # 추가
            "overlaps_detected": len(result.get("overlaps", []))
        }
        
        # 메인 일정표 (순서 검증 및 수정)
        schedules = []
        for schedule in result.get("schedules", []):
            schedule_data = {
                "period": schedule.get("period", ""),
                "task": schedule.get("task", ""),
                "details": schedule.get("details", ""),
                "month_range": self._extract_months(schedule.get("period", "")),
                "task_id": f"{crop}_{schedule.get('task', '')}_{len(schedules)+1:03d}",
                "search_keywords": self._extract_keywords(
                    schedule.get("task", ""), 
                    schedule.get("details", "")
                )
            }
            schedules.append(schedule_data)
        
        # 메인 일정표 순서 검증 및 수정
        schedules = self._validate_and_fix_schedule_order(schedules, crop)
        
        # 세분화된 작업 정보 추가 (월별 정렬 적용)
        detailed_tasks = []
        for task in result.get("detailed_tasks", []):
            task_data = {
                "task_name": task.get("task_name", ""),
                "month": task.get("month", ""),
                "subtask": task.get("subtask", ""),
                "period": task.get("period", ""),
                "description": task.get("description", ""),
                "methods": task.get("methods", []),
                "precautions": task.get("precautions", []),
                "tools_materials": task.get("tools_materials", []),
                "environmental_conditions": task.get("environmental_conditions", ""),
                "frequency": task.get("frequency", ""),
                "duration": task.get("duration", "")
            }
            detailed_tasks.append(task_data)
        
        # 메인 일정표 순서를 고려한 정렬 적용
        detailed_tasks = self._sort_detailed_tasks_list_by_main_schedule_order(detailed_tasks, crop, result.get("schedules", []))
        
        # 겹치는 일정
        overlaps = []
        for overlap in result.get("overlaps", []):
            overlap_data = {
                "month": overlap.get("month", ""),
                "period": overlap.get("period", ""),
                "tasks": overlap.get("tasks", []),
                "task1": overlap.get("task1", ""),
                "task2": overlap.get("task2", ""),
                "note": overlap.get("note", "")
            }
            overlaps.append(overlap_data)
        
        # 월별 캘린더
        monthly_calendar = self._populate_monthly_calendar(schedules)
        
        # JSON 데이터 구성
        json_data = {
            "crop": crop,
            "crop_name": crop_name,
            "generated_date": datetime.now().isoformat(),
            "location": location_info,
            "status": result.get("status", "unknown"),
            "processing_info": processing_info,
            "schedules": schedules,
            "detailed_tasks": detailed_tasks,  # 추가
            "overlaps": overlaps,
            "monthly_calendar": monthly_calendar
        }
        
        # JSON 파일 저장
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        return filename
    
    def export_both(self, result: Dict[str, Any], base_filename: str = None) -> tuple:
        """HTML과 JSON을 모두 내보내기"""
        crop = result.get("crop", "unknown")
        crop_name = result.get("crop_name", crop)  # 하드코딩 제거
        
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{crop_name}_schedule_{timestamp}"
        
        html_file = self.export_to_html(result, f"{base_filename}.html")
        json_file = self.export_to_json(result, f"{base_filename}.json")
        
        return html_file, json_file
    
    def _get_farm_info(self) -> Optional[Dict]:
        """경작지 정보 가져오기 (모듈 레벨 변수에서)"""
        return USER_FARM_INFO
    
    def _create_location_html(self, farm_info: Optional[Dict]) -> str:
        """위치 정보 HTML 생성"""
        if not farm_info:
            return ""
        
        return f"""
        <div class="location-info">
            <h3>📍 경작지 정보</h3>
            <p><strong>주소:</strong> {farm_info.get('road_address', 'N/A')}</p>
            <p><strong>법정동:</strong> {farm_info.get('legal_address', 'N/A')}</p>
            <p><strong>좌표:</strong> ({farm_info.get('longitude')}, {farm_info.get('latitude')})</p>
        </div>
        """
    
    def _create_main_table_html(self, result: Dict[str, Any]) -> str:
        """메인 일정표 HTML 생성 (순서 정렬 적용)"""
        main_df = result.get("main_df")
        if main_df is not None and not main_df.empty:
            # 메인 일정표 순서 정렬 적용
            main_df_sorted = self._sort_main_schedule_by_task_order(main_df, result.get("crop", ""))
            return main_df_sorted.to_html(index=False, classes="schedule-table", escape=False)
        return ""
    
    def _create_detailed_table_html(self, result: Dict[str, Any]) -> str:
        """월별 세분화 작업표 HTML 생성 (메인 일정표 순서 고려)"""
        detailed_df = result.get("detailed_df")
        if detailed_df is not None and not detailed_df.empty:
            crop = result.get("crop", "")
            main_schedules = result.get("schedules", [])
            
            # 메인 일정표 순서를 고려하여 정렬
            detailed_df_sorted = self._sort_detailed_tasks_by_main_schedule_order(
                detailed_df, crop, main_schedules
            )
            
            return f"""
            <h2>🔧 월별 세분화된 작업표 ({len(detailed_df_sorted)}개 항목)</h2>
            {detailed_df_sorted.to_html(index=False, classes="schedule-table", escape=False)}
            """
        return ""
    
    def _create_overlap_table_html(self, result: Dict[str, Any]) -> str:
        """겹침 일정표 HTML 생성"""
        overlap_df = result.get("overlap_df")
        if overlap_df is not None and not overlap_df.empty:
            return f"""
            <div class="overlap-section">
                <h3>⚠️ 겹치는 일정 ({len(overlap_df)}개)</h3>
                {overlap_df.to_html(index=False, classes="overlap-table", escape=False)}
            </div>
            """
        return ""
    
    def _create_monthly_calendar_html(self, result: Dict[str, Any]) -> str:
        """월별 캘린더 HTML 생성"""
        schedules = result.get("schedules", [])
        overlaps = result.get("overlaps", [])
        
        # 월별 작업 매핑
        monthly_tasks = {}
        for schedule in schedules:
            period = schedule.get("period", "")
            task = schedule.get("task", "")
            
            # 월 추출
            months = self._extract_months(period)
            for month in months:
                if month not in monthly_tasks:
                    monthly_tasks[month] = []
                monthly_tasks[month].append(task)
        
        # 겹침 정보 추가
        overlap_months = {}
        for overlap in overlaps:
            month = overlap.get("month")
            if month:
                overlap_months[month] = True
        
        # HTML 생성 (하드코딩 제거)
        month_names = [f"{i}월" for i in range(1, 13)]
        
        calendar_html = '<div class="calendar-grid">'
        
        for i, month_name in enumerate(month_names, 1):
            tasks = monthly_tasks.get(i, [])
            is_overlap = overlap_months.get(i, False)
            
            calendar_html += f'''
            <div class="month-card">
                <h4>{month_name}</h4>
            '''
            
            if tasks:
                for task in tasks:
                    overlap_class = "overlap-task" if is_overlap else ""
                    calendar_html += f'<div class="task-item {overlap_class}">{task}</div>'
            else:
                calendar_html += '<div style="color: #999; font-style: italic;">작업 없음</div>'
            
            calendar_html += '</div>'
        
        calendar_html += '</div>'
        
        return calendar_html
    
    def _extract_months(self, period: str) -> List[int]:
        """기간 문자열에서 월 추출"""
        months = []
        for i in range(1, 13):
            if f"{i}월" in period or str(i) in period:
                months.append(i)
        return months
    
    def _populate_schedules(self, json_data: Dict, result: Dict[str, Any]):
        """JSON 데이터에 일정 정보 추가"""
        schedules = result.get("schedules", [])
        for schedule in schedules:
            period = schedule.get("period", "")
            task = schedule.get("task", "")
            
            schedule_data = {
                "period": period,
                "task": task,
                "details": schedule.get("details", ""),
                "month_range": self._extract_months(period),
                "task_id": f"{json_data['crop']}_{task}_{len(json_data['schedules']) + 1:03d}",
                "search_keywords": self._extract_keywords(task, schedule.get("details", ""))
            }
            json_data["schedules"].append(schedule_data)
    
    def _populate_overlaps(self, json_data: Dict, result: Dict[str, Any]):
        """JSON 데이터에 겹침 정보 추가"""
        overlaps = result.get("overlaps", [])
        for overlap in overlaps:
            overlap_data = {
                "month": overlap.get("month"),
                "period": overlap.get("period", ""),
                "tasks": overlap.get("tasks", []),
                "task1": overlap.get("task1", ""),
                "task2": overlap.get("task2", ""),
                "note": overlap.get("note", "")
            }
            json_data["overlaps"].append(overlap_data)
    
    def _populate_monthly_calendar(self, schedules: List[Dict]) -> Dict:
        """월별 캘린더 정보 생성"""
        monthly_tasks = {}
        for schedule in schedules:  # schedules 리스트를 직접 사용
            for month in schedule["month_range"]:
                if month not in monthly_tasks:
                    monthly_tasks[month] = []
                monthly_tasks[month].append({
                    "task": schedule["task"],
                    "task_id": schedule["task_id"],
                    "period": schedule["period"]
                })
        
        return monthly_tasks
    
    def _sort_detailed_tasks_by_main_schedule_order(self, df: pd.DataFrame, crop: str, main_schedules: List[Dict] = None) -> pd.DataFrame:
        """메인 일정표 순서를 고려하여 세분화된 작업을 정렬"""
        
        # 실제 컬럼명 찾기
        task_col = None
        month_col = None
        subtask_col = None
        
        for col in df.columns:
            if '작업명' in col or 'task_name' in col.lower():
                task_col = col
            elif '월' in col or 'month' in col.lower():
                month_col = col
            elif '세부작업' in col or 'subtask' in col.lower():
                subtask_col = col
        
        if not task_col or not month_col:
            return df
        
        # 메인 일정표 작업 순서 매핑
        task_order = {}
        if main_schedules:
            for i, schedule in enumerate(main_schedules):
                task_name = schedule.get('task', '')
                if task_name:
                    task_order[task_name] = i
        
        def get_sort_key(row):
            # 1순위: 메인 일정표의 작업 순서
            task_name = str(row.get(task_col, ''))
            task_priority = task_order.get(task_name, 999)  # 없는 작업은 맨 뒤로
            
            # 2순위: 월별 순서 (11,12월은 다음 해로 간주)
            month_str = str(row.get(month_col, ''))
            month_num = extract_month_number(month_str)
            if month_num >= 11:  # 11월, 12월은 다음 해로 간주
                month_order = month_num - 12
            else:
                month_order = month_num
            
            # 3순위: 세부작업명 (같은 월 내에서 정렬)
            subtask = str(row.get(subtask_col, '')) if subtask_col else ''
            
            return (task_priority, month_order, subtask)
        
        df_sorted = df.copy()
        # 각 행에 대해 정렬 키를 계산하고 정렬
        sort_keys = [get_sort_key(row) for _, row in df_sorted.iterrows()]
        df_sorted['_sort_key'] = sort_keys
        df_sorted = df_sorted.sort_values('_sort_key').drop('_sort_key', axis=1)
        return df_sorted

    def _sort_detailed_tasks_by_month(self, df: pd.DataFrame) -> pd.DataFrame:
        """세분화된 작업을 월별로 정렬 (하드코딩 제거)"""
        def get_month_sort_key(month_str):
            # 범용 월 번호 추출
            month_num = extract_month_number(month_str)
            
            # 연간 순서 고려 (12월 다음이 1월)
            if month_num >= 11:  # 11월, 12월은 다음 해로 간주
                return month_num - 12
            return month_num
        
        # 실제 월 컬럼명 찾기
        month_col = None
        for col in df.columns:
            if '월' in col or 'month' in col.lower():
                month_col = col
                break
        
        if month_col:
            df_sorted = df.copy()
            df_sorted['_sort_key'] = df_sorted[month_col].apply(get_month_sort_key)
            df_sorted = df_sorted.sort_values('_sort_key').drop('_sort_key', axis=1)
            return df_sorted
        
        return df

    def _sort_detailed_tasks_list_by_month(self, detailed_tasks: List[Dict]) -> List[Dict]:
        """세분화된 작업 리스트를 월별로 정렬 (하드코딩 제거)"""
        def get_month_sort_key(task):
            month_str = task.get("month", "")
            return extract_month_number(month_str)
        
        return sorted(detailed_tasks, key=get_month_sort_key)

    def _sort_detailed_tasks_list_by_main_schedule_order(self, detailed_tasks: List[Dict], crop: str, main_schedules: List[Dict] = None) -> List[Dict]:
        """세분화된 작업 리스트를 메인 일정표 순서로 정렬"""
        
        # 메인 일정표 작업 순서 매핑
        task_order = {}
        if main_schedules:
            for i, schedule in enumerate(main_schedules):
                task_name = schedule.get('task', '')
                if task_name:
                    task_order[task_name] = i
        
        def get_sort_key(task):
            # 1순위: 메인 일정표의 작업 순서
            task_name = str(task.get("task_name", ""))
            task_priority = task_order.get(task_name, 999)
            
            # 2순위: 월별 순서 (11,12월은 다음 해로 간주)
            month_str = str(task.get("month", ""))
            month_num = extract_month_number(month_str)
            if month_num >= 11:  # 11월, 12월은 다음 해로 간주
                month_order = month_num - 12
            else:
                month_order = month_num
            
            # 3순위: 세부작업명 (같은 월 내에서 정렬)
            subtask = str(task.get("subtask", ""))
            
            return (task_priority, month_order, subtask)
        
        return sorted(detailed_tasks, key=get_sort_key)

    def _validate_and_fix_schedule_order(self, schedules: List[Dict], crop: str) -> List[Dict]:
        """메인 일정표의 순서를 검증하고 수정 (하드코딩 제거)"""
        # 메인 일정표가 이미 올바른 순서로 생성되었다면 그대로 사용
        # 추가적인 검증이나 수정이 필요한 경우에만 처리
        return schedules

    def _sort_main_schedule_by_task_order(self, df: pd.DataFrame, crop: str) -> pd.DataFrame:
        """메인 일정표를 작업 순서로 정렬 (하드코딩 제거)"""
        # 메인 일정표가 이미 올바른 순서로 생성되었다면 그대로 사용
        # 추가적인 정렬이 필요한 경우에만 처리
        return df

    def _extract_keywords(self, task: str, details: str) -> List[str]:
        """작업명과 상세내용에서 검색 키워드 추출 (하드코딩 제거)"""
        return extract_keywords_auto(task, details)
    
    def _get_html_template(self) -> str:
        """HTML 템플릿 반환"""
        return """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{crop_name} 재배 일정표</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #2c5530;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        
        h2 {{
            color: #4a6741;
            border-bottom: 2px solid #4a6741;
            padding-bottom: 10px;
            margin-top: 30px;
        }}
        
        h3 {{
            color: #6b8e6b;
            margin-top: 25px;
        }}
        
        .location-info {{
            background: #e8f5e8;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #28a745;
        }}
        
        .schedule-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        
        .schedule-table th {{
            background-color: #4a6741;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        
        .schedule-table td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        
        .schedule-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        .schedule-table tr:hover {{
            background-color: #f0f8f0;
        }}
        
        .overlap-section {{
            background: #fff3cd;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
        
        .overlap-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        
        .overlap-table th {{
            background-color: #ffc107;
            color: #212529;
            padding: 10px;
            text-align: left;
        }}
        
        .overlap-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        
        .calendar-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .month-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .month-card h4 {{
            margin: 0 0 10px 0;
            color: #4a6741;
            text-align: center;
            font-size: 1.2em;
        }}
        
        .task-item {{
            background: #e8f5e8;
            margin: 5px 0;
            padding: 8px;
            border-radius: 4px;
            font-size: 0.9em;
            border-left: 3px solid #28a745;
        }}
        
        .overlap-task {{
            background: #fff3cd;
            border-left-color: #ffc107;
        }}
        
        .status-info {{
            background: #d1ecf1;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #17a2b8;
            margin: 20px 0;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 15px;
            }}
            
            h1 {{
                font-size: 2em;
            }}
            
            .calendar-grid {{
                grid-template-columns: 1fr;
            }}
            
            .schedule-table {{
                font-size: 12px;
            }}
            
            .schedule-table th,
            .schedule-table td {{
                padding: 8px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌱 {crop_name} 재배 일정표</h1>
        
        {location_info}
        
        <div class="status-info">
            <h3>📊 처리 정보</h3>
            <p><strong>상태:</strong> {status}</p>
            <p><strong>검색된 문서:</strong> {documents_count}개</p>
            <p><strong>추출된 일정:</strong> {schedules_count}개</p>
            <p><strong>세분화된 작업:</strong> {detailed_tasks_count}개</p>
            <p><strong>겹치는 일정:</strong> {overlaps_count}개</p>
            <p><strong>생성일시:</strong> {current_time}</p>
        </div>
        
        <h2>📋 메인 재배 일정표</h2>
        {main_table_html}
        
        {detailed_table_html}
        
        {overlap_table_html}
        
        <h2>📅 월별 작업 캘린더</h2>
        {monthly_calendar}
        
        <div class="footer">
            <p>🌱 작물 재배 일정표 생성 시스템 | LangChain + LangGraph</p>
            <p>생성일시: {current_time_korean}</p>
        </div>
    </div>
</body>
</html>
"""

# 편의 함수들
def export_schedule_to_html(result: Dict[str, Any], filename: str = None) -> str:
    """일정표를 HTML로 내보내기 (편의 함수)"""
    exporter = ScheduleExporter()
    return exporter.export_to_html(result, filename)

def export_schedule_to_json(result: Dict[str, Any], filename: str = None) -> str:
    """일정표를 JSON으로 내보내기 (편의 함수)"""
    exporter = ScheduleExporter()
    return exporter.export_to_json(result, filename)

def export_schedule_both(result: Dict[str, Any], base_filename: str = None) -> tuple:
    """일정표를 HTML과 JSON으로 모두 내보내기 (편의 함수)"""
    exporter = ScheduleExporter()
    return exporter.export_both(result, base_filename)
