from transformers import AutoModelForImageClassification, AutoImageProcessor
from pathlib import Path
import os

# 현재 파일의 디렉토리를 기준으로 plant_classification_model 경로 찾기
_current_file = Path(__file__).resolve()
_module_dir = _current_file.parent
_base_dir = _module_dir.parent  # modelupdate 디렉토리
_model_dir = _base_dir / "plant_classification_model"

# 경로가 존재하는지 확인
if not _model_dir.exists():
    # 상위 디렉토리에서 찾기
    _base_dir = _base_dir.parent
    _model_dir = _base_dir / "modelupdate" / "plant_classification_model"

_model_path = str(_model_dir)

# 모델 로드 (경로가 존재할 때만)
if os.path.exists(_model_path):
    image_classification_model = AutoModelForImageClassification.from_pretrained(_model_path)
    image_classification_processor = AutoImageProcessor.from_pretrained(_model_path, use_fast=True)
    
    # 클래스 매핑 로드 (class_mapping.txt에서)
    all_classes = {}
    class_mapping_file = _model_dir / "class_mapping.txt"
    if class_mapping_file.exists():
        with open(class_mapping_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                
                # 클래스 매핑 파싱 (숫자: 클래스명 형식)
                if ":" in line and not line.startswith("=") and not line.startswith("클래스"):
                    try:
                        idx, name = line.split(": ", 1)
                        all_classes[int(idx)] = name
                    except ValueError:
                        continue  # 파싱 실패 시 무시
else:
    # 모델 경로가 없을 경우 None으로 설정 (이미지 분석 기능 비활성화)
    image_classification_model = None
    image_classification_processor = None
    all_classes = {}
    print(f"⚠️ 이미지 분류 모델을 찾을 수 없습니다: {_model_path}")
