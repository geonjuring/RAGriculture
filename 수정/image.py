from transformers import AutoModelForImageClassification, AutoImageProcessor

image_classification_model = AutoModelForImageClassification.from_pretrained("./plant_classification_model")
image_classification_processor = AutoImageProcessor.from_pretrained("./plant_classification_model")
    
    # 클래스 매핑 로드 (class_mapping.txt에서)
all_classes = {}
with open("./plant_classification_model/class_mapping.txt", "r", encoding="utf-8") as f:
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
