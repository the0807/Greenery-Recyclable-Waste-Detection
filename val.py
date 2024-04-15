from ultralytics import YOLO

# Load a model
model = YOLO('/home/students/cs/greenery/yolov8/separate_model/train/weights/best.pt')

# Customize validation settings
validation_results = model.val(data='/HDD/greenery/datasets/data.yaml',
                               imgsz=640,
                               batch=16,
                               conf=0.25,
                               iou=0.6,
                               device='0',
                               plots=True)