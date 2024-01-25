from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')

# Training
results = model.train(
   data='/home/students/cs/greenery/datas/data.yaml',
   imgsz=640,
   epochs=100,
   batch=32,
   project='/home/students/cs/greenery/yolov8/separate_model'
)