from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')

# Training
results = model.train(
   data='/home/students/cs/greenery/prep_datas/data.yaml',
   imgsz=640,
   epochs=1000,
   batch=32,
   device=[0,1],
   project='/home/students/cs/greenery/yolov8/separate_model'
)
