import comet_ml
from ultralytics import YOLO

comet_ml.init(project_name="Greenery Recyclable Waste Detection")

# Load the model
model = YOLO('yolov8n.pt')

# Training
results = model.train(
   data='/home/students/cs/greenery/prep_datas/data.yaml',
   imgsz=640,
   epochs=1000,
   batch=64,
   device=[0,1],
   plots=True,
   project='/home/students/cs/greenery/yolov8/separate_model'
)
