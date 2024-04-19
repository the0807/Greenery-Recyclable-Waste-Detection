#import comet_ml
from ultralytics import YOLO

#comet_ml.init(project_name="Greenery Recyclable Waste Detection")

# Load the model
model = YOLO('yolov8m.pt')

# Training
results = model.train(
   data='/home/students/cs/greenery/datasets/data.yaml',
   imgsz=640,
   epochs=1000,
   batch=24,
   device=0,
   workers=4,
   plots=True,
   project='/home/students/cs/greenery/yolov8/separate_model'
)
