from ultralytics import YOLO

# 1. Load the model
model = YOLO('yolo11n.pt') 

# 2. Trigger the auto-download and start training
# By simply using "construction-ppe.yaml", Ultralytics pulls the data for you
model.train(data="construction-ppe.yaml", epochs=10, imgsz=640)