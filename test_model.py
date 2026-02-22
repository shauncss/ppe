from ultralytics import YOLO

# The "Shield" for Windows multiprocessing
if __name__ == '__main__':
    
    # 1. Load the model
    model = YOLO('yolo11n.pt') 

    # 2. Train the model
    model.train(
        data="construction-ppe.yaml", 
        epochs=10, 
        imgsz=640,
        device=0
    )   