from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo26l.pt') 

    model.train(
        data="roboflow_dataset/data.yaml", 
        epochs=50, 
        imgsz=1024,
        batch=2,
        workers=2,
        device=0 
    )