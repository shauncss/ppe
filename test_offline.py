from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load YOUR newly trained Medium model
    model = YOLO('runs/detect/train/weights/best.pt')

    # 2. Run inference on a static image
    print("Testing Image...")
    model.predict(source='test_image.jpg', save=True)

    # 3. Run inference on a video file
    print("Testing Video...")
    model.predict(source='test_video.mp4', save=True)