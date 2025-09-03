from ultralytics import YOLO

def main():
    model = YOLO("yolov8s-seg.pt")  # or yolov8n-seg.pt if VRAM is tight
    results = model.train(
        data="configs/dataset.yaml",
        imgsz=640,
        epochs=300,
        batch=8,
        workers=4,
        project="runs",
        name="seg_v8s",
        pretrained=True
    )
    print(results)

if __name__ == "__main__":
    main()
