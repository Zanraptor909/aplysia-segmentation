# src/train.py
from ultralytics import YOLO
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]     # project root
    data_yaml = root / "configs" / "dataset.yaml"  # C:\...\TakeFive_Its_A_Vibe\configs\dataset.yaml

    model = YOLO("yolov8s-seg.pt")
    results = model.train(
        data=str(data_yaml),
        imgsz=640,
        epochs=300,
        batch=8,
        workers=4,
        project=str(root / "runs"),
        name="seg_v8s",
        pretrained=True,
    )
    print(results)

if __name__ == "__main__":
    main()
