from ultralytics import YOLO

model = YOLO("runs/train/brain_yolo_s_exp1/weights/best.pt")
model.predict(
    source="raw_data/videos/your_scan.mp4",  # EDIT ME: put the real video path here
    conf=0.4,
    save=True,
    project="outputs/predicted_videos",
    name="run1",
    stream=False,
)
