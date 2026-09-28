from ultralytics import YOLO

model = YOLO("runs/train/brain_yolo_s_exp1/weights/best.pt")
results = model.predict(
    source="path/to/test_image.png",   # EDIT ME: put the real image path here
    conf=0.4,
    save=True,
    project="outputs/predicted_images",
    name="run1",
)
