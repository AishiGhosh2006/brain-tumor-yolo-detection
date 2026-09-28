from ultralytics import YOLO

model = YOLO("runs/train/brain_yolo_s_exp1/weights/best.pt")
model.predict(source=0, conf=0.4, show=True, stream=True)  # source=0 -> default webcam
