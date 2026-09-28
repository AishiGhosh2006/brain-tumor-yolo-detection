from ultralytics import YOLO

model = YOLO("runs/train/brain_yolo_s_exp1/weights/best.pt")

# ONNX - good general-purpose export, works with onnxruntime on the Pi
model.export(format="onnx", opset=12, simplify=True, imgsz=416)

# NCNN - often fastest on ARM CPUs (Raspberry Pi), uncomment to use instead/also
# model.export(format="ncnn", imgsz=416)
