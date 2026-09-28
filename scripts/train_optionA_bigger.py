"""
Same training pipeline as Phase 6, but using a bigger YOLO11 variant
instead of nano for higher accuracy. Requires dataset/data.yaml already
created by prepare_dataset_optionA.py.

Run:
    python scripts\\train_optionA_bigger.py
"""

from ultralytics import YOLO

# ---- EDIT ME: pick your size ----
# "yolo11s.pt" -> recommended sweet spot for this project
# "yolo11m.pt" -> noticeably higher accuracy, ~2x slower, needs more VRAM
MODEL = "yolo11s.pt"

model = YOLO(MODEL)

results = model.train(
    data="dataset/data.yaml",
    imgsz=640,
    batch=16,            # EDIT ME: drop to 8 if you hit CUDA out-of-memory on 's', to 4-8 on 'm'
    epochs=150,
    patience=25,
    optimizer="auto",
    lr0=0.01,
    mosaic=1.0,
    mixup=0.0,
    degrees=5,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.0, hsv_s=0.2, hsv_v=0.2,
    project="runs/train",
    name="brain_yolo_s_exp1",
    device="cpu",           # use "cpu" if no GPU (will be slow for s/m)
    val=True,
)
