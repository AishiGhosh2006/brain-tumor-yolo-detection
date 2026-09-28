# Brain Tumor Detection using YOLO11n

A computer vision project for detecting **negative** and **positive** brain-tumor-related findings from brain images using the **YOLO11n object detection model** and the Ultralytics framework.

> **Important:** This is an academic/computer-vision project. The model output is not a medical diagnosis and should not be used as a substitute for evaluation by a qualified medical professional.

## 1. Project Overview

This project trains a YOLO object detection model on a brain tumor dataset. The trained model can be used to:

- Detect labeled regions in brain images.
- Classify detections into two classes: `negative` and `positive`.
- Evaluate the model using precision, recall, mAP@50, and mAP@50-95.
- Run inference on images, videos, or a webcam.
- Export the trained model for deployment.

The final successful training used **YOLO11n**. A larger YOLO11s CPU training attempt terminated with a Windows native process crash, while the YOLO11n CPU configuration completed successfully.

## 2. Technologies Used

| Technology | Version / Details |
|---|---|
| Python | 3.14.3 |
| Ultralytics | 8.4.163 |
| YOLO | YOLO11n |
| PyTorch | 2.14.0+cpu |
| Hardware | 13th Gen Intel Core i7-1355U |
| Device | CPU |
| IDE | Visual Studio Code |
| OS | Windows |

## 3. Dataset

The project uses the **Ultralytics Brain Tumor dataset**.

```text
brain-tumor/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── brain-tumor.yaml
└── LICENSE.txt
```

### Dataset split

- Training images: **893**
- Validation images: **223**
- Classes: **2**
- Class 0: `negative`
- Class 1: `positive`

### `dataset/data.yaml`

```yaml
path: C:/Users/aishi/Downloads/brain_yolo_project/brain_yolo_project/brain-tumor

train: images/train
val: images/val

names:
  0: negative
  1: positive
```

## 4. Project Structure

```text
brain_yolo_project/
├── .venv/
├── brain-tumor/
├── dataset/
│   └── data.yaml
├── outputs/
├── raw_data/
├── runs/
│   └── detect/
│       └── runs/
│           ├── test/
│           │   └── cpu_test/
│           └── final/
│               └── brain_tumor_yolo11n/
│                   ├── weights/
│                   │   ├── best.pt
│                   │   └── last.pt
│                   ├── results.png
│                   ├── confusion_matrix.png
│                   ├── confusion_matrix_normalized.png
│                   ├── PR_curve.png
│                   ├── F1_curve.png
│                   ├── P_curve.png
│                   └── R_curve.png
├── scripts/
│   ├── evaluate.py
│   ├── export_model.py
│   ├── infer_image.py
│   ├── infer_video.py
│   ├── infer_webcam.py
│   ├── prepare_dataset_optionA.py
│   ├── train_optionA_bigger.py
│   └── verify_annotations.py
├── README.md
└── requirements.txt
```

## 5. Environment Setup

Open PowerShell in the project directory:

```powershell
cd C:\Users\aishi\Downloads\brain_yolo_project\brain_yolo_project
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install requirements:

```powershell
pip install -r requirements.txt
```

Check Python:

```powershell
python --version
```

Check Ultralytics:

```powershell
python -c "import ultralytics; print(ultralytics.__version__)"
```

## 6. Dataset Verification

Run:

```powershell
python scripts\verify_annotations.py
```

The final dataset scan successfully found:

- 893/893 training images
- 223/223 validation images

## 7. Final Training Configuration

| Parameter | Value |
|---|---|
| Model | YOLO11n |
| Requested epochs | 100 |
| Image size | 320 × 320 |
| Batch size | 2 |
| Device | CPU |
| Workers | 0 |
| Patience | 20 |
| Dataset | `dataset/data.yaml` |

Training command:

```powershell
python -c "from ultralytics import YOLO; model=YOLO('yolo11n.pt'); model.train(data='dataset/data.yaml',epochs=100,imgsz=320,batch=2,device='cpu',workers=0,project='runs/final',name='brain_tumor_yolo11n',patience=20)"
```

## 8. Training Result

Early stopping ended the run before all 100 requested epochs:

- **Best epoch:** 26
- **Completed epochs:** 46
- **Patience:** 20
- **Training time:** approximately 22.499 hours
- **Best checkpoint:** `best.pt`
- **Last checkpoint:** `last.pt`

The training log reports that the best results were observed at epoch 26 and that training stopped after no improvement during the patience period.

## 9. Final Model

Best model:

```text
C:\Users\aishi\Downloads\brain_yolo_project\brain_yolo_project\runs\detect\runs\final\brain_tumor_yolo11n\weights\best.pt
```

Last checkpoint:

```text
C:\Users\aishi\Downloads\brain_yolo_project\brain_yolo_project\runs\detect\runs\final\brain_tumor_yolo11n\weights\last.pt
```

For final inference/evaluation, use `best.pt`.

## 10. Final Validation Results

Final validation was performed on **223 validation images** containing **241 labeled instances**.

| Metric | Result |
|---|---:|
| Precision | **44.5%** |
| Recall | **83.0%** |
| mAP@50 | **50.2%** |
| mAP@50-95 | **35.7%** |

### Class-wise results

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|
| negative | 142 | 154 | 0.551 | 0.786 | 0.562 | 0.387 |
| positive | 81 | 87 | 0.339 | 0.874 | 0.443 | 0.328 |
| **all** | **223** | **241** | **0.445** | **0.830** | **0.502** | **0.357** |

Recorded final validation speed was approximately:

- Preprocess: **0.3 ms/image**
- Inference: **20.5 ms/image**
- Postprocess: **0.7 ms/image**

## 11. Metric Explanation

### Precision

```text
Precision = True Positives / (True Positives + False Positives)
```

It indicates how many predicted detections were correct.

### Recall

```text
Recall = True Positives / (True Positives + False Negatives)
```

It indicates how many actual target detections were found.

### mAP@50

Mean Average Precision using an IoU threshold of 0.50.

### mAP@50-95

Mean Average Precision averaged over IoU thresholds from 0.50 through 0.95. It is stricter than mAP@50.

## 12. Training Graphs

Open the final run folder:

```powershell
explorer .\runs\detect\runs\final\brain_tumor_yolo11n
```

Important generated files:

```text
results.png
confusion_matrix.png
confusion_matrix_normalized.png
PR_curve.png
F1_curve.png
P_curve.png
R_curve.png
```

- `results.png` — training/validation loss and metric curves.
- `confusion_matrix.png` — prediction counts by class.
- `confusion_matrix_normalized.png` — normalized confusion matrix.
- `PR_curve.png` — precision-recall curve.
- `F1_curve.png` — F1 score across confidence thresholds.
- `P_curve.png` — precision across confidence thresholds.
- `R_curve.png` — recall across confidence thresholds.

## 13. Image Inference

The project includes:

```text
scripts/infer_image.py
```

The model path should be:

```text
runs/detect/runs/final/brain_tumor_yolo11n/weights/best.pt
```

Typical Ultralytics usage:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/runs/final/brain_tumor_yolo11n/weights/best.pt")
results = model("path/to/image.jpg")
```

## 14. Video Inference

Script:

```text
scripts/infer_video.py
```

Use the final model:

```text
runs/detect/runs/final/brain_tumor_yolo11n/weights/best.pt
```

Follow the input arguments/configuration already defined in `infer_video.py`.

## 15. Webcam Inference

Script:

```text
scripts/infer_webcam.py
```

Use `best.pt` for webcam detection. Since the model was trained and run on CPU, real-time performance can be limited compared with GPU inference.

## 16. Evaluation

Script:

```text
scripts/evaluate.py
```

Use:

```text
runs/detect/runs/final/brain_tumor_yolo11n/weights/best.pt
```

The reported results are validation results from this project and are not an independent clinical test.

## 17. Model Export

Script:

```text
scripts/export_model.py
```

Source model:

```text
runs/detect/runs/final/brain_tumor_yolo11n/weights/best.pt
```

The exact export format depends on the script configuration and installed dependencies.

## 18. Limitations

- Results are based on the available dataset split.
- The reported numbers are validation metrics, not clinical performance measurements.
- Precision is lower than recall in the final recorded validation results.
- The positive class recorded precision of 0.339 and recall of 0.874.
- CPU training is slow.
- The project has not established clinical validity.
- Performance may change on images from different scanners, hospitals, acquisition settings, or populations.
- The model should not be used as a standalone medical diagnostic system.

## 19. Future Improvements

1. Train on a GPU for faster experimentation.
2. Compare YOLO11n with larger YOLO models when hardware permits.
3. Tune image size, batch size, learning rate, and augmentation.
4. Study confidence thresholds using the PR and F1 curves.
5. Evaluate on a separate held-out test set if available.
6. Compare multiple runs under the same evaluation protocol.
7. Improve dataset quality and class balance where appropriate.
8. Build a web interface for image upload and detection.
9. Export and validate the model in the desired deployment format.

## 20. Quick Start

```powershell
cd C:\Users\aishi\Downloads\brain_yolo_project\brain_yolo_project
.\.venv\Scripts\Activate.ps1
python scripts\verify_annotations.py
```

Final model:

```text
runs\detect\runs\final\brain_tumor_yolo11n\weights\best.pt
```

Open results:

```powershell
explorer .\runs\detect\runs\final\brain_tumor_yolo11n
```

## 21. Final Project Summary

| Item | Final Detail |
|---|---|
| Project | Brain Tumor Detection |
| Framework | Ultralytics YOLO |
| Model | YOLO11n |
| Classes | negative, positive |
| Training images | 893 |
| Validation images | 223 |
| Validation instances | 241 |
| Requested epochs | 100 |
| Completed epochs | 46 |
| Best epoch | 26 |
| Image size | 320 |
| Batch size | 2 |
| Device | CPU |
| Precision | 44.5% |
| Recall | 83.0% |
| mAP@50 | 50.2% |
| mAP@50-95 | 35.7% |
| Best model | `best.pt` |
| Training time | ~22.499 hours |

## 22. Final Model Path

```text
C:\Users\aishi\Downloads\brain_yolo_project\brain_yolo_project\runs\detect\runs\final\brain_tumor_yolo11n\weights\best.pt
```

This README documents the final successful YOLO11n training run and its recorded validation results.
