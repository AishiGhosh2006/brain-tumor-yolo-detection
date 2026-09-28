import cv2, os, random

IMG_DIR = "dataset/images/train"
LBL_DIR = "dataset/labels/train"
SAMPLE_N = 15

files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
sample = random.sample(files, min(SAMPLE_N, len(files)))
os.makedirs("outputs/annotation_check", exist_ok=True)

for fname in sample:
    img_path = os.path.join(IMG_DIR, fname)
    lbl_path = os.path.join(LBL_DIR, os.path.splitext(fname)[0] + ".txt")
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = list(map(float, line.split()))
                if not parts:
                    continue
                cls, xc, yc, bw, bh = parts
                x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
                x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, str(int(cls)), (x1, max(y1-5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(os.path.join("outputs/annotation_check", fname), img)

print(f"Saved {len(sample)} annotated preview images to outputs/annotation_check/")
