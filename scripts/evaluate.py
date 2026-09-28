from ultralytics import YOLO

model = YOLO("runs/train/brain_yolo_s_exp1/weights/best.pt")
metrics = model.val(data="dataset/data.yaml", split="test")

p = metrics.box.mp
r = metrics.box.mr
map50 = metrics.box.map50
map5095 = metrics.box.map
f1 = 2 * p * r / (p + r + 1e-9)

print("\n=== FINAL TEST RESULTS (from actual model run) ===")
print(f"{'Metric':<20}{'Score':>10}")
print(f"{'Precision':<20}{p*100:>9.2f}%")
print(f"{'Recall':<20}{r*100:>9.2f}%")
print(f"{'F1-score':<20}{f1*100:>9.2f}%")
print(f"{'mAP@50':<20}{map50*100:>9.2f}%")
print(f"{'mAP@50-95':<20}{map5095*100:>9.2f}%")

print("\nPer-class mAP@50-95:")
class_names = model.names  # {0: 'negative', 1: 'positive'}
for i, cls_map in enumerate(metrics.box.maps):
    print(f"  {class_names[i]:<15}{cls_map*100:>6.2f}%")
