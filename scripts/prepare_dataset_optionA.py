"""
Prepares Option A: the Ultralytics official Brain Tumor dataset
(893 train / 223 val images, 2 classes: negative, positive)

Downloads it and reorganizes it into YOUR existing project structure:
    dataset/images/{train,val,test}
    dataset/labels/{train,val,test}
    dataset/data.yaml

Run this from the root of brain_yolo_project (same level as dataset/, scripts/).

Usage (PowerShell, venv active):
    python scripts\\prepare_dataset_optionA.py
"""

import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

# ---- EDIT ME if you want a different held-out test fraction ----
TEST_FRACTION = 0.15   # carve this fraction OUT of the original 893 train images
SEED = 42

DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip"
WORK_DIR = Path("raw_data/brain_tumor_download")
ZIP_PATH = WORK_DIR / "brain-tumor.zip"
EXTRACT_DIR = WORK_DIR / "extracted"

PROJECT_IMG = Path("dataset/images")
PROJECT_LBL = Path("dataset/labels")


def download_and_extract():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"Downloading {DATASET_URL} ...")
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        print("Download complete.")
    else:
        print("Zip already downloaded, skipping.")

    if not EXTRACT_DIR.exists():
        print("Extracting ...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)
    else:
        print("Already extracted, skipping.")

    # The zip layout is: brain-tumor/{train,valid}/{images,labels} OR
    # brain-tumor/images/{train,val} + brain-tumor/labels/{train,val}
    # depending on release version -- auto-detect both.
    root = next(EXTRACT_DIR.glob("brain-tumor*"), EXTRACT_DIR)
    return root


def find_split_dirs(root: Path):
    """Return dicts mapping split name -> (images_dir, labels_dir), handling
    both known folder layouts Ultralytics has shipped for this dataset."""
    candidates = {
        "train": [root / "train" / "images", root / "images" / "train"],
        "val":   [root / "valid" / "images", root / "images" / "val", root / "val" / "images"],
    }
    resolved = {}
    for split, opts in candidates.items():
        for img_dir in opts:
            if img_dir.exists():
                lbl_dir = Path(str(img_dir).replace("images", "labels"))
                resolved[split] = (img_dir, lbl_dir)
                break
    if "train" not in resolved or "val" not in resolved:
        raise FileNotFoundError(
            f"Could not locate train/val folders under {root}. "
            f"Contents: {list(root.rglob('*'))[:20]}"
        )
    return resolved


def copy_pairs(img_dir, lbl_dir, filenames, dest_split):
    (PROJECT_IMG / dest_split).mkdir(parents=True, exist_ok=True)
    (PROJECT_LBL / dest_split).mkdir(parents=True, exist_ok=True)
    n = 0
    for fname in filenames:
        img_src = img_dir / fname
        lbl_src = lbl_dir / (Path(fname).stem + ".txt")
        if not img_src.exists():
            continue
        shutil.copy(img_src, PROJECT_IMG / dest_split / fname)
        if lbl_src.exists():
            shutil.copy(lbl_src, PROJECT_LBL / dest_split / lbl_src.name)
        else:
            # negative-class images can legitimately have empty label files
            (PROJECT_LBL / dest_split / (Path(fname).stem + ".txt")).touch()
        n += 1
    return n


def main():
    random.seed(SEED)
    root = download_and_extract()
    splits = find_split_dirs(root)

    train_img_dir, train_lbl_dir = splits["train"]
    val_img_dir, val_lbl_dir = splits["val"]

    train_files = sorted([f.name for f in train_img_dir.glob("*") if f.is_file()])
    random.shuffle(train_files)

    n_test = int(len(train_files) * TEST_FRACTION)
    test_files = train_files[:n_test]
    final_train_files = train_files[n_test:]

    val_files = sorted([f.name for f in val_img_dir.glob("*") if f.is_file()])

    n_train = copy_pairs(train_img_dir, train_lbl_dir, final_train_files, "train")
    n_test_copied = copy_pairs(train_img_dir, train_lbl_dir, test_files, "test")
    n_val = copy_pairs(val_img_dir, val_lbl_dir, val_files, "val")

    print(f"\nDone.")
    print(f"  train: {n_train} images")
    print(f"  val:   {n_val} images")
    print(f"  test:  {n_test_copied} images  (held out from original train split, never used in training/tuning)")

    yaml_content = """path: dataset
train: images/train
val: images/val
test: images/test

names:
  0: negative
  1: positive
"""
    Path("dataset/data.yaml").write_text(yaml_content)
    print("\nWrote dataset/data.yaml")
    print("\nNext: python scripts\\train.py  (or the yolo CLI command from Phase 6)")


if __name__ == "__main__":
    main()
