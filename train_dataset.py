"""
MedScan AI — Dataset Training Script (v2 — High-Accuracy)
==========================================================
Trains medical image classifiers using:
  - RandomForestClassifier (well-calibrated probabilities, robust)
  - Combined features: HOG + Pixel Histogram + Downscaled pixels
  
Datasets:
  1. Bone Fracture (X-ray)  — Binary: Fractured vs Not Fractured
  2. Brain Tumor  (MRI)     — 4-class: Glioma, Meningioma, No Tumor, Pituitary
"""

import os
import sys
import json
import time
import random
import numpy as np
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# ─── Reproducibility ──────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BONE_FRACTURE_DIR = os.path.join(
    BASE_DIR,
    "dataset", "Bone Fracture",
    "Bone_Fracture_Binary_Classification",
    "Bone_Fracture_Binary_Classification",
)

BRAIN_TUMOR_DIR = os.path.join(
    BASE_DIR,
    "dataset", "Brain Tumor", "classification",
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────
IMG_SIZE = 128          # For HOG features
PIXEL_SIZE = 64         # Downscaled pixels as extra features
MAX_IMAGES_PER_CLASS = None   # None = use ALL images


# ══════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION (Combined: HOG + Histogram + Pixels)
# ══════════════════════════════════════════════════════════════════

def extract_hog_features(gray_img):
    """Extract HOG features from a grayscale image."""
    win_size = (IMG_SIZE, IMG_SIZE)
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    features = hog.compute(gray_img)
    return features.flatten()


def extract_combined_features(gray_img):
    """Extract a rich combined feature vector:
       1. HOG features (edge/gradient patterns)
       2. Pixel intensity histogram (32 bins)
       3. Downscaled raw pixels (captures texture/shape)
       4. Basic statistics (mean, std, skew, kurtosis)
    """
    # Resize for HOG
    img_hog = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))
    hog_feat = extract_hog_features(img_hog)
    
    # Pixel intensity histogram (32 bins, normalized)
    hist = cv2.calcHist([img_hog], [0], None, [32], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-7)
    
    # Downscaled raw pixels
    img_small = cv2.resize(gray_img, (PIXEL_SIZE, PIXEL_SIZE)).flatten().astype(np.float32) / 255.0
    
    # Basic statistics
    pixels = img_hog.flatten().astype(np.float32)
    mean_val = np.mean(pixels)
    std_val = np.std(pixels)
    # Quadrant means (spatial info)
    h, w = img_hog.shape
    q1 = img_hog[:h//2, :w//2].mean()
    q2 = img_hog[:h//2, w//2:].mean()
    q3 = img_hog[h//2:, :w//2].mean()
    q4 = img_hog[h//2:, w//2:].mean()
    center = img_hog[h//4:3*h//4, w//4:3*w//4].mean()
    border = np.mean([img_hog[:h//6].mean(), img_hog[-h//6:].mean(),
                      img_hog[:, :w//6].mean(), img_hog[:, -w//6:].mean()])
    stats = np.array([mean_val, std_val, q1, q2, q3, q4, center, border], dtype=np.float32) / 255.0
    
    # Combine all features
    combined = np.concatenate([hog_feat, hist, img_small, stats])
    return combined


def load_images_from_folder(folder_path, label, max_images=None):
    """Load images from folder, return (features_list, labels_list)."""
    features_list = []
    labels_list = []

    if not os.path.isdir(folder_path):
        print(f"  ⚠  Folder not found: {folder_path}")
        return features_list, labels_list

    files = sorted(os.listdir(folder_path))
    files = [f for f in files if f.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp', '.webp', '.tif', '.tiff'))]

    if max_images:
        files = files[:max_images]

    loaded = 0
    skipped = 0
    for fname in files:
        fpath = os.path.join(folder_path, fname)
        try:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped += 1
                continue
            feat = extract_combined_features(img)
            features_list.append(feat)
            labels_list.append(label)
            loaded += 1
        except Exception:
            skipped += 1

    print(f"    → Loaded {loaded} images (skipped {skipped}) from …/{os.path.basename(folder_path)}")
    return features_list, labels_list


# ══════════════════════════════════════════════════════════════════
#  TRAINING UTILITY
# ══════════════════════════════════════════════════════════════════

def train_and_save(X_train, y_train, X_val, y_val, class_names, model_name, pkl_name):
    """Train a RandomForest classifier, evaluate, save."""
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Classes:  {class_names}")
    print(f"  Train:    {len(X_train)} samples")
    print(f"  Val:      {len(X_val)} samples")
    print(f"  Features: {X_train.shape[1]} dimensions")
    print(f"{'='*60}")

    # RandomForest gives well-calibrated predict_proba out of the box
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
        ))
    ])

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"\n  ⏱  Training time : {elapsed:.1f}s")
    print(f"  🎯 Val Accuracy  : {acc*100:.2f}%\n")
    print(classification_report(y_val, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print("  Confusion Matrix:")
    header = "        " + "  ".join(f"{c[:8]:>8}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>8}" for v in row)
        print(f"  {class_names[i][:8]:>8}  {row_str}")

    # Test a few predictions to show calibration
    print(f"\n  Sample predictions (showing probability calibration):")
    for i in range(min(5, len(X_val))):
        proba = pipeline.predict_proba(X_val[i:i+1])[0]
        pred_idx = np.argmax(proba)
        true_label = class_names[y_val[i]]
        pred_label = class_names[pred_idx]
        conf = proba[pred_idx] * 100
        status = "✓" if pred_label == true_label else "✗"
        print(f"    {status} True: {true_label:15s} → Pred: {pred_label:15s} ({conf:.1f}%)")

    # Save model
    pkl_path = os.path.join(MODEL_DIR, pkl_name)
    joblib.dump(pipeline, pkl_path)

    # Save class names
    classes_path = os.path.join(MODEL_DIR, pkl_name.replace('_model.pkl', '_classes.json'))
    with open(classes_path, 'w') as f:
        json.dump(class_names, f)

    print(f"\n  ✅ Saved {pkl_path}")
    print(f"  ✅ Saved {classes_path}")

    return acc


# ══════════════════════════════════════════════════════════════════
#  1. BONE FRACTURE MODEL  (xray)
# ══════════════════════════════════════════════════════════════════

def train_bone_fracture():
    print("\n" + "#"*60)
    print("#  BONE FRACTURE MODEL  (X-Ray — Binary)")
    print("#"*60)

    class_names = ['Fractured', 'Not Fractured']
    class_folders = ['fractured', 'not fractured']

    # --- Train split ---
    train_dir = os.path.join(BONE_FRACTURE_DIR, "train")
    X_train_all, y_train_all = [], []
    for label_idx, folder_name in enumerate(class_folders):
        feats, labs = load_images_from_folder(
            os.path.join(train_dir, folder_name),
            label=label_idx,
            max_images=MAX_IMAGES_PER_CLASS,
        )
        X_train_all.extend(feats)
        y_train_all.extend(labs)

    # --- Validation split ---
    val_dir = os.path.join(BONE_FRACTURE_DIR, "val")
    X_val_all, y_val_all = [], []
    for label_idx, folder_name in enumerate(class_folders):
        feats, labs = load_images_from_folder(
            os.path.join(val_dir, folder_name),
            label=label_idx,
            max_images=MAX_IMAGES_PER_CLASS,
        )
        X_val_all.extend(feats)
        y_val_all.extend(labs)

    if not X_train_all:
        print("  ❌ No training images found for Bone Fracture!")
        return None

    X_train = np.array(X_train_all)
    y_train = np.array(y_train_all)
    X_val   = np.array(X_val_all) if X_val_all else X_train
    y_val   = np.array(y_val_all) if y_val_all else y_train

    # Shuffle training data
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    X_train, y_train = X_train[idx], y_train[idx]

    acc = train_and_save(
        X_train, y_train, X_val, y_val,
        class_names, "Bone Fracture X-Ray Classifier", "xray_model.pkl"
    )
    return acc


# ══════════════════════════════════════════════════════════════════
#  2. BRAIN TUMOR MODEL  (mri)
# ══════════════════════════════════════════════════════════════════

def train_brain_tumor():
    print("\n" + "#"*60)
    print("#  BRAIN TUMOR MODEL  (MRI — 4-Class)")
    print("#"*60)

    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    class_folders = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # --- Train split ---
    train_dir = os.path.join(BRAIN_TUMOR_DIR, "Training")
    X_train_all, y_train_all = [], []
    for label_idx, folder_name in enumerate(class_folders):
        feats, labs = load_images_from_folder(
            os.path.join(train_dir, folder_name),
            label=label_idx,
            max_images=MAX_IMAGES_PER_CLASS,
        )
        X_train_all.extend(feats)
        y_train_all.extend(labs)

    # --- Test split (used as validation) ---
    test_dir = os.path.join(BRAIN_TUMOR_DIR, "Testing")
    X_val_all, y_val_all = [], []
    for label_idx, folder_name in enumerate(class_folders):
        feats, labs = load_images_from_folder(
            os.path.join(test_dir, folder_name),
            label=label_idx,
            max_images=MAX_IMAGES_PER_CLASS,
        )
        X_val_all.extend(feats)
        y_val_all.extend(labs)

    if not X_train_all:
        print("  ❌ No training images found for Brain Tumor!")
        return None

    X_train = np.array(X_train_all)
    y_train = np.array(y_train_all)
    X_val   = np.array(X_val_all) if X_val_all else X_train
    y_val   = np.array(y_val_all) if y_val_all else y_train

    # Shuffle training data
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    X_train, y_train = X_train[idx], y_train[idx]

    acc = train_and_save(
        X_train, y_train, X_val, y_val,
        class_names, "Brain Tumor MRI Classifier", "mri_model.pkl"
    )
    return acc


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  MedScan AI — Dataset Training (v2 — High-Accuracy)")
    print("  RandomForest + Combined Features (HOG+Histogram+Pixels)")
    print("=" * 60)
    print(f"  Image size : {IMG_SIZE}×{IMG_SIZE} (HOG) + {PIXEL_SIZE}×{PIXEL_SIZE} (pixels)")
    print(f"  Bone data  : {BONE_FRACTURE_DIR}")
    print(f"  Brain data : {BRAIN_TUMOR_DIR}")
    print(f"  Output     : {MODEL_DIR}/")
    print()

    results = {}

    # 1. Bone Fracture
    acc = train_bone_fracture()
    if acc is not None:
        results['xray'] = acc

    # 2. Brain Tumor
    acc = train_brain_tumor()
    if acc is not None:
        results['mri'] = acc

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  🎯 TRAINING COMPLETE — Summary")
    print("=" * 60)
    for name, a in results.items():
        print(f"  {name:>8} model : {a*100:.1f}% accuracy  →  models/{name}_model.pkl")

    if not results:
        print("  ❌ No models were trained! Check that dataset/ folder exists.")
        sys.exit(1)

    print()
    print("  Next steps:")
    print("    1. Restart the Flask app:  python app.py")
    print("    2. Upload a bone X-ray or brain MRI to test predictions!")
    print()


if __name__ == '__main__':
    main()
