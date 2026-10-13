"""
MedScan AI — Model Training Script
Trains medical image classifiers using the provided dataset.

Dataset structure (archive (7)/):
  - AbdomenCT/   (10,000 images) → CT modality
  - BreastMRI/   (8,954 images)  → MRI modality
  - CXR/         (10,000 images) → X-Ray modality (Chest X-Ray)
  - ChestCT/     (10,000 images) → CT modality
  - Hand/        (10,000 images) → X-Ray modality (Hand X-Ray)
  - HeadCT/      (10,000 images) → CT modality

Training plan:
  1. xray_model: CXR vs Hand (2 classes)
  2. ct_model: AbdomenCT vs ChestCT vs HeadCT (3 classes)
  3. mri_model: BreastMRI detection (binary — uses subset of other modalities as negatives)

Uses HOG (Histogram of Oriented Gradients) features + SGDClassifier
for fast CPU training on 64x64 grayscale images.
"""

import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import joblib

# Ensure reproducibility
random.seed(42)
np.random.seed(42)

DATASET_DIR = "archive (7)"
MODEL_DIR = "models"
IMG_SIZE = 64

os.makedirs(MODEL_DIR, exist_ok=True)


def extract_hog_features(image_array):
    """Extract HOG features from a 64x64 grayscale image."""
    # HOG with small cells for 64x64
    win_size = (64, 64)
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    features = hog.compute(image_array)
    return features.flatten()


def load_images_from_folder(folder_path, max_images=None, label=None):
    """Load images from a folder and extract features."""
    images = []
    labels = []
    raw_pixels = []

    files = sorted(os.listdir(folder_path))
    if max_images:
        files = files[:max_images]

    for fname in files:
        if not fname.lower().endswith(('.jpeg', '.jpg', '.png')):
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # HOG features
            features = extract_hog_features(img)
            images.append(features)
            raw_pixels.append(img.flatten().astype(np.float32) / 255.0)
            if label is not None:
                labels.append(label)
        except Exception as e:
            pass

    return images, labels, raw_pixels


def train_model(X_train, y_train, X_test, y_test, class_names, model_name):
    """Train an SGDClassifier and evaluate."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.pipeline import Pipeline

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"{'='*60}")

    # Pipeline: scale → classify
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SGDClassifier(
            loss='modified_huber',  # gives probability estimates
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])

    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTraining time: {train_time:.1f}s")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return pipeline, accuracy


def main():
    print("=" * 60)
    print("MedScan AI — Model Training")
    print("=" * 60)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output: {MODEL_DIR}/")
    print()

    # ================================================================
    # 1. X-RAY MODEL: CXR vs Hand
    # ================================================================
    print("\n📋 Loading X-Ray dataset (CXR + Hand)...")

    xray_classes = ['Chest X-Ray', 'Hand X-Ray']
    cxr_features, cxr_labels, _ = load_images_from_folder(
        os.path.join(DATASET_DIR, "CXR"), max_images=5000, label=0
    )
    hand_features, hand_labels, _ = load_images_from_folder(
        os.path.join(DATASET_DIR, "Hand"), max_images=5000, label=1
    )

    X_xray = np.array(cxr_features + hand_features)
    y_xray = np.array(cxr_labels + hand_labels)

    # Shuffle and split
    indices = np.arange(len(X_xray))
    np.random.shuffle(indices)
    X_xray, y_xray = X_xray[indices], y_xray[indices]

    split = int(0.8 * len(X_xray))
    X_train_xray, X_test_xray = X_xray[:split], X_xray[split:]
    y_train_xray, y_test_xray = y_xray[:split], y_xray[split:]

    xray_model, xray_acc = train_model(
        X_train_xray, y_train_xray, X_test_xray, y_test_xray,
        xray_classes, "X-Ray Classifier"
    )

    # Save
    joblib.dump(xray_model, os.path.join(MODEL_DIR, "xray_model.pkl"))
    with open(os.path.join(MODEL_DIR, "xray_classes.json"), 'w') as f:
        json.dump(xray_classes, f)
    print(f"✅ Saved xray_model.pkl ({xray_acc*100:.1f}% accuracy)")

    # ================================================================
    # 2. CT MODEL: AbdomenCT vs ChestCT vs HeadCT
    # ================================================================
    print("\n📋 Loading CT dataset (Abdomen + Chest + Head)...")

    ct_classes = ['Abdomen CT', 'Chest CT', 'Head CT']
    abd_features, abd_labels, _ = load_images_from_folder(
        os.path.join(DATASET_DIR, "AbdomenCT"), max_images=4000, label=0
    )
    chest_features, chest_labels, _ = load_images_from_folder(
        os.path.join(DATASET_DIR, "ChestCT"), max_images=4000, label=1
    )
    head_features, head_labels, _ = load_images_from_folder(
        os.path.join(DATASET_DIR, "HeadCT"), max_images=4000, label=2
    )

    X_ct = np.array(abd_features + chest_features + head_features)
    y_ct = np.array(abd_labels + chest_labels + head_labels)

    indices = np.arange(len(X_ct))
    np.random.shuffle(indices)
    X_ct, y_ct = X_ct[indices], y_ct[indices]

    split = int(0.8 * len(X_ct))
    X_train_ct, X_test_ct = X_ct[:split], X_ct[split:]
    y_train_ct, y_test_ct = y_ct[:split], y_ct[split:]

    ct_model, ct_acc = train_model(
        X_train_ct, y_train_ct, X_test_ct, y_test_ct,
        ct_classes, "CT Classifier"
    )

    joblib.dump(ct_model, os.path.join(MODEL_DIR, "ct_model.pkl"))
    with open(os.path.join(MODEL_DIR, "ct_classes.json"), 'w') as f:
        json.dump(ct_classes, f)
    print(f"✅ Saved ct_model.pkl ({ct_acc*100:.1f}% accuracy)")

    # ================================================================
    # 3. MRI MODEL: BreastMRI vs non-MRI (binary)
    # ================================================================
    print("\n📋 Loading MRI dataset (BreastMRI + negatives)...")

    mri_classes = ['Breast MRI', 'Non-MRI']
    mri_features, mri_labels, _ = load_images_from_folder(
        os.path.join(DATASET_DIR, "BreastMRI"), max_images=4000, label=0
    )
    # Use a mix of CXR + CT as negatives
    neg1_features, neg1_labels, _ = load_images_from_folder(
        os.path.join(DATASET_DIR, "CXR"), max_images=2000, label=1
    )
    neg2_features, neg2_labels, _ = load_images_from_folder(
        os.path.join(DATASET_DIR, "ChestCT"), max_images=2000, label=1
    )

    X_mri = np.array(mri_features + neg1_features + neg2_features)
    y_mri = np.array(mri_labels + neg1_labels + neg2_labels)

    indices = np.arange(len(X_mri))
    np.random.shuffle(indices)
    X_mri, y_mri = X_mri[indices], y_mri[indices]

    split = int(0.8 * len(X_mri))
    X_train_mri, X_test_mri = X_mri[:split], X_mri[split:]
    y_train_mri, y_test_mri = y_mri[:split], y_mri[split:]

    mri_model, mri_acc = train_model(
        X_train_mri, y_train_mri, X_test_mri, y_test_mri,
        mri_classes, "MRI Classifier"
    )

    joblib.dump(mri_model, os.path.join(MODEL_DIR, "mri_model.pkl"))
    with open(os.path.join(MODEL_DIR, "mri_classes.json"), 'w') as f:
        json.dump(mri_classes, f)
    print(f"✅ Saved mri_model.pkl ({mri_acc*100:.1f}% accuracy)")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("🎯 TRAINING COMPLETE — Summary")
    print("=" * 60)
    print(f"  X-Ray Model:  {xray_acc*100:.1f}% accuracy  →  models/xray_model.pkl")
    print(f"  CT Model:     {ct_acc*100:.1f}% accuracy  →  models/ct_model.pkl")
    print(f"  MRI Model:    {mri_acc*100:.1f}% accuracy  →  models/mri_model.pkl")
    print()
    print("Models are ready for integration!")
    print("Restart the Flask server to use the trained models.")


if __name__ == '__main__':
    main()
