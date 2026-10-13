"""
Medical Image Classification Model Training Script
Uses transfer learning with ResNet50 for disease classification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import json


# Configuration
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# Model directories
MODELS_DIR = 'models'
DATA_DIR = 'data'

# Disease classes for each modality - will be dynamically updated based on actual data
DISEASE_CLASSES = {
    'xray': ['Normal', 'Pneumonia'],  # Available in training data
    'mri': ['Normal', 'Brain Tumor'],  # Available in training data
    'ct': ['Normal', 'Lung Tumor', 'Bone Fracture']  # Available in training data
}


def create_model(num_classes, input_shape=(224, 224, 3)):
    """
    Create CNN model from scratch (no pre-trained weights needed)
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_dummy_dataset(modality, samples_per_class=20):
    """
    Create synthetic dataset for demonstration
    In production, replace with actual medical images
    """
    print(f"\n{'='*50}")
    print(f"Creating synthetic dataset for {modality}")
    print(f"{'='*50}")
    
    classes = DISEASE_CLASSES[modality]
    
    # Create data directory
    train_dir = os.path.join(DATA_DIR, 'train', modality)
    val_dir = os.path.join(DATA_DIR, 'val', modality)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    for split_dir in [train_dir, val_dir]:
        for class_name in classes:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
    
    print(f"Dataset structure created at: {DATA_DIR}/train/{modality}")
    print(f"Classes: {classes}")
    print(f"\nTo train with real data:")
    print(f"1. Place {modality} images in data/train/{modality}/<class_name>/")
    print(f"2. Use naming: Normal, Pneumonia, Tuberculosis, COVID-19 (for xray)")
    print(f"3. Run: python train_model.py --modality {modality}")
    
    return train_dir, val_dir


def train_model(modality='xray'):
    """
    Train the model for a specific modality
    """
    print(f"\n{'='*60}")
    print(f"Training model for {modality.upper()} modality")
    print(f"{'='*60}")
    
    classes = DISEASE_CLASSES[modality]
    num_classes = len(classes)
    
    # Set up data directories
    train_dir = os.path.join(DATA_DIR, 'train', modality)
    val_dir = os.path.join(DATA_DIR, 'val', modality)
    
    # Check if data exists
    if not os.path.exists(train_dir):
        print(f"\nNo training data found for {modality}")
        create_dummy_dataset(modality)
        print("\nPlease add real medical images to train the model")
        return False
    
    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Print class mapping
    print(f"\nClass mapping: {train_generator.class_indices}")
    
    # Create model
    model = create_model(num_classes)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, f'{modality}_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    # Train
    print(f"\nStarting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Save class mapping
    class_indices = train_generator.class_indices
    reverse_mapping = {v: k for k, v in class_indices.items()}
    
    with open(os.path.join(MODELS_DIR, f'{modality}_classes.json'), 'w') as f:
        json.dump(reverse_mapping, f)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {MODELS_DIR}/{modality}_model.h5")
    
    # Fine-tune by unfreezing some layers
    print(f"\nFine-tuning model...")
    model.trainable = True
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_fine = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(os.path.join(MODELS_DIR, f'{modality}_model.h5'))
    print(f"Final model saved!")
    
    return True


def train_all_modalities():
    """
    Train models for all modalities
    """
    for modality in DISEASE_CLASSES.keys():
        print(f"\n\n{'#'*60}")
        print(f"# Training {modality.upper()} Model")
        print(f"{'#'*60}")
        
        success = train_model(modality)
        
        if not success:
            print(f"\nSkipping {modality} - no training data")


def create_sample_model():
    """
    Create a simple working model for demonstration
    This creates a basic model without training data
    """
    print("\nCreating sample model for demonstration...")
    
    for modality, classes in DISEASE_CLASSES.items():
        print(f"\n{modality}: {classes}")
        
        # Create model
        model = create_model(len(classes))
        
        # Save model architecture
        model.save(os.path.join(MODELS_DIR, f'{modality}_model.h5'))
        
        # Save class mapping
        class_indices = {name: idx for idx, name in enumerate(classes)}
        with open(os.path.join(MODELS_DIR, f'{modality}_classes.json'), 'w') as f:
            json.dump(class_indices, f)
        
        print(f"  Model saved: {modality}_model.h5")
    
    print("\nSample models created!")
    print("Note: These are untrained models. Add training data for accurate predictions.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Medical Image Classification Model')
    parser.add_argument('--modality', type=str, default='all', 
                        choices=['xray', 'mri', 'ct', 'all'],
                        help='Modality to train (xray, mri, ct, or all)')
    parser.add_argument('--demo', action='store_true',
                        help='Create demo model without training')
    
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if args.demo:
        create_sample_model()
    elif args.modality == 'all':
        train_all_modalities()
    else:
        train_model(args.modality)

