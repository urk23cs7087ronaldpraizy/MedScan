"""
Medical Image Dataset Downloader
Downloads sample medical datasets for training
"""

import os
import urllib.request
import zipfile
import shutil

# Configuration
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')

# Dataset URLs (public medical imaging datasets)
DATASETS = {
    'xray': {
        'name': 'Chest X-Ray Pneumonia',
        'description': 'Chest X-ray images for Pneumonia detection',
        'classes': ['Normal', 'Pneumonia'],
        # Note: These are placeholder URLs. In production, you would need to 
        # download from Kaggle or other sources
        'source': 'Kaggle: Chest X-Ray Images (Pneumonia)'
    },
    'mri': {
        'name': 'Brain MRI',
        'description': 'Brain MRI images for tumor detection',
        'classes': ['Normal', 'Brain Tumor'],
        'source': 'Kaggle: Brain Tumor Classification'
    },
    'ct': {
        'name': 'CT Scan',
        'description': 'CT scan images for lung tumor and fracture detection',
        'classes': ['Normal', 'Lung Tumor', 'Bone Fracture'],
        'source': 'Kaggle: COVID-19 CT Scan Dataset'
    }
}


def create_folder_structure():
    """Create the folder structure for datasets"""
    print("Creating folder structure...")
    
    for modality in DATASETS.keys():
        for split in ['train', 'val']:
            for class_name in DATASETS[modality]['classes']:
                path = os.path.join(DATA_DIR, split, modality, class_name)
                os.makedirs(path, exist_ok=True)
    
    print("Folder structure created!")


def create_sample_images():
    """
    Create sample placeholder images for demonstration
    In production, replace with actual medical images
    """
    import numpy as np
    from PIL import Image
    
    print("\nCreating sample placeholder images...")
    print("Note: These are synthetic images for structure demonstration.")
    print("Replace with real medical images for actual training.\n")
    
    for modality, info in DATASETS.items():
        print(f"Processing {modality}...")
        
        for class_name in info['classes']:
            # Create sample image (grayscale medical-like)
            for split in ['train', 'val']:
                # Number of sample images to create
                num_samples = 5 if split == 'train' else 2
                
                for i in range(num_samples):
                    # Create a simple grayscale image with some variation
                    img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
                    
                    # Add some structure to make it look more like medical image
                    # (This is just for demonstration)
                    center_y, center_x = 112, 112
                    for y in range(224):
                        for x in range(224):
                            dist = ((x - center_x)**2 + (y - center_y)**2)**0.5
                            if dist < 80:
                                img_array[y, x] = np.clip(img_array[y, x] + 30, 0, 255)
                    
                    # Save image
                    img = Image.fromarray(img_array)
                    filename = f"{class_name}_{i}.jpg"
                    filepath = os.path.join(DATA_DIR, split, modality, class_name, filename)
                    img.save(filepath)
        
        print(f"  Created sample images for {modality}")


def print_download_instructions():
    """Print instructions for downloading real datasets"""
    print("\n" + "="*60)
    print("INSTRUCTIONS FOR GETTING REAL MEDICAL DATASETS")
    print("="*60)
    
    print("""
To train accurate models, you need real medical image datasets.

OPTION 1: Download from Kaggle
--------------------------------
1. Go to https://www.kaggle.com/
2. Create an account (free)
3. Download these datasets:

   For X-Ray (Pneumonia/COVID-19/TB):
   - Search: "chest xray pneumonia"
   - Search: "covid-19 xray"
   - Search: "tuberculosis xray"
   
   For MRI (Brain Tumor/Alzheimer's):
   - Search: "brain tumor classification"
   - Search: "alzheimer mri"
   
   For CT Scan:
   - Search: "ct scan lung"
   - Search: "bone fracture ct"

4. Extract the files and place in:
   - Training images: data/train/<modality>/<class_name>/
   - Validation images: data/val/<modality>/<class_name>/

OPTION 2: Use Public Medical Image Repositories
------------------------------------------------
- NIH Clinical Center: https://nihcc.org/
- Kaggle Datasets: https://www.kaggle.com/datasets
- Open Medical Images: https://openmedicalimages.com/
- Cancer Imaging Archive: https://www.cancerimagingarchive.net/

OPTION 3: Use Sample Datasets Provided
---------------------------------------
Run: python train_model.py --modality xray --demo
This creates basic models (may not be accurate)
""")
    
    print("="*60)


def verify_data():
    """Verify that data exists"""
    print("\nVerifying data structure...")
    
    for modality in DATASETS.keys():
        print(f"\n{modality.upper()}:")
        for split in ['train', 'val']:
            split_path = os.path.join(DATA_DIR, split, modality)
            if not os.path.exists(split_path):
                print(f"  {split}: Missing")
                continue
            
            classes = os.listdir(split_path)
            print(f"  {split}: {len(classes)} classes")
            for cls in classes:
                cls_path = os.path.join(split_path, cls)
                count = len(os.listdir(cls_path)) if os.path.exists(cls_path) else 0
                print(f"    - {cls}: {count} images")


def main():
    """Main function"""
    print("="*60)
    print("Medical Image Dataset Setup")
    print("="*60)
    
    # Create folder structure
    create_folder_structure()
    
    # Create sample images for demonstration
    create_sample_images()
    
    # Verify data
    verify_data()
    
    # Print instructions
    print_download_instructions()
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Download real medical datasets (recommended)")
    print("2. Run: python train_model.py --modality xray")
    print("3. Run: python app.py")


if __name__ == '__main__':
    main()

