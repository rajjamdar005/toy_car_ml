"""
Train Custom YOLOv8 Model on Toy Car Data
Run this after labeling your images with prepare_training.py
"""

import os
import shutil
import random
from pathlib import Path

# Paths
DATASET_DIR = "dataset"
RAW_DIR = os.path.join(DATASET_DIR, "raw")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

# YOLO training structure
TRAIN_IMAGES = os.path.join(DATASET_DIR, "images", "train")
VAL_IMAGES = os.path.join(DATASET_DIR, "images", "val")
TRAIN_LABELS = os.path.join(DATASET_DIR, "labels", "train")
VAL_LABELS = os.path.join(DATASET_DIR, "labels", "val")


def setup_yolo_structure():
    """Create YOLO training directory structure"""
    for d in [TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS]:
        os.makedirs(d, exist_ok=True)


def split_dataset(val_ratio=0.2):
    """Split data into train/val sets"""
    # Get all labeled images
    labeled_images = []
    for img in os.listdir(RAW_DIR):
        if img.endswith('.jpg'):
            label_file = img.replace('.jpg', '.txt')
            label_path = os.path.join(LABELS_DIR, label_file)
            if os.path.exists(label_path):
                labeled_images.append(img)
    
    if len(labeled_images) == 0:
        print("ERROR: No labeled images found!")
        print("Run prepare_training.py first to label your images.")
        return False
    
    # Shuffle and split
    random.shuffle(labeled_images)
    val_count = int(len(labeled_images) * val_ratio)
    val_images = labeled_images[:val_count]
    train_images = labeled_images[val_count:]
    
    print(f"Total labeled images: {len(labeled_images)}")
    print(f"Training: {len(train_images)}")
    print(f"Validation: {len(val_images)}")
    
    # Copy to YOLO structure
    for img in train_images:
        shutil.copy(os.path.join(RAW_DIR, img), os.path.join(TRAIN_IMAGES, img))
        shutil.copy(os.path.join(LABELS_DIR, img.replace('.jpg', '.txt')), 
                   os.path.join(TRAIN_LABELS, img.replace('.jpg', '.txt')))
    
    for img in val_images:
        shutil.copy(os.path.join(RAW_DIR, img), os.path.join(VAL_IMAGES, img))
        shutil.copy(os.path.join(LABELS_DIR, img.replace('.jpg', '.txt')), 
                   os.path.join(VAL_LABELS, img.replace('.jpg', '.txt')))
    
    return True


def create_yaml_config():
    """Create YOLO dataset config file"""
    yaml_content = f"""# Toy Car Dataset for YOLO Training
path: {os.path.abspath(DATASET_DIR)}
train: images/train
val: images/val

# Classes
names:
  0: car
"""
    yaml_path = os.path.join(DATASET_DIR, "toycar.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Created config: {yaml_path}")
    return yaml_path


def train_model(epochs=100, img_size=320):
    """Train YOLOv8 on toy car data"""
    from ultralytics import YOLO
    
    # Load a pretrained YOLOv8 nano model (fastest)
    model = YOLO('yolov8n.pt')
    
    # Train on toy car data
    yaml_path = os.path.join(DATASET_DIR, "toycar.yaml")
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=8,
        name='toycar_model',
        patience=20,  # Early stopping
        save=True,
        plots=True
    )
    
    return results


if __name__ == "__main__":
    print("="*50)
    print("TRAIN CUSTOM TOY CAR YOLO MODEL")
    print("="*50)
    
    # Step 1: Setup directory structure
    print("\n1. Setting up YOLO directory structure...")
    setup_yolo_structure()
    
    # Step 2: Split dataset
    print("\n2. Splitting dataset...")
    if not split_dataset():
        exit(1)
    
    # Step 3: Create YAML config
    print("\n3. Creating YAML config...")
    create_yaml_config()
    
    # Step 4: Train
    print("\n4. Starting training...")
    print("   This may take 10-30 minutes depending on your GPU")
    print("-"*50)
    
    train_model(epochs=50, img_size=320)  # 50 epochs for small dataset
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("Best model saved to: runs/detect/toycar_model/weights/best.pt")
    print("="*50)
