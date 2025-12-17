"""
Fix Dataset and Retrain YOLO Model
This script:
1. Clears old train/val splits
2. Properly matches images with their labels
3. Splits the data correctly
4. Retrains the model
"""

import os
import shutil
import random
from pathlib import Path

# Paths
DATASET_DIR = "dataset"
RAW_DIR = os.path.join(DATASET_DIR, "raw")
ROOT_LABELS_DIR = os.path.join(DATASET_DIR, "labels")  # Your original labels here

# YOLO training structure
TRAIN_IMAGES = os.path.join(DATASET_DIR, "images", "train")
VAL_IMAGES = os.path.join(DATASET_DIR, "images", "val")
TRAIN_LABELS = os.path.join(DATASET_DIR, "labels", "train")
VAL_LABELS = os.path.join(DATASET_DIR, "labels", "val")


def analyze_dataset():
    """Analyze current dataset state"""
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    # Count raw images
    raw_images = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n📁 Raw images: {len(raw_images)}")
    
    # Count root labels (excluding subdirectories)
    root_labels = [f for f in os.listdir(ROOT_LABELS_DIR) 
                   if f.endswith('.txt') and os.path.isfile(os.path.join(ROOT_LABELS_DIR, f))]
    print(f"📁 Root labels: {len(root_labels)}")
    
    # Find matching pairs
    matched = []
    unmatched_images = []
    unmatched_labels = []
    
    for img in raw_images:
        base_name = os.path.splitext(img)[0]
        label_file = base_name + '.txt'
        label_path = os.path.join(ROOT_LABELS_DIR, label_file)
        
        if os.path.exists(label_path):
            # Check if label is not empty
            if os.path.getsize(label_path) > 0:
                matched.append(img)
            else:
                print(f"  ⚠️ Empty label: {label_file}")
                unmatched_images.append(img)
        else:
            unmatched_images.append(img)
    
    print(f"\n✅ Matched pairs (image + non-empty label): {len(matched)}")
    print(f"❌ Images without labels: {len(unmatched_images)}")
    
    return matched, unmatched_images


def clean_train_val_dirs():
    """Clear old train/val directories"""
    print("\n🧹 Cleaning old train/val directories...")
    
    for d in [TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    print("   Done!")


def split_dataset_properly(matched_images, val_ratio=0.2):
    """Split matched images into train/val"""
    print(f"\n📊 Splitting {len(matched_images)} images into train/val...")
    
    # Shuffle for randomness
    random.shuffle(matched_images)
    
    val_count = max(1, int(len(matched_images) * val_ratio))
    val_images = matched_images[:val_count]
    train_images = matched_images[val_count:]
    
    print(f"   Training set: {len(train_images)} images")
    print(f"   Validation set: {len(val_images)} images")
    
    # Copy training data
    print("\n📋 Copying training data...")
    for img in train_images:
        base_name = os.path.splitext(img)[0]
        
        # Copy image
        src_img = os.path.join(RAW_DIR, img)
        dst_img = os.path.join(TRAIN_IMAGES, img)
        shutil.copy(src_img, dst_img)
        
        # Copy label
        src_label = os.path.join(ROOT_LABELS_DIR, base_name + '.txt')
        dst_label = os.path.join(TRAIN_LABELS, base_name + '.txt')
        shutil.copy(src_label, dst_label)
    
    # Copy validation data
    print("📋 Copying validation data...")
    for img in val_images:
        base_name = os.path.splitext(img)[0]
        
        # Copy image
        src_img = os.path.join(RAW_DIR, img)
        dst_img = os.path.join(VAL_IMAGES, img)
        shutil.copy(src_img, dst_img)
        
        # Copy label
        src_label = os.path.join(ROOT_LABELS_DIR, base_name + '.txt')
        dst_label = os.path.join(VAL_LABELS, base_name + '.txt')
        shutil.copy(src_label, dst_label)
    
    return len(train_images), len(val_images)


def verify_setup():
    """Verify the dataset is set up correctly"""
    print("\n🔍 Verifying setup...")
    
    train_imgs = len([f for f in os.listdir(TRAIN_IMAGES) if f.endswith(('.jpg', '.png'))])
    train_lbls = len([f for f in os.listdir(TRAIN_LABELS) if f.endswith('.txt')])
    val_imgs = len([f for f in os.listdir(VAL_IMAGES) if f.endswith(('.jpg', '.png'))])
    val_lbls = len([f for f in os.listdir(VAL_LABELS) if f.endswith('.txt')])
    
    print(f"   Train: {train_imgs} images, {train_lbls} labels")
    print(f"   Val: {val_imgs} images, {val_lbls} labels")
    
    if train_imgs == train_lbls and val_imgs == val_lbls:
        print("   ✅ All images have matching labels!")
        return True
    else:
        print("   ❌ Mismatch detected!")
        return False


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
    print(f"\n📄 Created config: {yaml_path}")
    return yaml_path


def train_model(epochs=100, img_size=320):
    """Train YOLOv8 on toy car data"""
    from ultralytics import YOLO
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Load a pretrained YOLOv8 nano model
    model = YOLO('yolov8n.pt')
    
    yaml_path = os.path.join(DATASET_DIR, "toycar.yaml")
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=8,
        name='toycar_fixed',
        patience=30,  # More patience for larger dataset
        save=True,
        plots=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,  # Initial learning rate
        lrf=0.01,   # Final learning rate factor
        augment=True,  # Data augmentation
    )
    
    # Copy best model to root
    best_model = Path("runs/detect/toycar_fixed/weights/best.pt")
    if best_model.exists():
        shutil.copy(best_model, "toycar_best.pt")
        print(f"\n✅ Best model copied to: toycar_best.pt")
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("FIX DATASET AND RETRAIN YOLO MODEL")
    print("="*60)
    
    # Step 1: Analyze
    matched, unmatched = analyze_dataset()
    
    if len(matched) < 10:
        print("\n❌ ERROR: Not enough labeled images!")
        print("   You need at least 10 properly labeled images.")
        print(f"   Currently found: {len(matched)} matched pairs")
        print("\n   Please check that your label files in dataset/labels/")
        print("   have the same base name as your images in dataset/raw/")
        exit(1)
    
    # Step 2: Ask user to continue
    print(f"\n📊 Found {len(matched)} properly labeled images")
    response = input("Continue with training? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted.")
        exit(0)
    
    # Step 3: Clean and split
    clean_train_val_dirs()
    split_dataset_properly(matched)
    
    # Step 4: Verify
    if not verify_setup():
        print("Setup verification failed!")
        exit(1)
    
    # Step 5: Create YAML
    create_yaml_config()
    
    # Step 6: Train
    print("\n" + "="*60)
    print("Starting training with fixed dataset...")
    print("This may take 10-30 minutes depending on your GPU")
    print("="*60)
    
    train_model(epochs=100, img_size=320)  # More epochs for larger dataset
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("Best model saved to: toycar_best.pt")
    print("="*60)
