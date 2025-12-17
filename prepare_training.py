"""
Prepare Training Data for Custom Toy Car YOLO Model
Step 1: Collect raw images (you already have them in logs/)
Step 2: Create YOLO format labels
Step 3: Split into train/val
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path

# Paths
MODEL_DIR = "model"  # Your unlabeled images
DATASET_DIR = "dataset"
RAW_IMAGES_DIR = os.path.join(DATASET_DIR, "raw")  # Copy images here first

# Create directories
os.makedirs(RAW_IMAGES_DIR, exist_ok=True)


def copy_images_for_labeling():
    """Copy images from model/ to dataset/raw for labeling"""
    count = 0
    for filename in os.listdir(MODEL_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            src = os.path.join(MODEL_DIR, filename)
            dst = os.path.join(RAW_IMAGES_DIR, filename)
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                count += 1
    print(f"Copied {count} new images to {RAW_IMAGES_DIR}")
    print(f"Total images ready for labeling: {len(os.listdir(RAW_IMAGES_DIR))}")


def create_labeling_tool():
    """
    Simple tool to label toy cars in images
    Controls:
        - Click + drag: Draw bounding box
        - S: Save and next image
        - R: Reset current image
        - Q: Quit
    """
    images = sorted([f for f in os.listdir(RAW_IMAGES_DIR) if f.endswith('.jpg')])
    
    # Create labels directory
    labels_dir = os.path.join(DATASET_DIR, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    current_idx = 0
    boxes = []
    drawing = False
    start_point = None
    current_img = None
    display_img = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, boxes, display_img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            display_img = current_img.copy()
            # Draw existing boxes
            for box in boxes:
                cv2.rectangle(display_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # Draw current box
            cv2.rectangle(display_img, start_point, (x, y), (255, 0, 0), 2)
            
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = start_point
            x2, y2 = x, y
            # Normalize to ensure x1 < x2, y1 < y2
            boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
            display_img = current_img.copy()
            for box in boxes:
                cv2.rectangle(display_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    
    cv2.namedWindow("Label Toy Cars")
    cv2.setMouseCallback("Label Toy Cars", mouse_callback)
    
    while current_idx < len(images):
        img_path = os.path.join(RAW_IMAGES_DIR, images[current_idx])
        current_img = cv2.imread(img_path)
        display_img = current_img.copy()
        h, w = current_img.shape[:2]
        
        boxes = []
        
        # Check if label already exists
        label_path = os.path.join(labels_dir, images[current_idx].replace('.jpg', '.txt'))
        if os.path.exists(label_path):
            # Load existing labels
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, bw, bh = map(float, parts)
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        boxes.append([x1, y1, x2, y2])
            for box in boxes:
                cv2.rectangle(display_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        while True:
            # Show image info
            info_img = display_img.copy()
            cv2.putText(info_img, f"Image {current_idx+1}/{len(images)}: {images[current_idx]}", 
                       (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(info_img, f"Cars: {len(boxes)} | S=Save | R=Reset | Q=Quit", 
                       (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Label Toy Cars", info_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                # Save labels in YOLO format
                with open(label_path, 'w') as f:
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        # Convert to YOLO format (class cx cy w h) - normalized
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                print(f"Saved {len(boxes)} labels for {images[current_idx]}")
                current_idx += 1
                break
                
            elif key == ord('r'):
                boxes = []
                display_img = current_img.copy()
                
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return
    
    cv2.destroyAllWindows()
    print("\nLabeling complete!")
    print(f"Labels saved to: {labels_dir}")


if __name__ == "__main__":
    print("="*50)
    print("TOY CAR TRAINING DATA PREPARATION")
    print("="*50)
    
    print("\n1. Copying images from logs...")
    copy_images_for_labeling()
    
    print("\n2. Starting labeling tool...")
    print("   Draw boxes around each toy car")
    print("   S = Save & Next | R = Reset | Q = Quit")
    print("-"*50)
    
    create_labeling_tool()
