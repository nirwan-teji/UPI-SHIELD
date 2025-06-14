import os
import shutil
from pathlib import Path
import random

def convert_qr_to_yolo(src_dir, target_dir, max_samples_per_class=5000, train_ratio=0.8):
    src_dir = Path(src_dir)
    target_dir = Path(target_dir)
    
    # Create YOLO directories
    for split in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        (target_dir / split).mkdir(parents=True, exist_ok=True)

    # Gather images with class labels (limited to max_samples_per_class)
    image_label_pairs = []
    for class_name, class_id in [('benign', 0), ('malicious', 1)]:
        class_folder = src_dir / class_name
        class_images = list(class_folder.glob("*.*"))
        
        # Shuffle and limit to max_samples_per_class
        random.shuffle(class_images)
        class_images = class_images[:max_samples_per_class]
        
        # Add to pairs
        for img_file in class_images:
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            image_label_pairs.append((img_file, class_id))

    # Split into train/val
    random.shuffle(image_label_pairs)
    split_idx = int(len(image_label_pairs) * train_ratio)
    train_pairs = image_label_pairs[:split_idx]
    val_pairs = image_label_pairs[split_idx:]

    # Copy files
    for split, pairs in [('train', train_pairs), ('val', val_pairs)]:
        for img_file, class_id in pairs:
            # Copy image
            dest_img = target_dir / f"images/{split}/{img_file.name}"
            shutil.copy2(img_file, dest_img)
            
            # Create label file
            label_file = target_dir / f"labels/{split}/{img_file.stem}.txt"
            with open(label_file, "w") as f:
                f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")

if __name__ == "__main__":
    src_dir = "raw_qr_dataset"
    target_dir = "ml_models/data/datasets/qr_yolo_subset"  # New subset directory
    convert_qr_to_yolo(src_dir, target_dir, max_samples_per_class=5000)
    print("✅ Subset dataset created with 5k samples per class!")
