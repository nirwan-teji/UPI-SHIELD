import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

def create_subset_dataset():
    """Create subset dataset with 5k samples per class"""
    # Absolute paths for reliability
    raw_dir = Path("raw_qr_dataset").resolve()
    yolo_subset_dir = Path("ml_models/data/datasets/qr_yolo_subset").resolve()
    
    # Clean existing data
    if yolo_subset_dir.exists():
        shutil.rmtree(yolo_subset_dir)
    
    # Create directory structure
    (yolo_subset_dir / "images/train").mkdir(parents=True, exist_ok=True)
    (yolo_subset_dir / "images/val").mkdir(parents=True, exist_ok=True)
    (yolo_subset_dir / "labels/train").mkdir(parents=True, exist_ok=True)
    (yolo_subset_dir / "labels/val").mkdir(parents=True, exist_ok=True)

    # Process each class
    for class_name, class_id in [('benign', 0), ('malicious', 1)]:
        class_path = raw_dir / class_name
        if not class_path.exists():
            raise FileNotFoundError(f"Class directory not found: {class_path}")

        # Get all images and sample 5k
        all_images = [f for f in class_path.glob("*.*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        if not all_images:
            raise FileNotFoundError(f"No images found in {class_path}")
            
        sampled_images = random.sample(all_images, min(5000, len(all_images)))

        # Ensure minimum 20% validation samples
        min_val = max(1, int(len(sampled_images) * 0.2))
        split_idx = len(sampled_images) - min_val
        
        for i, img_path in enumerate(sampled_images):
            split = "train" if i < split_idx else "val"
            
            # Copy image
            dest_img = yolo_subset_dir / f"images/{split}/{img_path.name}"
            shutil.copy(img_path, dest_img)
            
            # Create label file
            label_path = yolo_subset_dir / f"labels/{split}/{img_path.stem}.txt"
            label_path.write_text(f"{class_id} 0.5 0.5 0.8 0.8\n")

    # Create dataset.yaml with absolute paths
    yaml_content = f"""path: {yolo_subset_dir}
train: images/train
val: images/val

names:
  0: benign_qr
  1: malicious_qr
"""
    (yolo_subset_dir / "dataset.yaml").write_text(yaml_content)
    print(f"✅ Created subset dataset at {yolo_subset_dir}")

def validate_dataset(dataset_dir):
    """Ensure dataset has required files"""
    dataset_dir = Path(dataset_dir)
    required = [
        dataset_dir / "images/train",
        dataset_dir / "images/val",
        dataset_dir / "labels/train",
        dataset_dir / "labels/val",
        dataset_dir / "dataset.yaml"
    ]
    
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required dataset component: {path}")
            
    # Check images exist
    train_images = list((dataset_dir / "images/train").glob("*"))
    val_images = list((dataset_dir / "images/val").glob("*"))
    
    if not train_images:
        raise FileNotFoundError("No training images found")
    if not val_images:
        raise FileNotFoundError("No validation images found")
        
    print(f"Dataset validated: {len(train_images)} train, {len(val_images)} val images")

def train_yolo_model():
    """Train YOLO model on subset dataset"""
    # Path configuration
    dataset_dir = Path("ml_models/data/datasets/qr_yolo_subset").resolve()
    model_save_dir = Path("ml_models/trained_models/yolo_qr_subset").resolve()
    
    # Validate dataset first
    validate_dataset(dataset_dir)
    
    # Initialize and train model
    model = YOLO('yolov8n.pt')
    results = model.train(
        data=str(dataset_dir / "dataset.yaml"),
        epochs=1,
        imgsz=640,
        batch=16,
        project=model_save_dir,
        name='qr_subset',
        patience=20,
        save=True,
        verbose=True,
        exist_ok=True  # Allow overwriting existing runs
    )

    # Verify and save best model
    best_model = model_save_dir / "qr_subset/weights/best.pt"
    if best_model.exists():
        final_path = model_save_dir / "best.pt"
        shutil.copy(best_model, final_path)
        print(f"✅ Best model saved to: {final_path}")
        return str(final_path)
    else:
        raise FileNotFoundError("Training failed - no best model found")

if __name__ == "__main__":
    try:
        # Step 1: Create subset dataset
        print("🔄 Creating subset dataset...")
        create_subset_dataset()
        
        # Step 2: Train model
        print("\n🚀 Starting training on subset data...")
        model_path = train_yolo_model()
        print(f"\n🎉 Training completed! Model available at: {model_path}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("💡 Troubleshooting tips:")
        print("- Verify raw_qr_dataset contains benign/ and malicious/ folders with images")
        print("- Check directory permissions")
        print("- Ensure at least 100 images per class for meaningful training")
        exit(1)
