import torch
from ultralytics import YOLO

# Configuration parameters
DATA_YAML = "data.yaml"
FREEZE_LAYERS = 15                   # Increased frozen layers
EPOCHS = 30                          # Reduced epochs
BATCH_SIZE = 4
IMG_SIZE = 640

def main():
    # Load the pretrained model
    model = YOLO('yolo11n.pt')
    print(f"Pretrained YOLOv11 model loaded")

    # Start training with adjusted parameters
    print("Starting training...")
    results = model.train(
        # Basic configuration
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        freeze=FREEZE_LAYERS,
        
        # Optimization parameters - adjusted to prevent overfitting
        lr0=0.001,              # Reduced learning rate
        lrf=0.001,              # Reduced final learning rate
        momentum=0.937,
        weight_decay=0.001,     # Increased weight decay
        warmup_epochs=3.0,
        
        # Loss function weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Enhanced data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        degrees=10.0,           # Added rotation
        translate=0.1,          # Added translation
        scale=0.5,              # Added scaling
        shear=2.0,              # Added shearing
        perspective=0.0001,     # Added perspective
        flipud=0.1,             # Added vertical flip
        
        # Training management - adjusted for early stopping
        patience=5,             # Reduced patience
        save_period=5,          # Save more frequently
        
        # Project organization
        project="yolov11_custom",
        name="run1",
        exist_ok=True,
        
        # Hardware utilization
        device=0,
        workers=8,
        amp=True,
        
        # Validation and visualization
        val=True,               # Validation is enabled by default
        plots=True              # Generate training plots
    )
    print("Training completed.")
    
    # Save the fine-tuned model
    model.save("yolov11_finetuned.pt")
    print("Fine-tuned model saved to yolov11_finetuned.pt.")

if __name__ == "__main__":
    main()
