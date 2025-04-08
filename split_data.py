import os
import splitfolders
from torchvision import transforms
from PIL import Image
import shutil

# First, verify the dataset path exists
input_dir = "Transformer_dataset"  # Update this if your folder has a different name
if not os.path.exists(input_dir):
    print(f"Error: {input_dir} directory not found. Please check the path.")
    # You might want to list available directories to help troubleshoot
    print("Available directories:", [d for d in os.listdir() if os.path.isdir(d)])
else:
    # Create output directory if it doesn't exist
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the dataset into train, validation, and test sets
    splitfolders.ratio(input_dir, output=output_dir, 
                       seed=42, ratio=(0.8, 0.1, 0.1), 
                       group_prefix=None)
    
    print(f"Dataset split into train/val/test in {output_dir}")
    
    # Define augmentation transforms
    augmentation_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1)
    ])
    
    # Apply augmentations to the training set
    train_dir = os.path.join(output_dir, "train")
    augmented_dir = os.path.join(output_dir, "train_augmented")
    
    if os.path.exists(train_dir):
        # Create augmented directory
        os.makedirs(augmented_dir, exist_ok=True)
        
        # Copy original training data to augmented directory
        for class_dir in os.listdir(train_dir):
            src_class_dir = os.path.join(train_dir, class_dir)
            dst_class_dir = os.path.join(augmented_dir, class_dir)
            
            if os.path.isdir(src_class_dir):
                os.makedirs(dst_class_dir, exist_ok=True)
                
                # Copy original images
                for img_file in os.listdir(src_class_dir):
                    if img_file.endswith(('jpg', 'png', 'jpeg')):
                        src_path = os.path.join(src_class_dir, img_file)
                        dst_path = os.path.join(dst_class_dir, img_file)
                        shutil.copy2(src_path, dst_path)
                
                # Generate augmented images
                for img_file in os.listdir(src_class_dir):
                    if img_file.endswith(('jpg', 'png', 'jpeg')):
                        try:
                            img_path = os.path.join(src_class_dir, img_file)
                            img = Image.open(img_path).convert('RGB')
                            
                            # Create 3 augmented versions of each image
                            for i in range(3):
                                aug_img = augmentation_transforms(img)
                                aug_path = os.path.join(dst_class_dir, f"aug_{i}_{img_file}")
                                aug_img.save(aug_path)
                        except Exception as e:
                            print(f"Error augmenting {img_file}: {e}")
        
        print(f"Augmented training data saved to {augmented_dir}")
    else:
        print(f"Error: Training directory {train_dir} not found after splitting")
