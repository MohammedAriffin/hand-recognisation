import os
import splitfolders
import multiprocessing
from functools import partial
from torchvision import transforms
from PIL import Image
import shutil
from tqdm import tqdm
import time

def process_class_dir(class_dir, train_dir, augmented_dir, augmentation_transforms):
    """Process a single class directory with multiprocessing"""
    src_class_dir = os.path.join(train_dir, class_dir)
    dst_class_dir = os.path.join(augmented_dir, class_dir)
    
    if not os.path.isdir(src_class_dir):
        return
        
    os.makedirs(dst_class_dir, exist_ok=True)
    
    # Copy original images first
    img_files = [f for f in os.listdir(src_class_dir) if f.endswith(('jpg', 'png', 'jpeg'))]
    for img_file in img_files:
        src_path = os.path.join(src_class_dir, img_file)
        dst_path = os.path.join(dst_class_dir, img_file)
        shutil.copy(src_path, dst_path)  # Faster than copy2
    
    # Generate augmentations
    for img_file in img_files:
        try:
            img_path = os.path.join(src_class_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            
            # Create 3 augmented versions of each image
            for i in range(3):
                aug_img = augmentation_transforms(img)
                aug_path = os.path.join(dst_class_dir, f"aug_{i}_{img_file}")
                aug_img.save(aug_path, quality=90)  # Slightly reduce quality for speed
                
        except Exception as e:
            print(f"Error augmenting {img_file}: {e}")
    
    return len(img_files)

def main():
    # Detect dataset directory - try common names
    potential_dirs = ["asl_alphabet_train", "Transformer_dataset", "asl_dataset", "dataset_raw"]
    input_dir = None
    
    for dir_name in potential_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            input_dir = dir_name
            break
    
    # If none found, ask user
    if input_dir is None:
        print("Could not automatically find dataset directory.")
        available_dirs = [d for d in os.listdir() if os.path.isdir(d)]
        print("Available directories:", available_dirs)
        input_dir = input("Enter the name of your dataset directory: ")
    
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} directory not found.")
        return
    
    # Count and display dataset statistics
    class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    print(f"Found {len(class_dirs)} class directories")
    
    total_images = 0
    for class_dir in class_dirs:
        class_path = os.path.join(input_dir, class_dir)
        images = [f for f in os.listdir(class_path) if f.endswith(('jpg', 'png', 'jpeg'))]
        total_images += len(images)
        print(f"Class {class_dir}: {len(images)} images")
    
    print(f"Total images found: {total_images}")
    print(f"Splitting dataset with ratio 75:10:15 (train:val:test)")
    
    # Create output directory
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the dataset into train, validation, and test sets
    # Custom ratio: 75% train, 10% validation, 15% test
    start_time = time.time()
    print("Starting dataset split...")
    splitfolders.ratio(input_dir, output=output_dir,
                      seed=42, ratio=(0.75, 0.1, 0.15),
                      group_prefix=None)
    
    print(f"Split completed in {time.time() - start_time:.1f} seconds")
    
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
        print("Creating augmented training dataset...")
        start_time = time.time()
        
        # Create augmented directory
        os.makedirs(augmented_dir, exist_ok=True)
        
        # Get class directories for parallelization
        class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        
        # Setup multiprocessing
        num_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
        print(f"Processing images using {num_cores} CPU cores")
        
        # Process all classes in parallel
        with multiprocessing.Pool(num_cores) as pool:
            process_func = partial(
                process_class_dir,
                train_dir=train_dir,
                augmented_dir=augmented_dir,
                augmentation_transforms=augmentation_transforms
            )
            results = list(tqdm(
                pool.imap(process_func, class_dirs),
                total=len(class_dirs),
                desc="Augmenting classes"
            ))
        
        print(f"Augmentation completed in {time.time() - start_time:.1f} seconds")
        
        # Verify dataset counts
        print("\nVerifying dataset counts:")
        train_count = sum(len(os.listdir(os.path.join(train_dir, d))) 
                          for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
        
        val_dir = os.path.join(output_dir, "val")
        val_count = sum(len(os.listdir(os.path.join(val_dir, d))) 
                        for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d)))
        
        test_dir = os.path.join(output_dir, "test")
        test_count = sum(len(os.listdir(os.path.join(test_dir, d))) 
                         for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)))
        
        aug_count = sum(len(os.listdir(os.path.join(augmented_dir, d))) 
                        for d in os.listdir(augmented_dir) if os.path.isdir(os.path.join(augmented_dir, d)))
        
        print(f"Original train images: {train_count}")
        print(f"Validation images: {val_count}")
        print(f"Test images: {test_count}")
        print(f"Augmented train images: {aug_count} (Original + augmentations)")
        
        total_ratio = train_count / total_images
        print(f"Train ratio: {total_ratio:.2f} (target: 0.75)")
        val_ratio = val_count / total_images
        print(f"Validation ratio: {val_ratio:.2f} (target: 0.10)")
        test_ratio = test_count / total_images
        print(f"Test ratio: {test_ratio:.2f} (target: 0.15)")
        
        print(f"\nDataset successfully prepared at {output_dir}!")
        print("- Use dataset/train_augmented for training")
        print("- Use dataset/val for validation")
        print("- Use dataset/test for testing")
    else:
        print(f"Error: Training directory {train_dir} not found after splitting")

if __name__ == "__main__":
    main()
