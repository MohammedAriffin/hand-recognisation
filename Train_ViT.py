import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Verify CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Print CUDA memory information if using GPU
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Data transformations with augmentation for hand signs
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # ViT normalization
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # ViT normalization
])

def sample_dataset(dataset, percentage=0.15):
    """Sample a specified percentage of a dataset"""
    sample_size = int(len(dataset) * percentage)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    sampled_indices = indices[:sample_size]
    return Subset(dataset, sampled_indices)

def main():
    start_time = time.time()
    
    # Set paths to your dataset folders
    data_dir = "dataset"  # UPDATE THIS PATH
    train_dir = os.path.join(data_dir, "train_augmented")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = ImageFolder(test_dir, transform=val_test_transform)
    
    # Print original dataset sizes
    print(f"Original dataset sizes:")
    print(f"Train: {len(train_dataset)} images")
    print(f"Validation: {len(val_dataset)} images")
    print(f"Test: {len(test_dataset)} images")
    
    # Sample datasets to 15% for faster training
    print("Sampling 15% of datasets for faster training...")
    train_dataset = sample_dataset(train_dataset, 0.15)
    val_dataset = sample_dataset(val_dataset, 0.15)
    test_dataset = sample_dataset(test_dataset, 0.15)
    
    # Print sampled dataset sizes
    print(f"Sampled dataset sizes:")
    print(f"Train: {len(train_dataset)} images")
    print(f"Validation: {len(val_dataset)} images")
    print(f"Test: {len(test_dataset)} images")
    
    # Access classes through the original dataset object
    classes = train_dataset.dataset.classes
    print(f"Classes: {classes}")
    print(f"Number of classes: {len(classes)}")
    
    # Create dataloaders with smaller batch size for limited resources
    batch_size = 8  # Reduced from 16 for your limited resources
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Mapping between class indices and class names
    id2label = {str(i): classes[i] for i in range(len(classes))}
    label2id = {label: id for id, label in id2label.items()}
    
    # Load pretrained ViT model and configure for our number of classes
    num_classes = len(classes)
    print(f"Loading model: google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',  # Use base model with 16x16 patch size
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  # Important when changing number of classes
    ).to(device)
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Fine-tuning configuration
    # We'll use different learning rates for different parts of the model
    optimizer = optim.AdamW([
        {'params': model.vit.parameters(), 'lr': 5e-6},  # Base model
        {'params': model.classifier.parameters(), 'lr': 5e-5}  # Classification head
    ], weight_decay=0.01)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    # Training function with mixed precision
    def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=10):
        scaler = torch.cuda.amp.GradScaler(enabled=True)  
        criterion = nn.CrossEntropyLoss()
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0.0
        patience = 0
        max_patience = 2  # Early stopping after 2 epochs without improvement
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # Mixed precision training
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(inputs).logits
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                _, preds = torch.max(outputs, 1)
                train_loss += loss.item() * inputs.size(0)
                train_correct += torch.sum(preds == labels.data)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs).logits
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    val_loss += loss.item() * inputs.size(0)
                    val_correct += torch.sum(preds == labels.data)
            
            # Calculate epoch metrics
            train_loss = train_loss / len(train_dataset)
            train_acc = train_correct.float() / len(train_dataset)
            val_loss = val_loss / len(val_dataset)
            val_acc = val_correct.float() / len(val_dataset)
            
            # Update scheduler
            scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc.item())
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc.item())
            
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.1f}s")
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                os.makedirs('best_vit_model', exist_ok=True)
                model.save_pretrained('best_vit_model')
                print(f"New best model saved with validation accuracy: {val_acc:.4f}")
            else:
                patience += 1
                print(f"No improvement. Patience: {patience}/{max_patience}")
                if patience >= max_patience:
                    print("Early stopping triggered!")
                    break
            
            # Check if we're approaching 30 minutes
            elapsed_time = time.time() - start_time
            if elapsed_time > 25 * 60:  # 25 minutes
                print(f"Training approaching time limit (30 min). Stopping at epoch {epoch+1}.")
                break
        
        return history
    
    # Train the model
    print("Starting ViT fine-tuning...")
    history = train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=10)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('vit_training_history.png')
    
    # Load best model for testing
    best_model = ViTForImageClassification.from_pretrained('best_vit_model').to(device)
    
    # Test final model
    best_model.eval()
    test_correct = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs).logits
            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels.data)
            
            # Per-class accuracy
            correct = (preds == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    
    test_acc = test_correct.float() / len(test_dataset)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Print per-class accuracy
    print("\nAccuracy by class:")
    for i in range(num_classes):
        if class_total[i] > 0:
            print(f'Accuracy of {id2label[str(i)]}: {100 * class_correct[i] / class_total[i]:.1f}%')
        else:
            print(f'No test samples for class {id2label[str(i)]}')
    
    # Save the final model
    os.makedirs('final_vit_model', exist_ok=True)
    best_model.save_pretrained('final_vit_model')
    print("Final model saved!")
    
    # Print total training time
    total_time = time.time() - start_time
    print(f"Total training time: {total_time/60:.2f} minutes")
    
    return model, history

if __name__ == "__main__":
    main()
