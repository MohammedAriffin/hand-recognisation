import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt
from tqdm import tqdm

# Verify CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def main():
    # Set paths to your dataset folders
    data_dir = "dataset"  # UPDATE THIS PATH
    train_dir = os.path.join(data_dir, "train_augmented")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    
    # Load datasets
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = ImageFolder(test_dir, transform=val_test_transform)
    
    print(f"Classes: {train_dataset.classes}")
    print(f"Number of training samples: {len(train_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    
    # Mapping between class indices and class names
    id2label = {str(i): train_dataset.classes[i] for i in range(len(train_dataset.classes))}
    label2id = {label: id for id, label in id2label.items()}
    
    # Load pretrained ViT model and configure for our number of classes
    num_classes = len(train_dataset.classes)
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',  # Use base model with 16x16 patch size
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  # Important when changing number of classes
    ).to(device)
    
    # Fine-tuning configuration
    # We'll use different learning rates for different parts of the model
    optimizer = optim.AdamW([
        {'params': model.vit.parameters(), 'lr': 5e-6},  # Base model
        {'params': model.classifier.parameters(), 'lr': 5e-5}  # Classification head
    ], weight_decay=0.01)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    # Training function with mixed precision
    def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=15):
        scaler = torch.amp.GradScaler('cuda')  
        criterion = nn.CrossEntropyLoss()
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # Mixed precision training
                with torch.amp.autocast('cuda'):
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
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct.float() / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_correct.float() / len(val_loader.dataset)
            
            # Update scheduler
            scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc.item())
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc.item())
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_pretrained('best_vit_model')
                print(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        return history
    
    # Train the model
    print("Starting ViT fine-tuning...")
    history = train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=15)
    
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
    
    # Test final model
    model.eval()
    test_correct = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits
            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels.data)
    
    test_acc = test_correct.float() / len(test_loader.dataset)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Save the final model
    model.save_pretrained('final_vit_model')
    print("Final model saved!")
    
    return model, history

if __name__ == "__main__":
    main()
