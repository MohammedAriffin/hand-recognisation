import torch
from torchvision import transforms, datasets
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader
import os
from sklearn.metrics import classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Define the benchmark_model function
def benchmark_model(model, data_loader, device):
    model.eval()
    total_time = 0
    total_params = sum(p.numel() for p in model.parameters())
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(images).logits
            total_time += time.time() - start_time
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_inference_time = total_time / len(data_loader)
    classification_report_dict = classification_report(all_labels, all_preds, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    # Save confusion matrix as an image
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data_loader.dataset.classes, yticklabels=data_loader.dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    return {
        'avg_inference_time': avg_inference_time,
        'total_params': total_params,
        'classification_report': classification_report_dict
    }

# Add this wrapper around your main code
if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = ViTForImageClassification.from_pretrained('best_vit_model').to(device)
    print("Model loaded successfully")

    # You DO need a test dataset - create a DataLoader for your test data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # ViT normalization
    ])

    # Point this to your test data directory
    test_dir = "dataset/test"  # Update this path to your test dataset
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Test dataset loaded with {len(test_dataset)} images across {len(test_dataset.classes)} classes")

    # Use the benchmark function
    benchmark_results = benchmark_model(model, test_loader, device)

    # Print results
    print(f"Average inference time: {benchmark_results['avg_inference_time']*1000:.2f} ms per batch")
    print(f"Total parameters: {benchmark_results['total_params']:,}")

    # Get overall metrics
    overall_metrics = benchmark_results['classification_report']['weighted avg']
    print(f"Overall precision: {overall_metrics['precision']:.4f}")
    print(f"Overall recall: {overall_metrics['recall']:.4f}")
    print(f"Overall F1-score: {overall_metrics['f1-score']:.4f}")

    print("Confusion matrix saved as 'confusion_matrix.png'")
