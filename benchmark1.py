import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

def benchmark_model_simplified(model_path, image_dir, device):
    """Simplified benchmark for ViT model performance"""
    # Load model
    model = ViTForImageClassification.from_pretrained(model_path).to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Print the model's label mappings to understand the expected format
    print("Model label mapping:")
    print(model.config.id2label)
    
    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images for benchmarking")
    
    # Process images - just measure inference time
    inference_times = []
    predictions = []
    
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                # Process image
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                # Time inference
                start_time = time.time()
                outputs = model(input_tensor).logits
                inference_time = time.time() - start_time
                inference_times.append(inference_time * 1000)  # ms
                
                # Get prediction index
                pred_idx = torch.argmax(outputs, dim=1).item()
                predictions.append(pred_idx)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Calculate statistics
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    
    # Plot prediction distribution
    plt.figure(figsize=(12, 8))
    plt.hist(predictions, bins=len(set(predictions)))
    plt.xticks(range(len(model.config.id2label)), 
               [model.config.id2label[str(i)] for i in range(len(model.config.id2label))],
               rotation=90)
    plt.title('Distribution of Predictions')
    plt.xlabel('Predicted Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('prediction_distribution.png')
    
    # Plot inference time distribution
    plt.figure(figsize=(10, 6))
    plt.hist(inference_times, bins=20)
    plt.title('Inference Time Distribution')
    plt.xlabel('Time (ms)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('inference_time_distribution.png')
    
    return {
        'avg_inference_time_ms': avg_time,
        'std_inference_time_ms': std_time,
        'total_images': len(image_files),
        'prediction_counts': {model.config.id2label[str(i)]: predictions.count(i) 
                             for i in range(len(model.config.id2label))}
    }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = 'best_vit_model'
    image_dir = 'paligemma_test_dataset'
    
    results = benchmark_model_simplified(model_path, image_dir, device)
    
    print("\n===== Benchmark Results =====")
    print(f"Total images processed: {results['total_images']}")
    print(f"Average inference time: {results['avg_inference_time_ms']:.2f} ms per image")
    print(f"Inference time std dev: {results['std_inference_time_ms']:.2f} ms")
    
    print("\n===== Prediction Distribution =====")
    for class_name, count in results['prediction_counts'].items():
        if count > 0:
            print(f"{class_name}: {count} images")
    
    print("\nPrediction distribution saved as 'prediction_distribution.png'")
    print("Inference time distribution saved as 'inference_time_distribution.png'")
