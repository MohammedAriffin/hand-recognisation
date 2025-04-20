import torch
from transformers import ViTForImageClassification
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms

class ClassifierService:
    def __init__(self, model_path="final_vit_model"):
        """Service for hand gesture classification using ViT"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load model
            self.model = ViTForImageClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Create transform pipeline manually
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # ViT usually expects 224x224 images
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Standard ViT normalization
            ])
            
            # Get class names
            if hasattr(self.model.config, 'id2label'):
                self.class_names = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
            else:
                self.class_names = [f"class_{i}" for i in range(self.model.config.num_labels)]
                
            print(f"Classification model loaded on {self.device} with {len(self.class_names)} classes")
            
        except Exception as e:
            print(f"Error loading classification model: {e}")
            raise
    
    def classify_image(self, image, confidence_threshold):
        """Classify a hand image and return the predicted class"""
        if image is None or image.size == 0:
            return "Invalid image"
            
        try:
            # Convert BGR to RGB
            #rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Apply transforms to preprocess the image
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Get predictions - pass tensor directly to the model
            with torch.no_grad():
                outputs = self.model(pixel_values=input_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get the maximum probability and corresponding index
            max_prob, predicted_idx = torch.max(probabilities, dim=-1)
            confidence = max_prob.item()
            
            # Only return prediction if confidence exceeds threshold
            if confidence >= confidence_threshold:
                predicted_class = self.class_names[predicted_idx.item()]
                return predicted_class
            else:
                return "Low confidence"
                
        except Exception as e:
            print(f"Error in classification: {e}")
            return "Classification error"
