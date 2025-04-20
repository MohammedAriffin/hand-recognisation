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
    
    def classify_image(self, image, confidence_threshold, return_confidence=False):
        """Classify a hand image and return the predicted class"""
        if image is None or image.size == 0:
            return ("Invalid image", 0.0) if return_confidence else "Invalid image"
            
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Apply transforms to preprocess the image
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
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
                return (predicted_class, confidence) if return_confidence else predicted_class
            else:
                return ("Low confidence", confidence) if return_confidence else "Low confidence"
                
        except Exception as e:
            print(f"Error in classification: {e}")
            return ("Classification error", 0.0) if return_confidence else "Classification error"


if __name__=="__main__":
    import os
    from tqdm import tqdm
    
    # Configuration - change these variables as needed
    input_folder = "segments"  # Path to folder containing hand images
    output_folder = "classified_images"  # Where to save annotated images
    output_file = "predictions.txt"  # Where to save the results
    confidence_threshold = 0.7  # Confidence threshold
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize classifier
    classifier = ClassifierService()
    
    # Get list of image files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        os.path.join(input_folder, f) for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f)) and 
        os.path.splitext(f)[1].lower() in valid_extensions
    ]
    
    if not image_files:
        print(f"No valid image files found in {input_folder}")
        exit()
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images and collect predictions
    predictions = []
    prediction_confidences = []
    
    # Open text file for writing results
    with open(output_file, 'w') as f:
        f.write("Filename,Prediction,Confidence\n")  # CSV header
        
        for img_path in tqdm(image_files):
            try:
                # Extract filename without extension
                filename = os.path.basename(img_path)
                name_only = os.path.splitext(filename)[0]
                
                # Load image (cv2.imread returns BGR)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Error reading image: {img_path}")
                    continue
                    
                # Convert BGR to RGB since saved images are in BGR format
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Classify with confidence
                result, confidence = classifier.classify_image(rgb_image, confidence_threshold, return_confidence=True)
                
                # Format the confidence as percentage
                confidence_pct = confidence * 100
                
                # Add to predictions if valid
                if result not in ["Classification error", "Invalid image", "Low confidence"]:
                    predictions.append(result)
                    prediction_confidences.append(confidence)
                    
                    # Create annotated image
                    annotated = image.copy()  # Use BGR image for OpenCV drawing
                    
                    # Draw text with prediction and confidence
                    text = f"{result} ({confidence_pct:.2f}%)"
                    cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 255, 0), 2)
                    
                    # Save the annotated image
                    output_path = os.path.join(output_folder, f"{name_only}_classified.png")
                    cv2.imwrite(output_path, annotated)
                    
                    print(f"{filename}: {result} ({confidence_pct:.2f}%)")
                else:
                    print(f"{filename}: {result} ({confidence_pct:.2f}%)")
                
                # Write to text file
                f.write(f"{filename},{result},{confidence_pct:.2f}%\n")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Save results
    print(f"\nProcessed folder: {input_folder}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Predictions saved to {output_file}")
    print(f"Annotated images saved to {output_folder}")
    
    if predictions:
        # Print the list of predictions
        print("\nPredicted classes:")
        for pred, conf in zip(predictions, prediction_confidences):
            print(f"{pred}: {conf*100:.2f}%")
        
        # Optionally, if you want to process the predictions with your NLP service
        try:
            from services.nlp import NLPService
            nlp_service = NLPService()
            segmented_text = nlp_service.segment_text(predictions)
            print(f"Segmented text: {segmented_text}")
        except ImportError:
            pass  # NLP service not available
