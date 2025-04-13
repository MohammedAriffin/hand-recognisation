import cv2
import torch
import time
from ultralytics import YOLO

class YoloService:
    def __init__(self, model_path="yolov11_finetuned.pt" , target_fps=12):
        """Service for hand detection using YOLOv11"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_fps = target_fps
        self.frame_interval = 1 / target_fps
        self.last_process = 0
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"YOLO model loaded on {self.device} at {target_fps} FPS")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
            
    def detect_hands(self, frame):
        """Detect hands in a frame and return cropped images"""
        if time.time() - self.last_process < self.frame_interval:
            return [], frame  # Skip processing
            
        # Convert BGR to RGB for YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = self.model.predict(rgb_frame)
        self.last_process = time.time()
        cropped_hands = []
        
        # Process detections if valid
        if results and hasattr(results[0], 'boxes') and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    cropped_hands.append(frame[y1:y2, x1:x2].copy())
        
        annotated_frame = results[0].plot() if results else frame.copy()
        return cropped_hands, annotated_frame
