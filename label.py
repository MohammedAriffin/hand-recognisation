import cv2
import os

def create_pseudo_labels(root_dir):
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
                    
                    # Full image bounding box
                    img = cv2.imread(img_path)
                    h, w = img.shape[:2]
                    
                    # YOLO format: class x_center y_center width height
                    yolo_line = f"{class_dir} 0.5 0.5 1.0 1.0\n"  # Full image box
                    
                    with open(label_path, 'w') as f:
                        f.write(yolo_line)

# Usage
create_pseudo_labels("dataset/train")
create_pseudo_labels("dataset/val")
create_pseudo_labels("dataset/test")
