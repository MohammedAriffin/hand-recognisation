import cv2
import torch
from ultralytics import YOLO
import time
import os

# Initialize the model using the YOLOv10s weights
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov10s.pt")
model.to(device)

# Folder to save cropped hand images
save_folder = "segment"
os.makedirs(save_folder, exist_ok=True)

def model_run(runtime=10, skip=6):
    """Open the webcam, process every 'skip'th frame, and if a hand is detected,
    crop & save it. The loop runs for 'runtime' seconds or until 'q' is pressed."""

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    start_time = time.time()
    saved_count = 0   # Count of saved crops
    frame_count = 0   # Total frame counter

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame capture failed.")
            break

        frame_count += 1

        # Only process every 'skip' frame
        if frame_count % skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame)

            # Ensure results contain valid detections
            if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                # Iterate through each detected box and save the crop
                for box in results[0].boxes:
                    # Retrieve bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # Crop the detected hand region from the original frame
                    cropped_hand = frame[y1:y2, x1:x2]

                    # Save the cropped image
                    filename = f"hand_crop_{saved_count}.jpg"
                    file_path = os.path.join(save_folder, filename)
                    cv2.imwrite(file_path, cropped_hand)
                    print(f"Saved cropped image: {file_path}")
                    saved_count += 1

                # Create an annotated frame with detection results
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame
        else:
            annotated_frame = frame

        # Show the processed frame
        cv2.imshow("YOLOv10-S Real-Time Detection", annotated_frame)

        # Exit when time exceeds 'runtime' seconds or when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > runtime:
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the function when the script is executed
if __name__ == "__main__":
    model_run(runtime=10, skip=6)
