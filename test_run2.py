import cv2
import torch
from ultralytics import YOLO  # Ensure you have installed this package

def main():
    # Determine the processing device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the YOLOv10-S model and move it to the selected device (GPU or CPU)
    model = YOLO("yolov10s.pt")
    model.to(device)

    # Open the webcam (0 is the default camera index)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return 

    # Optionally, print the webcam FPS for info
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Webcam FPS:", fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame capture failed.")
            break
        
        # Convert frame from BGR (used by OpenCV) to RGB (used by most deep learning models)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference and note that predict() returns a list of Results objects
        results = model.predict(rgb_frame)
        
        # Pick the first result from the list and use its plot() method to annotate the frame
        annotated_frame = results[0].plot() 

        # Display the annotated frame in a window titled "YOLOv10-S Real-Time Detection"
        cv2.imshow("YOLOv10-S Real-Time Detection", annotated_frame)
        
        # Exit if 'q' is pressed; wait 1ms between frames
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean-up: release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
