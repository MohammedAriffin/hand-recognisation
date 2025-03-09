import cv2
import torch
from ultralytics import YOLO # Ensure you have installed this package

def main():
    # Determine the processing device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("CUDA available:", torch.cuda.is_available())
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
        
        # Run inference 
        # Note: Depending on your YOLOv10 package API, model.predict() may directly accept numpy arrays.
        results = model.predict(rgb_frame)
        
        # Process results: Here we assume results.xyxy gives a tensor with rows:
        # [x1, y1, x2, y2, confidence, class]
        if hasattr(results, 'xyxy'):
            # Transfer results to CPU & convert to numpy array
            detections = results.xyxy.cpu().numpy()
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                # Draw a green rectangle around the detection
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Put the confidence score on the top-left corner of the bounding box
                cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # If your package provides a built-in result drawing method, you might use:
            results = model(frame)  # Perform inference
            frame = results.plot()  # Annotate the frame with detection results

        
        # Display the annotated frame in a window titled "YOLOv10-S Real-Time Detection"
        cv2.imshow("YOLOv10-S Real-Time Detection", frame)
        
        # Exit if 'q' is pressed; wait 1ms between frames
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean-up: release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()