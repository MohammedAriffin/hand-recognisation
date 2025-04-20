import cv2
import time
import argparse
import threading
import queue
from services.camera import CameraService
from services.yolo import YoloService
from services.vit import ClassifierService
from services.nlp import NLPService

def main(runtime=50, skip_frames=4, display=True, confidence_threshold=0.5, gesture_interval=0.2 , target_fps=20):
    classification_results = []
    stop_event = threading.Event()
    
    # Initialize services
    camera_service = CameraService()
    yolo_service = YoloService(target_fps=target_fps)
    classifier_service = ClassifierService()
    nlp_service = NLPService()
    
    try:
        camera_service.start()
    except Exception as e:
        print(f"Failed to start camera: {e}")
        return []

    def processing_thread():
        frame_count = 0
        last_classification_time = 0
        
        # Gesture stability tracking
        recent_gestures = []
        gesture_stability_count = {}
        stability_threshold = 3  # Number of consistent detections needed
        
        while not stop_event.is_set():
            frame = camera_service.get_frame()
            if frame is None:
                continue
                
            frame_count += 1
            current_time = time.time()
            
            if frame_count % skip_frames == 0:
                cropped_hands, annotated_frame = yolo_service.detect_hands(frame)
                
                if display:
                    cv2.imshow("Hand Detection", annotated_frame)
                    cv2.waitKey(1)
                
                # Only process if we have hands and enough time has passed
                if cropped_hands and current_time - last_classification_time >= gesture_interval:
                    # Select the largest hand (likely the most prominent gesture)
                    largest_hand = max(cropped_hands, key=lambda img: img.shape[0] * img.shape[1]) if cropped_hands else None
                    
                    if largest_hand is not None and largest_hand.size > 0:
                        # Classify with confidence threshold
                        classification = classifier_service.classify_image(largest_hand, confidence_threshold)
                        
                        # Skip invalid or low confidence classifications
                        if classification not in ["Classification error", "Invalid image", "Low confidence"]:
                            # Update stability tracking
                            if classification in gesture_stability_count:
                                gesture_stability_count[classification] += 1
                            else:
                                # Reset counts when a new gesture is detected
                                gesture_stability_count = {classification: 1}
                            
                            # Check if any gesture is stable
                            stable_gesture = None
                            for gesture, count in gesture_stability_count.items():
                                if count >= stability_threshold:
                                    stable_gesture = gesture
                                    break
                            
                            if stable_gesture:
                                # Only add stable gestures to results and reset stability counter
                                classification_results.append(stable_gesture)
                                last_classification_time = current_time
                                gesture_stability_count = {}
                                
                                # NLP processing
                                sentence = nlp_service.process_frame(stable_gesture)
                                if sentence:
                                    print(f"Detected: {stable_gesture} → Sentence: {sentence}")
                            
                            if display:
                                result_image = largest_hand.copy()
                                status = f"{classification} ({gesture_stability_count.get(classification, 0)}/{stability_threshold})"
                                cv2.putText(result_image, status, 
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.7, (0, 255, 0), 2)
                                cv2.imshow("Classification", result_image)
                                cv2.waitKey(1)

    process_thread = threading.Thread(target=processing_thread)
    process_thread.daemon = True
    process_thread.start()

    print(f"Running for {runtime} seconds...")
    time.sleep(runtime)
    stop_event.set()
    process_thread.join(timeout=1.0)
    
    # Final processing
    if classification_results:
        final_text = nlp_service.segment_text(classification_results)
        print("\nFinal reconstructed text:", final_text)
    else:
        print("\nNo stable gestures were detected.")
    
    camera_service.stop()
    cv2.destroyAllWindows()
    print(f"Completed. Classified {len(classification_results)} stable hand gestures.")
    return classification_results

if __name__ == "__main__":
    # Direct parameter assignment without argparse
    results = main(
        runtime=50,
        skip_frames=5,
        display=True,
        confidence_threshold=0.5,
        gesture_interval=0.2,
        target_fps=30,
    )
    
    print("\nStable classification results:")
    print(results)

