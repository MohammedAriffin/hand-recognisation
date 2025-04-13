# main.py (updated)
import cv2
import time
import argparse
import threading
import queue
from services.camera import CameraService
from services.yolo import YoloService
from services.vit import ClassifierService
from services.nlp import NLPService

def main(runtime=5, skip_frames=6, display=True):
    classification_results = []
    stop_event = threading.Event()
    
    # Initialize services
    camera_service = CameraService()
    yolo_service = YoloService(target_fps=12)
    classifier_service = ClassifierService()
    nlp_service = NLPService()
    
    try:
        camera_service.start()
    except Exception as e:
        print(f"Failed to start camera: {e}")
        return []

    def processing_thread():
        frame_count = 0
        while not stop_event.is_set():
            frame = camera_service.get_frame()
            if frame is None:
                continue
                
            frame_count += 1
            if frame_count % skip_frames == 0:
                cropped_hands, annotated_frame = yolo_service.detect_hands(frame)
                
                if display:
                    cv2.imshow("Hand Detection", annotated_frame)
                    cv2.waitKey(1)
                
                for hand_img in cropped_hands:
                    if hand_img.size > 0:
                        classification = classifier_service.classify_image(hand_img)
                        classification_results.append(classification)
                        
                        # NLP processing
                        sentence = nlp_service.process_frame(classification)
                        if sentence:
                            print(f"Current sentence: {sentence}")
                            
                        if display:
                            result_image = hand_img.copy()
                            cv2.putText(result_image, classification, 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.8, (0, 255, 0), 2)
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
    final_text = nlp_service.segment_text(''.join(classification_results))
    print("\nFinal reconstructed text:", final_text)
    
    camera_service.stop()
    cv2.destroyAllWindows()
    print(f"Completed. Classified {len(classification_results)} hand gestures.")
    return classification_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASL Recognition System")
    parser.add_argument("--runtime", type=int, default=10, help="Runtime in seconds")
    parser.add_argument("--skip", type=int, default=6, help="Process every Nth frame")
    parser.add_argument("--no-display", action="store_true", help="Disable visualization")
    args = parser.parse_args()
    
    results = main(
        runtime=args.runtime,
        skip_frames=args.skip,
        display=not args.no_display
    )
    
    print("\nClassification results:")
    print(results)
