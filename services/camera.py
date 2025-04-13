import cv2
import threading
import queue
import time

class CameraService:
    def __init__(self, camera_id=0, buffer_size=10):
        """Service to handle camera operations"""
        self.camera_id = camera_id
        self.is_running = False
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.cap = None
        
    def start(self):
        """Start camera capture in a separate thread"""
        if self.is_running:
            return
            
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
            
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        print("Camera service started")
        
    def _capture_loop(self):
        """Internal loop to continuously capture frames"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error capturing frame")
                time.sleep(0.1)
                continue
                
            # Remove oldest frame if buffer is full
            if self.frame_buffer.full():
                try:
                    self.frame_buffer.get_nowait()
                except queue.Empty:
                    pass
                    
            # Add new frame to buffer
            try:
                self.frame_buffer.put_nowait(frame)
            except queue.Full:
                pass
                
    def get_frame(self, timeout=1.0):
        """Get the latest frame from buffer"""
        try:
            return self.frame_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def stop(self):
        """Stop the camera service"""
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        print("Camera service stopped")
