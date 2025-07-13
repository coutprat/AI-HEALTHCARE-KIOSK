import cv2
import face_recognition
import os
import numpy as np
import time
import threading
from typing import Optional, Tuple

# Configuration - Use absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Go up one level from frontend
ENCODINGS_DIR = os.path.join(PROJECT_ROOT, "frontend", "encodings")
KNOWN_FACES_DIR = os.path.join(PROJECT_ROOT, "known_faces")

# Set environment variables for Windows compatibility
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

class AutomaticFaceRecognizer:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.recognition_active = False
        self.recognized_user = None
        self.confidence_threshold = 0.5
        self.consecutive_recognitions = 0
        self.required_consecutive = 5  # Require 5 consecutive recognitions
        self.last_recognized_name = None
        
    def load_known_faces(self) -> bool:
        """Load known faces from encodings directory"""
        self.known_encodings = []
        self.known_names = []
        
        print(f"[INFO] Loading face encodings from: {ENCODINGS_DIR}")
        
        if not os.path.exists(ENCODINGS_DIR):
            print(f"[ERROR] Encodings directory not found: {ENCODINGS_DIR}")
            return False
        
        try:
            files = os.listdir(ENCODINGS_DIR)
            encoding_files = [f for f in files if f.endswith('.npy')]
            
            if not encoding_files:
                print("[ERROR] No face encoding files found")
                return False
            
            for filename in encoding_files:
                try:
                    path = os.path.join(ENCODINGS_DIR, filename)
                    encoding = np.load(path)
                    name = os.path.splitext(filename)[0]
                    
                    self.known_encodings.append(encoding)
                    self.known_names.append(name)
                    print(f"[SUCCESS] Loaded encoding for: {name}")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to load {filename}: {e}")
            
            print(f"[INFO] Total loaded faces: {len(self.known_encodings)}")
            return len(self.known_encodings) > 0
            
        except Exception as e:
            print(f"[ERROR] Failed to load encodings: {e}")
            return False
    
    def recognize_face_in_frame(self, frame) -> Optional[Tuple[str, float]]:
        """Recognize face in a single frame"""
        try:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for faster processing
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            small_frame = np.ascontiguousarray(small_frame, dtype=np.uint8)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            
            for face_encoding in face_encodings:
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_encodings, 
                    face_encoding, 
                    tolerance=self.confidence_threshold
                )
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                
                if matches and any(matches):
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        confidence = 1 - face_distances[best_match_index]
                        name = self.known_names[best_match_index]
                        return name, confidence
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Face recognition error: {e}")
            return None
    
    def start_automatic_recognition(self, timeout=30) -> Optional[str]:
        """Start automatic face recognition with timeout"""
        if not self.load_known_faces():
            print("[ERROR] Cannot start recognition - no faces loaded")
            return None
        
        print(f"[INFO] Starting automatic face recognition...")
        print(f"[INFO] Loaded {len(self.known_encodings)} known faces: {self.known_names}")
        
        # Initialize camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("[WARNING] DirectShow failed, trying default backend")
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Could not open camera")
            return None
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Allow camera to warm up
        time.sleep(2)
        
        start_time = time.time()
        self.recognition_active = True
        self.recognized_user = None
        self.consecutive_recognitions = 0
        self.last_recognized_name = None
        
        print(f"[INFO] Looking for faces... (timeout: {timeout}s)")
        
        try:
            while self.recognition_active:
                # Check timeout
                if time.time() - start_time > timeout:
                    print(f"[INFO] Recognition timeout after {timeout} seconds")
                    break
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue
                
                # Recognize face in current frame
                result = self.recognize_face_in_frame(frame)
                
                if result:
                    name, confidence = result
                    
                    # Check for consecutive recognitions
                    if name == self.last_recognized_name:
                        self.consecutive_recognitions += 1
                    else:
                        self.last_recognized_name = name
                        self.consecutive_recognitions = 1
                    
                    # Draw recognition feedback
                    cv2.putText(frame, f"Recognizing: {name}", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", (50, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Progress: {self.consecutive_recognitions}/{self.required_consecutive}", 
                               (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # If we have enough consecutive recognitions, we're confident
                    if self.consecutive_recognitions >= self.required_consecutive:
                        self.recognized_user = name
                        print(f"[SUCCESS] Automatic recognition: {name} (confidence: {confidence:.2f})")
                        break
                else:
                    self.consecutive_recognitions = 0
                    self.last_recognized_name = None
                    cv2.putText(frame, "Looking for registered faces...", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Display the frame
                cv2.imshow('Automatic Face Recognition - Healthcare Kiosk', frame)
                
                # Allow early exit with 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Recognition cancelled by user")
                    break
        
        except Exception as e:
            print(f"[ERROR] Recognition failed: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.recognition_active = False
        
        return self.recognized_user

# Global recognizer instance
face_recognizer = AutomaticFaceRecognizer()

def automatic_face_checkin(timeout=30) -> Optional[str]:
    """Main function for automatic face check-in"""
    return face_recognizer.start_automatic_recognition(timeout)

if __name__ == "__main__":
    print("=== Automatic Face Recognition Check-in ===")
    recognized_user = automatic_face_checkin()
    
    if recognized_user:
        print(f"[SUCCESS] Welcome back, {recognized_user}!")
    else:
        print("[INFO] No user recognized automatically")
