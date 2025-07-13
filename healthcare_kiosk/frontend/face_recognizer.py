import face_recognition
import cv2
import os
import numpy as np
import time

# Configuration
KNOWN_FACES_DIR = os.path.join("healthcare_kiosk", "known_faces")
ENCODINGS_DIR = os.path.join("healthcare_kiosk", "frontend", "encodings")

TOLERANCE = 0.5
MODEL = "hog"  # or "cnn" (if GPU is available)

# Set environment variables for Windows compatibility
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# Initialize face database
known_faces = []
known_names = []

def load_known_faces():
    """Load known faces from both directory structure and encodings"""
    global known_faces, known_names
    
    known_faces = []
    known_names = []
    
    # First, try to load from encodings directory (preferred method)
    if os.path.exists(ENCODINGS_DIR):
        for filename in os.listdir(ENCODINGS_DIR):
            if filename.endswith(".npy"):
                try:
                    path = os.path.join(ENCODINGS_DIR, filename)
                    encoding = np.load(path)
                    name = os.path.splitext(filename)[0]
                    
                    known_faces.append(encoding)
                    known_names.append(name)
                    print(f"[INFO] Loaded encoding for {name}")
                except Exception as e:
                    print(f"[ERROR] Failed to load encoding {filename}: {e}")
    
    # Fallback: load from directory structure
    if not known_faces and os.path.exists(KNOWN_FACES_DIR):
        print("[INFO] No encodings found, loading from image files...")
        for name in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, name)
            if not os.path.isdir(person_dir):
                continue
            
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        image_path = os.path.join(person_dir, filename)
                        image = face_recognition.load_image_file(image_path)
                        
                        # Get face encodings
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            encoding = encodings[0]  # Take the first face found
                            known_faces.append(encoding)
                            known_names.append(name)
                            print(f"[INFO] Loaded face encoding for {name} from {filename}")
                            break  # Only load one image per person
                        else:
                            print(f"[WARNING] No face found in {image_path}")
                    except Exception as e:
                        print(f"[ERROR] Failed to load {image_path}: {e}")
    
    print(f"[INFO] Total loaded faces: {len(known_faces)}")

def recognize_face(timeout=30):
    """
    Recognize face from webcam with improved reliability
    Args:
        timeout (int): Maximum time to wait for recognition in seconds
    Returns:
        str: Name of recognized person or None
    """
    load_known_faces()
    
    if not known_faces:
        print("[ERROR] No known faces loaded. Please register faces first.")
        return None
    
    print(f"[INFO] Starting camera... Look into the webcam. Press 'q' to quit or wait {timeout}s.")
    
    # Initialize camera with DirectShow backend for Windows compatibility
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
    recognized_name = None
    consecutive_recognitions = 0
    required_consecutive = 3
    
    try:
        while True:
            # Check timeout
            if time.time() - start_time > timeout:
                print(f"[INFO] Recognition timeout after {timeout} seconds")
                break
            
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Failed to capture frame, retrying...")
                time.sleep(0.1)
                continue
            
            # Convert BGR to RGB (face_recognition expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
            
            try:
                # Find face locations and encodings
                face_locations = face_recognition.face_locations(rgb_frame, model=MODEL)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                current_recognition = None
                
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    # Compare with known faces
                    matches = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
                    face_distances = face_recognition.face_distance(known_faces, face_encoding)
                    
                    name = "Unknown"
                    
                    # Find the best match
                    if True in matches:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_names[best_match_index]
                            current_recognition = name
                    
                    # Draw rectangle and label
                    top, right, bottom, left = face_location
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Show recognition progress
                    if name != "Unknown":
                        label = f"{name} ({consecutive_recognitions + 1}/{required_consecutive})"
                    else:
                        label = name
                    
                    cv2.putText(frame, label, (left, top - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Check for consecutive recognitions
                if current_recognition:
                    if current_recognition == recognized_name:
                        consecutive_recognitions += 1
                    else:
                        recognized_name = current_recognition
                        consecutive_recognitions = 1
                    
                    # If we have enough consecutive recognitions, we're confident
                    if consecutive_recognitions >= required_consecutive:
                        print(f"[SUCCESS] Confidently recognized: {recognized_name}")
                        break
                else:
                    consecutive_recognitions = 0
                
            except Exception as e:
                print(f"[ERROR] Face recognition error: {e}")
                continue
            
            cv2.imshow("Face Recognition - Press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Recognition cancelled by user")
                break
    
    except Exception as e:
        print(f"[ERROR] Recognition failed: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return recognized_name if consecutive_recognitions >= required_consecutive else None

if __name__ == "__main__":
    user = recognize_face()
    if user:
        print(f"[SUCCESS] Recognized user: {user}")
    else:
        print("[INFO] No user recognized")
