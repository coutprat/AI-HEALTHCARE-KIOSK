import cv2
import numpy as np
import os
import face_recognition
import time
import atexit

# Configuration - Use absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Go up one level from frontend
KNOWN_FACES_DIR = os.path.join(PROJECT_ROOT, "known_faces")
ENCODINGS_DIR = os.path.join(PROJECT_ROOT, "frontend", "encodings")

# Set environment variables for Windows compatibility
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# Global camera reference for cleanup
_active_camera = None

def cleanup_camera():
    """Cleanup function to ensure camera is released"""
    global _active_camera
    if _active_camera is not None:
        try:
            _active_camera.release()
            cv2.destroyAllWindows()
            time.sleep(0.5)  # Give Windows time to release camera
        except:
            pass
        _active_camera = None

# Register cleanup function
atexit.register(cleanup_camera)

def register_face(username):
    """
    Register a face for the given username with automatic capture.
    """
    global _active_camera
    
    try:
        if not username or not username.strip():
            print("[ERROR] Username cannot be empty")
            return False
        
        username = username.strip()
        
        # Create directories if they don't exist
        user_dir = os.path.join(KNOWN_FACES_DIR, username)
        os.makedirs(user_dir, exist_ok=True)
        os.makedirs(ENCODINGS_DIR, exist_ok=True)
        
        print(f"[INFO] Starting automatic face registration for user: {username}")
        print(f"[INFO] Images will be saved to: {user_dir}")
        print(f"[INFO] Encodings will be saved to: {ENCODINGS_DIR}")
        
        # Ensure any existing camera is released
        cleanup_camera()
        time.sleep(1)
        
        # Initialize camera with DirectShow backend for Windows compatibility
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("[WARNING] DirectShow failed, trying default backend")
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Could not open camera")
            return False
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        _active_camera = cap
        
        # Allow camera to warm up
        time.sleep(2)
        
        count = 0
        max_captures = 5
        face_encodings_collected = []
        consecutive_detections = 0
        required_consecutive = 3
        
        print(f"[INFO] Look at the camera. Will automatically capture {max_captures} images.")
        print("[INFO] Press 'q' to quit early")
        
        frame_skip = 0
        last_capture_time = 0
        min_capture_interval = 2.0
        
        while count < max_captures:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("[WARNING] Frame capture failed, retrying...")
                time.sleep(0.1)
                continue
            
            # Skip frames for better processing
            frame_skip += 1
            if frame_skip % 5 != 0:
                cv2.imshow(f"Auto Register - User: {username} (Press 'q' to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Convert BGR to RGB explicitly
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
            
            if rgb_frame.dtype != np.uint8:
                print("[ERROR] Invalid image dtype")
                continue
            
            try:
                # Find face locations
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if face_locations:
                    consecutive_detections += 1
                    
                    # Check if enough time has passed since last capture
                    current_time = time.time()
                    time_since_last = current_time - last_capture_time
                    
                    # Auto-capture when face is stable and enough time has passed
                    if (consecutive_detections >= required_consecutive and 
                        time_since_last >= min_capture_interval):
                        
                        print(f"[INFO] Stable face detected! Auto-capturing image {count + 1}/{max_captures}")
                        
                        # Get face encodings
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        if face_encodings:
                            face_encodings_collected.append(face_encodings[0])
                            print(f"[INFO] Face encoding captured (shape: {face_encodings[0].shape})")
                        
                        # Save the face image
                        top, right, bottom, left = face_locations[0]
                        face_image = rgb_frame[top:bottom, left:right]
                        if face_image.size > 0:
                            file_path = os.path.join(user_dir, f"face_{count}.jpg")
                            face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(file_path, face_bgr)
                            print(f"[INFO] Saved image: {file_path}")
                        
                        count += 1
                        last_capture_time = current_time
                        consecutive_detections = 0
                        
                        # Brief pause to show capture feedback
                        time.sleep(0.5)
                    
                    # Draw rectangle around detected face
                    top, right, bottom, left = face_locations[0]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Show status
                    if consecutive_detections < required_consecutive:
                        status = f"Stabilizing... {consecutive_detections}/{required_consecutive}"
                        color = (0, 255, 255)  # Yellow
                    else:
                        remaining_time = max(0, min_capture_interval - time_since_last)
                        if remaining_time > 0:
                            status = f"Wait {remaining_time:.1f}s"
                            color = (0, 165, 255)  # Orange
                        else:
                            status = "Capturing..."
                            color = (0, 255, 0)  # Green
                    
                    cv2.putText(frame, status, (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"Captured: {count}/{max_captures}",
                               (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    consecutive_detections = 0
                    cv2.putText(frame, "No face detected - Please look at camera",
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            except Exception as e:
                print(f"[ERROR] Face detection error: {e}")
                consecutive_detections = 0
                continue
            
            cv2.imshow(f"Auto Register - User: {username} (Press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Registration cancelled by user")
                break
        
        # Proper camera cleanup
        cleanup_camera()
        
        # Save the best encoding
        if face_encodings_collected:
            # Use the first encoding
            best_encoding = face_encodings_collected[0]
            encoding_path = os.path.join(ENCODINGS_DIR, f"{username}.npy")
            
            try:
                np.save(encoding_path, best_encoding)
                print(f"[SUCCESS] Saved face encoding to: {encoding_path}")
                
                # Verify the file was created
                if os.path.exists(encoding_path):
                    file_size = os.path.getsize(encoding_path)
                    print(f"[INFO] Encoding file size: {file_size} bytes")
                else:
                    print(f"[ERROR] Encoding file was not created: {encoding_path}")
                    return False
                
            except Exception as e:
                print(f"[ERROR] Failed to save encoding: {e}")
                return False
            
            print(f"[SUCCESS] Successfully registered {count} face images for user '{username}'")
            return True
        else:
            print(f"[ERROR] Failed to capture any valid face encodings for user '{username}'")
            return False
    
    except Exception as e:
        print(f"[ERROR] Registration failed: {e}")
        return False
    finally:
        # Ensure cleanup happens
        cleanup_camera()

if __name__ == "__main__":
    user_name = input("Enter patient name for registration: ").strip()
    if user_name:
        success = register_face(user_name)
        if success:
            print(f"[SUCCESS] Registration completed for {user_name}")
        else:
            print(f"[ERROR] Registration failed for {user_name}")
    else:
        print("[ERROR] Name cannot be empty.")
