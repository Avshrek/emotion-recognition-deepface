import cv2
from deepface import DeepFace
import time
from firebase_utils import (
    initialize_firebase, 
    log_emotion_to_firestore, 
    check_and_trigger_rtdb_alert
)

# --- Configuration for the Main Loop ---
# Time interval in seconds to run the resource-intensive DeepFace analysis
WRITE_INTERVAL = 5 
LAST_WRITE_TIME = time.time()
# You would use a dynamic ID in a real system, but use a placeholder here.
CURRENT_STUDENT_ID = "S001_Demo_User" 

# --- 1. Initialize Firebase ---
if not initialize_firebase():
    print("Project cannot run without Firebase initialization.")
    exit()

# --- 2. Setup Camera ---
cap = cv2.VideoCapture(0) # Use 0 for primary camera

# --- 3. Main Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    
    # --- THROTTLING CHECK ---
    # Only run analysis and writes if the interval has passed
    if current_time - LAST_WRITE_TIME >= WRITE_INTERVAL:
        LAST_WRITE_TIME = current_time # Reset timer
        
        try:
            # 1. DeepFace Analysis
            analysis = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                enforce_detection=False # Recommended for real-time video feeds
            )

            # Check if any face was analyzed successfully
            if analysis and len(analysis) > 0:
                result = analysis[0]
                dominant_emotion = result['dominant_emotion']
                confidence = result['emotion'][dominant_emotion] # Get the specific confidence value

                # --- Firebase Writes ---
                
                # 2. WRITE TO FIRESTORE (Log ALL emotions)
                log_emotion_to_firestore(dominant_emotion, confidence, CURRENT_STUDENT_ID)

                # 3. WRITE TO REALTIME DB (Trigger only for high risk)
                check_and_trigger_rtdb_alert(dominant_emotion, CURRENT_STUDENT_ID)
            
            # else:
                # print("No face detected in the frame.")

        except Exception as e:
            # Catches errors like "No face detected" or deepface issues
            # We skip writing to Firebase if the analysis fails
            print(f"DeepFace/Analysis Error (Skipping write): {e}")

    # --- Display Frame (Optional) ---
    # Optional: You can draw the emotion text on the frame here for visual feedback
    cv2.imshow('Emotion Monitoring System', frame)
    
    # Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Application closed.")