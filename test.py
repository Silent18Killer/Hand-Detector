import cv2
from cvzone.HandTrackingModule import HandDetector 
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.5, maxHands=2)

desired_width = 1080
desired_height = 800

# Timeout settings
timeout_duration = 10  # seconds
last_hand_detected_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam.")
        break
    
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)
    resized_img = cv2.resize(img, (desired_width, desired_height))
    
    current_time = time.time()
    
    if hands:
        last_hand_detected_time = current_time
        for hand in hands:
            hand_type = hand['type']
            if hand_type == 'Left':
                cv2.putText(resized_img, "Left Hand Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif hand_type == 'Right':
                cv2.putText(resized_img, "Right Hand Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        if current_time - last_hand_detected_time > timeout_duration:
            print("No hands detected for 10 seconds. Closing window.")
            break
    
    cv2.imshow("Image", resized_img)
    key = cv2.waitKey(1)
    
    # Exit on close button or 'q' key press
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1 or key & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
