import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  # Optimize FPS
screen_w, screen_h = pyautogui.size()

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Smooth movement parameters
smoothening = 5
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Gesture timing variables
last_gesture_time = time.time()
gesture_cooldown = 0.8  # 800ms delay for gestures
active_gesture = None

def fingers_up(hand_landmarks):
    """Detect which fingers are up (1 = Up, 0 = Down)."""
    fingers = [0, 0, 0, 0, 0]  # Thumb, Index, Middle, Ring, Pinky

    # Index Finger - (8 is above 6)
    fingers[1] = 1 if hand_landmarks[8][1] < hand_landmarks[6][1] else 0

    # Other Fingers - (Closed if tips are below their lower joint)
    fingers[2] = 1 if hand_landmarks[12][1] < hand_landmarks[10][1] else 0  # Middle
    fingers[3] = 1 if hand_landmarks[16][1] < hand_landmarks[14][1] else 0  # Ring
    fingers[4] = 1 if hand_landmarks[20][1] < hand_landmarks[18][1] else 0  # Pinky

    return fingers

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the camera
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmark positions
            lm_list = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in hand_landmarks.landmark]

            fingers = fingers_up(lm_list)

            # Cursor Movement: Only when index finger is up and all others are down
            if fingers == [0, 1, 0, 0, 0]:  
                index_x, index_y = lm_list[8]  # Index Finger Tip
                curr_x = np.interp(index_x, [100, frame_w - 100], [0, screen_w])
                curr_y = np.interp(index_y, [100, frame_h - 100], [0, screen_h])

                # Smooth movement
                prev_x = (prev_x * (smoothening - 1) + curr_x) / smoothening
                prev_y = (prev_y * (smoothening - 1) + curr_y) / smoothening
                pyautogui.moveTo(prev_x, prev_y, duration=0.05)

            # Left Click: Thumb + Index Finger Pinch
            thumb_tip, index_tip = lm_list[4], lm_list[8]
            pinch_distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

            if fingers[0] == 1 and fingers[1] == 1 and pinch_distance < 30:
                pyautogui.click()
                time.sleep(0.2)  # Prevent multiple clicks

            # Right Click: Thumb + Middle Finger Pinch
            middle_tip = lm_list[12]
            pinch_distance_middle = np.linalg.norm(np.array(thumb_tip) - np.array(middle_tip))

            if fingers[0] == 1 and fingers[2] == 1 and pinch_distance_middle < 30:
                pyautogui.rightClick()
                time.sleep(0.2)

            # Scrolling: Move Two Fingers Up/Down
            if fingers[1] == 1 and fingers[2] == 1:
                scroll_movement = lm_list[8][1] - lm_list[12][1]
                if abs(scroll_movement) > 20:
                    pyautogui.scroll(-5 if scroll_movement < 0 else 5)

            # Zoom In/Out: Pinch Gesture
            if pinch_distance < 30:
                pyautogui.hotkey('ctrl', '+')
                time.sleep(0.2)
            elif pinch_distance > 100:
                pyautogui.hotkey('ctrl', '-')
                time.sleep(0.2)

            # Swipe Left (Go Back) & Swipe Right (Go Forward)
            palm_x, palm_y = lm_list[0]
            hand_movement_x = palm_x - prev_x

            swipe_threshold = 120  # Pixels for swipe
            current_time = time.time()

            if abs(hand_movement_x) > swipe_threshold:
                if hand_movement_x > 0:  # Moving right (Go Forward)
                    if active_gesture != "forward" or (current_time - last_gesture_time) > gesture_cooldown:
                        pyautogui.hotkey('alt', 'right')
                        active_gesture = "forward"
                        last_gesture_time = current_time
                else:  # Moving left (Go Back)
                    if active_gesture != "back" or (current_time - last_gesture_time) > gesture_cooldown:
                        pyautogui.hotkey('alt', 'left')
                        active_gesture = "back"
                        last_gesture_time = current_time

            # Draw Hand Landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the Camera Feed
    cv2.imshow("Hand Gesture Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
