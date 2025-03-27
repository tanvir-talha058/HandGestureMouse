import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS to optimize performance
screen_w, screen_h = pyautogui.size()

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Smoothing parameters
smoothening = 5
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Gesture tracking variables
last_gesture_time = time.time()
gesture_cooldown = 1  # 1-second delay for swipe gestures
active_gesture = None

def fingers_up(hand_landmarks):
    """Detect which fingers are up (1 = Up, 0 = Down)."""
    fingers = [0, 0, 0, 0, 0]  # Thumb, Index, Middle, Ring, Pinky

    # Index Finger - (8 is above 6)
    if hand_landmarks[8][1] < hand_landmarks[6][1]:
        fingers[1] = 1  # Index Finger is Up

    # Other Fingers - (Closed if their tips are below their lower joint)
    fingers[2] = 0 if hand_landmarks[12][1] > hand_landmarks[10][1] else 1  # Middle
    fingers[3] = 0 if hand_landmarks[16][1] > hand_landmarks[14][1] else 1  # Ring
    fingers[4] = 0 if hand_landmarks[20][1] > hand_landmarks[18][1] else 1  # Pinky

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
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                x, y = int(lm.x * frame_w), int(lm.y * frame_h)
                lm_list.append((x, y))

            fingers = fingers_up(lm_list)

            #  move cursor
            if fingers == [0, 1, 0, 0, 0]:  
                index_x, index_y = lm_list[8]  # Index Finger Tip
                curr_x = np.interp(index_x, [100, frame_w - 100], [0, screen_w])
                curr_y = np.interp(index_y, [100, frame_h - 100], [0, screen_h])

                # Smoothing cursor movement
                prev_x = (prev_x * (smoothening - 1) + curr_x) / smoothening
                prev_y = (prev_y * (smoothening - 1) + curr_y) / smoothening
                pyautogui.moveTo(prev_x, prev_y, duration=0.1)

            # Left Click (Thumb + Index Finger Pinch)
            if fingers[1] == 1 and fingers[0] == 1 and fingers[2] == 0:
                pyautogui.click()
                time.sleep(0.2)

            # Right Click (Thumb + Middle Finger Pinch)
            if fingers[1] == 0 and fingers[2] == 1 and fingers[0] == 1:
                pyautogui.rightClick()
                time.sleep(0.2)

            # Scroll (Move Two Fingers Up/Down)
            if fingers[1] == 1 and fingers[2] == 1:
                pyautogui.scroll(-10 if lm_list[8][1] < lm_list[12][1] else 10)

            # Zoom In/Out (Pinch Open/Close)
            pinch_distance = abs(lm_list[4][0] - lm_list[8][0])
            if pinch_distance < 30:
                pyautogui.hotkey('ctrl', '+')
                time.sleep(0.2)
            elif pinch_distance > 100:
                pyautogui.hotkey('ctrl', '-')
                time.sleep(0.2)

            # Detect Swipe Gestures for Navigation
            palm_x, palm_y = lm_list[0]  # Palm Center
            prev_palm_x, prev_palm_y = prev_x, prev_y  # Store previous palm position
            hand_movement_x = palm_x - prev_palm_x

            swipe_threshold = 120  # Pixels for swipe
            current_time = time.time()

            if abs(hand_movement_x) > swipe_threshold:
                if hand_movement_x > 0:  #Go Forward
                    if active_gesture != "forward" or (current_time - last_gesture_time) > gesture_cooldown:
                        pyautogui.hotkey('alt', 'right')
                        active_gesture = "forward"
                        last_gesture_time = current_time
                else:  # Go Back
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
