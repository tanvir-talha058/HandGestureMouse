import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time


# Initialize Video
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
click_cooldown = 0.3  # Prevent multiple clicks within 300ms
last_click_time = time.time()

def get_finger_openness(hand_landmarks):
    """Calculate how open each finger is (0-100%)."""
    openness = [0] * 5  # Thumb, Index, Middle, Ring, Pinky

    # Calculate openness percentage
    for i, (tip, base) in enumerate([(4, 2), (8, 6), (12, 10), (16, 14), (20, 18)]):
        tip_y, base_y = hand_landmarks[tip][1], hand_landmarks[base][1]
        openness[i] = int(max(0, min(100, ((base_y - tip_y) / abs(base_y - hand_landmarks[0][1])) * 100)))

    return openness

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

            openness = get_finger_openness(lm_list)

            # Cursor Movement: Only when index finger is open and others are closed
            if openness[1] > 70 and all(op < 30 for i, op in enumerate(openness) if i != 1):  
                index_x, index_y = lm_list[8]  # Index Finger Tip
                curr_x = np.interp(index_x, [100, frame_w - 100], [0, screen_w])
                curr_y = np.interp(index_y, [100, frame_h - 100], [0, screen_h])

                # Smooth movement
                prev_x = (prev_x * (smoothening - 1) + curr_x) / smoothening
                prev_y = (prev_y * (smoothening - 1) + curr_y) / smoothening
                pyautogui.moveTo(prev_x, prev_y, duration=0.05)

            # Pinch Click: Thumb + Index Finger (for left click)
            thumb_tip, index_tip = lm_list[4], lm_list[8]
            pinch_distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

            current_time = time.time()
            if openness[0] > 70 and openness[1] > 70 and pinch_distance < 40:  # Pinch Distance Threshold
                if (current_time - last_click_time) > click_cooldown:  # Prevent multiple clicks
                    pyautogui.click()
                    last_click_time = current_time

            # Right Click: Thumb + Middle Finger Pinch
            middle_tip = lm_list[12]
            pinch_distance_middle = np.linalg.norm(np.array(thumb_tip) - np.array(middle_tip))

            if openness[0] > 70 and openness[2] > 70 and pinch_distance_middle < 40:
                if (current_time - last_click_time) > click_cooldown:
                    pyautogui.rightClick()
                    last_click_time = current_time

            # Scrolling: Move Two Fingers Up/Down
            if openness[1] > 70 and openness[2] > 70:
                scroll_movement = lm_list[8][1] - lm_list[12][1]
                if abs(scroll_movement) > 20:
                    pyautogui.scroll(-5 if scroll_movement < 0 else 5)

            # Draw Hand Landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display Finger Openness Bar Graph
            bar_x, bar_y = 20, frame_h - 100
            bar_width, bar_height = 300, 15

            for i, percent in enumerate(openness):
                cv2.rectangle(frame, (bar_x + i * 60, bar_y), (bar_x + i * 60 + 50, bar_y + bar_height), (255, 255, 255), 2)
                cv2.rectangle(frame, (bar_x + i * 60, bar_y), (bar_x + i * 60 + 50, bar_y + int(bar_height * (percent / 100))), (0, 255, 0), -1)
                cv2.putText(frame, f"{percent}%", (bar_x + i * 60 + 5, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the Camera Feed
    cv2.imshow("Hand Gesture Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()

