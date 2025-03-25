import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize hand tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Screen size
screen_width, screen_height = pyautogui.size()
smooth_factor = 5  # Smooth movement

# Variables to store previous locations
prev_x, prev_y = 0, 0
click_threshold = 30  # Distance to trigger click

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # Get index finger tip position
            index_x, index_y = int(landmarks[8].x * screen_width), int(landmarks[8].y * screen_height)
            thumb_x, thumb_y = int(landmarks[4].x * screen_width), int(landmarks[4].y * screen_height)

            # Smooth movement
            curr_x = prev_x + (index_x - prev_x) / smooth_factor
            curr_y = prev_y + (index_y - prev_y) / smooth_factor
            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            # Click detection (if thumb and index finger are close)
            distance = np.hypot(index_x - thumb_x, index_y - thumb_y)
            if distance < click_threshold:
                pyautogui.click()

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
