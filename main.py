import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize hand tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Screen size
screen_width, screen_height = pyautogui.size()
smooth_factor = 5

# Variables to store previous locations
prev_x, prev_y = 0, 0
click_threshold = 30

# Gesture tracking variables
gesture_start_time = time.time()
gesture_cooldown = 1.0  # Prevent double activations
prev_gesture = None

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

            # Get important points
            index_x, index_y = int(landmarks[8].x * screen_width), int(landmarks[8].y * screen_height)
            thumb_x, thumb_y = int(landmarks[4].x * screen_width), int(landmarks[4].y * screen_height)
            middle_x, middle_y = int(landmarks[12].x * screen_width), int(landmarks[12].y * screen_height)
            ring_x, ring_y = int(landmarks[16].x * screen_width), int(landmarks[16].y * screen_height)
            pinky_x, pinky_y = int(landmarks[20].x * screen_width), int(landmarks[20].y * screen_height)
            palm_x, palm_y = int(landmarks[0].x * screen_width), int(landmarks[0].y * screen_height)

            # Move Cursor
            curr_x = prev_x + (index_x - prev_x) / smooth_factor
            curr_y = prev_y + (index_y - prev_y) / smooth_factor
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Click detection (Pinch - Index & Thumb)
            distance_click = np.hypot(index_x - thumb_x, index_y - thumb_y)
            if distance_click < click_threshold:
                pyautogui.click()

            # Right Click (Pinch - Middle & Thumb)
            distance_right_click = np.hypot(middle_x - thumb_x, middle_y - thumb_y)
            if distance_right_click < click_threshold:
                pyautogui.rightClick()

            # Drag & Drop (Hold Pinch)
            if distance_click < click_threshold:
                pyautogui.mouseDown()
            else:
                pyautogui.mouseUp()

            # Copy (Double Tap - Thumb & Index)
            if distance_click < click_threshold:
                if prev_gesture != "copy" or (time.time() - gesture_start_time) > gesture_cooldown:
                    pyautogui.hotkey('ctrl', 'c')
                    prev_gesture = "copy"
                    gesture_start_time = time.time()

            # Paste (Double Tap - Thumb & Middle)
            if distance_right_click < click_threshold:
                if prev_gesture != "paste" or (time.time() - gesture_start_time) > gesture_cooldown:
                    pyautogui.hotkey('ctrl', 'v')
                    prev_gesture = "paste"
                    gesture_start_time = time.time()

            # Scroll (Swipe Two Fingers Up/Down)
            scroll_distance = abs(middle_y - index_y)
            if scroll_distance > 50:
                if middle_y > index_y:
                    pyautogui.scroll(5)  # Scroll Up
                else:
                    pyautogui.scroll(-5)  # Scroll Down

            # Zoom In/Out (Pinch Open/Close with Index & Thumb)
            pinch_distance = np.hypot(index_x - thumb_x, index_y - thumb_y)
            if pinch_distance > 100:
                pyautogui.hotkey('ctrl', '+')  # Zoom In
            elif pinch_distance < 20:
                pyautogui.hotkey('ctrl', '-')  # Zoom Out

            # Go Back/Forward (Swipe Hand Left/Right)
            hand_movement = palm_x - prev_x
            if abs(hand_movement) > 100:
                if hand_movement > 0:
                    if prev_gesture != "forward" or (time.time() - gesture_start_time) > gesture_cooldown:
                        pyautogui.hotkey('alt', 'right')  # Go Forward
                        prev_gesture = "forward"
                        gesture_start_time = time.time()
                else:
                    if prev_gesture != "back" or (time.time() - gesture_start_time) > gesture_cooldown:
                        pyautogui.hotkey('alt', 'left')  # Go Back
                        prev_gesture = "back"
                        gesture_start_time = time.time()

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Advanced Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
