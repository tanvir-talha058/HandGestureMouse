import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize hand tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Screen size
screen_width, screen_height = pyautogui.size()
smooth_factor = 7  # Increased for smoother movement

# Gesture Control Variables
prev_x, prev_y = 0, 0
click_threshold = 30  # Distance for pinch click
gesture_cooldown = 1.0  # Prevent rapid gesture activation
last_gesture_time = 0
active_gesture = None
dragging = False  # Track dragging state

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    current_time = time.time()  # Track gesture timing

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # Get important points
            index_x, index_y = int(landmarks[8].x * screen_width), int(landmarks[8].y * screen_height)
            thumb_x, thumb_y = int(landmarks[4].x * screen_width), int(landmarks[4].y * screen_height)
            middle_x, middle_y = int(landmarks[12].x * screen_width), int(landmarks[12].y * screen_height)
            palm_x, palm_y = int(landmarks[0].x * screen_width), int(landmarks[0].y * screen_height)

            # Cursor Movement smooth transition
            curr_x = prev_x + (index_x - prev_x) / smooth_factor
            curr_y = prev_y + (index_y - prev_y) / smooth_factor
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Click detection (Pinch - Index & Thumb)
            distance_click = np.hypot(index_x - thumb_x, index_y - thumb_y)
            if distance_click < click_threshold:
                if active_gesture != "click" or (current_time - last_gesture_time) > gesture_cooldown:
                    pyautogui.click()
                    active_gesture = "click"
                    last_gesture_time = current_time

            # Right Click (Pinch - Middle & Thumb)
            distance_right_click = np.hypot(middle_x - thumb_x, middle_y - thumb_y)
            if distance_right_click < click_threshold:
                if active_gesture != "right_click" or (current_time - last_gesture_time) > gesture_cooldown:
                    pyautogui.rightClick()
                    active_gesture = "right_click"
                    last_gesture_time = current_time

            # Drag & Drop (Hold Pinch)
            if distance_click < click_threshold:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # Scroll (Swipe Two Fingers Up/Down)
            scroll_distance = abs(middle_y - index_y)
            if scroll_distance > 50:
                if middle_y > index_y:
                    pyautogui.scroll(10)  # Scroll Up
                else:
                    pyautogui.scroll(-10)  # Scroll Down

            # Zoom In/Out (Pinch Open/Close with Index & Thumb)
            pinch_distance = np.hypot(index_x - thumb_x, index_y - thumb_y)
            if pinch_distance > 120:
                if active_gesture != "zoom_in" or (current_time - last_gesture_time) > gesture_cooldown:
                    pyautogui.hotkey('ctrl', '+')  # Zoom In
                    active_gesture = "zoom_in"
                    last_gesture_time = current_time
            elif pinch_distance < 20:
                if active_gesture != "zoom_out" or (current_time - last_gesture_time) > gesture_cooldown:
                    pyautogui.hotkey('ctrl', '-')  # Zoom Out
                    active_gesture = "zoom_out"
                    last_gesture_time = current_time

            # Go Back/Forward (Swipe Hand Left/Right)
            hand_movement = palm_x - prev_x
            if abs(hand_movement) > 120:  # Increased threshold to prevent accidental activation
                if hand_movement > 0:
                    if active_gesture != "forward" or (current_time - last_gesture_time) > gesture_cooldown:
                        pyautogui.hotkey('alt', 'right')  # Go Forward
                        active_gesture = "forward"
                        last_gesture_time = current_time
                else:
                    if active_gesture != "back" or (current_time - last_gesture_time) > gesture_cooldown:
                        pyautogui.hotkey('alt', 'left')  # Go Back
                        active_gesture = "back"
                        last_gesture_time = current_time

            # Copy (Double Tap - Thumb & Index)
            if distance_click < click_threshold:
                if active_gesture != "copy" or (current_time - last_gesture_time) > gesture_cooldown:
                    pyautogui.hotkey('ctrl', 'c')
                    active_gesture = "copy"
                    last_gesture_time = current_time

            # Paste (Double Tap - Thumb & Middle)
            if distance_right_click < click_threshold:
                if active_gesture != "paste" or (current_time - last_gesture_time) > gesture_cooldown:
                    pyautogui.hotkey('ctrl', 'v')
                    active_gesture = "paste"
                    last_gesture_time = current_time

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Refined Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
