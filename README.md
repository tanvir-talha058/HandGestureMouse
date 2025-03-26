
## **🖐 Hand Gesture-Based Virtual Mouse**  
### **Control Your PC Like Tony Stark! 🚀**  

This project uses **computer vision and hand tracking** to turn your hand into a virtual mouse, allowing you to control your entire PC with hand gestures—just like in Iron Man! 🦾  


---

## **✨ Features**  
✅ **Move Cursor** – Control the mouse by moving your hand  
✅ **Left Click** – Pinch thumb & index finger  
✅ **Right Click** – Pinch thumb & middle finger  
✅ **Drag & Drop** – Hold pinch gesture  
✅ **Scroll** – Move two fingers up/down  
✅ **Zoom In/Out** – Open/close pinch gesture  
✅ **Go Back & Forward** – Swipe left/right  
✅ **Copy & Paste** – Double-tap gestures  
✅ **Enhanced Accuracy** – Smooth movement and reduced misclicks  

---

## **📌 How It Works**  

This project uses **MediaPipe Hand Tracking** and **OpenCV** to detect your hand gestures. Based on the distance and position of your fingers, it sends keyboard and mouse commands using **PyAutoGUI**.  

### **📷 Hand Landmarks Used:**  
- **Index Finger (8)** – Cursor Control  
- **Thumb (4)** – Clicks & Gestures  
- **Middle Finger (12)** – Right Click  
- **Palm Center (0)** – Navigation  

---

## **⚡ Installation**  

1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/tanvir-talha058/HandGestureMouse.git


2️⃣ **Install Dependencies**  
```bash
pip install opencv-python mediapipe pyautogui numpy
```

3️⃣ **Run the Program**  
```bash
python virtual_mouse.py
```

---

## **🖥️ Controls & Gestures**  

| Gesture | Action |
|---------|--------|
| Move Hand | Move Cursor |
| Pinch (Thumb + Index) | Left Click |
| Pinch (Thumb + Middle) | Right Click |
| Hold Pinch | Drag & Drop |
| Two-Finger Scroll | Scroll Up/Down |
| Pinch Open | Zoom In |
| Pinch Close | Zoom Out |
| Swipe Left | Go Back |
| Swipe Right | Go Forward |
| Double Tap (Index + Thumb) | Copy |
| Double Tap (Middle + Thumb) | Paste |

---


## **📝 Credits**  
Made with ❤️ by **[Tanvir Ahmed](https://github.com/tanvir-talha058)**  

🚀 **If you like this project, don't forget to ⭐ the repo!**  

---
