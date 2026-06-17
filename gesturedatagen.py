import mediapipe as mp
from mediapipe import solutions
import cv2
import numpy as np
import csv
import threading
import queue
from pynput import keyboard



mp_hands = solutions.hands
mp_drawing = mp.solutions.drawing_utils
datapath = ("data\mock_gesture_data.csv")
saverowqueue = queue.Queue()

capture = cv2.VideoCapture(1)
key_map = {
    ord('1'): 'point',
    ord('2'): 'point_up',
    ord('3'): 'point_down',
    ord('4'): 'fist',
    ord('5'): 'about_to_pinch',
    ord('6'): 'pinch',
    ord('7'): 'full_pinch',
    ord('8'): 'open_hand',
    ord('9'): 'closed_hand',
    ord('0'): 'thumbs_up',
}
counts = {g: 0 for g in key_map.values()}

current_key = None
last_key = None

def on_press(k):
    global current_key, last_key
    try:
        if k.char != last_key:  # only update if its a new key
            current_key = ord(k.char)
            last_key = k.char
    except:
        pass

def on_release(k):
    global current_key, last_key
    current_key = None
    last_key = None

keyboard.Listener(on_press=on_press, on_release=on_release).start()

def saverow():
    with open(datapath, 'a', newline='') as f:
        writer = csv.writer(f)
        while True:
            row = saverowqueue.get()
            if row is None: 
                break
            writer.writerow(row)
            f.flush()

def draw_hud(frame, current_label=None):
    hud = frame.copy()
    cv2.rectangle(hud, (0, 0), (250, 320), (0, 0, 0), -1)
    cv2.addWeighted(hud, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, "HOLD KEY TO RECORD", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for i, (key, label) in enumerate(key_map.items()):
        colour = (0, 255, 0) if label == current_label else (200, 200, 200)
        text = f"{chr(key)}: {label} ({counts[label]})"
        cv2.putText(frame, text, (10, 55 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    cv2.putText(frame, "Q: quit", (10, 310),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)


saverowthread = threading.Thread(target=saverow, daemon=True)
saverowthread.start()

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence =0.5
) as hands:


        while capture.isOpened():
            ret, frame = capture.read()
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            detected_image = hands.process(image)
            cv2.waitKey(1) 
            key = current_key
            current_label = key_map.get(key)
            if detected_image.multi_hand_landmarks: # each hand
                for hand in detected_image.multi_hand_landmarks: # for each landmarkqa
                    
                    for landmark in hand.landmark:
                        h, w , _ =  frame.shape
                        x = int(landmark.x*w)
                        y = int(landmark.y*h)
                        cv2.circle(frame,(x,y), 5,(0,255,0),-1)

                    if key in key_map:
                        label = key_map[key]
                        raw = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand.landmark])
                        raw = raw - raw[0]
                        row = [label] + raw.flatten().tolist()
                        saverowqueue.put(row)
                        counts[label] +=1
                        print(f"Captured: {label}")

            draw_hud(frame, current_label)
            cv2.imshow('Webcam', frame)
    
            if key == ord('q'):
                break

capture.release()
cv2.destroyAllWindows()



