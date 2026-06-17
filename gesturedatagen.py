import mediapipe as mp
from mediapipe import solutions
import cv2
import numpy as np
import csv

mp_hands = solutions.hands
mp_drawing = mp.solutions.drawing_utils
datapath = ("data\mock_gesture_data.csv")

capture = cv2.VideoCapture(0)
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

        key =  cv2.waitKey(1) & 0xFF


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
                    with open(datapath, 'a', newline='') as f:
                        csv.writer(f).writerow(row)
                    print(f"Saved: {label}")


        cv2.imshow('Webcam', frame)
  
        if key == ord('q'):
          break

capture.release()
cv2.destroyAllWindows()

