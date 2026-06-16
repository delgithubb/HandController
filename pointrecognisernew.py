import mediapipe as mp
import time
import cv2

capture = cv2.VideoCapture(0)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class HandTracker:
    def __init__(self):
        self.frame=None
        self.annotated_frame =None

    def on_result(self, detected_image, output_image,timestamp_ms):
        if self.frame is None:
            return
        self.annotated_frame = self.frame.copy() 
        if detected_image.hand_landmarks  :
                for hand in detected_image.hand_landmarks:
                    for point in hand:
                        print('point')
                        h, w , _ =  self.frame.shape
                        x = int(point.x*w)
                        y = int(point.y*h)
                        cv2.circle(self.frame,(x,y), 5,(0,255,0),-1)

handTracker = HandTracker()
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='.\landmarkermodel\hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence =0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
    result_callback=handTracker.on_result)




with HandLandmarker.create_from_options(options) as landmarker:
    while capture.isOpened():
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        handTracker.frame = frame

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        # show annotated if available, else raw frame
        display = handTracker.annotated_frame if handTracker.annotated_frame is not None else frame
        cv2.imshow('Webcam', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

capture.release()
cv2.destroyAllWindows()

