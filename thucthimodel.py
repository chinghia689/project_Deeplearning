import pandas as pd
import joblib
import cv2
import mediapipe as mp

load_model = joblib.load('D:\\AI_DataScience\\duantest\\deep_learnig.joblib')

cap=cv2.VideoCapture(0)

mp_hands=mp.solutions.hands
mp_drawings=mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    
)
display_text=''
while True:
    ret,frame=cap.read()
    if not ret:
        print("Không nhận diện được Camera hoặc Camera đang bị chiếm dụng!")
        # Thử bỏ qua vòng lặp này để chờ camera, hoặc break để thoát
        continue
    frame=cv2.flip(frame,1)
    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawings.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

        row = []
        for lm in hand_landmarks.landmark:
           row.extend([lm.x, lm.y, lm.z])

        
        y_val=load_model.predict([row])

        label_name=y_val[0]
        display_text = label_name

        cv2.rectangle(frame, (0, 0), (300, 60), (0, 0, 0), -1) 
        cv2.putText(frame, f"Chu: {label_name}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Webcam',frame)


    key=cv2.waitKey(1)
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
for i in range(1, 5):
    cv2.waitKey(1)