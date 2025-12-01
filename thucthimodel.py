import pandas as pd
import joblib #thu vien load model
import cv2 #thu vien opencv
import mediapipe as mp #thu vien ve khung xuong
import pyttsx3 #thu vien giong doc
import threading #thu vien chay luong rieng
import pythoncom #thu vien de window ho tro

load_model = joblib.load('D:\\AI_DataScience\\duantest\\deep_learnig.joblib')

cap=cv2.VideoCapture(0)
#khoi tao ve khung xuong
mp_hands=mp.solutions.hands
mp_drawings=mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    
)
#khoi tao giong doc
last_label = ""
def speak_func(text):
    try:
#tao khai bao luong 
        pythoncom.CoInitialize()
#khai bao engine
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
#giai phong luong
        pythoncom.CoUninitialize()
    except Exception as e:
        print("Lỗi giọng nói:", e)


display_text=''
#vong lap mo camara
while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
#doi bgr thanh rgb
    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#gan ket qua xu ly duoc khi lay hinh grb vao result(khung xuong)
    result=hands.process(frame_rgb)
#ve khung xuong
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawings.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
#tao du lieu arr
        row = []
        for lm in hand_landmarks.landmark:
           row.extend([lm.x, lm.y, lm.z])

 #chay model du doan       
        y_val=load_model.predict([row])
#lay ket qua du doan
        label_name=y_val[0]
#giong doc
        if label_name != last_label:
            t = threading.Thread(target=speak_func, args=(label_name,))
            t.start()
            last_label = label_name

#ve khung tren camera
        cv2.rectangle(frame, (0, 0), (300, 60), (0, 0, 0), -1) 
        cv2.putText(frame, f"Chu: {label_name}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#show camera
    cv2.imshow('Webcam',frame)

#de thoat camera
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
#giai phong 
cap.release()
cv2.destroyAllWindows()
#tranh do vscode
for i in range(1, 5):
    cv2.waitKey(1)