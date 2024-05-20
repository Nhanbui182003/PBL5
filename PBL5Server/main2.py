import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time
import urllib.request
import requests
import socket


mixer.init()
sound = mixer.Sound('alarm.wav')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
model = load_model(os.path.join("models", "model.keras"))

lbl = ['Close', 'Open']

url = "http://192.168.1.8/capture"

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('0.0.0.0', 8585 ))
s.listen(0)                 
                        

while True:
    try:
        # Lấy ảnh từ ESP32-CAM
        with urllib.request.urlopen(url) as response:
            img_array = np.array(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))
        eyes = eye_cascade.detectMultiScale(gray, minNeighbors=1, scaleFactor=1.1)

        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        for (x, y, w, h) in eyes:
            eye = frame[y:y + h, x:x + w]
            eye = cv2.resize(eye, (80, 80))
            eye = eye / 255
            eye = eye.reshape(80, 80, 3)
            eye = np.expand_dims(eye, axis=0)
            prediction = model.predict(eye)

            # Condition for Close
            if prediction[0][0] > 0.30:
                cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)
                cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                if score <5 :
                    score = score + 1
                if score == 5:
                    client, addr = s.accept()
                    client.send(b'1')
                    client.close()   
                        
            else:
                if score > 0:
                    score -= 1

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print("Lỗi:", e)
        break

cv2.destroyAllWindows()
