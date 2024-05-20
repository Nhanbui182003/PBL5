import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time
import face_recognition
import pickle

mixer.init()
sound = mixer.Sound('alarm.wav')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
model = load_model(os.path.join("models", "model.keras"))

lbl=['Close', 'Open']

path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0

def authencation():
    # Set default values
    encodings_path = "models/encodings.pickle"  # Path to the serialized db of facial encodings
    detection_method = "cnn"  # Change to "hog" if you prefer

    # Load the known faces and encodings
    print("[INFO] loading encodings...")
    with open(encodings_path, "rb") as file:
        data = pickle.load(file)

    # Open a connection to the webcam
    video_capture = cv2.VideoCapture(0)

    # Capture a single frame from the webcam
    ret, frame = video_capture.read()

    # Release the webcam
    video_capture.release()
    cv2.destroyAllWindows()

    if not ret:
        print("[ERROR] Failed to capture image from webcam.")
        exit(1)

    # Convert the image from BGR to RGB color
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb, model=detection_method)
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []
    # Loop over the face encodings
    for encoding in encodings:
        # Check for matches between the known faces and the detected face encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding, 0.4)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)
   
    # Draw bounding boxes and labels on the image
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

    # Display the resulting image
    cv2.imshow("Image", frame)
    # Wait for 5 seconds or until a key is pressed
    start_time = cv2.getTickCount()
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 5:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
    return name

DRIVER_NAME = authencation()
print(DRIVER_NAME)

if (DRIVER_NAME!="Unknown"):

    while(True):
        ret, frame = cap.read()
        height,width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,minNeighbors = 3,scaleFactor = 1.1,minSize=(25,25))
        eyes = eye_cascade.detectMultiScale(gray,minNeighbors = 1,scaleFactor = 1.1)

        cv2.rectangle(frame, (0,height-50) , (500,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (255,0,0) , 3 )

        for (x,y,w,h) in eyes:

            eye = frame[y:y+h,x:x+w]
            #eye = cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
            eye = cv2.resize(eye,(80,80))
            eye = eye/255
            eye = eye.reshape(80,80,3)
            eye = np.expand_dims(eye,axis=0)
            prediction = model.predict(eye)
            # print(prediction)
        #Condition for Close
            if prediction[0][0]>0.30:
                # cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                # cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

                # Clear the text region
                cv2.rectangle(frame, (0, height - 50), (500, height), (0, 0, 0), thickness=cv2.FILLED)

                # Draw the updated text
                cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, 'Score:' + str(score) + " , ", (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, 'Driver Name:' + DRIVER_NAME, (200, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                score=score+1
                #print("Close Eyes")
                if(score > 15):
                    try:
                        sound.play()
                    except:  # isplaying = False
                        pass
            else :
                if score>0 : score -=1
            
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()