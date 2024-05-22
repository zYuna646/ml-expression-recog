import cv2
import os
import numpy as np
from keras.models import load_model

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.abspath(os.path.join(current_dir, '../../models'))

model = load_model(os.path.join(model_dir, 'fer2013_model.h5'))

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def preprocess_frame(frame, target_size=(48, 48)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, target_size[0], target_size[1], 1))
    return reshaped

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        processed_frame = preprocess_frame(face_roi)
        predictions = model.predict(processed_frame)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotion_labels[max_index]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Facial Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
