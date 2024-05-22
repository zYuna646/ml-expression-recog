import cv2
import numpy as np
from keras.models import load_model

model = load_model('models/fer2013_model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def preprocess_frame(frame, target_size=(48, 48)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, target_size[0], target_size[1], 1))
    return reshaped

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = preprocess_frame(frame)

    predictions = model.predict(processed_frame)
    max_index = np.argmax(predictions[0])
    predicted_emotion = emotion_labels[max_index]

    cv2.putText(frame, predicted_emotion, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Facial Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
