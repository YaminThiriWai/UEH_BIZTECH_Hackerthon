import cv2
import numpy as np
from keras.models import load_model
import random
import time

# Load model and face detector
model = load_model('model_file_30epochs.h5')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion labels
labels_dict = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 
    3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'
}

# Friendly responses
responses = {
    "Angry": ["Whoa! Summoning a dragon? 🐉🔥", "Lotus tea break? 🍵🌸"],
    "Disgust": ["Shrimp paste face! 😖🦐", "First time with durian? 😅"],
    "Fear": ["Saw a village ghost? 👻🌕", "Don't worry, I got you! 🧵🛡️"],
    "Happy": ["Tết vibes! 🌼🥳", "Lucky money smile! 💌💰"],
    "Neutral": ["Lotus pond calm 🌿🍃", "Waiting backstage 🧍‍♂️💧"],
    "Sad": ["Rainy Hanoi feels ☔💔", "I'll dance for you 🎭🧵"],
    "Surprise": ["Water puppet shock! 😲💦", "Free rice cake?! 🎁🥮"]
}

# State to manage response updates
last_update_time = 0
current_response = ""
last_emotion = ""  # Track the last detected emotion

# Start webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped, verbose=0)
        label = np.argmax(result, axis=1)[0]
        emotion = labels_dict[label]

        # Update response when emotion changes OR every 30 seconds
        current_time = time.time()
        should_update = (emotion != last_emotion) or (current_time - last_update_time > 30)
        
        if should_update:
            current_response = random.choice(responses.get(emotion, ["I'm not sure how you're feeling 🤖"]))
            last_update_time = current_time
            last_emotion = emotion

        # Draw bounding box and emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-60), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, emotion, (x + 5, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, current_response, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show video frame
    cv2.imshow("Facial Emotion Recognition", frame)

    # Break loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()