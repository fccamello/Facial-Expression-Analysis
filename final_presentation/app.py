from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import mediapipe as mp
import os
import pickle

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "mp_sequential_model.h5")
encoder_path = os.path.join(BASE_DIR, "label_encoder.pkl")

model = load_model(model_path)

with open(encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def extract_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        coords = []
        for landmark in landmarks.landmark:
            coords.extend([landmark.x, landmark.y, landmark.z])
        return np.array(coords)
    else:
        return None

@app.route('/')
def index():
    return render_template('game.html')


@app.route('/predict_video', methods=['POST'])
def predict_video():
    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    landmarks = extract_landmarks(frame)

    if landmarks is None:
        return jsonify({'error': 'No face detected'})

    prediction = model.predict(np.array([landmarks]))
    predicted_class = np.argmax(prediction, axis=1)[0]
    label = label_encoder.inverse_transform([predicted_class])[0]
    confidence = float(np.max(prediction))

    return jsonify({
        'label': label,
        'confidence': f"{confidence * 100:.2f}%"
    })


if __name__ == '__main__':
    app.run(debug=True)
