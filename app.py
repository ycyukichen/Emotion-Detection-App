import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
import random
from PIL import Image

# ====== STREAMLIT UI SETUP ======
st.set_page_config(page_title="Real-Time Emotion Detection", page_icon="😊", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #666;
    }
    .instructions {
        font-size: 18px;
        text-align: center;
        color: #333;
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">😊 Real-Time Emotion Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect emotions in real-time using AI</p>', unsafe_allow_html=True)
st.markdown('<p class="instructions">📸 Click "Take a Photo" and then the system will detect emotions automatically.</p>', unsafe_allow_html=True)

# ====== SIDEBAR CONTROLS ======
st.sidebar.header("🎛 Controls")
start_button = st.sidebar.button("▶ Start Webcam", use_container_width=True)
stop_button = st.sidebar.button("⏹ Stop Webcam", use_container_width=True)

# ====== SESSION STATE ======
if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False
if "emotion_classifier" not in st.session_state:
    st.session_state.emotion_classifier = None

# Load model only when "Start" is pressed
if start_button:
    st.session_state.run_webcam = True
    if st.session_state.emotion_classifier is None:
        model_path = "emotion_model_finetuned_2.keras"
        st.session_state.emotion_classifier = tf.keras.models.load_model(model_path)
        st.sidebar.success("✅ Model loaded successfully!")

if stop_button:
    st.session_state.run_webcam = False

# ====== STREAMLIT LIVE FEED ======
stframe = st.empty()

# ====== FACE DETECTION SETUP ======
mp_face_detection = mp.solutions.face_detection.FaceDetection()

# Generate unique colors for detected faces
def get_random_color():
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

face_colors = {}

# ====== WEBCAM FUNCTIONALITY ======
if st.session_state.run_webcam:
    image = st.camera_input("Take a picture")

    if image is not None:
        img = Image.open(image)
        img = np.array(img)
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces using Mediapipe
        results = mp_face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        human_faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                human_faces.append((x, y, w, h))

        for i, (x, y, w, h) in enumerate(human_faces):
            # Extract face ROI
            face_roi = gray_frame[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype("float32") / 255.0
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi = np.expand_dims(face_roi, axis=-1)

            # Predict emotion
            preds = st.session_state.emotion_classifier.predict(face_roi)
            emotion_index = np.argmax(preds)
            emotion_text = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'][emotion_index]
            confidence = round(float(np.max(preds)) * 100, 1)

            # Assign unique colors
            if i not in face_colors:
                face_colors[i] = get_random_color()
            color = face_colors[i]

            # Draw bounding box
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)

            # Display emotion text with confidence
            font_scale = 1.5
            thickness = 3
            label = f"{emotion_text} ({confidence}%)"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = x + (w - text_size[0]) // 2
            cv2.putText(img, label, (text_x, y-20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        # Convert to RGB for Streamlit
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display in Streamlit
        stframe.image(img, channels="RGB")
