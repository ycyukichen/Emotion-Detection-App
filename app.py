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
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">😊 Real-Time Emotion Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect emotions in real-time using AI</p>', unsafe_allow_html=True)

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

# Function to verify if the detected face is human
def is_human_face(image, face_box):
    x, y, w, h = face_box
    face_roi = image[y:y+h, x:x+w]
    results = mp_face_detection.process(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
    return results.detections is not None

# Generate unique colors for detected faces
def get_random_color():
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

face_colors = {}

# ====== WEBCAM FUNCTIONALITY ======
if st.session_state.run_webcam:
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.sidebar.error("❌ Error: Could not access webcam.")
    else:
        while st.session_state.run_webcam:
            ret, frame = video_capture.read()
            if not ret:
                st.sidebar.error("❌ Error: Failed to grab frame.")
                break

            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces using Mediapipe
            results = mp_face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            human_faces = []
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
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
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

                # Display emotion text with confidence
                font_scale = 1.5
                thickness = 3
                label = f"{emotion_text} ({confidence}%)"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = x + (w - text_size[0]) // 2
                cv2.putText(frame, label, (text_x, y-20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

            # Convert to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display in Streamlit
            stframe.image(frame, channels="RGB")

        # Release webcam when "Stop" is clicked
        video_capture.release()
        cv2.destroyAllWindows()
