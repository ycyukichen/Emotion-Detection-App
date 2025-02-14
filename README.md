# 😊 Real-Time Emotion Detection App
🚀 **AI-powered real-time emotion recognition using OpenCV, TensorFlow, and Streamlit**.  
Detects **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral** emotions from a live webcam feed.


---

## 🎯 **Features**
✅ **Real-time Emotion Detection** using Deep Learning  
✅ **Smooth UI with Streamlit**  
✅ **Supports Multiple Faces with Unique Colors**  
✅ **Confidence Score for Each Emotion**  
✅ **Works on Web, Desktop, and Cloud Platforms**  

---

## 🚀 **Live Demo**
🔴 Try it now: **[Real-Time Emotion Detection](https://realtime-emotion-detection-app.streamlit.app/)**  

---

## 🛠 **Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/ycyukichen/Emotion-Detection-App.git
cd Emotion-Detection-App
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the App**
```bash
streamlit run app.py
```
🚀 The app will open in your browser at `http://localhost:8501/`.


---


## 📌 **App Versions**
- `app.py` → Uses **Mediapipe** for face detection and takes a **photo for emotion recognition**. Optimized for **Streamlit Cloud Deployment**.  
- `app1.py` → Uses **face-recognition** (with `dlib`) and **streams video for live emotion detection**. This version is **only for local use** due to deployment issues with `dlib` on Streamlit Cloud.


---


## ⚙️ **Requirements**
- Python 3.8+
- OpenCV
- TensorFlow
- Streamlit
- Face Recognition / Mediapipe (depending on the version used)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🏗 **Project Structure**
```
📂 Emotion-Detection-App
│── 📜 app.py          # Main application script using Mediapipe (photo-based detection)
│── 📜 app1.py         # Alternative script using Face-Recognition (video streaming detection)
│── 📜 requirements.txt # Dependencies
│── 📜 README.md        # Project documentation
│── 📂 models           # Pre-trained model for emotion detection
```

---

## 💡 **How It Works**
1. **Face Detection** → OpenCV detects faces.
2. **Emotion Prediction** → TensorFlow classifies emotions.
3. **Bounding Boxes & Labels** → Faces are marked with different colors.
4. **Confidence Score** → Displays prediction accuracy.

---

## 🎯 **Next Steps**
🔹 Deploy on **Hugging Face Spaces**  
🔹 Improve Face Detection using **Mediapipe**  
🔹 Optimize for **Mobile & Edge Devices**  

---

## 📜 **License**
📄 This project is **MIT Licensed**.

