# emotion-detection
Real-time facial emotion detection using OpenCV and DeepFace
# AI Mental Health Emotion Detection

A real-time facial emotion detection app that uses a webcam feed to classify emotional states, built as a team project.

## What it does
Captures live webcam video, detects the face using OpenCV, and uses the DeepFace library to classify the detected emotion (happy, sad, angry, neutral, etc.) in real time. Wrapped in a Streamlit interface so it's usable without touching code.

## Tech Stack
- Python
- OpenCV — face detection
- DeepFace — pretrained deep learning emotion classification
- Streamlit — web interface
- NumPy

## How it works
1. Webcam feed is captured frame by frame.
2. OpenCV locates the face region in each frame.
3. DeepFace analyzes the face crop and predicts the emotion.
4. Result is displayed live on the Streamlit app.

## How to run
pip install opencv-python deepface streamlit numpy
streamlit run app.py

## My contribution
Led the core development — writing and implementing the code end-to-end, then iteratively testing and refining it against the output until the emotion detection performed reliably.

## Team
Built as a 5-member team project.
