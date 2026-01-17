# AI-Powered Emotion Recognition System

This project implements a real-time emotion recognition pipeline using
computer vision and deep learning. Facial emotion analysis is performed
using the DeepFace framework on live webcam input.

## Features
- Real-time facial emotion detection
- Webcam-based video processing
- Emotion classification using pretrained deep learning models
- Local inference (no cloud dependency)

## Tech Stack
- Python
- OpenCV
- DeepFace
- TensorFlow

## Setup Instructions

1. Clone the repository
   ```bash
   git clone https://github.com/Avshrek/emotion-recognition-deepface.git
   
2. Install dependencies
   pip install -r requirements.txt


3. Run the application
   python src/emotion_detector.py


Note: DeepFace automatically downloads required pretrained models
on first execution. These models are not included in the repository.

Output

The system detects faces from live video input and classifies emotions
such as happy, sad, angry, neutral, and surprised in real time.


---

## 🔹 STEP 6 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Emotion recognition using DeepFace"
git branch -M main
git remote add origin https://github.com/Avshrek/emotion-recognition-deepface.git
git push -u origin main
