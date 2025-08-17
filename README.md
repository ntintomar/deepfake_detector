DeepFake Detector 🔍
A lightweight FastAPI backend with a Streamlit frontend for detecting deepfakes in images and videos using an Autoencoder reconstruction signal fused with a classifier score. Supports GPU acceleration when available and provides an interactive UI plus clean REST endpoints.

What It Does
Detects deepfakes in single images and videos

Returns verdict (REAL/FAKE), confidence, and detailed metrics

Interactive UI to upload media and view results

Endpoints for health checks and model statistics

Tech Stack
Backend: FastAPI, PyTorch, OpenCV

Frontend: Streamlit

Model: Autoencoder + Classifier fusion

Optional CUDA GPU support

API Endpoints
GET /health — service status, device, model load state

GET /stats — parameters, device, supported formats

POST /detect/image — multipart image upload (jpg, jpeg, png, bmp, tiff)

POST /detect/video — multipart video upload (mp4, avi, mov, mkv)
pip install -r requirements.txt
# optional weights: trained_models/deepfake_detector.pth

# run backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# run frontend (in another terminal)
streamlit run streamlit_app.py
How It Works
Face detection and preprocessing

Autoencoder reconstruction → anomaly score (error)

Classifier → deepfake probability

Fusion → final verdict + confidence (HIGH >0.70, MEDIUM 0.40–0.70, LOW ≤0.40)
Notes
Some deps (dlib/face-recognition) may require CMake/build tools

Large videos take longer; tune frame sampling in VideoProcessor

CORS is open for development; restrict and add auth in production

Roadmap
Batch analysis

Training/evaluation scripts

Explainability overlays (heatmaps)

Improved fusion/calibration
