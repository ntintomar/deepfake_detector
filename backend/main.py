from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import tempfile
import os
from PIL import Image
import io
import logging
from typing import Dict, Any

from models.autoencoder import DeepFakeAutoEncoder, DeepFakeDetector
from utils.preprocessing import load_and_preprocess_image
from utils.video_processing import VideoProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="DeepFake Detector API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
detector_model = None
video_processor = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global detector_model, video_processor
    
    try:
        # Initialize autoencoder
        autoencoder = DeepFakeAutoEncoder()
        
        # Initialize detector
        detector_model = DeepFakeDetector(autoencoder)
        
        # Load pre-trained weights (you'll need to train and save these)
        try:
            checkpoint = torch.load('trained_models/deepfake_detector.pth', map_location=device)
            detector_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded pre-trained model successfully")
        except FileNotFoundError:
            logger.warning("No pre-trained model found. Using randomly initialized weights.")
        
        detector_model.to(device)
        detector_model.eval()
        
        # Initialize video processor
        video_processor = VideoProcessor(detector_model, device)
        
        logger.info(f"Models loaded successfully on device: {device}")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "DeepFake Detector API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": detector_model is not None
    }

@app.post("/detect/image")
async def detect_image_deepfake(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Detect deepfake in uploaded image"""
    
    if not detector_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image
        input_tensor = load_and_preprocess_image(io.BytesIO(image_bytes))
        input_tensor = input_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            output = detector_model(input_tensor)
        
        # Extract results
        reconstruction_error = output['reconstruction_error'].cpu().numpy()
        classification_score = output['classification_score'].cpu().numpy()
        is_fake = output['is_fake'].cpu().numpy()
        
        # Calculate confidence
        confidence = float(max(reconstruction_error.mean(), classification_score.mean()))
        
        result = {
            "filename": file.filename,
            "is_deepfake": bool(is_fake.any()),
            "confidence": confidence,
            "reconstruction_error": float(reconstruction_error.mean()),
            "classification_score": float(classification_score.mean()),
            "faces_detected": len(input_tensor),
            "analysis": {
                "verdict": "FAKE" if is_fake.any() else "REAL",
                "confidence_level": "HIGH" if confidence > 0.7 else "MEDIUM" if confidence > 0.4 else "LOW"
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect/video")
async def detect_video_deepfake(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Detect deepfake in uploaded video"""
    
    if not video_processor:
        raise HTTPException(status_code=503, detail="Video processor not loaded")
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process video
            result = video_processor.process_video(tmp_file_path)
            
            # Add metadata
            result.update({
                "filename": file.filename,
                "analysis": {
                    "verdict": "FAKE" if result.get('is_deepfake', False) else "REAL",
                    "confidence_level": "HIGH" if result.get('confidence', 0) > 0.7 else "MEDIUM" if result.get('confidence', 0) > 0.4 else "LOW"
                }
            })
            
            return result
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/stats")
async def get_model_stats():
    """Get model statistics and information"""
    if not detector_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in detector_model.parameters())
    trainable_params = sum(p.numel() for p in detector_model.parameters() if p.requires_grad)
    
    return {
        "model_info": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(device),
            "architecture": "Autoencoder + Classifier"
        },
        "supported_formats": {
            "images": ["jpg", "jpeg", "png", "bmp", "tiff"],
            "videos": ["mp4", "avi", "mov", "mkv"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
