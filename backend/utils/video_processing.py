import cv2
import numpy as np
import torch
from .preprocessing import FaceExtractor

class VideoProcessor:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.face_extractor = FaceExtractor()
        
    def process_video(self, video_path, max_frames=30):
        """Process video and return detection results"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // max_frames)
        
        results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                try:
                    # Extract faces from frame
                    faces = self.face_extractor.extract_faces(frame)
                    faces = faces.to(self.device)
                    
                    # Get prediction
                    with torch.no_grad():
                        output = self.model(faces)
                    
                    # Store results
                    frame_result = {
                        'frame_number': frame_count,
                        'reconstruction_error': output['reconstruction_error'].cpu().numpy(),
                        'classification_score': output['classification_score'].cpu().numpy(),
                        'is_fake': output['is_fake'].cpu().numpy()
                    }
                    results.append(frame_result)
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    
            frame_count += 1
            
        cap.release()
        
        # Aggregate results
        if results:
            avg_reconstruction_error = np.mean([r['reconstruction_error'].mean() for r in results])
            avg_classification_score = np.mean([r['classification_score'].mean() for r in results])
            fake_frame_ratio = np.mean([r['is_fake'].any() for r in results])
            
            return {
                'total_frames_analyzed': len(results),
                'avg_reconstruction_error': float(avg_reconstruction_error),
                'avg_classification_score': float(avg_classification_score),
                'fake_frame_ratio': float(fake_frame_ratio),
                'is_deepfake': fake_frame_ratio > 0.3,  # Threshold for video classification
                'confidence': max(avg_classification_score, fake_frame_ratio)
            }
        else:
            return {
                'error': 'No frames could be processed',
                'is_deepfake': False,
                'confidence': 0.0
            }
