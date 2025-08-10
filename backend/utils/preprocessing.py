import cv2
import numpy as np
import face_recognition
from PIL import Image
import torch
from torchvision import transforms

class FaceExtractor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_faces(self, image):
        """Extract faces from image"""
        # Convert PIL to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert RGB to BGR for face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_image)
        faces = []
        
        for (top, right, bottom, left) in face_locations:
            # Extract face region
            face_image = rgb_image[top:bottom, left:right]
            
            # Convert to PIL Image
            face_pil = Image.fromarray(face_image)
            
            # Apply transforms
            face_tensor = self.transform(face_pil)
            faces.append(face_tensor)
        
        if faces:
            return torch.stack(faces)
        else:
            # If no face detected, return the whole image resized
            image_pil = Image.fromarray(rgb_image)
            return self.transform(image_pil).unsqueeze(0)
    
    def preprocess_video_frame(self, frame):
        """Preprocess a single video frame"""
        return self.extract_faces(frame)

def load_and_preprocess_image(image_path_or_file):
    """Load and preprocess image for the model"""
    if isinstance(image_path_or_file, str):
        image = Image.open(image_path_or_file)
    else:
        image = Image.open(image_path_or_file)
    
    extractor = FaceExtractor()
    return extractor.extract_faces(image)
