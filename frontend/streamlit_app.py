import streamlit as st
import requests
import json
from PIL import Image
import io
import time

# Configure Streamlit page
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your FastAPI server URL

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_and_analyze_image(uploaded_file):
    """Upload image to API and get analysis"""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    
    with st.spinner("Analyzing image... This may take a few moments."):
        try:
            response = requests.post(f"{API_BASE_URL}/detect/image", files=files, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.Timeout:
            st.error("Request timed out. The image might be too large or the server is busy.")
            return None
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None

def upload_and_analyze_video(uploaded_file):
    """Upload video to API and get analysis"""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    
    with st.spinner("Analyzing video... This may take several minutes for large files."):
        try:
            response = requests.post(f"{API_BASE_URL}/detect/video", files=files, timeout=300)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.Timeout:
            st.error("Request timed out. The video might be too large or processing is taking too long.")
            return None
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None

def display_results(results, media_type="image"):
    """Display analysis results"""
    
    # Main verdict
    if results.get('is_deepfake', False):
        st.error("🚨 **DEEPFAKE DETECTED**")
        verdict_color = "red"
    else:
        st.success("✅ **AUTHENTIC MEDIA**")
        verdict_color = "green"
    
    # Confidence and metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence = results.get('confidence', 0)
        st.metric(
            label="Confidence",
            value=f"{confidence:.2%}",
            help="Higher values indicate stronger evidence"
        )
    
    with col2:
        if media_type == "image":
            faces_count = results.get('faces_detected', 0)
            st.metric(
                label="Faces Detected",
                value=str(faces_count),
                help="Number of faces found in the media"
            )
        else:
            frames_analyzed = results.get('total_frames_analyzed', 0)
            st.metric(
                label="Frames Analyzed",
                value=str(frames_analyzed),
                help="Number of video frames processed"
            )
    
    with col3:
        confidence_level = results.get('analysis', {}).get('confidence_level', 'UNKNOWN')
        st.metric(
            label="Confidence Level",
            value=confidence_level,
            help="HIGH: >70%, MEDIUM: 40-70%, LOW: <40%"
        )
    
    # Detailed metrics
    st.subheader("📊 Detailed Analysis")
    
    if media_type == "image":
        metrics_data = {
            "Reconstruction Error": f"{results.get('reconstruction_error', 0):.4f}",
            "Classification Score": f"{results.get('classification_score', 0):.4f}",
            "Final Verdict": results.get('analysis', {}).get('verdict', 'UNKNOWN')
        }
    else:
        metrics_data = {
            "Avg Reconstruction Error": f"{results.get('avg_reconstruction_error', 0):.4f}",
            "Avg Classification Score": f"{results.get('avg_classification_score', 0):.4f}",
            "Fake Frame Ratio": f"{results.get('fake_frame_ratio', 0):.2%}",
            "Final Verdict": results.get('analysis', {}).get('verdict', 'UNKNOWN')
        }
    
    # Display metrics in an expandable section
    with st.expander("View Technical Details"):
        for key, value in metrics_data.items():
            st.write(f"**{key}:** {value}")
        
        # Raw JSON data
        st.write("**Raw API Response:**")
        st.json(results)

def get_model_stats():
    """Get model statistics from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🔍 DeepFake Detector")
    st.markdown("""
    This application uses advanced autoencoder neural networks to detect manipulated images and videos.
    Upload your media files to analyze their authenticity.
    """)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ **Backend API is not running!**")
        st.markdown("""
        Please make sure the FastAPI backend is running on `http://localhost:8000`.
        
        To start the backend:
        ```
        cd backend
        uvicorn main:app --reload --host 0.0.0.0 --port 8000
        ```
        """)
        return
    else:
        st.success("✅ Backend API is running")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ Information")
        
        # Model stats
        model_stats = get_model_stats()
        if model_stats:
            st.subheader("🤖 Model Info")
            model_info = model_stats.get('model_info', {})
            st.write(f"**Architecture:** {model_info.get('architecture', 'N/A')}")
            st.write(f"**Parameters:** {model_info.get('total_parameters', 'N/A'):,}")
            st.write(f"**Device:** {model_info.get('device', 'N/A')}")
        
        st.subheader("📝 How it works")
        st.markdown("""
        1. **Face Detection**: Extracts faces from media
        2. **Autoencoder Analysis**: Reconstructs faces and measures errors
        3. **Classification**: Uses neural networks for final prediction
        4. **Verdict**: Combines multiple signals for final decision
        """)
        
        st.subheader("⚠️ Limitations")
        st.markdown("""
        - Works best with clear facial images
        - May have false positives/negatives
        - Performance depends on training data
        - Large files may take time to process
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🎥 Video Analysis", "📊 Batch Analysis"])
    
    with tab1:
        st.header("Image DeepFake Detection")
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image to analyze for deepfake content"
        )
        
        if uploaded_image is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 2])
            
            with col1:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Image info
                st.write(f"**Filename:** {uploaded_image.name}")
                st.write(f"**Size:** {len(uploaded_image.getvalue())} bytes")
                st.write(f"**Dimensions:** {image.size}")
            
            with col2:
                if st.button("🔍 Analyze Image", type="primary"):
                    results = upload_and_analyze_image(uploaded_image)
                    if results:
                        display_results(results, "image")
    
    with tab2:
        st.header("Video DeepFake Detection")
        
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video to analyze for deepfake content"
        )
        
        if uploaded_video is not None:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display video info
                st.video(uploaded_video)
                st.write(f"**Filename:** {uploaded_video.name}")
                st.write(f"**Size:** {len(uploaded_video.getvalue())} bytes")
            
            with col2:
                st.warning("⚠️ Video analysis may take several minutes depending on file size.")
                
                if st.button("🔍 Analyze Video", type="primary"):
                    results = upload_and_analyze_video(uploaded_video)
                    if results:
                        display_results(results, "video")
    
    with tab3:
        st.header("Batch Analysis")
        st.markdown("*Coming Soon: Upload multiple files for batch processing*")
        
        uploaded_files = st.file_uploader(
            "Choose multiple files",
            type=['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'],
            accept_multiple_files=True,
            help="Upload multiple files for batch analysis"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files")
            
            for i, file in enumerate(uploaded_files):
                st.write(f"{i+1}. {file.name} ({len(file.getvalue())} bytes)")
            
            if st.button("🔍 Analyze All Files", type="primary"):
                st.info("Batch processing feature coming soon!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🔍 <strong>DeepFake Detector</strong> | Built with FastAPI & Streamlit | 
        <a href='https://github.com' target='_blank'>View Source Code</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
