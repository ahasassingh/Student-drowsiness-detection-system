import streamlit as st
import cv2
import numpy as np
from detector import DrowsinessDetector

st.set_page_config(page_title="Drowsiness Detection", page_icon="😴")

st.title("Real-Time Drowsiness Detection System")
st.markdown("This system uses your webcam and MediaPipe Face Mesh to detect signs of drowsiness and sleep.")

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

if run:
    # Use session state to cache the detector natively and persist detection states across redraws
    if 'detector' not in st.session_state:
        st.session_state.detector = DrowsinessDetector()
        
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture stream from webcam.")
            break
            
        frame = cv2.flip(frame, 1)
        
        # Process frame
        processed_frame = st.session_state.detector.process_frame(frame)
        
        # Display frame
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(processed_frame)

    camera.release()
    
if not run and 'detector' in st.session_state:
    st.session_state.detector.release()
    del st.session_state['detector']
elif not run:
    st.write("Click 'Start Webcam' to begin tracking.")
