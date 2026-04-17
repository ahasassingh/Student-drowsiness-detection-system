import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
from detector import DrowsinessDetector

# RTC Configuration for STUN servers (needed for peer-to-peer connection in some networks)
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
        ]
    }
)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = DrowsinessDetector()
        self.frame_count = 0

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        # Only process every 2nd frame to save CPU on Streamlit Cloud
        if self.frame_count % 2 == 0:
            # Note: detector.process_frame expects a BGR image and returns a BGR image
            processed_img = self.detector.process_frame(img)
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        
        # For skipped frames, just return the original BGR translation
        # This keeps the video smooth while saving CPU cycles
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def on_ended(self):
        self.detector.release()

def main():
    st.set_page_config(page_title="Web Sleep Detection", page_icon="😴")
    st.title("Remote Sleep Detection System")
    st.markdown("""
    This app uses **WebRTC** to process your camera feed in the browser.
    It detects:
    - **EAR** (Eye Aspect Ratio) for sleep/drowsiness
    - **MAR** (Mouth Aspect Ratio) for yawning
    - **Blinks** tracking
    """)

    webrtc_ctx = webrtc_streamer(
        key="sleep-detection",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        async_processing=False,  # Changed to False for stability on Streamlit Cloud
        media_stream_constraints={
            "video": {"width": {"ideal": 320}, "height": {"ideal": 240}},
            "audio": False
        },
    )

    st.sidebar.markdown("### System Log")
    if st.button("Clear Log File"):
        if 'detector' in st.session_state:
            # This is tricky with webrtc threads, but we can clear the file
            with open("drowsiness_log.txt", "w") as f:
                f.write("=== Drowsiness Detection Log ===\n")
            st.sidebar.success("Log cleared!")

    st.sidebar.info("The system logs events to `drowsiness_log.txt` on the server.")

if __name__ == "__main__":
    main()
