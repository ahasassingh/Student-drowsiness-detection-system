import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from utils import eye_aspect_ratio, mouth_aspect_ratio, estimate_head_pose
from alert_system import AlertSystem

class DrowsinessDetector:
    def __init__(self, ear_threshold=0.25, ear_frames=20, mar_threshold=0.5, sleep_frames=50):
        # Initialize Face Landmarker using Tasks API
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.ear_frames = ear_frames
        self.sleep_frames = sleep_frames
        
        # State variables
        self.counter = 0
        self.sleep_counter = 0
        self.yawn_counter = 0
        self.total_blinks = 0
        self.state = "Awake"
        self.alert_system = AlertSystem()
        
        # Landmark indices logic (Legacy indices still apply to the 478 points mesh)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH = [13, 14, 78, 308] # Top, Bottom, Left, Right

    def extract_landmarks(self, face_landmarks, indices, frame_w, frame_h):
        # face_landmarks is a list of NormalizedLandmark objects
        return [(int(face_landmarks[i].x * frame_w), int(face_landmarks[i].y * frame_h)) for i in indices]

    def process_frame(self, frame):
        h, w, _ = frame.shape
        # Tasks API expects mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run detection
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            # Get the first face
            face_landmarks = detection_result.face_landmarks[0]
            
            # 1. Extract eye landmarks
            left_eye_points = self.extract_landmarks(face_landmarks, self.LEFT_EYE, w, h)
            right_eye_points = self.extract_landmarks(face_landmarks, self.RIGHT_EYE, w, h)
            
            # 2. Calculate EAR
            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)
            ear = (left_ear + right_ear) / 2.0
            
            # 3. Process Mouth for yawning
            mouth_points = self.extract_landmarks(face_landmarks, self.MOUTH, w, h)
            mar = mouth_aspect_ratio(mouth_points)
            
            # Draw landmarks for visualization
            for p in left_eye_points + right_eye_points + mouth_points:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)
            
            # 4. State Machine Logic
            # Check yawning
            if mar > self.mar_threshold:
                self.yawn_counter += 1
            else:
                self.yawn_counter = 0
                
            # Check EAR mapping
            if ear < self.ear_threshold:
                self.counter += 1
                self.sleep_counter += 1
                
                if self.sleep_counter >= self.sleep_frames:
                    self.state = "Sleeping"
                    self.alert_system.start_alarm()
                elif self.counter >= self.ear_frames:
                    self.state = "Drowsy"
                    self.alert_system.stop_alarm()
            else:
                if self.counter > 2 and self.counter < self.ear_frames:
                    self.total_blinks += 1 # Short closure is a blink
                self.counter = 0
                self.sleep_counter = 0
                self.state = "Awake"
                self.alert_system.stop_alarm()
            
            # UI Overlays
            color = (0, 255, 0)
            if self.state == "Drowsy":
                color = (0, 165, 255)
            elif self.state == "Sleeping":
                color = (0, 0, 255)
                
            # Add bounding box around face (approximate using eyes and mouth range)
            all_pts = left_eye_points + right_eye_points + mouth_points
            x_pts = [p[0] for p in all_pts]
            y_pts = [p[1] for p in all_pts]
            cv2.rectangle(frame, (min(x_pts)-20, min(y_pts)-40), (max(x_pts)+20, max(y_pts)+40), color, 2)
                
            cv2.putText(frame, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Blinks: {self.total_blinks}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if self.state == "Sleeping":
                cv2.putText(frame, "WAKE UP!", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        return frame

    def release(self):
        self.alert_system.stop_alarm()
