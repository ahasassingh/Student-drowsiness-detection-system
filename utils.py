import math
import numpy as np
import cv2

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two 2D points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def eye_aspect_ratio(eye_landmarks):
    """
    Calculate the Eye Aspect Ratio (EAR) given an array of 6 eye landmarks.
    Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    # Vertical distances
    A = calculate_distance(eye_landmarks[1], eye_landmarks[5])
    B = calculate_distance(eye_landmarks[2], eye_landmarks[4])
    # Horizontal distance
    C = calculate_distance(eye_landmarks[0], eye_landmarks[3])
    
    if C == 0:
        return 0.0
    
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth_landmarks):
    """
    Calculate the Mouth Aspect Ratio (MAR) to detect yawning.
    Given top, bottom, left, and right lips landmarks.
    MAR = ||top - bottom|| / ||left - right||
    """
    top_lip = mouth_landmarks[0]
    bottom_lip = mouth_landmarks[1]
    left_lip = mouth_landmarks[2]
    right_lip = mouth_landmarks[3]
    
    vertical_dist = calculate_distance(top_lip, bottom_lip)
    horizontal_dist = calculate_distance(left_lip, right_lip)
    
    if horizontal_dist == 0:
        return 0.0
    
    return vertical_dist / horizontal_dist

def estimate_head_pose(face_landmarks, image_width, image_height, camera_matrix, dist_coeffs):
    """
    Estimate head pose using PnP.
    Returns Pitch, Yaw, Roll.
    """
    # 3D model points of standard human face
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    
    # 2D image points from landmarks
    # Indices in MediaPipe face mesh (approx):
    # Nose tip: 1, Chin: 152, Left eye left: 33, Right eye right: 263
    # Left mouth: 61, Right mouth: 291
    image_points = np.array([
        face_landmarks[1],    # Nose tip
        face_landmarks[152],  # Chin
        face_landmarks[33],   # Left eye left corner
        face_landmarks[263],  # Right eye right corner
        face_landmarks[61],   # Left mouth corner
        face_landmarks[291]   # Right mouth corner
    ], dtype="double")
    
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
    # Not implementing full Euler angle conversion here to keep it simple, 
    # but could be added based on rotation_vector
    return rotation_vector, translation_vector
    
