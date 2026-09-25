import cv2
import mediapipe as mp
import math
import numpy as np
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Sense Component: Detect joints using the camera
# Things you need to improve: Make the skeleton tracking smoother and robust to errors.
class Sense:

    def __init__(self, session, model_path=None):
        if session == 'ex1':
            model_path = Path(model_path or Path(__file__).parents[1] / 'models' / 'hand_landmarker.task')
            if not model_path.is_file():
                raise FileNotFoundError(
                    f'MediaPipe hand model not found at {model_path}. '
                    'Restore models/hand_landmarker.task before running the application.'
                )

            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mp_pose = vision.HandLandmarker.create_from_options(options)
            self.timestamp_ms = 0

        else:
            model_path = Path(model_path or Path(__file__).parents[1] / 'models' / 'pose_landmarker_lite.task')
            if not model_path.is_file():
                raise FileNotFoundError(
                    f'MediaPipe hand model not found at {model_path}. '
                    'Restore models/pose_landmarker_lite.task before running the application.'
                )

            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mp_pose = vision.PoseLandmarker.create_from_options(options)
            self.timestamp_ms = 0
            

        # used later for having a moving avergage
        self.angle_window = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.previous_angle = -1

    def detect_joints(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.timestamp_ms += 1
        return self.mp_pose.detect_for_video(image, self.timestamp_ms)

    def extract_joint_coordinates(self, session, landmarks, joint):
        """
        Extracts the (x, y) coordinates of a specific joint.

        Parameters:
        - landmarks: The list of pose landmarks from MediaPipe
        - joint: The name of the joint (e.g., 'left_elbow')

        Returns:
        - A tuple of (x, y) coordinates of the specified joint
        """

        
        if session == 'ex1':
            joint_index_map = {
                'wrist': 0,
                'thumb_cmc': 1,
                'thumb_mcp': 2,
                'thumb_ip': 3,
                'thumb_tip': 4,
                'index_finger_mcp': 5,
                'index_finger_pip': 6,
                'index_finger_dip': 7,
                'index_finger_tip': 8,
                'middle_finger_mcp': 9,
                'middle_finger_pip': 10,
                'middle_finger_dip': 11,
                'middle_finger_tip': 12,
                'ring_finger_mcp': 13,
                'ring_finger_pip': 14,
                'ring_finger_dip': 15,
                'ring_finger_tip': 16,
                'pinky_mcp': 17,
                'pinky_pip': 18,
                'pinky_dip': 19,
                'pinky_tip': 20
                }
        elif session == 'ex2':
            joint_index_map = {
                'left_shoulder': vision.PoseLandmark.LEFT_SHOULDER,
                'right_shoulder': vision.PoseLandmark.RIGHT_SHOULDER,
                'left_elbow': vision.PoseLandmark.LEFT_ELBOW,
                'right_elbow': vision.PoseLandmark.RIGHT_ELBOW,
                'left_wrist': vision.PoseLandmark.LEFT_WRIST,
                'right_wrist': vision.PoseLandmark.RIGHT_WRIST,
                'left_hip': vision.PoseLandmark.LEFT_HIP,
                'right_hip': vision.PoseLandmark.RIGHT_HIP,
                'left_knee': vision.PoseLandmark.LEFT_KNEE,
                'right_knee': vision.PoseLandmark.RIGHT_KNEE,
                'left_ankle': vision.PoseLandmark.LEFT_ANKLE,
                'right_ankle': vision.PoseLandmark.RIGHT_ANKLE
            }

        landmark = landmarks[joint_index_map[joint]]

        return landmark.x, landmark.y
