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

    def __init__(self, model_path=None):
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

        # used later for having a moving avergage
        self.angle_window = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.previous_angle = -1

    def detect_joints(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.timestamp_ms += 1
        return self.mp_pose.detect_for_video(image, self.timestamp_ms)

    def calculate_angle(self, joint1, joint2, joint3):
        """
        Calculates the angle between three joints.

        Parameters:
        - joint1: Tuple of (x, y) for the first joint (e.g., shoulder)
        - joint2: Tuple of (x, y) for the middle joint (e.g., elbow)
        - joint3: Tuple of (x, y) for the last joint (e.g., wrist)

        Returns:
        - Angle in degrees between the three joints
        """
        # Calculate vectors
        vector1 = [joint1[0] - joint2[0], joint1[1] - joint2[1]]
        vector2 = [joint3[0] - joint2[0], joint3[1] - joint2[1]]

        # Calculate the dot product and magnitude of the vectors
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        # Calculate the angle in radians and convert to degrees
        angle = math.acos(dot_product / (magnitude1 * magnitude2))

        # get a moving average
        self.angle_window.pop(0)
        self.angle_window.append(angle)
        # Use np.convolve to calculate the moving average
        window_size = 10
        angle_mvg = np.convolve(np.asarray(self.angle_window), np.ones(window_size) / window_size, mode='valid')

        return math.degrees(float(angle_mvg[0]))

    def extract_joint_coordinates(self, landmarks, joint):
        """
        Extracts the (x, y) coordinates of a specific joint.

        Parameters:
        - landmarks: The list of pose landmarks from MediaPipe
        - joint: The name of the joint (e.g., 'left_elbow')

        Returns:
        - A tuple of (x, y) coordinates of the specified joint
        """
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

        landmark = landmarks[joint_index_map[joint]]

        return landmark.x, landmark.y
