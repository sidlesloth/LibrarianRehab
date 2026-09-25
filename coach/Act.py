# Act Component: Provide feedback to the user

import cv2
import numpy as np
import random
import pyttsx3
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils


# Act Component: Visualization to motivate user, visualization such as the skeleton and debugging information.
# Things to add: Other graphical visualization, a proper GUI, more verbal feedback
class Act:

    def __init__(self):
        self.engine = pyttsx3.init()

        self.motivating_utterances = ['keep on going', 'you are doing great. I see it', 'only a few left', 'that is awesome', 'you have almost finished the exercise']
        # Handles balloon inflation and reset after explosion

    def provide_feedback(self, session, decision, frame, joints):
        """
        Displays the skeleton and some text using open cve.

        :param decision: The decision in which state the user is from the think component.
        :param frame: The currently processed frame form the webcam.
        :param joints: The joints extracted from mediapipe from the current frame.

        """

        if (session =='ex1'):
            drawing_utils.draw_landmarks(
                frame,
                joints.hand_landmarks[0],
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
            )

        elif session=='ex2':
            drawing_utils.draw_landmarks(
                frame,
                joints.pose_landmarks[0],
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
                drawing_styles.get_default_pose_landmarks_style(),
            )

        if session == 'ex1':    
            if decision == "squish":
                text = "Great! Squish as much as you can, then release"
            elif decision == "stretch":
                text = "Fantastic! Keep Stretching!"
        
        elif session == 'ex2':
            if decision == "squish":
                text = "Great! Squish as much as you can, then release"
            elif decision == "stretch":
                text = "Keep Stretching!"
        
        elif session == 'end':
            text = 'Great job today! Press X to exit'

        self.display_text(frame, text)

        
    def display_text(self, frame, text):
        # Set the position, font, size, color, and thickness for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = .9
        font_color = (0, 0, 0)  # White color in BGR
        thickness = 2
        # Define the position for the number and text
        text_position = (50, 50)

        # Draw the text on the image
        cv2.putText(frame, text, text_position, font, font_scale, font_color, thickness)
        
