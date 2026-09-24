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

    def provide_feedback(self, decision, frame, joints):
        """
        Displays the skeleton and some text using open cve.

        :param decision: The decision in which state the user is from the think component.
        :param frame: The currently processed frame form the webcam.
        :param joints: The joints extracted from mediapipe from the current frame.
        :param elbow_angle_mvg: The moving average from the left elbow angle.

        """

        if joints.hand_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                joints.hand_landmarks[0],
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
            )


        # Set the position, font, size, color, and thickness for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = .9
        font_color = (0, 0, 0)  # White color in BGR
        thickness = 2

        # Define the position for the number and text
        text_position = (50, 50)

        # Draw the text on the image
        cv2.putText(frame, "myre", text_position, font, font_scale, font_color, thickness)

        # Display the frame (for debugging purposes)
        cv2.imshow('Sport Coaching Program', frame)
