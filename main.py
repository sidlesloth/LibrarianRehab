import cv2
import mediapipe as mp
from coach import Sense
from coach import Think
from coach import Act

import numpy as np


# Main Program Loop
def main():
    """
    Main function to initialize the exercise tracking application.

    This function sets up the webcam feed, initializes the Sense, Think, and Act components,
    and starts the main loop to continuously process frames from the webcam.
    """
    
    # Initialize the components: Sense for input, Think for decision-making, Act for output
    sense = Sense.Sense()
    act = Act.Act()
    think = Think.Think(act)


    # Search and print available camera devices (may take a while to complete)
    #searchValidCameraIndexes()
    
    # Initialize the webcam capture
    cap = cv2.VideoCapture(0)  # Use the default camera (0) or change to a different index if multiple cameras are connected to system
    
    
    # Main loop to process video frames
    while cap.isOpened():
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect
        frame = cv2.resize(frame, None, fx=2, fy=2)
        if not ret:
            print("Failed to grab frame")
            break

        # Sense: Detect joints
        joints = sense.detect_joints(frame)
        landmarks = joints.hand_landmarks[0] if joints.hand_landmarks else None

        # If landmarks are detected, calculate
        if landmarks:
            # Extract joint coordinates for the left arm
            # For this example, we will use specific landmark indexes for shoulder, elbow, and wrist
            
            jointslist = []
            wrist = sense.extract_joint_coordinates(landmarks, 'wrist'); jointslist.append(wrist) #index 0 

            thumbtip = sense.extract_joint_coordinates(landmarks, 'thumb_tip'); jointslist.append(thumbtip) #index 1

            indextip = sense.extract_joint_coordinates(landmarks, 'index_finger_tip'); jointslist.append(indextip) #2
            indexmid = sense.extract_joint_coordinates(landmarks, 'index_finger_pip'); jointslist.append(indexmid) #3
            middletip = sense.extract_joint_coordinates(landmarks, 'middle_finger_tip'); jointslist.append(middletip) #4
            middlemid = sense.extract_joint_coordinates(landmarks, 'middle_finger_pip'); jointslist.append(middlemid) #5
            ringtip = sense.extract_joint_coordinates(landmarks, 'ring_finger_tip'); jointslist.append(ringtip) #6
            ringmid = sense.extract_joint_coordinates(landmarks, 'ring_finger_pip'); jointslist.append(ringmid) #7
            pinkytip = sense.extract_joint_coordinates(landmarks, 'pinky_tip'); jointslist.append(pinkytip) #8
            pinkymid = sense.extract_joint_coordinates(landmarks, 'pinky_pip'); jointslist.append(pinkymid) #9


            think.update_state(jointslist)
            decision = think.state

            # Act: Provide feedback to the user.
            act.provide_feedback(decision, frame=frame, joints=joints)
            

        # Exit if the 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def searchValidCameraIndexes():
    # checks the first 10 indexes. May take a while to complete
    
    print(f"Searching available camera index nrs")
    valid_cams = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            print(f"Warning: unable to open video source: {i}")
        else:
            print(f"Found valid video source: {i}")
            valid_cams.append(i)
            
    print(f"Available camera index nrs: {valid_cams}")

if __name__ == "__main__":
    main()