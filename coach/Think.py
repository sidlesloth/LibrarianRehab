from transitions import Machine
import math
import time


class Think(object):
    session = ['start', 'ex1', 'break', 'ex2', 'end']

    def __init__(self, act_component):
        self.session = 'start'
        self.state = 'stretch'
        self.previous_state = self.state

        self.squish_to_stretch_count = 0
        self.stretch_to_squish_count = 0

        self.act_component = act_component

        # EX2 variables
        self.ex2_phase = 'raising'
        self.ex2_rounds = 0
        self.ex2_hold_start = None

        states = ['squish', 'stretch']
        self.machine = Machine(model=self, states=states, initial='stretch')

        self.machine.add_transition(trigger='squishing', source='stretch', dest='squish',
                                    after='increment_squish')
        self.machine.add_transition(trigger='stretching', source='squish', dest='stretch',
                                    after='increment_stretch')

    def squish_or_stretch(self, joints):
        indextip = joints[2][1]
        middletip = joints[4][1]
        ringtip = joints[6][1]
        pinkytip = joints[8][1]
        thumbtip = joints[1][1]

        maximum = max(indextip, middletip, ringtip, pinkytip, thumbtip)
        minimum = min(indextip, middletip, ringtip, pinkytip, thumbtip)
        diff = maximum - minimum

        if diff < 0.05:
            return True
        if diff > 0.20:
            return False

    def update_state(self, joints):

        if(self.session == 'ex1'):
            index_bent  = joints[3][1] < joints[2][1]
            middle_bent = joints[5][1] < joints[4][1]
            ring_bent   = joints[7][1] < joints[6][1]
            pinky_bent  = joints[9][1] < joints[8][1]
            handup = joints[0][1] > joints[1][1]

            if(index_bent and middle_bent and ring_bent and pinky_bent and handup):
                if(self.state == "stretch" and self.squish_or_stretch(joints)):
                    self.squishing()

            elif(not index_bent and not middle_bent and not ring_bent and not pinky_bent and handup):
                if(self.state == "squish" and not self.squish_or_stretch(joints)):
                    self.stretching()

        elif(self.session == 'ex2'):

            # joints:
            # 0 = left shoulder
            # 1 = left elbow
            # 2 = left wrist
            # 3 = left hip

            shoulder = joints[0]
            elbow = joints[1]
            wrist = joints[2]
            hip = joints[3]

            # ---------------------------------------------------------
            # Calculate arm flexion angle
            #
            # Vector 1: shoulder -> hip
            # Vector 2: shoulder -> elbow
            #
            # When the arm is down, flexion is close to 0 degrees.
            # When the arm is raised, flexion becomes larger.
            # ---------------------------------------------------------

            torso_vector = (
                hip[0] - shoulder[0],
                hip[1] - shoulder[1]
            )

            upper_arm_vector = (
                elbow[0] - shoulder[0],
                elbow[1] - shoulder[1]
            )

            torso_length = math.sqrt(
                torso_vector[0] ** 2 +
                torso_vector[1] ** 2
            )

            upper_arm_length = math.sqrt(
                upper_arm_vector[0] ** 2 +
                upper_arm_vector[1] ** 2
            )

            if torso_length == 0 or upper_arm_length == 0:
                return

            dot_product = (
                torso_vector[0] * upper_arm_vector[0] +
                torso_vector[1] * upper_arm_vector[1]
            )

            cos_angle = dot_product / (torso_length * upper_arm_length)

            cos_angle = max(-1.0, min(1.0, cos_angle))

            flexion = math.degrees(math.acos(cos_angle))

            # ---------------------------------------------------------
            # Calculate elbow angle
            #
            # This follows the same form-checking idea as Eleanor:
            # shoulder -> elbow -> wrist
            # ---------------------------------------------------------

            shoulder_to_elbow = (
                shoulder[0] - elbow[0],
                shoulder[1] - elbow[1]
            )

            wrist_to_elbow = (
                wrist[0] - elbow[0],
                wrist[1] - elbow[1]
            )

            vector1_length = math.sqrt(
                shoulder_to_elbow[0] ** 2 +
                shoulder_to_elbow[1] ** 2
            )

            vector2_length = math.sqrt(
                wrist_to_elbow[0] ** 2 +
                wrist_to_elbow[1] ** 2
            )

            if vector1_length == 0 or vector2_length == 0:
                return

            elbow_dot = (
                shoulder_to_elbow[0] * wrist_to_elbow[0] +
                shoulder_to_elbow[1] * wrist_to_elbow[1]
            )

            elbow_cos = elbow_dot / (vector1_length * vector2_length)

            elbow_cos = max(-1.0, min(1.0, elbow_cos))

            elbow_angle = math.degrees(math.acos(elbow_cos))

            # ---------------------------------------------------------
            # EX2 exercise state machine
            # ---------------------------------------------------------

            if self.ex2_phase == 'raising':

                # Arm has been raised sufficiently
                if flexion >= 60:

                    self.ex2_phase = 'holding'
                    self.ex2_hold_start = time.time()

            elif self.ex2_phase == 'holding':

                # User lowered the arm before completing the hold
                if flexion < 55:

                    self.ex2_phase = 'raising'
                    self.ex2_hold_start = None

                else:

                    # Hold for 2 seconds
                    hold_time = time.time() - self.ex2_hold_start

                    if hold_time >= 2:

                        self.ex2_rounds += 1
                        self.ex2_phase = 'lowering'
                        self.ex2_hold_start = None

            elif self.ex2_phase == 'lowering':

                # Arm has returned close to the starting position
                if flexion < 25:

                    if self.ex2_rounds >= 5:

                        self.session = 'break'

                    else:

                        self.ex2_phase = 'raising'

    def increment_squish(self):
        self.stretch_to_squish_count += 1

        if self.stretch_to_squish_count == 5:
            self.session = 'break'

    def increment_stretch(self):
        self.squish_to_stretch_count += 1

        if self.stretch_to_squish_count == 5:
            self.session = 'break'