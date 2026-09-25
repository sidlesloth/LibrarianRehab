from transitions import Machine

# Think Component: Decision Making
# Things you need to improve: Add states and transitions according to your intervention/rehabilitation coaching design

class Think(object):

    session =['start', 'ex1', 'break', 'ex2', 'end']
    def __init__(self, act_component):
        """
        Initializes the state machine and sets up the transition logic.
        :param act_component: Reference to the Act component to trigger visual feedback
        :param flexion_threshold: threshold for entering the flexion state
        :param extension_threshold: threshold for entering the extension state
        """

        self.session = 'start'
        # Define initial state and thresholds for transitions
        self.state = 'stretch'  # Initial state
        self.previous_state = self.state
        

        # Transition counter
        self.squish_to_stretch_count = 0  
        self.stretch_to_squish_count = 0  

        # Act component for visualization
        self.act_component = act_component

        # Define the state machine with states 'flexion' and 'extension'
        states = ['squish', 'stretch']

        # Initialize the state machine
        self.machine = Machine(model=self, states=states, initial='stretch')

        # Define transitions
        self.machine.add_transition(trigger='squishing', source='stretch', dest='squish',
                                    after='increment_squish')
        # conditions='is_flexion_threshold_reached',

        self.machine.add_transition(trigger='stretching', source='squish', dest='stretch',
                                    after='increment_stretch')
        # conditions='is_extension_threshold_reached',
        
    def squish_or_stretch(self, joints):#does job of both threshold reached funcs. True = squish
        
        indextip = joints[2][1]; middletip = joints[4][1]; ringtip = joints[6][1]; pinkytip = joints[8][1]; thumbtip = joints[1][1]
        maximum = max(indextip, max(middletip, max(ringtip, max(pinkytip, thumbtip))))
        minimum = min(indextip, min(middletip, min(ringtip, min(pinkytip, thumbtip))))
        diff = maximum-minimum
        if diff < 0.05:
            return True
        if diff > 0.20:
            return False

    def update_state(self, joints):
        """
        Updates the state machine based on the current angle (flexion or extension).

        :param current_angle: The current elbow joint angle (in degrees)
        :param previous_angle: The previous elbow joint angle (in degrees)
        """
        index_bent  = joints[3][1] < joints[2][1] #y coord of indextip less than y coord of indexmiddle
        middle_bent = joints[5][1] < joints[4][1]
        ring_bent   = joints[7][1] < joints[6][1]
        pinky_bent  = joints[9][1] < joints[8][1]

        handup = joints[0][1] > joints[1][1] #y coord of wrist less than thumb tip
        if(index_bent and middle_bent and ring_bent and pinky_bent and handup):
             if(self.state == "stretch" and self.squish_or_stretch(joints)): 
                  self.squishing()
        
        elif(not index_bent and not middle_bent and not ring_bent and not pinky_bent and handup):
            if(self.state == "squish" and not self.squish_or_stretch(joints)):
               self.stretching()


    def increment_squish(self):
        self.stretch_to_squish_count+=1;
        if self.stretch_to_squish_count == 5: self.session = 'break';
    def increment_stretch(self):
        self.squish_to_stretch_count+=1;
        if self.stretch_to_squish_count == 5: self.session = 'break';