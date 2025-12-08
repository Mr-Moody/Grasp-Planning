import numpy as np
import pybullet as p

from Grippers.Gripper import Gripper

class ThreeFingerGripper(Gripper):
    GRASP_JOINTS = [1, 4, 7]
    PRESHAPE_JOINTS = [2, 5, 8]
    UPPER_JOINTS = [3, 6, 9]
    
    def __init__(self, position:np.ndarray=np.array([0,0,0]), orientation:np.ndarray=np.array([0,0,0,1])) -> None:
        # Initialise base Gripper class with joint positions for three finger gripper
        super().__init__(name="Gripper", 
                         urdf_file="/Models/ThreeFingerGripper/sdh.urdf", 
                         position=position, 
                         orientation=orientation,
                         joint_positions=[])
        
        self.hand_base_constraint = None
        self.num_joints = 0
        self.is_open = True
        
    def preshape(self):
        """Move fingers into preshape pose."""
        for i in [2, 5, 8]:
            p.setJointMotorControl2(self.gripper_id, i, p.POSITION_CONTROL,
                                    targetPosition=0.4, maxVelocity=2, force=1)
        self.is_open = False

    def open(self):
        """Gradually open fingers until fully open."""
        closed, iteration = True, 0
        
        while closed and not self.is_open:
            joints = self.getJointPositions()
            closed = False
            
            for i in range(self.num_joints):
                if i in [2, 5, 8] and joints[i] >= 0.9:
                    self.moveJoint(i, joints[i] - 0.05)
                    closed = True
                elif i in [3, 6, 9] and joints[i] <= 0.9:
                    self.moveJoint(i, joints[i] - 0.05)
                    closed = True
                elif i in [1, 4, 7] and joints[i] <= 0.9:
                    self.moveJoint(i, joints[i] - 0.05)
                    closed = True
                    
            iteration += 1
            if iteration > 10000:
                break
            p.stepSimulation()
            
        self.is_open = True

    def moveJoint(self, joint, target):
        p.setJointMotorControl2(self.gripper_id, joint, p.POSITION_CONTROL, targetPosition=target, maxVelocity=2, force=5)

    def getJointPositions(self):
        return [p.getJointState(self.gripper_id, i)[0] for i in range(self.num_joints)]
