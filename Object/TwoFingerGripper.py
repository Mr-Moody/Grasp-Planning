import pybullet as p
import numpy as np

from Object.Gripper import Gripper

class TwoFingerGripper(Gripper):
    def __init__(self, position:np.ndarray=np.array([0,0,0]), orientation:np.ndarray=np.array([0,0,0,1])) -> None:
        # Initialise base Gripper class with joint positions for two finger gripper
        super().__init__(name="Gripper", 
                         urdf_file="pr2_gripper.urdf", 
                         position=position, 
                         orientation=orientation,
                         joint_positions=[0.550569, 0.0, 0.549657, 0.0])
        
    def load(self):
        """
        Load the Gripper into the simulation.
        """
        super().load()

        p.changeConstraint(self.gripper_constraint,
                           jointChildFrameOrientation=p.getQuaternionFromEuler([0, np.pi/2, 0]))
        
        # Add friction to all links including fingers
        num_joints = p.getNumJoints(self.body_id)
        for i in range(-1, num_joints):
            p.changeDynamics(self.body_id, i, lateralFriction=100.0, rollingFriction=1.0, spinningFriction=1.0)


    def open(self) -> None:
        self.joints[0].move(1)
        self.joints[2].move(1)

    def close(self) -> None:
        self.joints[0].move(0)
        self.joints[2].move(0)