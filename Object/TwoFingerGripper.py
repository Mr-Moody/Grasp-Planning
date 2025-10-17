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

    def open(self) -> None:
        self.joints[0].move(0.5)
        self.joints[2].move(0.5)

    def close(self) -> None:
        self.joints[0].move(0.1)
        self.joints[2].move(0.1)