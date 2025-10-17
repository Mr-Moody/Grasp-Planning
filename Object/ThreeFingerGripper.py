import numpy as np

from Object.Gripper import Gripper

class ThreeFingerGripper(Gripper):
    def __init__(self, position:np.ndarray=np.array([0,0,0]), orientation:np.ndarray=np.array([0,0,0,1])) -> None:
        # Initialise base Gripper class with joint positions for three finger gripper
        super().__init__(name="Gripper", 
                         urdf_file="", 
                         position=position, 
                         orientation=orientation,
                         joint_positions=[])
