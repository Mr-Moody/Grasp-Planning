import pybullet as p
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

from Object.GameObject import GameObject
from Object.Joint import Joint

FINGER_JOINTS = [0,2]

class Gripper(GameObject):
    def __init__(self, position:np.ndarray=np.array([0,0,0]), orientation:np.ndarray=np.array([0,0,0,1])) -> None:
        # Initialise GameObject for Gripper
        super().__init__(name="Gripper", 
                         urdf_file="pr2_gripper.urdf", 
                         position=position, 
                         orientation=orientation)
        
        self.gripper_constraint = p.createConstraint(parentBodyUniqueId=self.body_id,
                                                     parentLinkIndex=-1,
                                                     childBodyUniqueId=-1,
                                                     childLinkIndex=-1,
                                                     jointType=p.JOINT_FIXED,
                                                     jointAxis=[0, 0, 0],
                                                     parentFramePosition=[0.2, 0, 0],
                                                     childFramePosition=[0, 0, 0])
        
        joint_positions = [0.550569, 0.0, 0.549657, 0.0]

        self.joints = [Joint(self.body_id,i,pos) for i,pos in enumerate(joint_positions)]

    def open(self) -> None:
        self.joints[0].move(0.5)
        self.joints[2].move(0.5)

    def close(self) -> None:
        self.joints[0].move(0)
        self.joints[2].move(0)

    def orientationToTarget(self, target:np.ndarray=np.array([0,0,0])) -> np.ndarray:
        """
        Compute quaternion for gripper to face location from current position.
        """

        direction = target - self.getPosition()
        
        if np.linalg.norm(direction) < 0.0001:  # If too close to target
            return np.array([0, 0, 0, 1])  # Identity quaternion
            
        # normalised forward vector
        x_axis = direction / np.linalg.norm(direction)
        
        world_z = np.array([0.0, 0.0, 1.0]) 
        
        y_axis_temp = np.cross(x_axis, world_z)
        
        if np.linalg.norm(y_axis_temp) < 0.0001:
            # If parallel define a new World_Z
            world_y = np.array([0.0, 1.0, 0.0])
            y_axis_temp = np.cross(x_axis, world_y)
            
        y_axis = y_axis_temp / np.linalg.norm(y_axis_temp)

        z_axis = np.cross(y_axis, x_axis)

        rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)

        r = R.from_matrix(rotation_matrix)
        
        quaternion = r.as_quat(False) 
        
        return np.array(quaternion.tolist())