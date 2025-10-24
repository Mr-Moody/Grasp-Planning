import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R
from abc import ABC, abstractmethod

from Object.GameObject import GameObject
from Object.Joint import Joint

FINGER_JOINTS = [0,2]

class Gripper(GameObject, ABC):
    def __init__(self, name:str="Gripper", urdf_file:str="pr2_gripper.urdf", position:np.ndarray=np.array([0,0,0]), orientation:np.ndarray=np.array([0,0,0,1]), joint_positions:list[float]=[]) -> None:
        # Initialise GameObject for Gripper
        super().__init__(name=name, 
                         urdf_file=urdf_file, 
                         position=position, 
                         orientation=orientation)
        
        self.joint_positions = joint_positions

    def load(self) -> None:
        """
        Load the Gripper into the simulation.
        """
        super().load()
        
        self.gripper_constraint = p.createConstraint(parentBodyUniqueId=self.body_id,
                                                     parentLinkIndex=-1,
                                                     childBodyUniqueId=-1,
                                                     childLinkIndex=-1,
                                                     jointType=p.JOINT_FIXED,
                                                     jointAxis=[0, 0, 0],
                                                     parentFramePosition=[0, 0, 0],
                                                     childFramePosition=[0, 0, 0])
        
        if len(self.joint_positions) > 0:
            self.joints = [Joint(self.body_id,i,pos) for i,pos in enumerate(self.joint_positions)]

    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError(f"open() method not implemented for {type(self).__name__}.")

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError(f"close() method not implemented for {type(self).__name__}.")
    
    def setPosition(self, new_position:np.ndarray=np.array([0,0,0]), new_orientation:np.ndarray=np.array([0,0,0,1])) -> None:
        """
        Move gripper to new position and orientation.
        """
        if new_position is None:
            new_position = self.position

        if new_orientation is None:
            new_orientation = self.orientation

        super().setPosition(new_position, new_orientation)

        p.changeConstraint(self.gripper_constraint,
                            new_position,
                            new_orientation,
                            500)
        

    def orientationToTarget(self, target:np.ndarray=np.array([0,0,0])) -> np.ndarray:
        """
        Compute quaternion for gripper to face location from current position.
        """

        direction = target - self.getPosition()
        
        norm_x = np.linalg.norm(direction)
        
        if norm_x < 1e-6:
            return R.identity().as_quat()
        
        new_x = direction / norm_x
        
        
        world_up = np.array([0, 0, 1]) 

        new_y = np.cross(world_up, new_x)

        if np.linalg.norm(new_y) < 1e-6:
            # if looking up/down manually set new x to the world X
            new_y = np.cross(new_x, [0,1,0])
            
        new_y = new_y / np.linalg.norm(new_y)
        
        new_z = np.cross(new_x, new_y)

        rotation_matrix = np.column_stack((new_x, new_y, new_z))

        rotation = R.from_matrix(rotation_matrix)
        
        quaternion_xyzw = rotation.as_quat(True)
        
        return quaternion_xyzw
    
    def debugDrawOrientation(self, target):
        quaternion = self.orientationToTarget(target)
        position = self.getPosition()
        rotation = R.from_quat(quaternion)
        axes = rotation.as_matrix()

        # Draw lines for x,y,z axes (red,green,blue)
        p.addUserDebugLine(position, position + 0.1 * axes[:,0], [1,0,0], 2)
        p.addUserDebugLine(position, position + 0.1 * axes[:,1], [0,1,0], 2)
        p.addUserDebugLine(position, position + 0.1 * axes[:,2], [0,0,1], 2)