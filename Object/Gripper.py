import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from abc import ABC, abstractmethod
from typing import Optional
import time

from Object.GameObject import GameObject

class Gripper(GameObject, ABC):
    def __init__(self, name:str="Gripper", urdf_file:str="pr2_gripper.urdf", position:np.ndarray=np.array([0,0,0]), orientation:np.ndarray=np.array([0,0,0,1]), joint_positions:list[float]=[], offset:list=[0,0,0]) -> None:
        # Initialise GameObject for Gripper
        super().__init__(name=name, 
                         urdf_file=urdf_file, 
                         position=position, 
                         orientation=orientation)
        
        self.joint_positions = joint_positions
        self.offset = offset

    def load(self) -> None:
        """
        Load the Gripper into the simulation.
        """
        super().load()
        
        self.gripper_constraint = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=self.offset,
            childFramePosition=self.position,
            childFrameOrientation=self.orientation
        )
        

    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError(f"open() method not implemented for {type(self).__name__}.")

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError(f"close() method not implemented for {type(self).__name__}.")
    
    def lockPosition(self) -> None:
        """
        Lock the gripper position by updating the constraint with current position.
        This helps prevent movement during closing.
        """
        position, orientation = self.getPositionAndOrtientation()
        p.changeConstraint(self.gripper_constraint,
                          jointChildPivot=position,
                          jointChildFrameOrientation=orientation,
                          maxForce=50)
    
    def setPosition(self, new_position:Optional[np.ndarray]=None, new_orientation:Optional[np.ndarray]=None) -> None:
        """
        Override setPosition to also update the constraint.
        This ensures the gripper constraint stays synchronized with the gripper position.
        """
        if new_position is None:
            new_position = self.position

        if new_orientation is None:
            new_orientation = self.orientation

        # Update the base position/orientation
        super().setPosition(new_position, new_orientation)

        # Update the constraint to match the new position/orientation
        if hasattr(self, 'gripper_constraint') and self.gripper_constraint is not None:
            p.changeConstraint(self.gripper_constraint,
                                jointChildPivot=new_position,
                                jointChildFrameOrientation=new_orientation,
                                maxForce=50)

    def moveToPosition(self, target_position:np.ndarray, target_orientation:Optional[np.ndarray]=None, duration:float=1.0, steps:int=240) -> None:
        """
        Move the object to a target position and orientation over a specified duration.
        """
        position, orientation = self.getPositionAndOrtientation()

        if target_orientation is None:
            target_orientation = orientation

        for step in range(steps):
            self.close()
            t = (step + 1) / steps
            new_position = position * (1 - t) + target_position * t
            # Spherical linear interpolation (slerp) for smooth rotation
            slerp = Slerp([0, 1], R.from_quat([orientation, target_orientation]))
            slerped_rot = slerp(t)
            new_orientation = slerped_rot.as_quat(canonical=True)

            self.setPosition(new_position=new_position, new_orientation=new_orientation)
            p.stepSimulation()
            time.sleep(duration / steps)
        

    def orientationToTarget(self, target:np.ndarray=np.array([0,0,0])) -> np.ndarray:
        """
        Compute quaternion for gripper to face location from current position.
        """

        direction = target - self.getPosition()
        
        norm_x = np.linalg.norm(direction)
        
        if norm_x < 1e-6:
            return R.identity().as_quat(canonical=True)
        
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

    def updateCamera(self, z, yaw):
        p.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=50 + (yaw * 180 / 3.1416),
            cameraPitch=-60,
            cameraTargetPosition=[0.5, 0.3, z]
        )