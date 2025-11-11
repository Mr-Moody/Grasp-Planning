import pybullet as p
import numpy as np
import time
from typing import Optional
from scipy.spatial.transform import Rotation as R, Slerp

class GameObject():
    count = 0

    def __init__(self, name, urdf_file, position:np.ndarray=np.array([0,0,0]), orientation:np.ndarray=np.array([0,0,0,1])) -> None:
        # encapsulated attributes so they aren't modified directly
        self.__name = name
        self.__urdf_file = urdf_file
        self.__position = position
        self.__orientation = orientation
        self.__id = None
        self.__constraint_id = None

        self.grasp_offset = np.array([0,0,0])
        
        GameObject.count += 1

    def __del__(self) -> None:
        GameObject.count -= 1

    def __repr__(self):
        return f"<GameObject {self.__name} at {self.__position}>"

    # property getters so that attributes can be read from outside the class
    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, new_name:str):
        self.__name = new_name
    
    @property
    def urdf_file(self):
        return self.__urdf_file
    
    @property
    def position(self):
        return self.__position
    
    @property
    def orientation(self):
        return self.__orientation
    
    @property
    def id(self):
        return self.__id
    
    def load(self) -> None:
        """
        Load the GameObject into the simulation.
        """
        self.__id = p.loadURDF(self.__urdf_file, list(self.__position), list(self.__orientation))
        self.name = f"{self.__name}_{self.__id}"


    def unload(self) -> None:
        """
        Remove the GameObject from the simulation.
        """

        if self.__constraint_id is not None:
            p.removeConstraint(self.__constraint_id)

        if self.__id is not None:
            p.removeBody(self.__id)
            self.__id = None


    def setPosition(self, new_position:np.ndarray=np.array([0,0,0]), new_orientation:np.ndarray=np.array([0,0,0,1])) -> None:
        """
        Move object to new position and orientation.
        """
        if new_position is None:
            new_position = self.__position

        if new_orientation is None:
            new_orientation = self.__orientation

        self.__position = new_position
        self.__orientation = new_orientation

        p.resetBasePositionAndOrientation(self.__id, new_position, new_orientation)

    def getPositionAndOrientation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the current position and orientation of the GameObject.
        """
        position, orientation = p.getBasePositionAndOrientation(self.__id)
        self.__position = np.array(position)
        self.__orientation = np.array(orientation)

        return self.__position, self.__orientation

    def getPosition(self) -> np.ndarray:
        """
        Returns the current position of the GameObject.
        """
        position, _ = self.getPositionAndOrientation()

        return position
    
    def getOrientation(self) -> np.ndarray:
        """
        Returns the current orientation of the GameObject.
        """
        _, orientation = self.getPositionAndOrientation()

        return orientation

    def applyForce(self, force:np.ndarray, rel_pos:np.ndarray=np.array([0,0,0])) -> None:
        """
        Apply a force at a relative position.
        """
        p.applyExternalForce(self.__id, -1, force, rel_pos, p.WORLD_FRAME)

    def moveToPosition(self, target_position:np.ndarray, target_orientation:Optional[np.ndarray]=None, duration:float=1.0, steps:int=240) -> None:
        """
        Move the object to a target position and orientation over a specified duration.
        """
        position, orientation = self.getPositionAndOrtientation()

        if target_orientation is None:
            target_orientation = orientation

        for step in range(steps):
            t = (step + 1) / steps
            new_position = position * (1 - t) + target_position * t
            # Spherical linear interpolation (slerp) for smooth rotation
            slerp = Slerp([0, 1], R.from_quat([orientation, target_orientation]))
            slerped_rot = slerp(t)
            new_orientation = slerped_rot.as_quat(canonical=True)

            self.setPosition(new_position=new_position, new_orientation=new_orientation)
            p.stepSimulation()
            time.sleep(duration / steps)

    
