import pybullet as p
import numpy as np

class GameObject():
    count = 0

    def __init__(self, name, urdf_file, position:np.ndarray=np.array([0,0,0]), orientation:np.ndarray=np.array([0,0,0,1])) -> None:
        # encapsulated attributes so they aren't modified directly
        self.__name = name
        self.__urdf_file = urdf_file
        self.__position = position
        self.__orientation = orientation
        self.__body_id = p.loadURDF(self.__urdf_file, list(self.__position), self.__orientation)

        GameObject.count += 1

    def __del__(self) -> None:
        GameObject.count -= 1

    def __repr__(self):
        return f"<GameObject {self.__name} at {self.__position}>"

    # property getters so that attributes can be read from outside the class
    @property
    def name(self):
        return self.__name
    
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
    def body_id(self):
        return self.__body_id

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

        p.resetBasePositionAndOrientation(self.__body_id, new_position, new_orientation)

    def getPosition(self) -> np.ndarray:
        """
        Returns the current position of the GameObject.
        """
        pos, _ = p.getBasePositionAndOrientation(self.__body_id)
        self.__position = np.array(pos)

        return self.__position

    def applyForce(self, force:np.ndarray, rel_pos:np.ndarray=np.array([0,0,0])) -> None:
        """
        Apply a force at a relative position.
        """
        p.applyExternalForce(self.__body_id, -1, force, rel_pos, p.WORLD_FRAME)

    
