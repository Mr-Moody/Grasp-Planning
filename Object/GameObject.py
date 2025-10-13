import pybullet as p
import numpy as np

class GameObject():
    count = 0

    def __init__(self, name, urdf_file, position:np.ndarray=np.array([0,0,0]), orientation:np.ndarray=np.array([0,0,0,1])) -> None:
        self.name = name
        self.urdf_file = urdf_file
        self.position = position
        self.orientation = orientation
        self.body_id = p.loadURDF(self.urdf_file, list(self.position), self.orientation)

        GameObject.count += 1

    def __del__(self) -> None:
        GameObject.count -= 1

    def __repr__(self):
        return f"<GameObject {self.name} at {self.position}>"

    def setPosition(self, new_position:np.ndarray=np.array([0,0,0]), new_orientation:np.ndarray=np.array([0,0,0,1])) -> None:
        """
        Move object to new position and orientation.
        """
        if new_position is None:
            new_position = self.position

        if new_orientation is None:
            new_orientation = self.orientation

        self.position = new_position
        self.orientation = new_orientation

        p.resetBasePositionAndOrientation(self.body_id, new_position, new_orientation)

    def getPosition(self) -> np.ndarray:
        """
        Returns the current position of the GameObject.
        """
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        self.position = np.array(pos)

        return self.position

    def applyForce(self, force:np.ndarray, rel_pos:np.ndarray=np.array([0,0,0])) -> None:
        """
        Apply a force at a relative position.
        """
        p.applyExternalForce(self.body_id, -1, force, rel_pos, p.WORLD_FRAME)

    
