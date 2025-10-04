import pybullet as p
from DataType.Vector3 import Vector3

class GameObject:
    count = 0

    def __init__(self, name, urdf_file, position:Vector3=Vector3(0,0,0), orientation=[0,0,0,1]) -> None:
        self.name = name
        self.urdf_file = urdf_file
        self.position = position
        self.orientation = orientation
        self.body_id = p.loadURDF(self.urdf_file, list(self.position), self.orientation)

        GameObject.count += 1

    def __del__(self) -> None:
        GameObject.count -= 1

    def move(self, new_position:Vector3=Vector3(0,0,0), new_orientation=None) -> None:
        """
        Move object to new position and orientation.
        """
        if new_position is None:
            new_position = self.position

        if new_orientation is None:
            new_orientation = self.orientation

        self.position = new_position
        self.orientation = new_orientation

        p.resetBasePositionAndOrientation(self.body_id, list(new_position), new_orientation)

    def get_position(self) -> Vector3:
        """
        Returns the current position of the GameObject.
        """
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        self.position = Vector3(*pos)

        return self.position

    def apply_force(self, force:Vector3, rel_pos:Vector3=Vector3(0,0,0)):
        """
        Apply a force at a relative position.
        """
        p.applyExternalForce(self.body_id, -1, list(force), list(rel_pos), p.WORLD_FRAME)

    def __repr__(self):
        return f"<GameObject {self.name} at {self.position}>"
