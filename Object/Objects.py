import numpy as np

from Object.GameObject import GameObject

class Box(GameObject):
    def __init__(self, position):
        super().__init__(name="Box", urdf_file="cube_small.urdf", position=position)
        self.grasp_offset = np.array([0,0,0.01])
        self.name = f"Box_{self.id}"


class Cylinder(GameObject):
    def __init__(self, position):
        super().__init__(name="Cylinder", urdf_file="cylinder.urdf", position=position)
        self.grasp_offset = np.array([0,0,0.01])
        self.name = f"Cylinder_{self.id}"
        
class Duck(GameObject):
    def __init__(self, position):
        super().__init__(name="Duck", urdf_file="duck_vhacd.urdf", position=position)
        self.grasp_offset = np.array([0,0,0.01])
        self.name = f"Duck_{self.id}"