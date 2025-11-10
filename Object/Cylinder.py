from Object.GameObject import GameObject

class Cylinder(GameObject):
    def __init__(self, position):
        super().__init__(name="Cylinder", urdf_file="cylinder.urdf", position=position)
        self.grasp_height = 0.1
        self.name = f"Cylinder_{self.id}"