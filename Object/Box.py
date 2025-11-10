from Object.GameObject import GameObject

class Box(GameObject):
    def __init__(self, position):
        super().__init__(name="Box", urdf_file="cube_small.urdf", position=position)
        self.grasp_height = 0.03
        self.name = f"Box_{self.id}"