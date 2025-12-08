import math
import numpy as np

from util import drawGizmo, removeGizmo

class FibonacciSphere():
    def __init__(self, samples:int=100, radius:float=1.0, cone_angle:float=-2*math.pi, cone_origin:np.ndarray=np.array([0,0,1])) -> None:
        """
        Generate points on a sphere using the Fibonacci method.
        Args:
            samples (int): Number of points to generate.
            radius (float): Radius of the sphere.
            cone_angle (float): Cone angle in radians to limit the points. Default is -2pi (full sphere).
            cone_origin (Vector3): Direction of the cone origin. Default is (0,0,1) (pointing up).
        """
        self.samples = samples
        self.radius = radius
        self.cone_angle = cone_angle
        self.cone_origin = cone_origin
        self.vertices = self.generateVertices()
        self.gizmos = []

    def generateVertices(self) -> list[np.ndarray]:
        # Reference: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere#comment12186258_9600801
        points = []
        phi = math.pi * (3 - math.sqrt(5))  # golden angle in radians

        up = self.cone_origin / np.linalg.norm(self.cone_origin)
        angle_limit = math.cos(self.cone_angle / 2)

        for i in range(self.samples):
            y = 1 - (i / float(self.samples - 1)) * 2  # y from 1 to -1
            radius = math.sqrt(1 - y * y) # radius at y (imagine spiralling outwards from the pole)

            theta = phi * i  # increment golden angle 

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            v = np.array([x, y, z])

            v = v / np.linalg.norm(v)

            dp = np.dot(v, up)

            if dp >= angle_limit:
                points.append(v * self.radius)

        return points
    
    def visualise(self) -> None:
        self.gizmos = []
        for v in self.vertices:
            self.gizmos.append(drawGizmo(v))

    def removeVisualisation(self) -> None:
        if self.gizmos is None:
            return

        if len(self.gizmos) < 1:
            return

        for gizmo_id in self.gizmos:
            removeGizmo(gizmo_id)