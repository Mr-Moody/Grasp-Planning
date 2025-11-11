import pybullet as p
import pybullet_data
import time
import math
import numpy as np

from Object.TwoFingerGripper import TwoFingerGripper
from Object.Objects import Box, Cylinder, Duck
from Planning.Sphere import FibonacciSphere
from util import drawGizmo, setupEnvironment, pause
from constants import TIME, TICK_RATE, NUM_TICKS

if __name__ == "__main__":
    plane_id = setupEnvironment()

    gripper_start = np.array([0,0,1])
    object_start = np.array([0,0,0.06])

    s = FibonacciSphere(samples=50, radius=0.6, cone_angle=math.pi)
    s.visualise()
    p.stepSimulation()

    for v in s.vertices:
        # Initialise gripper and object
        gripper = TwoFingerGripper(position=v)
        object = Box(position=object_start)
        
        gripper.load()
        object.load()

        # Let object settle
        for _ in range(50):
            p.stepSimulation()
        
        target = object.getPosition()
        orientation = gripper.orientationToTarget(target)
        gripper.setPosition(new_position=v, new_orientation=orientation)
        p.stepSimulation()
        
        gripper.open()
        p.stepSimulation()

        gripper.graspObject(object)

        gripper.unload()
        object.unload()

    s.removeVisualisation()

    pause(100)

    p.disconnect()