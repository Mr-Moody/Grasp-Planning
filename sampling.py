import pybullet as p
import pybullet_data
import time
import math
import numpy as np

from Object.TwoFingerGripper import TwoFingerGripper
from Object.Objects import Box
from Planning.Sphere import FibonacciSphere
from util import drawGizmo, setupEnvironment, pause
from constants import TIME, TICK_RATE, NUM_TICKS

if __name__ == "__main__":
    plane_id = setupEnvironment()

    gripper_start = np.array([0,0,1])
    box_start = np.array([0,0,0.03])

    # Initialise gripper and object
    gripper = TwoFingerGripper(position=gripper_start)
    object = Box(position=box_start)

    s = FibonacciSphere(samples=50, radius=0.6, cone_angle=math.pi)
    s.visualise()

    p.stepSimulation()

    s.vertices = [np.array([0,0,1]) for _ in range(50)]

    for v in s.vertices:
        gripper.load()
        object.load()


        #update position of gripper
        gripper.setPosition(new_position=v)
        gripper.open()
        pause(0.2)

        gripper.graspObject(object)

        gripper.unload()
        object.unload()

    pause(100)

    p.disconnect()
