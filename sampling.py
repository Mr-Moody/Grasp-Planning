import pybullet as p
import pybullet_data
import time
import math
import numpy as np

from Object.GameObject import GameObject
from Object.TwoFingerGripper import TwoFingerGripper
from Object.ThreeFingerGripper import ThreeFingerGripper
from Planning.sphere import FibonacciSphere
from util import drawGizmo, setupEnvironment, pause
from constants import TIME, TICK_RATE, NUM_TICKS

if __name__ == "__main__":
    plane_id = setupEnvironment()

    start = np.array([0,0,1])
    target = np.array([0,0,0.03])

    # Initialise gripper and object
    gripper = TwoFingerGripper(position=start)
    object = GameObject(name="cube", urdf_file="cube_small.urdf", position=target)

    s = FibonacciSphere(samples=50, radius=0.6, cone_angle=math.pi)
    s.visualise()

    p.stepSimulation()

    s.vertices = [np.array([0,0,1]) for _ in range(50)]

    for v in s.vertices:
        gripper.load()
        object.load()

        gripper.open()

        #update position of gripper
        gripper.setPosition(new_position=v)
        p.stepSimulation()

        gripper.graspObject(object)

        gripper.unload()
        object.unload()

    pause(100)

    p.disconnect()
