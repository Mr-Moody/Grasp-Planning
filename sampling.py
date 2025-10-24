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
    target = np.array([0,0,0.05])

    # Initialise gripper and object
    gripper = TwoFingerGripper(position=start)
    object = GameObject(name="cube", urdf_file="cube_small.urdf", position=target)

    s = FibonacciSphere(samples=50, radius=0.6, cone_angle=math.pi)
    s.visualise()

    p.stepSimulation()

    for v in s.vertices:
        
        gripper.load()
        object.load()

        #update position of gripper
        gripper.setPosition(new_position=v)
        p.stepSimulation()

        #calculate orientation to object picking up
        target_position = object.getPosition()
        orientation = gripper.orientationToTarget(target=target_position)
        gripper.setPosition(new_position=v, new_orientation=orientation)

        #visualise axes
        #gripper.debugDrawOrientation(target_position)

        p.stepSimulation()
        time.sleep(1)

        direction = target_position - v

        distance = np.linalg.norm(direction)
        direction = direction / distance

        gripper.moveToPosition(target_position=(target_position - 0.29*direction), target_orientation=orientation, duration=0.5, steps=50)

        gripper.close()
        p.stepSimulation()
        pause(2)

        position = gripper.getPosition()
        position[2] += 0.3

        gripper.moveToPosition(position, duration=2, steps=50)

        gripper.unload()
        object.unload()

    pause(100)

    p.disconnect()
