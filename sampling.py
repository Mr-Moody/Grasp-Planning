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
    target = np.array([0,0,0.025])

    # Initialise gripper and object
    gripper = TwoFingerGripper(position=start)
    object = GameObject(name="cube", urdf_file="cube_small.urdf", position=target)

    s = FibonacciSphere(samples=50, radius=0.6, cone_angle=math.pi)
    s.visualise()

    p.stepSimulation()

    # s.vertices = [np.array([0,0,1]) for _ in range(50)]

    for v in s.vertices:
        gripper.load()
        object.load()

        gripper.open()

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
        pause(0.5)

        direction = target_position - v

        distance = np.linalg.norm(direction)
        direction = direction / distance

        offset = -0.3 * direction
        
        gripper.moveToPosition(target_position=(target_position + offset), target_orientation=orientation, duration=0.2, steps=10)
        
        # Allow contacts to settle before closing
        for i in range(20):
            p.stepSimulation()
            time.sleep(TICK_RATE)

        # Lock gripper position before closing to prevent movement
        gripper.lockPosition()
        
        # Close slowly and maintain position
        gripper.close()
        # Let gripper close and settle for better contact - more steps for stability
        for i in range(150):
            gripper.close()  # Keep closing command active
            gripper.lockPosition()  # Continuously lock position to prevent drift
            p.stepSimulation()
            time.sleep(TICK_RATE)

        time.sleep(0.2)

        position = gripper.getPosition()
        position[2] += 0.2

        gripper.moveToPosition(position, duration=2, steps=40)

        gripper.unload()
        object.unload()

    pause(100)

    p.disconnect()
