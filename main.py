import pybullet as p
import pybullet_data
import time
import random
import math

from DataType.Vector3 import Vector3
from Object.GameObject import GameObject
from Object.Gripper import Gripper

TIME = 10 #seconds
TICK_RATE = 1./240.
NUM_TICKS = math.ceil(TIME / TICK_RATE)

def pause(wait_time:float):
    """
    Run simulation for time seconds.
    """
    t0 = time.time()
    while ((time.time() - t0) < wait_time):
        p.stepSimulation()
        time.sleep(TICK_RATE)

if __name__ == "__main__":
    # Connect to physics server
    cid = p.connect(p.SHARED_MEMORY)
    if cid < 0:
        p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, -10)

    # Create a floor
    plane_id = p.loadURDF("plane.urdf")

    gripper = Gripper(position=Vector3(0, 0, 0.1))
 
    gripper.close()
    pause(2)

    gripper.open()
    pause(2)

    # Keep simulation open for 10 before closing
    pause(10)

    p.disconnect()