import pybullet as p
import pybullet_data
import time
import random
import math

from DataType.Vector3 import Vector3
from Object.GameObject import GameObject

TIME = 10 #seconds
TICK_RATE = 1./240.
NUM_TICKS = math.ceil(TIME / TICK_RATE)

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

    objects = []

    # Random number of boxes and cylinders (up to 10 each)
    num_boxes = random.randint(1, 10)
    num_cylinders = random.randint(1, 10)

    for i in range(num_boxes):
        pos = Vector3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.5, 1.5))
        obj = GameObject(f"Box_{i+1}", "Models/cube_small.urdf", position=pos)
        objects.append(obj)

    for i in range(num_cylinders):
        pos = Vector3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.5, 1.5))
        obj = GameObject(f"Cylinder_{i+1}", "Models/cylinder.urdf", position=pos)
        objects.append(obj)

    for obj in objects:
        displacement = Vector3(0,0,random.uniform(0.1, 0.5))
        obj.move(displacement)

    # Print total objects using the class variable
    print("Total objects in scene:", GameObject.count)

    # Run simulation for a while so objects settle on the floor
    for _ in range(NUM_TICKS):
        p.stepSimulation()
        time.sleep(TICK_RATE)

    p.disconnect()