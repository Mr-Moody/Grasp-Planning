import pybullet as p
import pybullet_data
import time
import math
import numpy as np

from Object.GameObject import GameObject
from Object.TwoFingerGripper import TwoFingerGripper
from Object.ThreeFingerGripper import ThreeFingerGripper
from Planning.sphere import FibonacciSphere
from util import drawGizmo

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
    p.setGravity(0, 0, -9.81)

    # Create floor plane
    plane_id = p.loadURDF("plane.urdf")

    #v_down = Vector3(0, 1 * math.pi / 2, 0) # Pointing downwards
    v = np.array([0,0,1])
    orientation = p.getQuaternionFromEuler(v)

    # Spawn gripper and object
    gripper = TwoFingerGripper(position=np.array([0, 0, 0.5]), orientation=orientation)
    object = GameObject(name="cube", urdf_file="cube_small.urdf", position=np.array([0, 0, 0])) #Object at origin for ease

    s = FibonacciSphere(samples=50, radius=0.4, cone_angle=math.pi)
    [drawGizmo(v) for v in s.vertices]

    pause(2)

    #gripper.close() # Assume has found object and has approached it correctly
    #pause(2)

    # Move gripper directly upwards for 10 seconds and see if object is slipping
    pre_obj_pos = object.getPosition()
    pre_grip_pos = gripper.getPosition()

    for v in s.vertices:
        orientation = gripper.orientationToTarget(target=np.array([0,0,0]))
        gripper.setPosition(new_position=v, new_orientation=orientation)
        p.addUserDebugLine(v, [0,0,0], [1,0,0])
        p.stepSimulation()
        time.sleep(1)

    pause(5)
    isSlipping = False
        
    for i in range(NUM_TICKS):
        gripper.setPosition(new_position=np.array([0, 0, 0.1 + (i * (0.5 / NUM_TICKS))]))
        p.stepSimulation()
        time.sleep(TICK_RATE)

        # Determine slipping by calculating and comparing acceleration of object and gripper
        obj_vec = object.getPosition() - pre_obj_pos
        grip_vec = gripper.getPosition() - pre_grip_pos

        obj_acc = obj_vec / (TICK_RATE ** 2)
        grip_acc = grip_vec / (TICK_RATE ** 2)

        # Comparing the dot product of the two accelerations to see if they are in the same direction
        if abs(np.dot(obj_acc, grip_acc)) < 0.9: #If the object acceleration is not in the same direction as the gripper acceleration, it is slipping
            isSlipping = True
            break
        else:
            pre_obj_pos = object.getPosition()
            pre_grip_pos = gripper.getPosition()
            isSlipping = False

    gripper.open()
    pause(2)

    # Keep simulation open before closing
    pause(100)

    p.disconnect()
