import pybullet as p
import pybullet_data
import time
import math
import numpy as np

from Object.GameObject import GameObject
from Object.TwoFingerGripper import TwoFingerGripper
from Object.ThreeFingerGripper import ThreeFingerGripper
from Object.roboticArm import RoboticArm
from Planning.sphere import FibonacciSphere
from util import drawGizmo, setupEnvironment, pause
from constants import TIME, TICK_RATE, NUM_TICKS


if __name__ == "__main__":
    setupEnvironment()

    # Initialise arm
    arm = RoboticArm(gripper_class = TwoFingerGripper)

    v = np.array([0,0,1])
    orientation = p.getQuaternionFromEuler(v)

    # Spawn gripper and object
    # gripper = TwoFingerGripper(position=np.array([0, 0, 1]), orientation=orientation)
    # p.stepSimulation()
    object = GameObject(name="cube", urdf_file="cube_small.urdf", position=np.array([0, 0, 0.05]))
    object.load()

    s = FibonacciSphere(samples=50, radius=0.4, cone_angle=math.pi)
    s.visualise()

    p.stepSimulation()

    good_grasps = []
    bad_grasps = []

    endEffectorLinkIndex = 6
    object_pos = object.getPosition()

    # Move gripper directly upwards for 10 seconds and see if object is slipping
    # pre_obj_pos = object.getPosition()
    # pre_grip_pos = gripper.getPosition()

    # for v in s.vertices:
    #     orientation = gripper.orientationToTarget(target=pre_obj_pos)
    #     gripper.setPosition(new_position=v, new_orientation=orientation)
    #     #p.addUserDebugLine(v, [0,0,0], [1,0,0])
    #     gripper.debugDrawOrientation(pre_obj_pos)
    #     p.stepSimulation()
    #     time.sleep(1)

    # pause(5)
    # isSlipping = False
        
    # for i in range(NUM_TICKS):
    #     gripper.setPosition(new_position=np.array([0, 0, 0.1 + (i * (0.5 / NUM_TICKS))]))
    #     p.stepSimulation()
    #     time.sleep(TICK_RATE)

    #     # Determine slipping by calculating and comparing acceleration of object and gripper
    #     obj_vec = object.getPosition() - pre_obj_pos
    #     grip_vec = gripper.getPosition() - pre_grip_pos

    #     obj_acc = obj_vec / (TICK_RATE ** 2)
    #     grip_acc = grip_vec / (TICK_RATE ** 2)

    #     # Comparing the dot product of the two accelerations to see if they are in the same direction
    #     if abs(np.dot(obj_acc, grip_acc)) < 0.9: #If the object acceleration is not in the same direction as the gripper acceleration, it is slipping
    #         isSlipping = True
    #         break
    #     else:
    #         pre_obj_pos = object.getPosition()
    #         pre_grip_pos = gripper.getPosition()
    #         isSlipping = False

    # gripper.open()
    # pause(2)

    # Code for robotic arm implementation
    for v in s.vertices:
        candidate_pos = object_pos + v
        
        # Calculate gripper orientation to face object
        target_orientation = arm.gripper.orientationToTarget(object_pos)

        arm.moveArmToPose(endEffectorLinkIndex, candidate_pos, target_orientation, steps=100, duration=2.0)

        above_object_pos = candidate_pos.copy()
        above_object_pos[2] = object_pos[2] + 0.1
        arm.moveArmToPose(endEffectorLinkIndex, above_object_pos, target_orientation, steps=50, duration=1.0)
        
        # Move down to object
        grasp_pos = above_object_pos
        grasp_pos[2] = object_pos[2] + 0.01
        arm.moveArmToPose(endEffectorLinkIndex, grasp_pos, target_orientation, steps=50, duration=1.0)

        arm.gripper.close()
        time.sleep(1)

        lift_pos = grasp_pos.copy()
        lift_pos += 0.3
        arm.moveArmToPose(endEffectorLinkIndex, lift_pos, target_orientation, steps=100, duration=2.0)

        # Slipping detection
        pre_obj_pos = object_pos
        pre_grip_pos = grasp_pos.copy()

        isSlipping = False
        for i in range(NUM_TICKS):
            p.stepSimulation()
            time.sleep(TICK_RATE)
            cur_obj_pos = object.getPosition()
            cur_grip_pos = p.getLinkState(arm.robotId, endEffectorLinkIndex)[4]
            obj_vec = np.array(cur_obj_pos) - np.array(pre_obj_pos)
            grip_vec = np.array(cur_grip_pos) - np.array(pre_grip_pos)

            obj_acc = obj_vec / (TICK_RATE ** 2)
            grip_acc = grip_vec / (TICK_RATE ** 2)

            if abs(np.dot(obj_acc, grip_acc)) < 0.9:
                isSlipping = True
                break   
            else:
                pre_obj_pos = cur_obj_pos
                pre_grip_pos = cur_grip_pos
        
        if isSlipping:
            bad_grasps.append(v)
        else:
            good_grasps.append(v)

        # Reset gripper for next attempt
        arm.gripper.open()
        time.sleep(1)
 
    # Keep simulation open before closing
    pause(100)

    p.disconnect()
