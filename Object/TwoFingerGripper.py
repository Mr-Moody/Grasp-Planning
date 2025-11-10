import pybullet as p
import numpy as np
import time

from Object.Gripper import Gripper
from Object.GameObject import GameObject
from util import pause
from constants import TICK_RATE

class TwoFingerGripper(Gripper):
    INITIAL_POSITIONS = [0.550569, 0.0, 0.549657, 0.0]
    INITIAL_ORIENTATION = p.getQuaternionFromEuler([0, np.pi/2, 0])
    JOINTS = [0,2]
    MAX_FORCE = 400
    MAX_VELOCITY = 2

    def __init__(self, position:np.ndarray=np.array([0,0,0]), orientation:np.ndarray=np.array([0,0,0,1])) -> None:
        # Initialise base Gripper class
        super().__init__(name="Gripper", 
                         urdf_file="pr2_gripper.urdf", 
                         position=position, 
                         orientation=orientation)
        
    def load(self):
        """
        Load the Gripper into the simulation.
        """
        super().load()

        p.changeConstraint(self.gripper_constraint,
                           jointChildFrameOrientation=p.getQuaternionFromEuler([0, np.pi/2, 0]))


        p.changeDynamics(self.id, -1, lateralFriction=2.0, rollingFriction=0.1, spinningFriction=0.1)


        for i, pos in enumerate(self.INITIAL_POSITIONS):
            p.resetJointState(self.id, i, pos)
    

    def open(self) -> None:
        for joint_idx in self.JOINTS:
            p.setJointMotorControl2(self.id, joint_idx, p.POSITION_CONTROL, targetPosition=0.5, maxVelocity=2, force=300)

    def close(self) -> None:
        for joint_idx in self.JOINTS:
            p.setJointMotorControl2(self.id, joint_idx, p.POSITION_CONTROL, targetPosition=0.1, maxVelocity=2, force=300)

    def graspObject(self, object:GameObject) -> None:

        if not isinstance(object, GameObject):
            raise TypeError("Object must be a GameObject")

        p.changeDynamics(object.id, -1, lateralFriction=2.0, rollingFriction=0.1, spinningFriction=0.1)
        p.changeDynamics(self.id, -1, lateralFriction=2.0, rollingFriction=0.1, spinningFriction=0.1)

        target_position = object.getPosition()
        orientation = self.orientationToTarget(target=target_position)
        self.setPosition(new_orientation=orientation)

        self.open()

        pause(0.5)

        direction = target_position - self.position

        distance = np.linalg.norm(direction)
        direction = direction / distance

        offset = -0.3 * direction + np.array([0,0,0.001])
        
        self.moveToPosition(target_position=(target_position + offset), target_orientation=orientation, duration=0.5, steps=20)
        

        # Lock gripper position before closing to prevent movement
        self.lockPosition()

        for _ in range(100):
            self.close()
            p.stepSimulation()
            time.sleep(TICK_RATE)


        start_pos = self.getPosition()
        lift_target = np.array(start_pos)
        lift_target[2] += 0.2
        steps = 40

        for step in range(steps):
            t = (step + 1) / steps
            intermediate = start_pos * (1 - t) + lift_target * t

            self.setPosition(new_position=intermediate)

            self.close()
            p.stepSimulation()
            time.sleep((2.0 / steps))

