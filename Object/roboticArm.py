import pybullet as p
import pybullet_data
import time
import numpy as np

class RoboticArm():

    def __init__(self, robot_urdf="kuka_iiwa/model.urdf", gripper_class=None, gripper_urdf_position=[0,0,0]):
        # Load robot file
        self.robotId = p.loadURDF(robot_urdf, [0.6, 0, 0], useFixedBase=True)

        # Load gripper instance
        self.gripper = None
        if gripper_class is not None:
            self.gripper = gripper_class(position=np.array(gripper_urdf_position))
            self.gripper.load()

            # Attach gripper to robot
            self.gripper_constraint = p.createConstraint(parentBodyUniqueId=self.robotId,
                                                        parentLinkIndex=6,
                                                        childBodyUniqueId=self.gripper.body_id,
                                                        childLinkIndex=-1,
                                                        jointType=p.JOINT_FIXED,
                                                        jointAxis=[0,0,0],
                                                        parentFramePosition=[0,0,0],
                                                        childFramePosition=[0,0,0],
                                                        parentFrameOrientation=[0,0,0,1],
                                                        childFrameOrientation=p.getQuaternionFromEuler([0, np.pi/2, 0]))
            
    def moveArmToPose(self, endEffectorLinkIndex, targetPos, targetOrn, steps = 100, duration = 1.0):
        # Moves arm to desired location
        currentPos, currentOrn = p.getLinkState(self.robotId, endEffectorLinkIndex)[4:6]
        for step in range(steps):
            alpha = step / (steps - 1)
            interpPos = (1-alpha) * np.array(currentPos) + alpha * np.array(targetPos)
            interpOrn = p.getQuaternionSlerp(currentOrn, targetOrn, alpha)
            jointPoses = p.calculateInverseKinematics(self.robotId, endEffectorLinkIndex, interpPos.tolist(), list(interpOrn))
            for i, jointPos in enumerate(jointPoses):
                p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, jointPos)
            p.stepSimulation()
            time.sleep(duration / steps)