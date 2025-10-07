import pybullet as p

from Object.GameObject import GameObject
from Object.Joint import Joint
from DataType.Vector3 import Vector3

class Gripper(GameObject):
    def __init__(self, position:Vector3=Vector3.zero(), orientation:list=[0,0,0,1]) -> None:
        # Initialise GameObject for Gripper
        super().__init__(name="Gripper", 
                         urdf_file="pr2_gripper.urdf", 
                         position=position, 
                         orientation=orientation)
        
        self.gripper_constraint = p.createConstraint(parentBodyUniqueId=self.body_id,
                                                     parentLinkIndex=-1,
                                                     childBodyUniqueId=-1,
                                                     childLinkIndex=-1,
                                                     jointType=p.JOINT_FIXED,
                                                     jointAxis=[0, 0, 0],
                                                     parentFramePosition=[0.2, 0, 0],
                                                     childFramePosition=[0.5, 0.3, 0.7])
        
        joint_positions = [0.550569, 0.0, 0.549657, 0.0]

        self.joints = [Joint(self.body_id,i,pos) for i,pos in enumerate(joint_positions)]

    def open(self) -> None:
        for joint in self.joints:
            joint.move(0.5)

    def close(self) -> None:
        for joint in self.joints:
            joint.move(0.0)
