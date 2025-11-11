import pybullet as p
import numpy as np
import time
from scipy.spatial.transform import Rotation as R, Slerp

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

    def __init__(self, position:np.ndarray=np.array([0,0,0]), orientation:np.ndarray=np.array([0,0,0,1])):
        super().__init__(name="Gripper", urdf_file="pr2_gripper.urdf", position=position, orientation=orientation, offset=np.array([0.3, 0, 0]))

    def load(self):
        """Open gripper at start."""
        super().load()

        for i, pos in enumerate(self.INITIAL_POSITIONS):
            p.resetJointState(self.id, i, pos)

    def open(self):
        """Open the gripper fingers."""
        for joint in self.JOINTS:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.5, maxVelocity=2, force=300)

    def close(self):
        """Close the gripper fingers."""
        for joint in self.JOINTS:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.1, maxVelocity=2, force=300)



    def graspObject(self, object:GameObject) -> None:

        if not isinstance(object, GameObject):
            raise TypeError("Object must be a GameObject")


        # Set friction on contact surfaces
        p.changeDynamics(object.id, -1, lateralFriction=2.0, rollingFriction=0.1, spinningFriction=0.1)
        p.changeDynamics(self.id, -1, lateralFriction=2.0, rollingFriction=0.1, spinningFriction=0.1)

        target = object.getPosition()
        orientation = self.orientationToTarget(target)
        self.setPosition(new_orientation=orientation)
        p.stepSimulation()

        # Move towards object - use more steps for smoother, more direct movement
        self.moveToPosition(target + object.grasp_offset, duration=0.2, steps=25)
        pause(0.25)
        
        
        # Close gripper
        self.close()
        pause(1.25)


        start_position = object.getPosition()
        start_orientation = self.getOrientation()
        target_position = start_position + np.array([0,0,0.2])
        target_orientation = start_orientation
        duration = 0.5
        steps = 25

        # Create Slerp interpolator once outside the loop
        key_rots = R.from_quat([start_orientation, target_orientation])
        slerp = Slerp([0, 1], key_rots)

        # Calculate simulation steps per movement step
        sim_steps_per_update = max(5, int(240 * duration / steps))

        # Lift object - use fixed start position to avoid oscillation
        for step in range(steps):
            t = (step + 1) / steps
            # Interpolate from fixed start position, not current position
            new_position = start_position * (1 - t) + target_position * t
            # Spherical linear interpolation (slerp) for smooth rotation
            slerped_rot = slerp(t)
            new_orientation = slerped_rot.as_quat(canonical=True)

            # Update constraint with high force for direct movement
            p.changeConstraint(
                self.constraint_id,
                jointChildPivot=new_position,
                jointChildFrameOrientation=new_orientation,
                maxForce=300) 

            # Run multiple simulation steps to allow constraint to settle
            for _ in range(sim_steps_per_update):
                p.stepSimulation()
            
            time.sleep(duration / steps)

