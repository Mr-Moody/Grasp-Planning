import pybullet as p
import pybullet_data
import time
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import math

from Object.GameObject import GameObject
from Object.Objects import Box, Cylinder, Duck
from Planning import sphere

from constants import TICK_RATE
from util import setupEnvironment, pause
 

# ---------- Gripper Classes ----------
class Gripper():
    """Base gripper class that defines common gripper behavior."""
    def __init__(self, name, urdf_file, position, orientation, offset):
        self.urdf_file = urdf_file
        self.position = position
        self.orientation = orientation
        self.offset = offset
        self.id = None
        self.constraint_id = None
        self.grasp_moving = False

    def load(self):
        """Load gripper into the PyBullet world."""
        self.id = p.loadURDF(self.urdf_file, *self.position)

        # Set moderate damping to reduce oscillation while allowing smooth movement
        p.changeDynamics(self.id, -1, linearDamping=0.5, angularDamping=0.5)

        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=self.offset,
            childFramePosition=self.position
        )

        return self.id
    
    def unload(self):
        """Remove gripper from the PyBullet world."""
        if self.constraint_id is not None:
            p.removeConstraint(self.constraint_id)
            self.constraint_id = None
        if self.id is not None:
            p.removeBody(self.id)
            self.id = None
        

    def moveToPosition(self, target_position, target_orientation=None, duration:float=1, steps:int=20):
        """Move gripper to a new position and orientation."""
        
        if self.constraint_id is None:
            raise ValueError("Gripper must be fixed before moving.")

        start_position, start_orientation = self.getPositionAndOrientation()

        if target_orientation is None:
            target_orientation = start_orientation

        # Create Slerp interpolator once outside the loop
        key_rots = R.from_quat([start_orientation, target_orientation])
        slerp = Slerp([0, 1], key_rots)

        # Calculate simulation steps per movement step (multiple steps help constraint settle)
        sim_steps_per_update = max(5, int(240 * duration / steps))  # Ensure smooth constraint resolution

        for step in range(steps):
            t = (step + 1) / steps
            # Interpolate from fixed start position to avoid oscillation
            new_position = start_position * (1 - t) + target_position * t
            # Spherical linear interpolation (slerp) for smooth rotation
            slerped_rot = slerp(t)
            new_orientation = slerped_rot.as_quat(canonical=True)

            # Update constraint with higher force for direct movement
            p.changeConstraint(
                self.constraint_id,
                jointChildPivot=new_position,
                jointChildFrameOrientation=new_orientation,
                maxForce=500)  # Increased force for more direct movement

            # Run multiple simulation steps to allow constraint to settle
            for _ in range(sim_steps_per_update):
                p.stepSimulation()
            
            time.sleep(duration / steps)
        
        
    
    def setPosition(self, new_position:np.ndarray=None, new_orientation:np.ndarray=None) -> None:
        """
        Move object to new position and orientation.
        """
        if new_position is None:
            new_position = self.__position

        if new_orientation is None:
            new_orientation = self.__orientation

        self.__position = new_position
        self.__orientation = new_orientation

        p.resetBasePositionAndOrientation(self.id, new_position, new_orientation)
        
        # Update constraint to match new position
        if self.constraint_id is not None:
            p.changeConstraint(
                self.constraint_id,
                jointChildPivot=new_position,
                jointChildFrameOrientation=new_orientation,
                maxForce=500)

    def getPositionAndOrientation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the current position and orientation of the GameObject.
        """
        position, orientation = p.getBasePositionAndOrientation(self.id)
        self.__position = np.array(position)
        self.__orientation = np.array(orientation)

        return self.__position, self.__orientation

    def getPosition(self) -> np.ndarray:
        """
        Returns the current position of the GameObject.
        """
        position, _ = self.getPositionAndOrientation()

        return position
    
    def getOrientation(self) -> np.ndarray:
        """
        Returns the current orientation of the GameObject.
        """
        _, orientation = self.getPositionAndOrientation()

        return orientation
    

    def orientationToTarget(self, target:np.ndarray=np.array([0,0,0])) -> np.ndarray:
        """
        Compute quaternion for gripper to face location from current position.
        """

        direction = target - self.getPosition()
        
        norm_x = np.linalg.norm(direction)
        
        if norm_x < 1e-6:
            return R.identity().as_quat(canonical=True)
        
        new_x = direction / norm_x
        
        
        world_up = np.array([0, 0, 1]) 

        new_y = np.cross(world_up, new_x)

        if np.linalg.norm(new_y) < 1e-6:
            # if looking up/down manually set new x to the world X
            new_y = np.cross(new_x, [0,1,0])
            
        new_y = new_y / np.linalg.norm(new_y)
        
        new_z = np.cross(new_x, new_y)

        rotation_matrix = np.column_stack((new_x, new_y, new_z))

        rotation = R.from_matrix(rotation_matrix)
        
        quaternion_xyzw = rotation.as_quat(True)
        
        return quaternion_xyzw
    
    def debugDrawOrientation(self, target):
        quaternion = self.orientationToTarget(target)
        position = self.getPosition()
        rotation = R.from_quat(quaternion)
        axes = rotation.as_matrix()

        # Draw lines for x,y,z axes (red,green,blue)
        p.addUserDebugLine(position, position + 0.1 * axes[:,0], [1,0,0], 2)
        p.addUserDebugLine(position, position + 0.1 * axes[:,1], [0,1,0], 2)
        p.addUserDebugLine(position, position + 0.1 * axes[:,2], [0,0,1], 2)


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
        pause(0.5)
        
        
        # Close gripper
        self.close()
        pause(1)


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



#---------- Main Simulation ----------
if __name__ == "__main__":
    plane_id = setupEnvironment()

    gripper_start = np.array([0,0,1])
    object_start = np.array([0,0,0.03])

    s = sphere.FibonacciSphere(samples=50, radius=0.6, cone_angle=math.pi)
    s.visualise()
    p.stepSimulation()

    #s.vertices = [np.array([0,0,1]) for _ in range(50)]

    for v in s.vertices:
        # Initialise gripper and object
        gripper = TwoFingerGripper(position=v)
        object = Box(position=object_start)
        
        gripper.load()
        object.load()
        p.stepSimulation()
        
        target = object.getPosition()
        orientation = gripper.orientationToTarget(target)
        gripper.setPosition(new_position=v, new_orientation=orientation)
        p.stepSimulation()
        
        gripper.open()
        p.stepSimulation()


        gripper.graspObject(object)

        gripper.unload()
        object.unload()

    pause(100)

    p.disconnect()