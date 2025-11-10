import pybullet as p
import pybullet_data
import time
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import math

from Object.GameObject import GameObject
from Object.Box import Box
from Object.Cylinder import Cylinder
from Planning.Sphere import FibonacciSphere

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

        position, orientation = self.getPositionAndOrientation()

        if target_orientation is None:
            target_orientation = orientation

        for step in range(steps):
            t = (step + 1) / steps
            new_position = position * (1 - t) + target_position * t
            # Spherical linear interpolation (slerp) for smooth rotation
            slerp = Slerp([0, 1], R.from_quat([orientation, target_orientation]))
            slerped_rot = slerp(t)
            new_orientation = slerped_rot.as_quat(canonical=True)

            
            p.changeConstraint(
            self.constraint_id,
            jointChildPivot=position,
            jointChildFrameOrientation=orientation,
            maxForce=50)

            p.stepSimulation()
            time.sleep(duration / steps)
        
        
    
    def setPosition(self, new_position:np.ndarray=np.array([0,0,0]), new_orientation:np.ndarray=np.array([0,0,0,1])) -> None:
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
        super().__init__(name="Gripper", urdf_file="pr2_gripper.urdf", position=position, orientation=orientation, offset=[0.27, 0, 0])

    def load(self):
        """Open gripper at start."""
        super().load()

        for i, pos in enumerate(self.INITIAL_POSITIONS):
            p.resetJointState(self.id, i, pos)

    def open(self):
        """Open the gripper fingers."""
        for joint in self.JOINTS:
            p.setJointMotorControl2(self.id, joint, p.POSITION_CONTROL,
                                    targetPosition=0.0, maxVelocity=2, force=300)

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

        # --- Move above object by calling the move function inherited from the parent
        self.moveToPosition(target, orientation, duration=0.3, steps=20)
        for _ in range(100):
            p.stepSimulation()
            time.sleep(TICK_RATE)

        # --- Lower onto object ---
        self.moveToPosition(target, orientation, duration=0.2, steps=10)
        print("\033[93mmove to the grasp height")
        for _ in range(100):
            p.stepSimulation()
            time.sleep(TICK_RATE)
        

        for _ in range(100):
            self.close()
            p.stepSimulation()
            time.sleep(TICK_RATE)


        position, orientation = self.getPositionAndOrientation()
        lift_target = position + np.array([0,0,0.2])
        steps = 100

        self.moveToPosition(target_position=position, target_orientation=orientation, duration=1, steps=100)



#---------- Main Simulation ----------
if __name__ == "__main__":
    setupEnvironment()

    gripper_start = np.array([0,0,1])
    object_start = np.array([0,0,0.07])

    gripper = TwoFingerGripper(position=gripper_start)
    object = Box(position=object_start)
 
    s = FibonacciSphere(samples=50, radius=0.6, cone_angle=math.pi)
    s.visualise()

    p.stepSimulation()

    #s.vertices = [np.array([0,0,1]) for _ in range(50)]

    for v in s.vertices:
        object.load()
        gripper.load()

        gripper.setPosition(new_position=v)
        pause(0.2)

        # Grasp and lift the box
        gripper.graspObject(object)

        # Keep GUI open briefly
        for _ in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        print(f"\033[91mdone grasping: {object.name}")

        gripper.unload()
        object.unload()

        # short pause before loading next one
        time.sleep(0.5)

    p.disconnect()



# ---------- Environment Setup ----------
def setup_environment():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(0)
    plane_id = p.loadURDF("plane.urdf")
    return plane_id


# ---------- Main Simulation ----------
if __name__ == "__main__":
    plane_id = setupEnvironment()

    gripper_start = np.array([0,0,1])
    box_start = np.array([0,0,0.03])

    # Initialise gripper and object
    gripper = TwoFingerGripper(position=gripper_start)
    object = Box(position=box_start)

    s = FibonacciSphere(samples=50, radius=0.6, cone_angle=math.pi)
    s.visualise()

    p.stepSimulation()

    #s.vertices = [np.array([0,0,1]) for _ in range(50)]

    for v in s.vertices:
        gripper.load()
        object.load()

        gripper.open()

        #update position of gripper
        gripper.setPosition(new_position=v)
        pause(0.2)

        gripper.graspObject(object)

        gripper.unload()
        object.unload()

    pause(100)

    p.disconnect()
