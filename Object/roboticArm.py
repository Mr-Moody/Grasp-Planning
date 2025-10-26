import pybullet as p
import pybullet_data
import time

# 1. Connect to PyBullet (GUI mode)
p.connect(p.GUI)

# 2. Add PyBullet’s data path (for plane.urdf, kuka, etc.)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3. Load environment
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

# 4. Set gravity
p.setGravity(0, 0, -9.8)

# 5. Position the camera
p.resetDebugVisualizerCamera(
    cameraDistance=3.5,
    cameraYaw=50,
    cameraPitch=-35,
    cameraTargetPosition=[0, 0, 0]
)

# 6. Print robot joints info
num_joints = p.getNumJoints(robotId)
print("Number of joints:", num_joints)
for j in range(num_joints):
    print(p.getJointInfo(robotId, j))

# 7. Add a cube above the plane
# cubeId = p.loadURDF("cube.urdf", [0.5, 0, 0.5])

# 7. Move one joint (joint #5 in this case)
target_pos = 1.0  # radians
p.setJointMotorControl2(robotId,
                        jointIndex=5,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target_pos)
