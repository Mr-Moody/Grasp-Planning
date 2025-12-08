import pybullet as p
import pybullet_data
import time
import math
from collections import namedtuple

# ---------- Setup ----------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0,0,-10)
plane_id = p.loadURDF("plane.urdf")
p.resetDebugVisualizerCamera(
    cameraDistance=0.5,
    cameraYaw=40,
    cameraPitch=-30,
    cameraTargetPosition=[0.6, 0.3, 0.2]
)

# ---------- Load Box ----------
box_pos = [0.6, 0.3, 0.025]
box_id = p.loadURDF("cube_small.urdf", box_pos)

# ---------- Load Free-Flying Gripper ----------
gripper_start_pos = [0.6,0.3,0.5]
gripper_start_ori = p.getQuaternionFromEuler([3.1416,0,0])
gripper_id = p.loadURDF("robotiq_2f_85/robotiq.urdf", gripper_start_pos, 
                         gripper_start_ori, useFixedBase=False)

# ---------- Parse joints ----------
num_joints = p.getNumJoints(gripper_id)
JointInfo = namedtuple('JointInfo',['id','name','type','lower','upper','maxForce'])
joints = []
for i in range(num_joints):
    info = p.getJointInfo(gripper_id, i)
    jid = info[0]
    name = info[1].decode()
    jtype = info[2]
    lower = info[8]
    upper = info[9]
    maxForce = info[10]
    joints.append(JointInfo(jid,name,jtype,lower,upper,maxForce))
    p.setJointMotorControl2(gripper_id,jid,p.VELOCITY_CONTROL,targetVelocity=0,force=0)

# ---------- Mimic Joint Setup ----------
mimic_parent_name = 'finger_joint'
mimic_children_names = {'right_outer_knuckle_joint':1,
                        'left_inner_knuckle_joint':1,
                        'right_inner_knuckle_joint':1,
                        'left_inner_finger_joint':-1,
                        'right_inner_finger_joint':-1}

mimic_parent_id = [j.id for j in joints if j.name==mimic_parent_name][0]
mimic_child_multiplier = {j.id: mimic_children_names[j.name] for j in joints if j.name in mimic_children_names}

# Create mimic constraints
for joint_id, multiplier in mimic_child_multiplier.items():
    c = p.createConstraint(gripper_id,mimic_parent_id,
                           gripper_id,joint_id,
                           jointType=p.JOINT_GEAR,
                           jointAxis=[0,1,0],
                           parentFramePosition=[0,0,0],
                           childFramePosition=[0,0,0])
    p.changeConstraint(c,gearRatio=-multiplier,maxForce=100,erp=1)

gripper_range = [0,0.085]  # min open, max open

def move_gripper(open_length):
    open_angle = 0.715 - math.asin((open_length-0.010)/0.1143)
    p.setJointMotorControl2(gripper_id,mimic_parent_id,p.POSITION_CONTROL,
                            targetPosition=open_angle, force=60)  # increase force
    return open_angle

def set_gripper_position(pos):
    p.resetBasePositionAndOrientation(gripper_id,pos,gripper_start_ori)

# ---------- Pick-Up Sequence ----------

# Open gripper
move_gripper(gripper_range[1])
for _ in range(100): p.stepSimulation(); time.sleep(1./240.)

# Close gripper and continuously maintain grip while lifting
target_grip = gripper_range[0]
for step in range(150):
    move_gripper(target_grip)
    p.stepSimulation()
    time.sleep(1./240.)

# target_grip = move_gripper(target_grip)

print("Closed gripper")

# Lift the box while keeping fingers closed
for i in range(100):
    # Apply smooth upward velocity to the gripper base
    current_pos, current_ori = p.getBasePositionAndOrientation(gripper_id)
    target_velocity = [0, 0, 0.5]  # Slow upward movement
    p.resetBaseVelocity(gripper_id, target_velocity, [0, 0, 0])
    move_gripper(target_grip)  # keep applying grasp
    p.stepSimulation()
    time.sleep(1./240.)

# Keep GUI open
time.sleep(2)
p.disconnect()
