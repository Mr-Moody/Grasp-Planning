import pybullet as p
import pybullet_data
import numpy as np
from Planning.Sphere import FibonacciSphere 
import time
import os

def load_environment(): # Pybullet setup
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.loadURDF("plane.urdf")  
    p.setGravity(0, 0, -9.81)

def load_panda(): # Load Panda robotic arm
    panda_urdf = os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")
    robot_id = p.loadURDF(panda_urdf, [-0.6, 0, 0], useFixedBase=True)
    return robot_id

def get_gripper_indices(robot_id): # Finds joint indices for gripper fingers for further use
    return [i for i in range(p.getNumJoints(robot_id))
            if 'finger_joint' in p.getJointInfo(robot_id, i)[1].decode('utf-8')]

def reset_arm_pose(robot_id): # Resets arn to neutral position and opens grippers fully
    neutral_positions = [0, -np.pi / 4, 0, -np.pi / 2, 0, np.pi / 3, 0]
    for i, pos in enumerate(neutral_positions):
        p.resetJointState(robot_id, i, pos)
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, pos, force=100)

    gripper_joints = get_gripper_indices(robot_id)
    for gj in gripper_joints:
        p.resetJointState(robot_id, gj, 0.04)  # open gripper
        p.setJointMotorControl2(robot_id, gj, p.POSITION_CONTROL, 0.04, force=100)
    p.stepSimulation()

def set_gripper(robot_id, indices, width): # Control gripper fingers to close or open to given width
    target = width / 2.0
    for i in indices:
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=target, force=200)

def move_to_pose(robot_id, ee_link_idx, target_pos, target_orn, steps=100): # Move end-effector to given position
    visualize_target_pose(target_pos, target_orn)  # Visual debug target pose

    current_pose = p.getLinkState(robot_id, ee_link_idx)
    current_pos = np.array(current_pose[0])
    current_orn = np.array(current_pose[1])

    for alpha in np.linspace(0, 1, steps):
        interp_pos = (1 - alpha) * current_pos + alpha * np.array(target_pos)
        interp_orn = p.getQuaternionSlerp(current_orn, target_orn, alpha)

        joints = p.calculateInverseKinematics(
            robot_id,
            ee_link_idx,
            interp_pos,
            interp_orn,
            jointDamping=[0.1] * p.getNumJoints(robot_id),  # smoother IK
        )
        for i, q in enumerate(joints):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, q)
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

def draw_sphere_point(pos): # Draws debug line at target position of end-effector
    p.addUserDebugLine(pos, np.array(pos) + [0, 0, 0], lineColorRGB=[1, 0, 0], lifeTime=2)

def grasp_pose_from_point(point, cube_pos, offset=0.0): # Computes approach for gripper
    direction = np.array(point)
    norm = np.linalg.norm(direction)
    if norm == 0:
        norm = 1  # to avoid division by zero
    approach_vec = direction / norm
    approach_pos = np.array(cube_pos) + approach_vec * offset
    approach_orn = p.getQuaternionFromEuler([np.pi, 0, 0])  # wrist facing down
    return approach_pos, approach_orn

def do_grasp_and_evaluate(robot_id, gripper_indices, ee_link_idx, approach_pos, approach_orn, cube_id): # Executes grasp and movement
    move_to_pose(robot_id, ee_link_idx, approach_pos, approach_orn)

    # Delay before closing gripper (for stability)
    for _ in range(120):  # ~0.5 second delay at 240Hz
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # Close gripper
    set_gripper(robot_id, gripper_indices, 0.00)
    for _ in range(60):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # Lift the gripper upward by 15 cm
    lift_pos = approach_pos + np.array([0, 0, 0.15])
    move_to_pose(robot_id, ee_link_idx, lift_pos, approach_orn)

    for _ in range(60):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # Evaluate grasp success
    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    block_height = cube_pos[2]
    block_in_hand = (block_height > 0.10) and (
        np.linalg.norm(np.array(cube_pos[:2]) - np.array(lift_pos[:2])) < 0.05
    )
    return block_in_hand

def visualize_target_pose(position, orientation): # Visual debug for target position of end-effector
    p.addUserDebugText("Target", position, textColorRGB=[1, 0, 0], lifeTime=2)
    rot_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    origin = np.array(position)
    axis_length = 0.05
    p.addUserDebugLine(origin, origin + rot_matrix[:, 0] * axis_length, [1, 0, 0], 2, lifeTime=2)
    p.addUserDebugLine(origin, origin + rot_matrix[:, 1] * axis_length, [0, 1, 0], 2, lifeTime=2)
    p.addUserDebugLine(origin, origin + rot_matrix[:, 2] * axis_length, [0, 0, 1], 2, lifeTime=2)

def grasp_sampling(): # Main loop
    load_environment()
    robot_id = load_panda()
    reset_arm_pose(robot_id)  # neutral initial configuration

    gripper_indices = get_gripper_indices(robot_id)
    ee_link_idx = 11  # Panda end-effector link index

    cube_start_pos = [0, 0, 0]
    cube_id = p.loadURDF("cube_small.urdf", cube_start_pos)

    sphere_obj = FibonacciSphere(samples=50, radius=0.20)
    approach_points = np.array(sphere_obj.vertices)
    results = [] # RESULTS: approach point, success bool

    fixed_cube_pos = np.array([0, 0, 0])  # Fixed manual cube center

    for pt in approach_points:
        approach_pos, approach_orn = grasp_pose_from_point(pt, fixed_cube_pos, offset=0.0)
        draw_sphere_point(approach_pos)  # visualize approach point for debug

        success = do_grasp_and_evaluate(robot_id, gripper_indices, ee_link_idx, approach_pos, approach_orn, cube_id)
        results.append({"approach": pt.tolist(), "success": success})

        # Reset cube and robot between trials
        p.resetBasePositionAndOrientation(cube_id, fixed_cube_pos, [0, 0, 0, 1])
        reset_arm_pose(robot_id)
        time.sleep(0.5)

    print(results)

if __name__ == "__main__":
    grasp_sampling()
    p.disconnect()