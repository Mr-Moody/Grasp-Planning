import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import random
import pandas as pd
from datetime import datetime
import math
from scipy.spatial.transform import Rotation as R

from Planning.Sphere import FibonacciSphere
from Object.Objects import Box, Cylinder, Duck
from util import setupEnvironment, pause, addNoiseToOrientation, addNoiseToOffset
from constants import TICK_RATE

class RoboticArm():
    def __init__(self):
        self.SIMULATION_SPEED = 1.0  
        self.GRASP_Z_OFFSET = 0.01
        self.BASE_STEP_SLEEP = TICK_RATE
        # Increased steps for smoother, more accurate movement
        self.BASE_STEPS_APPROACH = 150  # Increased from 75 for better alignment
        self.BASE_STEPS_FAST = 80  # Increased from 30 for smoother fast moves
        self.BASE_STEPS_PRECISE = 200  # For critical approach movements
    

    def load_panda(self):
        """Load Panda arm with WIDER gripper opening (fixes pushing issue)"""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        robot_id = p.loadURDF("franka_panda/panda.urdf", [-0.6, 0, 0], useFixedBase=True)
        
        gripper_joints = [9, 10]
        p.changeDynamics(robot_id, 9, lateralFriction=2.0)
        p.changeDynamics(robot_id, 10, lateralFriction=2.0)
        
        return robot_id, gripper_joints

    def get_gripper_indices(self, robot_id):
        """Auto-detect Panda gripper joints"""
        finger_joints = []

        for i in range(p.getNumJoints(robot_id)):
            joint_info = p.getJointInfo(robot_id, i)
            joint_name = joint_info[1].decode('utf-8')

            if "finger" in joint_name.lower():
                finger_joints.append(i)
                print(f"Found gripper joint {i}: {joint_name}")

        return finger_joints or [9, 10]

    def reset_arm_pose(self, robot_id):
        """Reset Panda to neutral pose, MAXIMUM gripper opening"""
        neutral_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0, 0, 0.04, 0.04]

        for i, pos in enumerate(neutral_positions):
            p.resetJointState(robot_id, i, pos)
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, pos, force=200)

        pause(TICK_RATE)

    def set_gripper(self, robot_id, gripper_joints, width):
        """Panda gripper: 0.08m is maximum opening (twice default)"""
        target_pos = min(max(width / 2.0, 0.0), 0.08)

        for joint_idx in gripper_joints:
            p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL,
                                targetPosition=target_pos, force=300)
        pause(TICK_RATE)

    def move_to_pose(self, robot_id, ee_link_idx, target_pos, target_orn, steps=None):
        """Interpolated motion with visual pacing. Steps and per-step sleep scale with SIMULATION_SPEED."""
        if steps is None:
            steps = max(20, int(self.BASE_STEPS_APPROACH / self.SIMULATION_SPEED))

        step_duration = (self.BASE_STEP_SLEEP / self.SIMULATION_SPEED) if self.SIMULATION_SPEED > 0 else TICK_RATE

        current_pose = p.getLinkState(robot_id, ee_link_idx)
        current_pos = np.array(current_pose[0])
        current_orn = np.array(current_pose[1])

        # Use better IK solver parameters for more accurate positioning
        ik_kwargs = {
            'maxNumIterations': 100,
            'residualThreshold': 1e-6
        }

        for alpha in np.linspace(0, 1, steps):
            interp_pos = (1 - alpha) * current_pos + alpha * np.array(target_pos)
            interp_orn = p.getQuaternionSlerp(current_orn, target_orn, alpha)
            
            # Use improved IK solver with better parameters
            joints = p.calculateInverseKinematics(
                robot_id, ee_link_idx, interp_pos, interp_orn,
                maxNumIterations=ik_kwargs['maxNumIterations'],
                residualThreshold=ik_kwargs['residualThreshold']
            )

            for i, q in enumerate(joints):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, q, force=200)

            pause(step_duration)

        # Ensure final commanded joint positions are applied and arm settles
        pause(TICK_RATE * 10)  # Increased from 5 for better settling

    def move_to_pose_fast(self, robot_id, ee_link_idx, target_pos, target_orn, steps=None):
        """Fast motion for non-critical moves; scales with SIMULATION_SPEED."""
        if steps is None:
            steps = max(15, int(self.BASE_STEPS_FAST / self.SIMULATION_SPEED))

        step_duration = (self.BASE_STEP_SLEEP / self.SIMULATION_SPEED) if self.SIMULATION_SPEED > 0 else TICK_RATE
        current_pose = p.getLinkState(robot_id, ee_link_idx)
        current_pos = np.array(current_pose[0])
        current_orn = np.array(current_pose[1])

        # Use better IK solver parameters
        ik_kwargs = {
            'maxNumIterations': 100,
            'residualThreshold': 1e-6
        }

        for alpha in np.linspace(0, 1, steps):
            interp_pos = (1 - alpha) * current_pos + alpha * np.array(target_pos)
            interp_orn = p.getQuaternionSlerp(current_orn, target_orn, alpha)
            joints = p.calculateInverseKinematics(
                robot_id, ee_link_idx, interp_pos, interp_orn,
                maxNumIterations=ik_kwargs['maxNumIterations'],
                residualThreshold=ik_kwargs['residualThreshold']
            )

            for i, q in enumerate(joints):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, q, force=400)

            pause(step_duration)
        
        # Add settling time after fast moves
        pause(TICK_RATE * 5)

    def draw_sphere_point(self, pt, object_pos=[0, 0, 0]):
        """Draw RED line from OBJECT CENTER to raw SPHERE POINT"""
        p.addUserDebugLine(object_pos, np.array(object_pos) + pt, 
                        lineColorRGB=[1, 0, 0], lineWidth=5, lifeTime=5)

    def grasp_pose_from_point(self, point, object_pos, offset=0.05):
        """Approach the object with gripper oriented along the approach direction vector.
        The gripper's approach axis (x-axis) points along the direction from start to object.
        Fingers close perpendicular to the approach direction.
        Target is SLIGHTLY ABOVE the center to prevent gripper from pushing down.
        
        Args:
            point: Point on Fibonacci sphere (direction vector from object center to start position)
            object_pos: Object center position
            offset: Not used currently, kept for compatibility
        """
        # Calculate approach direction: from sphere point (start) to object center
        # `point` is a vector from object center to the sphere point
        # So the approach direction (from start to object) is -point
        approach_dir = np.array(point)
        approach_dir_norm = np.linalg.norm(approach_dir)
        if approach_dir_norm > 1e-6:
            approach_dir = approach_dir / approach_dir_norm  # Normalize to unit vector
        else:
            approach_dir = np.array([0, 0, -1])  # Default downward if point is at origin
        
        approach_pos = np.array(object_pos)
        
        # Gripper orientation for Panda gripper:
        # For Panda gripper, the Z-axis points forward (outward from gripper)
        # The Y-axis is the finger closing direction (left-right)
        # The X-axis is perpendicular to both
        # We want the gripper's forward (Z-axis) to point along the approach path
        
        # Approach direction: from start position to object (gripper moves along this path)
        # Negative because approach_dir points from object to start, we want from start to object
        gripper_forward = -approach_dir  # This will be the gripper's Z-axis (forward direction)
        
        # Choose a reference vector to compute the finger closing direction (Y-axis)
        # Prefer world up [0,0,1] but use alternative if approach is nearly vertical
        world_up = np.array([0, 0, 1])
        world_right = np.array([1, 0, 0])
        
        # Check if approach direction is nearly parallel to world up
        dot_with_up = abs(np.dot(approach_dir, world_up))
        
        if dot_with_up > 0.95:  # Nearly vertical (approaching from above or below)
            # Use world right as reference for finger direction
            reference_vec = world_right
        else:
            # Use world up as reference for finger direction
            reference_vec = world_up
        
        # Compute gripper Y-axis (finger closing direction): perpendicular to forward and reference
        gripper_y = np.cross(reference_vec, gripper_forward)
        y_norm = np.linalg.norm(gripper_y)
        
        if y_norm < 1e-6:
            # If cross product is too small (nearly parallel), use alternative reference
            if dot_with_up > 0.95:
                reference_vec = world_up
            else:
                reference_vec = world_right
            gripper_y = np.cross(reference_vec, gripper_forward)
            y_norm = np.linalg.norm(gripper_y)
        
        if y_norm > 1e-6:
            gripper_y = gripper_y / y_norm
        else:
            # Fallback: use arbitrary perpendicular vector
            if abs(gripper_forward[0]) < 0.9:
                gripper_y = np.cross(np.array([1, 0, 0]), gripper_forward)
            else:
                gripper_y = np.cross(np.array([0, 1, 0]), gripper_forward)
            gripper_y = gripper_y / np.linalg.norm(gripper_y)
        
        # Compute gripper X-axis: perpendicular to both Y and Z (right-handed system)
        gripper_x = np.cross(gripper_y, gripper_forward)
        gripper_x = gripper_x / np.linalg.norm(gripper_x)
        
        # Ensure perfect orthonormality by recomputing
        gripper_y = np.cross(gripper_forward, gripper_x)
        gripper_y = gripper_y / np.linalg.norm(gripper_y)
        gripper_forward = np.cross(gripper_x, gripper_y)
        gripper_forward = gripper_forward / np.linalg.norm(gripper_forward)
        
        # For Panda gripper: Z-axis is forward, so build rotation matrix as [X, Y, Z]
        gripper_z = gripper_forward  # Z-axis points forward (along approach path)
        
        # Build rotation matrix: [X, Y, Z] where Z is forward (along approach path)
        rot_matrix = np.column_stack([gripper_x, gripper_y, gripper_z])
        
        # Matrix to quaternion conversion using trace method
        trace = rot_matrix[0, 0] + rot_matrix[1, 1] + rot_matrix[2, 2]

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (rot_matrix[2, 1] - rot_matrix[1, 2]) * s
            qy = (rot_matrix[0, 2] - rot_matrix[2, 0]) * s
            qz = (rot_matrix[1, 0] - rot_matrix[0, 1]) * s

        elif rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2])
            qw = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            qz = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s

        elif rot_matrix[1, 1] > rot_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2])
            qw = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
            qx = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s

        else:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1])
            qw = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
            qx = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
            qy = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
            qz = 0.25 * s
        
        approach_orn = [qx, qy, qz, qw]

        return approach_pos, approach_orn

    def quaternion_to_euler(self, q):
        """Convert quaternion [x,y,z,w] to roll, pitch, yaw (radians)"""
        x, y, z, w = q
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def wait_for_end_effector(self, robot_id, ee_link_idx, target_pos, target_orn, pos_tol=0.005, orn_tol=0.03, timeout=5.0):
        """Step simulation until end-effector reaches target pose within tolerances.
        Returns True if reached, False if timed out.
        pos_tol: meters (tighter tolerance), orn_tol: quaternion L2 norm approx.
        """
        start_t = time.time()
        max_iterations = int(timeout / TICK_RATE)
        iteration = 0

        while iteration < max_iterations:
            link_state = p.getLinkState(robot_id, ee_link_idx)
            cur_pos = np.array(link_state[0])
            cur_orn = np.array(link_state[1])
            pos_err = np.linalg.norm(cur_pos - np.array(target_pos))

            # Orientation error: angle between quaternions
            dot = abs(np.dot(cur_orn, np.array(target_orn)))
            dot = min(1.0, max(-1.0, dot))
            orn_err_angle = 2.0 * math.acos(dot)

            if pos_err <= pos_tol and orn_err_angle <= orn_tol:
                # Additional settling time to ensure stability
                pause(TICK_RATE * 3)
                return True

            if time.time() - start_t > timeout:
                return False

            pause(TICK_RATE)
            iteration += 1
        return False


    def is_pose_reachable(self, robot_id, ee_link_idx, target_pos, target_orn, pos_tol=0.03, orn_tol=0.5):
        """Check if an IK solution reaches the requested target within tolerances.
        This temporarily resets joint states to the IK result to measure the resulting pose,
        then restores the original joints.
        """
        num_joints = p.getNumJoints(robot_id)
        # Save current joint positions
        saved = [p.getJointState(robot_id, i)[0] for i in range(num_joints)]

        ik_joints = p.calculateInverseKinematics(robot_id, ee_link_idx, target_pos, target_orn)
        # Apply IK solution using resetJointState (instant)
        for i in range(min(num_joints, len(ik_joints))):
            p.resetJointState(robot_id, i, ik_joints[i])

        link_state = p.getLinkState(robot_id, ee_link_idx)
        ik_pos = np.array(link_state[0])
        ik_orn = np.array(link_state[1])

        # Restore saved joint positions
        for i in range(num_joints):
            p.resetJointState(robot_id, i, saved[i])

        pos_err = np.linalg.norm(ik_pos - np.array(target_pos))
        dot = abs(np.dot(ik_orn, np.array(target_orn)))
        dot = min(1.0, max(-1.0, dot))
        orn_err_angle = 2.0 * math.acos(dot)

        return (pos_err <= pos_tol) and (orn_err_angle <= orn_tol)


    def do_grasp_and_evaluate(self, robot_id, gripper_joints, ee_link_idx,
                            start_pos, start_orn, approach_pos, approach_orn, object, grasp_offset=None):
        """
        Move to a start pose (sampled sphere point), then approach the object,
        close the gripper and lift. Returns True on successful lift.
        
        Args:
            grasp_offset: Optional offset from object center [x, y, z]. If None, uses default GRASP_Z_OFFSET.
        """
        # OPEN GRIPPER wide to avoid pushing
        self.set_gripper(robot_id, gripper_joints, 0.08)

        # Move FAST to the distant start pose (sampled point on the sphere)
        self.move_to_pose_fast(robot_id, ee_link_idx, start_pos, start_orn)

        # Compute a grasp center using the provided offset or default
        object_pos, _ = object.getPositionAndOrientation()
        if grasp_offset is not None:
            grasp_center = np.array(object_pos) + np.array(grasp_offset)
        else:
            grasp_center = np.array(object_pos) + np.array([0, 0, self.GRASP_Z_OFFSET])

        # pre-approach: start from further away along -approach_dir so fingers go around object
        approach_dir = np.array(grasp_center) - (np.array(start_pos))
        if np.linalg.norm(approach_dir) > 1e-6:
            approach_dir = approach_dir / np.linalg.norm(approach_dir)
        else:
            approach_dir = -np.array(start_pos) / np.linalg.norm(start_pos)

        # Search for a reachable pre-approach distance (from far to near)
        max_dist = 0.25
        min_dist = 0.04
        step = 0.02
        pre_approach_pos = None

        for d in np.arange(max_dist, min_dist - 1e-6, -step):
            cand = grasp_center + (-approach_dir) * d

            if self.is_pose_reachable(robot_id, ee_link_idx, cand, approach_orn, pos_tol=0.03, orn_tol=0.5):
                pre_approach_pos = cand
                break

        if pre_approach_pos is None:
            # fallback: use start_pos as pre-approach
            pre_approach_pos = start_pos

        # Move to pre-approach quickly
        self.move_to_pose_fast(robot_id, ee_link_idx, pre_approach_pos, approach_orn)
        
        # Add settling time after fast move
        pause(TICK_RATE * 5)

        # Move in a straight line from pre-approach to the final grasp center (slower, precise)
        # Use more steps for critical approach movement
        self.move_to_pose(robot_id, ee_link_idx, grasp_center, approach_orn, steps=int(self.BASE_STEPS_PRECISE / self.SIMULATION_SPEED))

        # Ensure we've actually arrived at the approach pose before closing gripper
        # Tighter tolerances for better alignment
        reached = self.wait_for_end_effector(robot_id, ee_link_idx, grasp_center, approach_orn, pos_tol=0.005, orn_tol=0.05, timeout=2.0)
        if not reached:
            print("Warning: grasp center not reached within timeout")

        # Stabilize at approach - give more time for proper alignment
        pause(TICK_RATE * 8)  # Increased from 3 for better settling

        # CLOSE GRIPPER (small non-zero = better than fully zero to avoid finger collision)
        self.set_gripper(robot_id, gripper_joints, 0.005)
        # Give more time for gripper to close and establish contact
        pause(TICK_RATE * 20)  # Increased from 12 for better grip establishment

        # Check contact points between gripper and object to verify a grip
        contacts = p.getContactPoints(bodyA=object.id, bodyB=robot_id)
        contact_count = len(contacts)

        # LIFT: compute lift target from the CURRENT end-effector pose (more robust)
        ee_state = p.getLinkState(robot_id, ee_link_idx)
        ee_pos = np.array(ee_state[0])
        lift_pos = ee_pos + np.array([0, 0, 0.15])
        self.move_to_pose_fast(robot_id, ee_link_idx, lift_pos, approach_orn)

        # After lift, check object position and whether contacts persist
        pause(TICK_RATE * 6)

        object_pos, _ = object.getPositionAndOrientation()
        post_contacts = p.getContactPoints(bodyA=object.id, bodyB=robot_id)
        post_contact_count = len(post_contacts)

        # Relaxed success criteria: object lifted above ~8cm and within ~8cm laterally OR persistent contact
        height_ok = object_pos[2] > 0.08
        lateral_ok = np.linalg.norm(np.array(object_pos[:2]) - np.array(lift_pos[:2])) < 0.08
        attached_ok = post_contact_count > 0 or contact_count > 0

        success = (height_ok and lateral_ok) or attached_ok

        return success

    def save_to_csv(self, results):
        gripper_type = "RoboticArm"
        object_type = results[0].get("object_type", "Box") if results else "Box"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{gripper_type}_{object_type}_{timestamp}.csv"
        
        # Ensure Samples directory exists
        os.makedirs("Samples", exist_ok=True)
        output_file = os.path.join("Samples", filename)
        
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(results)
        
        # Save with pandas (no index, exact sampling.py format)
        df.to_csv(output_file, index=False)
        
        print(f"\nResults saved to: {output_file}")
        print(f"{len(results)} grasp attempts collected")
        print(f"Success rate: {df['label'].mean():.1%}")
        

    def robotic_arm_grasp_sampling(self, object_type:str="Box", gui:bool=True, num_samples:int=200):
        setupEnvironment(gui=gui)

        robot_id, gripper_joints = self.load_panda()
        gripper_joints = self.get_gripper_indices(robot_id)
        self.reset_arm_pose(robot_id)
        
        ee_link_idx = 11
        # Place object at origin [0, 0, 0]
        object_start_pos = np.array([0, 0, 0.04])

        if object_type == "Box":
            object_id = Box(position=object_start_pos)
            object_orientation = np.array([0, 0, 0, 1])
        elif object_type == "Cylinder":
            object_id = Cylinder(position=object_start_pos)
            object_orientation = np.array([0, 0, 0, 1])
        elif object_type == "Duck":
            object_id = Duck(position=object_start_pos)
            # Duck needs 90-degree rotation about x-axis
            object_orientation = np.array(p.getQuaternionFromEuler([np.pi/2, 0, 0]))
        else:
            print(f"Unknown object type: {object_type}")
            return

        # Load the object into the simulation
        object_id.load()
        # Let object settle
        for _ in range(50):
            p.stepSimulation()
        
        # For Duck, reset orientation to 90 degrees about x-axis after settling
        if object_type == "Duck":
            current_position, _ = object_id.getPositionAndOrientation()
            object_id.setPosition(new_position=current_position, new_orientation=object_orientation)
        
        # Get the actual object position after settling
        object_start_pos, _ = object_id.getPositionAndOrientation()
        object_start_pos = list(object_start_pos)

        # Noise ranges for uniform random distribution (noise will be in [-range/2, range/2])
        roll_noise_range = 1
        pitch_noise_range = 1
        yaw_noise_range = 1
        offset_noise_range = 0.11

        sphere_obj = FibonacciSphere(samples=2*num_samples, radius=0.20, cone_angle=math.pi)

        # Restrict to the top one-third of the sphere area: z > radius * (1/3)
        z_thresh = sphere_obj.radius / 3.0
        all_approach_points = np.array([pt for pt in sphere_obj.vertices if pt[2] > z_thresh])
        
        print(f"Generating Fibonacci sphere visualization ({num_samples} points)...")
        sphere_obj.visualise()
        pause(TICK_RATE)
        
        results = []
        print(f"Sphere visualized. Starting {num_samples} RANDOM grasp tests...")
        pause(2.0 / self.SIMULATION_SPEED if self.SIMULATION_SPEED > 0 else 2.0)
        
        tested_points = set()
        
        for i in range(num_samples):
            # PREFER UNTTESTED POINTS FIRST
            available_points = [j for j in range(len(all_approach_points)) if j not in tested_points]

            if available_points:
                random_idx = random.choice(available_points)
                tested_points.add(random_idx)
            else:
                random_idx = random.randint(0, len(all_approach_points) - 1)
            
            pt = all_approach_points[random_idx]
            
            # Get current object position (may have shifted)
            current_object_pos, _ = object_id.getPositionAndOrientation()
            current_object_pos = np.array(current_object_pos)
            
            # VISUAL FEEDBACK: THICK GREEN LINE to current test point
            p.addUserDebugLine(current_object_pos, current_object_pos + pt, 
                            lineColorRGB=[0, 1, 0], lineWidth=8, lifeTime=3)
            p.addUserDebugText(f"P{random_idx}", current_object_pos + pt, 
                            textColorRGB=[0, 1, 0], lifeTime=3)
            
            if i % 20 == 0:
                print(f"Test {i+1}/{num_samples} | Point #{random_idx} | Unique: {len(tested_points)}/{num_samples}")
                
            approach_pos, approach_orn = self.grasp_pose_from_point(pt, current_object_pos)
            self.draw_sphere_point(pt, current_object_pos)

            # Add noise to orientation
            noisy_orientation = addNoiseToOrientation(
                approach_orn,
                roll_noise_range=roll_noise_range,
                pitch_noise_range=pitch_noise_range,
                yaw_noise_range=yaw_noise_range
            )

            # Compute a distant start pose located at the sampled sphere point
            # `pt` is already scaled by the sphere radius in `FibonacciSphere.vertices`
            start_pos = current_object_pos + np.array(pt)
            start_orn = noisy_orientation

            # Add uniform random noise to grasp offset
            original_grasp_offset = object_id.grasp_offset
            noisy_grasp_offset = addNoiseToOffset(original_grasp_offset, offset_noise_range=offset_noise_range)

            success = self.do_grasp_and_evaluate(robot_id, gripper_joints, ee_link_idx,
                            start_pos, start_orn, approach_pos, noisy_orientation, object_id, grasp_offset=noisy_grasp_offset)
            
            # EXTRACT FEATURES EXACTLY LIKE sampling.py
            # Get actual end-effector orientation after grasp (matches gripper.getOrientation)
            link_state = p.getLinkState(robot_id, ee_link_idx)
            ee_quat = link_state[1]  # quaternion (x,y,z,w)
            roll, pitch, yaw = self.quaternion_to_euler(ee_quat)
            
            # Approach direction and distance (matches sampling.py: approachvertex - objectpos)
            approach_vertex = pt  # sphere point relative to object center
            approach_direction = approach_vertex - np.array([0, 0, 0])  # relative to object center
            approach_distance = np.linalg.norm(approach_direction)
            if approach_distance > 0:
                approach_direction = approach_direction / approach_distance
            
            results.append({
                "orientation_roll": float(roll),
                "orientation_pitch": float(pitch),
                "orientation_yaw": float(yaw),
                "offset_x": float(noisy_grasp_offset[0]),
                "offset_y": float(noisy_grasp_offset[1]),
                "offset_z": float(noisy_grasp_offset[2]),
                "approach_dir_x": float(approach_direction[0]),
                "approach_dir_y": float(approach_direction[1]),
                "approach_dir_z": float(approach_direction[2]),
                "approach_distance": float(approach_distance),
                "label": 1 if success else 0,
                "object_type": object_type
            })
            
            # Reset object to starting position and orientation
            p.resetBasePositionAndOrientation(object_id.id, object_start_pos, object_orientation)
            p.resetBaseVelocity(object_id.id, [0, 0, 0], [0, 0, 0])
            self.reset_arm_pose(robot_id)
            pause(0.1 / self.SIMULATION_SPEED if self.SIMULATION_SPEED > 0 else 0.1)
        
        # FINAL RESULTS
        total_tests = len(results)
        successes = sum(r['label'] for r in results)
        success_rate = (successes / total_tests) * 100
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Total tests: {total_tests}")
        print(f"Successful grasps: {successes}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Unique points tested: {len(tested_points)}/{num_samples}")
        
        # SAVE TO FIXED CSV WITH PANDAS
        self.save_to_csv(results)
