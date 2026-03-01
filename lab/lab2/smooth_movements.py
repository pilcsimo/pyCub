import os
import sys
try:
    from icub_pybullet.pycub import pyCub
    from icub_pybullet.utils import Pose
except:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from icub_pybullet.pycub import pyCub
    from icub_pybullet.utils import Pose
import matplotlib.pyplot as plt
import numpy as np


def set_axes_equal(ax):
    try:
        ax.set_aspect('equal', adjustable='box')
    except:
        """Set equal aspect ratio for 3D plots"""
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        z_limits = ax.get_zlim()

        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]

        max_range = max(x_range, y_range, z_range) / 2.0

        mid_x = np.mean(x_limits)
        mid_y = np.mean(y_limits)
        mid_z = np.mean(z_limits)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


def show_points(points):
    points = np.array(points)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    set_axes_equal(ax)
    plt.show()


def move(client, action="line", axis=[0], r=[0.05]):
    """
    The main move function that moves the end effector in a line or circle.

    :param client: instance of pyCub
    :type client: pointer to pyCub
    :param action: name of the action; "line" or "circle"
    :type action: str
    :param axis: axes of the action; list of 1 or 2 elements
    :type axis: list of int
    :param r: radius/lengths of the action; list of 1 or 2 elements
    :type r: list of float
    :return: tuple of (start_pose, end_pose)
    :rtype: tuple
    """
    
    # Define constants for smooth movement
    STEPS = 500
    POSITION_THRESHOLD = 0.001  # Tighter threshold for better radius accuracy
    MAX_TARGET_CHECKS = 500      # More checks to ensure reaching target
    DEBUG = False 
    
    start_pose = client.end_effector.get_position()
    initial_pos = np.array(start_pose.pos, dtype=float)
    initial_ori = start_pose.ori
    vector = np.zeros(3, dtype=float)
    for idx, axis_idx in enumerate(axis):
        vector[axis_idx] = r[idx]
    
    # Track actual positions for the circle action
    actual_positions = []

    def _command_target(pos: np.ndarray, label: str) -> None:
        target_pose = Pose(pos.tolist(), initial_ori)
        try:
            client.move_cartesian(target_pose, wait=False, velocity=0.3, check_collision=False, timeout=10)
        except Exception as exc:
            return

        for check in range(MAX_TARGET_CHECKS):
            if not client.is_alive():
                break
            client.update_simulation()
            current_pos = np.array(client.end_effector.get_position().pos, dtype=float)
            
            # Track actual positions during circle motion
            if action == "circle" and check % 5 == 0:  # Sample every 5 checks to avoid too many points
                actual_positions.append(current_pos.copy())
            
            if np.linalg.norm(current_pos - pos) <= POSITION_THRESHOLD or client.motion_done():
                break

    if action == "line":
        for step in range(1, STEPS + 1):
            progress = step / STEPS
            target_pos = initial_pos.copy()
            for idx, axis_idx in enumerate(axis):
                target_pos[axis_idx] = initial_pos[axis_idx] + vector[axis_idx] * progress
            _command_target(target_pos, f"line step {step}")

    elif action == "circle":
        axis_idx = int(axis[0]) if axis else 0
        axis_idx = axis_idx % 3
        radius = abs(r[0]) if r else 0.0
        axis_vec = np.zeros(3, dtype=float)
        axis_vec[axis_idx] = 1.0

        # Create orthonormal basis for the plane perpendicular to axis_vec
        # Find a vector not parallel to axis_vec
        if abs(axis_vec[0]) < 0.9:
            temp = np.array([1.0, 0.0, 0.0])
        else:
            temp = np.array([0.0, 1.0, 0.0])

        # Gram-Schmidt: make temp orthogonal to axis_vec
        plane_u = temp - np.dot(temp, axis_vec) * axis_vec
        plane_u = plane_u / np.linalg.norm(plane_u)

        # Second vector: cross product ensures orthogonality
        plane_v = np.cross(axis_vec, plane_u)
        plane_v = plane_v / np.linalg.norm(plane_v)

        for step in range(STEPS + 1):
            angle = step / STEPS * 2 * np.pi
            offset = radius * (np.cos(angle) * plane_u + np.sin(angle) * plane_v)
            target_pos = initial_pos + offset
            _command_target(target_pos, f"circle step {step}")

        # DEBUG: Visualise the target circle and actual path, and analyze radius accuracy
        if actual_positions and DEBUG:
            target_positions = [initial_pos + radius * (np.cos(step / STEPS * 2 * np.pi) * plane_u + np.sin(step / STEPS * 2 * np.pi) * plane_v) for step in range(STEPS + 1)]
            
            fig = plt.figure(figsize=(12, 5))
            
            # Plot 1: Target circle
            ax1 = fig.add_subplot(121, projection='3d')
            target_positions = np.array(target_positions)
            ax1.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 'b-', linewidth=2, label='Target circle')
            ax1.scatter(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], c='blue', s=10, alpha=0.5)
            ax1.set_title('Target Circle')
            ax1.legend()
            set_axes_equal(ax1)
            
            # Plot 2: Actual vs Target
            ax2 = fig.add_subplot(122, projection='3d')
            actual_positions_arr = np.array(actual_positions)
            ax2.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 'b-', linewidth=2, label='Target')
            ax2.plot(actual_positions_arr[:, 0], actual_positions_arr[:, 1], actual_positions_arr[:, 2], 'r-', linewidth=2, label='Actual')
            ax2.set_title('Target vs Actual Path')
            ax2.legend()
            set_axes_equal(ax2)
            
            # Calculate radius error
            center_to_actual = np.linalg.norm(actual_positions_arr - initial_pos, axis=1)
            actual_radius = np.mean(center_to_actual)
            radius_error = np.abs(actual_radius - radius)
            threshold = radius * 0.1
            
            print(f"\n=== Circle Motion Analysis ===")
            print(f"Target radius: {radius:.6f} m")
            print(f"Actual average radius: {actual_radius:.6f} m")
            print(f"Radius error: {radius_error:.6f} m ({100*radius_error/radius:.2f}%)")
            print(f"Threshold (r*0.1): {threshold:.6f} m")
            print(f"Pass: {'Yes' if radius_error < threshold else 'No'}")
            print(f"Radius std dev: {np.std(center_to_actual):.6f} m")
            print(f"Min/Max radius: {np.min(center_to_actual):.6f} / {np.max(center_to_actual):.6f} m")
            
            plt.tight_layout()
            plt.show()

    end_pose = client.end_effector.get_position()
    return start_pose, end_pose


if __name__ == "__main__":
    client = pyCub(config="smooth_movements.yaml")
    client.start_pose = None

    move(client, action="circle", axis=[0], r=[0.05])

    while client.is_alive() and not client.motion_done():
        client.update_simulation()
