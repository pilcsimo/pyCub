"""
Exercise 2 Smooth Movements Assignment

:Author: Lukas Rustler
"""

from __future__ import annotations
from icub_pybullet.pycub import pyCub
from icub_pybullet.utils import Pose
import numpy as np
import scipy
import pybullet


def move(client: pyCub, action: str, axis: list, r: list) -> tuple:
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
    steps = 100
    start_pose = client.end_effector.get_position()
    initial_pos = np.array(start_pose.pos, dtype=float)
    initial_ori = start_pose.ori
    if not hasattr(client, "pose_logger"):
        client.pose_logger = []
    client.pose_logger.clear()
    client.pose_logger.append(start_pose)
    client.log_pose = True
    vector = np.zeros(3, dtype=float)
    for idx, axis_idx in enumerate(axis):
        vector[axis_idx] = r[idx]

    POSITION_THRESHOLD = 0.005
    MAX_TARGET_CHECKS = 250

    def _command_target(pos: np.ndarray, label: str) -> None:
        target_pose = Pose(pos.tolist(), initial_ori)
        try:
            client.move_cartesian(target_pose, wait=False, velocity=0.5, check_collision=False, timeout=10)
        except Exception as exc:
            client.logger.warning("Failed to command %s: %s", label, exc)
            return

        for check in range(MAX_TARGET_CHECKS):
            if not client.is_alive():
                break
            client.update_simulation()
            current_pos = np.array(client.end_effector.get_position().pos, dtype=float)
            if np.linalg.norm(current_pos - pos) <= POSITION_THRESHOLD or client.motion_done():
                break

    if action == "line":
        for step in range(1, steps + 1):
            progress = step / steps
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

        null_space = scipy.linalg.null_space(axis_vec.reshape(1, 3))
        if null_space.shape[1] < 2:
            null_space = np.eye(3)[:, np.arange(2)]
        plane_u = null_space[:, 0]
        plane_v = null_space[:, 1]

        plane_u = plane_u / np.linalg.norm(plane_u)
        plane_v = plane_v / np.linalg.norm(plane_v)

        for step in range(steps + 1):
            angle = step / steps * 2 * np.pi
            offset = radius * (np.cos(angle) * plane_u + np.sin(angle) * plane_v)
            target_pos = initial_pos + offset
            _command_target(target_pos, f"circle step {step}")

    end_pose = client.end_effector.get_position()
    return start_pose, end_pose


if __name__ == "__main__":
    client = pyCub(config="exercise_2.yaml")

    move(client, action="line", axis=[0], r=[0.05])
