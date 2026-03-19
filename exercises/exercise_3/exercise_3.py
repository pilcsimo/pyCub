"""
Exercise 3 Gaze Assignment

:Author: Lukas Rustler
"""
from icub_pybullet.pycub import pyCub
import numpy as np


def gaze(client: pyCub, head_direction: np.array, head_ball_direction: np.array) -> None:
    """
    Function that moves the neck to look at the ball based on the angle between head-to-ball and head vectors.

    :param client: instance of pyCub class
    :type client: pointer to pyCub
    :param head_direction: direction of the head; where the robot is looking
    :type head_direction: np.array
    :param head_ball_direction: vector between the head and the ball; where the robot should be looking
    :type head_ball_direction: np.array
    :return:
    :rtype:
    """

    # Normalize the vectors to ensure proper angle calculations
    head_direction = head_direction / np.linalg.norm(head_direction)
    head_ball_direction = head_ball_direction / np.linalg.norm(head_ball_direction)

    # Calculate yaw (horizontal angle) and pitch (vertical angle) differences
    yaw_angle = np.arctan2(head_ball_direction[1], head_ball_direction[0]) - np.arctan2(head_direction[1], head_direction[0])
    pitch_angle = np.arctan2(head_ball_direction[2], np.linalg.norm(head_ball_direction[:2])) - np.arctan2(head_direction[2], np.linalg.norm(head_direction[:2]))

    # Normalize angles to [-pi, pi]
    yaw_angle = (yaw_angle + np.pi) % (2 * np.pi) - np.pi
    pitch_angle = (pitch_angle + np.pi) % (2 * np.pi) - np.pi


    neck_yaw_state = client.get_joint_state(["neck_yaw"])
    neck_pitch_state = client.get_joint_state(["neck_pitch"])

    current_neck_yaw_position = neck_yaw_state[0]  # Extract current yaw position
    current_neck_pitch_position = neck_pitch_state[0]  # Extract current pitch position

    # Calculate new positions for the neck joints
    new_neck_yaw_position = current_neck_yaw_position + yaw_angle
    new_neck_pitch_position = current_neck_pitch_position + pitch_angle

    # Clamp positions to joint limits
    neck_yaw_limits = (-0.872664625997, 0.872664625997)  # Limits for neck_yaw
    neck_pitch_limits = (-0.698131700798, 0.383972435439)  # Limits for neck_pitch

    new_neck_yaw_position = max(neck_yaw_limits[0], min(neck_yaw_limits[1], new_neck_yaw_position))
    new_neck_pitch_position = max(neck_pitch_limits[0], min(neck_pitch_limits[1], new_neck_pitch_position))

    # Move the joints to the new positions
    client.move_position("neck_yaw", new_neck_yaw_position, velocity=5, wait=False)
    client.move_position("neck_pitch", new_neck_pitch_position, velocity=5, wait=False)
