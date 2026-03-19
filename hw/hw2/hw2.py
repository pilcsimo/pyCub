import os
import sys
try:
    from icub_pybullet.pycub import pyCub
except:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from icub_pybullet.pycub import pyCub

import time
import numpy as np

def get_poses(client):
    # get ball position and orientation
    ball_pos, ball_ori = client.getBasePositionAndOrientation(client.free_objects[0])

    # Get head link position and orientation
    head_state = client.getLinkState(client.robot, 96, computeLinkVelocity=0, computeForwardKinematics=0)
    head_pos, head_ori = head_state[0], head_state[1]

    return ball_pos, ball_ori, head_pos, head_ori


def gaze(client):
    """
    Function that moves the neck yaw and pitch to look at the ball based on the angle between head-to-ball and head vectors.
    
    :param client: instance of pyCub class
     """
    
    # Solution step by step:
    # 1. Get the position and orientation of the ball and the head
    # 2. Compute the vector from the head to the ball and the head's forward vector using the head's orientation
    # 3. Compute the angle between the head's forward vector and the vector from the head to the ball using atan2
    # 4. Decompose the angle into yaw and pitch components
    # 5. Control the neck joints via PD controller towards the desired yaw and pitch angles.

    ball_pos, ball_ori, head_pos, head_ori = get_poses(client)

    # Compute the vector from head to ball
    head_ball_direction = np.array(ball_pos) - np.array(head_pos)

    # Compute head's forward vector using the head's orientation
    R = np.eye(4)
    R[:3, :3] = np.reshape(client.getMatrixFromQuaternion(head_ori), (3, 3))
    head_direction = np.matmul(R, [0, 0, 1, 1])[:3] 

    # Normalise to be sure we are working with unit vectors for angle calculations
    head_direction = head_direction / np.linalg.norm(head_direction)
    head_ball_direction = head_ball_direction / np.linalg.norm(head_ball_direction)

    # Calculate yaw (horizontal angle) and pitch (vertical angle) differences
    yaw_angle = np.arctan2(head_ball_direction[1], head_ball_direction[0]) - np.arctan2(head_direction[1], head_direction[0])
    # Pitch is a bit more complex, we need to consider the vertical component and the horizontal distance (= norm of x y components)
    pitch_angle = np.arctan2(head_ball_direction[2], np.linalg.norm(head_ball_direction[:2])) - np.arctan2(head_direction[2], np.linalg.norm(head_direction[:2]))

    # Normalize angles to [-pi, pi] (Unfortunately, atan2 is too smart for what the neck can do)
    yaw_angle_error = (yaw_angle + np.pi) % (2 * np.pi) - np.pi
    pitch_angle_error = (pitch_angle + np.pi) % (2 * np.pi) - np.pi

    # PD controller for the neck joints to smoothly follow the ball
    Kp = 1.0  # Proportional gain
    Kd = 0.2  # Derivative gain

    # Add attributes to store previous errors for the derivative term
    if not hasattr(gaze, "prev_yaw_error"):
        gaze.prev_yaw_error = 0
    if not hasattr(gaze, "prev_pitch_error"):
        gaze.prev_pitch_error = 0

    # Calculate PD control terms
    delta_yaw_error = yaw_angle_error - gaze.prev_yaw_error
    delta_pitch_error = pitch_angle_error - gaze.prev_pitch_error

    yaw_correction = Kp * yaw_angle_error + Kd * delta_yaw_error
    pitch_correction = Kp * pitch_angle_error + Kd * delta_pitch_error

    # Update previous errors
    gaze.prev_yaw_error = yaw_angle_error
    gaze.prev_pitch_error = pitch_angle_error

    # Get the current states of the neck joints
    current_neck_yaw_position = client.get_joint_state(["neck_yaw"])[0]
    current_neck_pitch_position = client.get_joint_state(["neck_pitch"])[0]

    # Calculate new positions for the neck joints
    new_neck_yaw_position = current_neck_yaw_position + yaw_correction
    new_neck_pitch_position = current_neck_pitch_position + pitch_correction

    # Move the joints to the new positions - use a higher velocity for faster response?
    client.move_position("neck_yaw", new_neck_yaw_position, velocity=10, wait=False)
    client.move_position("neck_pitch", new_neck_pitch_position, velocity=10, wait=False)


if __name__ == "__main__":
    client = pyCub(config="hw2.yaml")
    # look down and move arms from the view
    client.move_position("neck_pitch", -0.54, wait=False)
    client.move_position(["l_shoulder_pitch", "l_shoulder_roll", "r_shoulder_pitch", "r_shoulder_roll"],
                         [-1.5, 1.5, -1.5, 1.5], wait=False)
    while not client.motion_done():
        client.update_simulation(None)

    # apply force to the ball so it moves
    K = -5
    client.applyExternalForce(client.free_objects[0], -1, [K, K, 0], [0, 0, 0], client.WORLD_FRAME)

    time_step = client.config.simulation_step
    t = time.time()
    last_steps = None # Initial gaze command to set the head position
    while client.is_alive() and time.time()-t < 5:
        if last_steps is None or last_steps != client.steps_done:
            gaze(client)
            last_steps = client.steps_done

        client.update_simulation(time_step)


