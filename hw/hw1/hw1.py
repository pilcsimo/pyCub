"""
Template for HRO HW1
"""
import os
import sys
try:
    from icub_pybullet.pycub import pyCub
except:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from icub_pybullet.pycub import pyCub

import time
import numpy as np

def push_the_ball(client):
    """
    Function to push the ball from the table with joint control for HW1.

    :param client: instance of pyCub
    :type client: pyCub
    :return: 0 if successful
    :rtype: int
    """

    # move torso, so the hand hits the ball with the back of the hand
    client.move_position("torso_yaw", -0.87)

    # move the left arm to the torse, so it doesnt collide with the table
    client.move_position("l_elbow", np.deg2rad(20), wait=False)
    client.move_position("l_shoulder_pitch", np.deg2rad(0), wait=False)
    client.move_position("l_shoulder_roll", np.deg2rad(10), wait=False)

    # line up the hand with the ball, position the palm flat with the table
    client.move_position("r_elbow", np.deg2rad(65), wait=False)
    client.move_position("r_wrist_prosup", np.deg2rad(60), wait=False, velocity=10)
    client.move_position("r_wrist_pitch", np.deg2rad(-20), wait=False, velocity=10)

    # move torso pitch so the robot bends over the table
    client.move_position("torso_pitch", np.deg2rad(30))

    # move torso and hand to push the ball, use the combined force of the joints of the arm
    client.move_position("torso_pitch", np.deg2rad(27.5), wait=False, velocity=10)
    client.move_position("r_elbow", np.deg2rad(60), wait=False, velocity=10)
    client.move_position("r_shoulder_roll", np.deg2rad(90), wait=False, velocity=10)
    client.move_position("r_elbow", np.deg2rad(20), wait=False, velocity=10)
    client.move_position("torso_yaw", 0.87, wait=False, velocity=10)

    # wait manually
    while not client.motion_done():
        client.update_simulation()

    client.logger.info("Moved the ball!")
    return 0



def evaluate(client):
    c = client.getClosestPoints(client.other_objects[1][0], client.other_objects[2][0], np.Inf)
    min_dist = np.Inf
    for _ in c:
        d = _[client.contactPoints["DISTANCE"]]
        if d < min_dist:
            min_dist = d
    min_dist = np.round(min_dist, 3)
    score = np.round(np.min([min_dist*2, 5]), 2)
    client.logger.info(f"You moved the ball {min_dist}m away from the table. Your score is {score}.")


if __name__ == "__main__":
    # load the robot with correct world/config
    client = pyCub(config="hw1.yaml")

    push_the_ball(client)

    start_step = client.steps_done
    while client.is_alive():
        client.update_simulation()
        if int((client.steps_done - start_step) / (1/client.config.simulation_step)) >= 1:
            break

    evaluate(client)