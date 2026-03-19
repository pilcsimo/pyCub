#!/usr/bin/env python3
import os
import sys
try:
    from icub_pybullet.pycub import pyCub
except:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from icub_pybullet.pycub import pyCub
import numpy as np

def get_body_part_from_skin_part(client, skin_part):
    for chain_name, chain_links in client.chains.items():
        if skin_part in chain_links:
            return chain_name


def process(client):
    # client.activated_skin_points = dictionary with skin part as key and list of activated points as value
    # client.activated_skin_normals = dictionary with skin part as key and list of activated normals as value
    # body_part = get_body_part_from_skin_part(client, skin_part) returns the body part (the one you want to move) that the skin part belongs to
    # jac, joints = client.compute_jacobian(body_part, end=skin_part) returns the Jacobian matrix and the joint indexes of the body part
    # client.move_velocity(joints, joint_vel) moves the robot in velocity space
    # client.stop_robot() stops the robot

    ACTIVATED_SKIN_PARTS = client.activated_skin_points.items()
    max_points = 0
    for skin_part, activated_points in ACTIVATED_SKIN_PARTS:
        if len(activated_points) > 0:
            points = client.activated_skin_points[skin_part]
            normals = client.activated_skin_normals[skin_part]
            max_points = 
            break
        
        if max_points > 0:

            body_part = get_body_part_from_skin_part(client, activated_skin_parts[0][0])

            jac, joints = client.compute_jacobian(body_part, end=activated_skin_parts[0][0])

            # TODO: Calculate the joint velocity joint_vel using RRMC
            # For example, you can use the pseudoinverse of the Jacobian to calculate the joint velocity that would move the end effector in the direction of the activated skin points
            # You can also use the normals of the activated skin points to calculate the desired direction of
            # movement, for example by taking the average of the normals and using it as the desired direction
            # For simplicity, we will just use the first activated skin point and its normal to calculate the desired direction of movement
            desired_direction = -activated_skin_parts[0][2][0]  # Use the normal
            # Calculate the joint velocity using the pseudoinverse of the Jacobian
            jac_pinv = np.linalg.pinv(jac)
            joint_vel = jac_pinv @ desired_direction  # This is a simple RRMC implementation
            client.logger.info(f"Activated skin part: {activated_skin_parts[0][0]}, Desired direction: {desired_direction}, Joint velocity: {joint_vel}")

            client.move_velocity(joints, joint_vel)
        else:
            client.stop_robot()


if __name__ == "__main__":
    client = pyCub(config="rrmc.yaml")


    while client.is_alive():
        process(client)
        client.update_simulation()