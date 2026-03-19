from __future__ import annotations
import os
from typing import Optional
import cv2
import numpy as np
from transforms3d import quaternions
try:
    from icub_pybullet.pycub import EndEffector, pyCub
except:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from icub_pybullet.pycub import EndEffector, pyCub
import open3d.visualization.rendering as rendering
import open3d as o3d


class Grasper:
    def __init__(self, client: pyCub, fake_vision: bool=False, idx: int=0):
        """
        Class for grasping the ball

        :param client: instance of pyCub
        :type client: point to pyCub class
        :param fake_vision: whether to use fake vision (saved RGB image and deprojected 3D points) or real vision (get images from the camera)
        :type fake_vision: bool
        :param idx: index of the saved RGB image and deprojected 3D points to use in fake vision mode
        :type idx: int
        """
        self.client = client
        self.fake_vision = fake_vision
        # If vision is not available, use the saved RGB image and matrix of corresponding deprojected 3D points
        if self.fake_vision:
            self.rgb = cv2.imread(os.path.join(os.path.realpath(os.path.dirname(__file__)), "data", f"rgb_{idx}.png"))
            self.td_points = np.load(os.path.join(os.path.realpath(os.path.dirname(__file__)), "data", f"td_points_{idx}.npy"), allow_pickle=False)
        self.eye = "l_eye"
        for link in self.client.links:
            if link.name == f"{self.eye}_pupil":
                self.eye_link_id = link.robot_link_id
        self.mat = rendering.MaterialRecord()
        self.mat.shader = 'defaultLit'

    def get_rgb(self) -> np.array:
        """
        Get the RGB image from the camera

        :return: numpy array representing the image
        :rtype: np.array
        """
        if self.fake_vision:
            return self.rgb
        else:
            return self.client.get_camera_images(self.eye)[self.eye]

    def get_depth(self) -> np.array:
        """
        Get the depth image from the camera

        :return: numpy array representing the image
        :rtype: np.array
        """
        if not self.fake_vision:
            return self.client.get_camera_depth_images(self.eye)[self.eye]
        else:
            # You cannot get this in fake vision mode, do not use
            raise NotImplementedError("Depth image is not available in fake vision mode")

    def get_3d_point(self, u: int, v: int, d: float=-1) -> np.array:
        """
        Function to deproject point in camera image into 3D point in world coordinates

        :param u: u pixel coordinate in the image
        :type u: int
        :param v: v pixel coordinate in the image
        :type v: int
        :param d: depth values (in meters)
        :type d: float
        :return: 3D point in world coordinates
        :rtype: np.array
        """
        if not self.fake_vision:
            ew = self.client.visualizer.eye_windows[self.eye]
            return ew.unproject(v, u, d) # unproject uses (v, u) instead of (u, v) for pixel coordinates
        else:
            return self.td_points[u, v]

    def move_fingers(self, closure: Optional[float] = 1.0, hand: Optional[str] = "right",
                     timeout: Optional[float] = 10) -> None:
        """

        :param closure: 0 = open, 1 = close
        :type closure: float
        :param hand: which hand to move
        :type hand: str
        :param timeout: time to wait for the fingers to move
        :type timeout: float
        :return:
        :rtype:
        """

        # joint to move
        hand = "r" if hand == "right" else "l"
        joints = [f"{hand}_hand_thumb_2_joint", f"{hand}_hand_thumb_3_joint",
                  f"{hand}_hand_index_2_joint", f"{hand}_hand_index_3_joint",
                  f"{hand}_hand_middle_2_joint", f"{hand}_hand_middle_3_joint",
                  f"{hand}_hand_ring_2_joint", f"{hand}_hand_ring_3_joint",
                  f"{hand}_hand_little_2_joint", f"{hand}_hand_little_3_joint"]

        # Move all the joints
        for joint in joints:
            joint_handle = self.get_joint_handle(joint)
            self.client.move_position(joint, joint_handle.lower_limit + closure *
                                      (joint_handle.upper_limit - joint_handle.lower_limit), wait=False,
                                      check_collision=False, timeout=timeout, velocity=5)

        # wait for completion and do not care about collisions
        while not self.client.motion_done(check_collision=False):
            self.client.update_simulation()

    def get_pupil_vectors(self, point: np.array) -> tuple:
        """
        Function to compute vector from pupil to ball (where the robot should be looking)
        and vector from the pupil (where to robot is looking)

        :param point: 3D position of the ball
        :type point: numpy array
        :return: vectors from pupil to ball and from pupil
        :rtype: tuple
        """

        # Get head link position and orientation
        head_state = self.client.getLinkState(self.client.robot, self.eye_link_id, computeLinkVelocity=0, computeForwardKinematics=0)
        head_pos, head_ori = head_state[0], head_state[1]

        # The direction of a vector from the ball to the head
        pupil_ball_direction = np.array(point) - head_pos
        pupil_ball_len = np.linalg.norm(pupil_ball_direction)
        pupil_ball_direction /= pupil_ball_len  # normalize

        R_l = np.eye(4)
        R_l[:3, :3] = np.reshape(self.client.getMatrixFromQuaternion(head_ori), (3, 3))
        pupil_direction = np.matmul(R_l, [0, 0, 1, 1])[:3]  # Rotate Z-axis vector to point in direction of head
        pupil_direction /= np.linalg.norm(pupil_direction)  # normalize

        return pupil_ball_direction, pupil_direction

    def get_joint_handle(self, joint_name: str) -> pyCub.Joint:
        """
        Help function to get the handle of the joint by name

        :param joint_name: name of a joint
        :type joint_name: str
        :return: handle to the joint
        :rtype: pycub.Joint
        """
        for joint in self.client.joints:
            if joint.name == joint_name:
                return joint

    @staticmethod
    def quaternion_swap(q: list | np.array, to: Optional[str] = "wxyz") -> list:
        """
        Help function to convert between different quaternion representations

        YOU DO NOT HAVE TO USE THIS. It is useful for transforms3d.quaternions that represent quaternions as wxyz but pyCub is xyzw

        :param q: quaternion
        :type q: list or numpy array
        :param to: to which representation to convert; "wxyz" or "xyzw"; there is no check whether the input quaternion in the other format!
        :type to: str
        :return: quaternion in desired format
        :rtype: list
        """
        if to == "wxyz":
            return [q[3], q[0], q[1], q[2]]
        elif to == "xyzw":
            return [q[1], q[2], q[3], q[0]]

    def show_ee_axes(self):
        """
        Shows axes of the end-effector.
        The axes are shown at one place and do not move/rotate with the end-effector. To do so, you must call this function again.
        """
        ee_info = self.client.getLinkState(self.client.robot, self.client.end_effector.link_id, computeLinkVelocity=0,
                                    computeForwardKinematics=0)
        pos, ori = ee_info[0], ee_info[1]
        R = np.eye(4)
        R[:3, :3] = np.reshape(self.client.getMatrixFromQuaternion(ori), (3, 3))
        cf = o3d.geometry.TriangleMesh().create_coordinate_frame(origin=[0, 0, 0], size=0.5)
        cf.transform(R)
        cf.translate(pos, relative=True)
        self.client.visualizer.vis.remove_geometry("ee")
        self.client.visualizer.vis.add_geometry("ee", geometry=cf, material=self.mat)

    def set_ee(self, link_name):
        """
        Change the end-effector to the given link

        :param link_name: name of the link to set as end-effector. I.e., "r_hand" or "l_hand"
        :type link_name: str

        """
        self.client.end_effector = EndEffector(link_name, self.client)


    def find_the_ball(self) -> tuple:
        """
        Function to find the ball in the image

        HINTS:
            - The ball is green like (0, 255, 0) in RGB
                - but there is sun in the visualizer, so the color might be slightly different
            - The class contains a method to get the RGB image

        :return: 2D position of center of the ball in image plane
        :rtype: tuple
        """

        # TODO find the ball in the image and return its 2D center

        return center

    def grasp(self, center: tuple | list | np.array) -> int:
        """
        Function to grasp the ball based on 2D center.

        HINTS:
            - The class contains a method to get the depth image
            - The class contains a method to move the fingers
            - The class contains a method to get the 3D position of the ball
            - The class contains a method to get vector from the camera (pupil) to ball
            - The ball's radius is 2.5 cm
            - It is better to move few cm above the ball before grasping it
            - It may be useful to bend the wrist before grasp


        :param center: 2D position of center of the ball in image plane
        :type center: tuple
        :return:
        :rtype:
        """

        # TODO get 3D position of the ball, look at it and grasp it

        return 0


def main(pos, idx, fake_vision=True) -> tuple[int, float, float]:
    # get client and grasper
    client = pyCub(config="hw3.yaml")
    client.resetBasePositionAndOrientation(client.other_objects[1][0], pos, [0, 0, 0, 1])
    grasper = Grasper(client, fake_vision, int(idx.split("_")[-2]))

    #WARMUP for images
    while client.steps_done < 10:
        client.update_simulation()

    # try to find and grasp the ball
    try:
        center = grasper.find_the_ball()
        grasper.grasp(center)
    except Exception as e:
        print(e)
        return 0, 0, 0

    # get closest points between ball and table
    c = client.getClosestPoints(client.other_objects[1][0], client.other_objects[2][0], np.Inf)
    min_dist = np.Inf
    for _ in c:
        d = _[client.contactPoints["DISTANCE"]]
        if d < min_dist:
            min_dist = d

    # get closest points between ball and hand
    c = client.getClosestPoints(client.other_objects[1][0], client.robot, np.Inf, -1, client.end_effector.link_id)
    min_dist_to_hand = np.Inf
    for _ in c:
        d = _[client.contactPoints["DISTANCE"]]
        if d < min_dist_to_hand:
            min_dist_to_hand = d

    min_dist = np.abs(np.round(min_dist, 3))
    min_dist_to_hand = np.abs(np.round(min_dist_to_hand, 3))

    score = 5 if min_dist > 0.05 > min_dist_to_hand else 0

    return score, min_dist * 100, min_dist_to_hand*100


if __name__ == "__main__":
    FAKE_VISION = True
    REPS = 3
    results_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), "results")
    positions = [[-0.35, 0.175, -0.1], [-0.35, 0, -0.1], [-0.35, -0.175, -0.1]]

    score = 0
    total_score = 0
    for position_id in range(0, len(positions)):
        pose_score = 0
        for rep in range(REPS):
            idx = f"{position_id}_{rep}"
            position = positions[position_id]
            score_temp, min_dist, min_dist_to_hand = main(position, idx, FAKE_VISION)
            pose_score += score_temp
            print(f"Test {idx} completed with score {score_temp} with dist to table {min_dist:.3f} cm and dist to hand {min_dist_to_hand:.3f} cm")
        score += np.min([pose_score, 5])
        total_score += pose_score
        print(f"Pose {position_id} completed with score {np.min([pose_score, 5])}.")
    if total_score == REPS*3*5:
        score = 17
        print(f"Congratulations! You have successfully grasped the ball in all {REPS*3} tests -> +2 points. Your total score is {score}.")
    else:
        print(f"Total score: {score}")