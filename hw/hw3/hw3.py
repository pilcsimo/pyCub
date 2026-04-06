from __future__ import annotations
import os
from typing import Optional
import cv2
import numpy as np
from transforms3d import quaternions
try:
    from icub_pybullet.pycub import EndEffector, pyCub, Pose
except:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from icub_pybullet.pycub import EndEffector, pyCub, Pose
import open3d.visualization.rendering as rendering
import open3d as o3d


class Grasper:
    def __init__(self, client: pyCub, fake_vision: bool=False, idx: int=0, parent_path=os.path.realpath(os.path.dirname(__file__))):
        """
        Class for grasping the ball

        :param client: instance of pyCub
        :type client: point to pyCub class
        :param fake_vision: whether to use fake vision (saved RGB image and deprojected 3D points) or real vision (get images from the camera)
        :type fake_vision: bool
        :param idx: index of the saved RGB image and deprojected 3D points to use in fake vision mode
        :type idx: int
        :param parent_path: calling script path
        :type parent_path: os.path | string
        """
        self.client = client
        self.fake_vision = fake_vision
        # If vision is not available, use the saved RGB image and matrix of corresponding deprojected 3D points
        if self.fake_vision:
            self.rgb = cv2.imread(os.path.join(parent_path, "data", f"rgb_{idx}.png"))
            self.td_points = np.load(os.path.join(parent_path, "data", f"td_points_{idx}.npy"), allow_pickle=False)
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
            # td_points shape is (height, width, 3) = (row, col, xyz) — numpy row-major
            # u = col (horizontal), v = row (vertical) → index as [v, u]
            return self.td_points[v, u]

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


    def _mark(self, name: str, pos, color, ori=None, frame_size: float = 0.15):
        """
        Draw a colored sphere + world-aligned coordinate frame at `pos` in the visualizer.

        :param name:       unique geometry name (old geometry with same name is replaced)
        :param pos:        [x, y, z] world position
        :param color:      [R, G, B]  0-1 floats
        :param ori:        optional xyzw quaternion; if given, draws the frame in that orientation
                           instead of the world-aligned identity frame
        :param frame_size: axis arrow length in metres
        """
        um = rendering.MaterialRecord()
        um.shader = 'defaultUnlit'

        # Sphere at origin
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
        sphere.translate(np.array(pos))
        sphere.paint_uniform_color(color)
        self.client.visualizer.vis.remove_geometry(f"{name}_sphere")
        self.client.visualizer.vis.add_geometry(f"{name}_sphere", geometry=sphere, material=um)

        # Coordinate frame
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=frame_size)
        if ori is not None:
            R4 = np.eye(4)
            R4[:3, :3] = np.reshape(self.client.getMatrixFromQuaternion(ori), (3, 3))
            cf.transform(R4)
        cf.translate(np.array(pos), relative=True)
        self.client.visualizer.vis.remove_geometry(f"{name}_frame")
        self.client.visualizer.vis.add_geometry(f"{name}_frame", geometry=cf, material=self.mat)


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

        # Get the RGB image from camera
        rgb_image = self.get_rgb()
        
        # Convert RGB to BGR for OpenCV (OpenCV uses BGR by default)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # Convert BGR to HSV for better color detection
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        # Define range for green color in HSV
        # Green hue is approximately 35-85 in OpenCV (hue range is 0-180)
        # Adjust saturation and value ranges to account for lighting variations
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create binary mask for green color
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            raise ValueError("No green ball found in the image")
        
        # Get the largest contour (assuming it's the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the center of the ball using moments
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            raise ValueError("Could not calculate ball center from contour")
        
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        return (center_x, center_y)

    def clutch(self, amount: float, hand: Optional[str] = "right", timeout: Optional[float] = 10, wait: bool = True) -> None:
        """
        Open or close the hand in a natural clutch shape.

        Joint layout per finger (all joints: [lower, upper] deg):
          _0_joint : abduction / opposition
                     thumb  : [0°,  90°]  0 = pointing up/back, 90 = fully opposing fingers
                     index  : [-20°, 0°]  spread joint (leave at 0 for neutral)
                     middle : [0°,  20°]  spread joint (leave at 0 for neutral)
                     ring   : [0°,  20°]  spread joint (leave at 0 for neutral)
                     little : [0°,  20°]  spread joint (leave at 0 for neutral)
          _1_joint : MCP  – base knuckle  [0°, 90°]  (this is what drives the main curl)
          _2_joint : PIP  – middle knuckle [0°, 90°]
          _3_joint : DIP  – tip knuckle    [0°, 90°]

        :param amount: 0.0 = fully open, 1.0 = fully closed
        :type amount: float
        :param hand: "right" or "left"
        :type hand: str
        :param timeout: seconds to wait per step
        :type timeout: float
        :param wait: whether to wait for motion to complete (default True)
        :type wait: bool
        """
        h = "r" if hand == "right" else "l"

        # --- TUNING CONSTANTS ---
        # Thumb opposition: how far thumb_0 rotates toward the other fingers.
        # 0.0 = thumb points straight up/back (no opposition)
        # 1.0 = thumb fully opposes (rotated 90° to face the other fingertips)
        THUMB_OPPOSITION = 0.9

        # Per-joint-group flex scale: each value multiplies `amount` before
        # mapping onto that joint's full range. 1.0 = uses full range.
        THUMB_MCP_SCALE   = 1.0   # thumb  _1_joint
        THUMB_PIP_SCALE   = 1.0   # thumb  _2_joint
        THUMB_DIP_SCALE   = 1.0   # thumb  _3_joint
        FINGER_MCP_SCALE  = 1.0   # index/middle/ring/little  _1_joint  (base knuckle)
        FINGER_PIP_SCALE  = 1.0   # index/middle/ring/little  _2_joint
        FINGER_DIP_SCALE  = 1.0   # index/middle/ring/little  _3_joint
        # -------------------------

        # Step 1: bring thumb into opposition FIRST (before curling fingers)
        th0 = self.get_joint_handle(f"{h}_hand_thumb_0_joint")
        opp_target = th0.lower_limit + THUMB_OPPOSITION * (th0.upper_limit - th0.lower_limit)
        self.client.move_position(f"{h}_hand_thumb_0_joint", opp_target,
                                  wait=True, check_collision=False, timeout=timeout)

        # Step 2: flex all remaining joints simultaneously
        flex_joints = [
            # (joint_name,                        scale)
            (f"{h}_hand_thumb_1_joint",   amount * THUMB_MCP_SCALE),
            (f"{h}_hand_thumb_2_joint",   amount * THUMB_PIP_SCALE),
            (f"{h}_hand_thumb_3_joint",   amount * THUMB_DIP_SCALE),
            (f"{h}_hand_index_1_joint",   amount * FINGER_MCP_SCALE),
            (f"{h}_hand_index_2_joint",   amount * FINGER_PIP_SCALE),
            (f"{h}_hand_index_3_joint",   amount * FINGER_DIP_SCALE),
            (f"{h}_hand_middle_1_joint",  amount * FINGER_MCP_SCALE),
            (f"{h}_hand_middle_2_joint",  amount * FINGER_PIP_SCALE),
            (f"{h}_hand_middle_3_joint",  amount * FINGER_DIP_SCALE),
            (f"{h}_hand_ring_1_joint",    amount * FINGER_MCP_SCALE),
            (f"{h}_hand_ring_2_joint",    amount * FINGER_PIP_SCALE),
            (f"{h}_hand_ring_3_joint",    amount * FINGER_DIP_SCALE),
            (f"{h}_hand_little_1_joint",  amount * FINGER_MCP_SCALE),
            (f"{h}_hand_little_2_joint",  amount * FINGER_PIP_SCALE),
            (f"{h}_hand_little_3_joint",  amount * FINGER_DIP_SCALE),
        ]

        for joint_name, scaled_amount in flex_joints:
            jh = self.get_joint_handle(joint_name)
            target = jh.lower_limit + scaled_amount * (jh.upper_limit - jh.lower_limit)
            target = np.clip(target, jh.lower_limit, jh.upper_limit)
            self.client.move_position(joint_name, target, wait=False,
                                      check_collision=False, timeout=timeout, velocity=10)

        if wait:
            while not self.client.motion_done(check_collision=False):
                self.client.update_simulation()

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

        PREGRIP    = 0.20   # TUNE: pre-shape before approach
        FINAL_GRIP = 0.80   # TUNE: grip closure at grasp

        # ── Geometry ──────────────────────────────────────────────────────────
        UP_HEIGHT      = 0.15  # metres straight up in world Z after grasping

        # Per-hand offset applied to the target position (relative to ball centre).
        # Left hand is a 180°Y mirror of right → negate both X and Y components.
        RIGHT_OFFSET = np.array([ 0.12,  0.0, 0.0])  # TUNE
        LEFT_OFFSET  = np.array([RIGHT_OFFSET[0], RIGHT_OFFSET[1], -RIGHT_OFFSET[2]])

        # ── Orientations ──────────────────────────────────────────────────────
        # EE Y axis = "nail through the palm". For a side approach the palm should
        # face the ball, so EE_Y must point FROM hover TOWARD ball (±world Y).
        #
        # Reference (confirmed working for top-down right):
        #   [0.707, 0, 0, 0.707] → EE_Y = world +Z (up)
        #
        # For side approach we rotate that 90° in world to lay EE_Y along world ±Y:
        #   right hand approaches from +Y side → EE_Y = world −Y → [0, 0, 1, 0]  (180° around Z)
        #   left hand  approaches from −Y side → EE_Y = world +Y → [0, 0, 0, 1]  (identity)
        #
        # These are START POINTS — tune until the _mark frame matches desired palm direction.
        # Derivation: RIGHT flipped from [0,0,1,0] by ⊗180°Z → [0,0,0,1] (identity, EE_Y→+world_Y = away from ball)
        #             LEFT = RIGHT ⊗ [0,1,0,0] (known left-arm correction) → [0,1,0,0] (180°Y, EE_Y→+world_Y)
        RIGHT_ORI = [0, 0, 0, 1]   # right-hand: EE_Y = +world_Y (away from ball, hover at +Y side)
        # LEFT must have EE_Y = −world_Y (away from ball, hover at −Y side).
        # RIGHT with EE_Y→−world_Y = [0,0,1,0]; LEFT = [0,0,1,0] ⊗ [0,1,0,0] = [-1,0,0,0] ≡ [1,0,0,0]
        LEFT_ORI  = [0, 0, 1, 0]   # left-hand: EE_Y = −world_Y (away from ball, hover at −Y side)

        # ── 1. Ball world position ────────────────────────────────────────────
        u, v = int(center[0]), int(center[1])
        if self.fake_vision:
            ball_pos = np.array(self.get_3d_point(u, v))
        else:
            depth_img = self.get_depth()
            ball_pos  = np.array(self.get_3d_point(u, v, float(depth_img[v, u])))

        # ── 1.5. Enforce symmetric initial joint configuration ────────────────
        # Ensure both arms are in the same pose for balanced approach
        initial_joint_angles = {
            "r_shoulder_pitch": np.deg2rad(0),
            "r_shoulder_roll":  np.deg2rad(30),
            "r_shoulder_yaw":   np.deg2rad(0),
            "r_elbow":          np.deg2rad(30),
            "r_wrist_prosup":   np.deg2rad(0),
            "r_wrist_pitch":    np.deg2rad(0),
            "r_wrist_yaw":      np.deg2rad(15),
            "l_shoulder_pitch": np.deg2rad(0),  # ← symmetric with right
            "l_shoulder_roll":  np.deg2rad(30),
            "l_shoulder_yaw":   np.deg2rad(0),
            "l_elbow":          np.deg2rad(30),
            "l_wrist_prosup":   np.deg2rad(0),
            "l_wrist_pitch":    np.deg2rad(0),
            "l_wrist_yaw":      np.deg2rad(15),
            "neck_pitch":       np.deg2rad(-35),
            "r_hand_thumb_0_joint": np.deg2rad(0),
        }

        for joint_name, target_angle in initial_joint_angles.items():
            self.client.move_position(joint_name, target_angle, wait=False,
                                      check_collision=False, timeout=5)
        
        while not self.client.motion_done(check_collision=False):
            self.client.update_simulation()
        print("Initialized symmetric arm configuration")

        # ── 2. Both hands converge to ball symmetrically ──────────────────────
        # Calculate target positions for each hand using their own offsets/orientations
        target_pos_right = ball_pos + RIGHT_OFFSET
        target_pos_left  = ball_pos + LEFT_OFFSET
        
        up_pos_right = target_pos_right.copy()
        up_pos_right[2] += UP_HEIGHT
        
        up_pos_left = target_pos_left.copy()
        up_pos_left[2] += UP_HEIGHT

        # ── 3. Debug marks ────────────────────────────────────────────────────
        self._mark("target_r", target_pos_right, [1, 0.5, 0], ori=RIGHT_ORI)  # orange – right grasp
        self._mark("target_l", target_pos_left,  [1, 0.5, 0.5], ori=LEFT_ORI)   # magenta – left grasp
        self.client.update_simulation()

        # ── 4. Pre-shape both hands (parallel) ────────────────────────────────
        self.clutch(PREGRIP, hand="right", timeout=5, wait=False)
        self.clutch(PREGRIP, hand="left", timeout=5, wait=True)  # last one waits
        print(f"Pre-shaped both to {PREGRIP:.0%}")

        # ── 5. IK → target for both hands (iterative convergence, parallel) ───
        # Keep hammering each hand in parallel, with mirrored approach offsets
        # to avoid bumping the ball back and forth like ping-pong.
        
        max_iterations = 5  # TUNE: how many back-and-forth iterations
        convergence_threshold = 0.01  # metres - how close is "good enough"
        
        # Approach offset: how far to stand back from target before final positioning
        approach_offset = 0.05  # metres  # TUNE
        
        # Initialize adjusted targets (will be updated each iteration)
        # Right hand starts offset in -Y (away from target, since it approaches from +Y side)
        # Left hand starts offset in +Y (away from target, since it approaches from -Y side)
        target_pos_right_adjusted = target_pos_right + np.array([0, -approach_offset, 0])
        target_pos_left_adjusted = target_pos_left + np.array([0, +approach_offset, 0])
        
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            
            # ── Move both hands in parallel ────────────────────────────────────
            # Capture base state before moving
            base_pos_before, _ = self.client.getBasePositionAndOrientation(self.client.robot)
            base_pos_before = np.array(base_pos_before)
            
            # Compute IK for right hand
            self.set_ee("r_hand")
            ik_solution_r = np.array(self.client.calculateInverseKinematics(self.client.robot, 
                                                                             self.client.end_effector.link_id, 
                                                                             target_pos_right_adjusted,
                                                                             RIGHT_ORI,
                                                                             lowerLimits=self.client.IK_config["lower_limits"],
                                                                             upperLimits=self.client.IK_config["upper_limits"],
                                                                             jointRanges=self.client.IK_config["joint_ranges"],
                                                                             restPoses=self.client.IK_config["rest_poses"]))
            
            # Compute IK for left hand
            self.set_ee("l_hand")
            ik_solution_l = np.array(self.client.calculateInverseKinematics(self.client.robot, 
                                                                             self.client.end_effector.link_id, 
                                                                             target_pos_left_adjusted,
                                                                             LEFT_ORI,
                                                                             lowerLimits=self.client.IK_config["lower_limits"],
                                                                             upperLimits=self.client.IK_config["upper_limits"],
                                                                             jointRanges=self.client.IK_config["joint_ranges"],
                                                                             restPoses=self.client.IK_config["rest_poses"]))
            
            # Move both arms by issuing separate move_position calls for right and left arm joints
            movable_joints = self.client.IK_config["movable_joints"]
            
            # Separate joints by arm
            right_arm_indices = []
            left_arm_indices = []
            for idx, joint_id in enumerate(movable_joints):
                joint_name = None
                for joint in self.client.joints:
                    if joint.joints_id == joint_id:
                        joint_name = joint.name
                        break
                if joint_name:
                    if "l_" in joint_name and "l_hand" not in joint_name:
                        left_arm_indices.append(idx)
                    elif "r_" in joint_name and "r_hand" not in joint_name:
                        right_arm_indices.append(idx)
            
            # Move right arm (wait=False to allow parallel execution)
            right_joints = [movable_joints[i] for i in right_arm_indices]
            right_positions = [ik_solution_r[i] for i in right_arm_indices]
            if right_joints:
                self.client.move_position(right_joints, right_positions, wait=False,
                                          check_collision=False, timeout=8)
            
            # Move left arm (wait=False to allow parallel execution)
            left_joints = [movable_joints[i] for i in left_arm_indices]
            left_positions = [ik_solution_l[i] for i in left_arm_indices]
            if left_joints:
                self.client.move_position(left_joints, left_positions, wait=False,
                                          check_collision=False, timeout=8)
            
            # Wait for both to complete
            while not self.client.motion_done(check_collision=False):
                self.client.update_simulation()
            
            # Measure base drift after both hands moved
            base_pos_after, _ = self.client.getBasePositionAndOrientation(self.client.robot)
            base_pos_after = np.array(base_pos_after)
            base_drift = base_pos_after - base_pos_before
            
            # Get current hand positions and compute errors (from actual target, not offset target)
            r_hand_state = self.client.getLinkState(self.client.robot, 
                                                     [link.robot_link_id for link in self.client.links 
                                                      if link.name == "r_hand"][0], 0, 0)
            r_ee_pos = np.array(r_hand_state[0])
            r_error = np.linalg.norm(r_ee_pos - target_pos_right)
            
            l_hand_state = self.client.getLinkState(self.client.robot, 
                                                     [link.robot_link_id for link in self.client.links 
                                                      if link.name == "l_hand"][0], 0, 0)
            l_ee_pos = np.array(l_hand_state[0])
            l_error = np.linalg.norm(l_ee_pos - target_pos_left)
            
            print(f"  Right: error={r_error:.4f}m")
            print(f"  Left:  error={l_error:.4f}m")
            print(f"  Base drift: {np.linalg.norm(base_drift):.4f}m")
            
            # Update targets for next iteration: compensate for base drift, but maintain mirrored offsets
            target_pos_right_adjusted = target_pos_right + np.array([0, -approach_offset, 0]) + base_drift
            target_pos_left_adjusted = target_pos_left + np.array([0, +approach_offset, 0]) + base_drift
            
            # Check convergence
            if r_error < convergence_threshold and l_error < convergence_threshold:
                print(f"\nConverged after {iteration + 1} iterations!")
                break
        
        print("Both hands converged at target")

        # ── 5.5. Fine-tune grasp positioning with joint deltas ────────────────
        # Right hand
        grasp_deltas_r = {
            "r_shoulder_pitch": np.deg2rad(0),       # TUNE
            "r_shoulder_roll":  np.deg2rad(-10),     # TUNE
            "r_shoulder_yaw":   np.deg2rad(0),       # TUNE
            "r_elbow":          np.deg2rad(-10),     # TUNE
            "r_wrist_prosup":   np.deg2rad(0),       # TUNE
            "r_wrist_pitch":    np.deg2rad(0),       # TUNE
            "r_wrist_yaw":      np.deg2rad(0),       # TUNE
        }
        
        # Left hand (mirrored deltas)
        grasp_deltas_l = {
            "l_shoulder_pitch": np.deg2rad(0),       # TUNE (same as right)
            "l_shoulder_roll":  np.deg2rad(-10),     # TUNE (same as right)
            "l_shoulder_yaw":   np.deg2rad(0),       # TUNE (same as right)
            "l_elbow":          np.deg2rad(-10),     # TUNE (same as right)
            "l_wrist_prosup":   np.deg2rad(0),       # TUNE (same as right)
            "l_wrist_pitch":    np.deg2rad(0),       # TUNE (same as right)
            "l_wrist_yaw":      np.deg2rad(0),       # TUNE (same as right)
        }
        
        # Apply deltas to both hands in parallel
        for joint_name, delta_angle in grasp_deltas_r.items():
            current = self.client.get_joint_state(joint_name)[0]
            target = current + delta_angle
            self.client.move_position(joint_name, target,
                                      check_collision=False, timeout=3)
        
        for joint_name, delta_angle in grasp_deltas_l.items():
            current = self.client.get_joint_state(joint_name)[0]
            target = current + delta_angle
            self.client.move_position(joint_name, target,
                                      check_collision=False, timeout=3)
        
        while not self.client.motion_done(check_collision=False):
            self.client.update_simulation()
        print("Positioned both for grasp")

        # ── 6. Grasp both hands (parallel) ───────────────────────────────────
        self.clutch(FINAL_GRIP, hand="right", timeout=8, wait=False)
        self.clutch(FINAL_GRIP, hand="left", timeout=8, wait=True)  # last one waits
        print(f"Both grasped at {FINAL_GRIP:.0%}")

        # ── 7. Lift both straight up in world Z (iterative convergence, parallel) 
        # Keep hammering each hand in parallel with mirrored approach offsets
        
        # Initialize adjusted up positions with mirrored offsets
        up_pos_right_adjusted = up_pos_right + np.array([0, -approach_offset, 0])
        up_pos_left_adjusted = up_pos_left + np.array([0, +approach_offset, 0])
        
        for iteration in range(max_iterations):
            print(f"\n--- Lift Iteration {iteration + 1}/{max_iterations} ---")
            
            # ── Lift both hands in parallel ────────────────────────────────────
            # Capture base state before moving
            base_pos_before, _ = self.client.getBasePositionAndOrientation(self.client.robot)
            base_pos_before = np.array(base_pos_before)
            
            # Compute IK for right hand
            self.set_ee("r_hand")
            ik_solution_r = np.array(self.client.calculateInverseKinematics(self.client.robot, 
                                                                             self.client.end_effector.link_id, 
                                                                             up_pos_right_adjusted,
                                                                             RIGHT_ORI,
                                                                             lowerLimits=self.client.IK_config["lower_limits"],
                                                                             upperLimits=self.client.IK_config["upper_limits"],
                                                                             jointRanges=self.client.IK_config["joint_ranges"],
                                                                             restPoses=self.client.IK_config["rest_poses"]))
            
            # Compute IK for left hand
            self.set_ee("l_hand")
            ik_solution_l = np.array(self.client.calculateInverseKinematics(self.client.robot, 
                                                                             self.client.end_effector.link_id, 
                                                                             up_pos_left_adjusted,
                                                                             LEFT_ORI,
                                                                             lowerLimits=self.client.IK_config["lower_limits"],
                                                                             upperLimits=self.client.IK_config["upper_limits"],
                                                                             jointRanges=self.client.IK_config["joint_ranges"],
                                                                             restPoses=self.client.IK_config["rest_poses"]))
            
            # Move both arms by issuing separate move_position calls for right and left arm joints
            movable_joints = self.client.IK_config["movable_joints"]
            
            # Separate joints by arm
            right_arm_indices = []
            left_arm_indices = []
            for idx, joint_id in enumerate(movable_joints):
                joint_name = None
                for joint in self.client.joints:
                    if joint.joints_id == joint_id:
                        joint_name = joint.name
                        break
                if joint_name:
                    if "l_" in joint_name and "l_hand" not in joint_name:
                        left_arm_indices.append(idx)
                    elif "r_" in joint_name and "r_hand" not in joint_name:
                        right_arm_indices.append(idx)
            
            # Move right arm (wait=False to allow parallel execution)
            right_joints = [movable_joints[i] for i in right_arm_indices]
            right_positions = [ik_solution_r[i] for i in right_arm_indices]
            if right_joints:
                self.client.move_position(right_joints, right_positions, wait=False,
                                          check_collision=False, timeout=5)
            
            # Move left arm (wait=False to allow parallel execution)
            left_joints = [movable_joints[i] for i in left_arm_indices]
            left_positions = [ik_solution_l[i] for i in left_arm_indices]
            if left_joints:
                self.client.move_position(left_joints, left_positions, wait=False,
                                          check_collision=False, timeout=5)
            
            # Wait for both to complete
            while not self.client.motion_done(check_collision=False):
                self.client.update_simulation()
            
            # Measure base drift after both hands moved
            base_pos_after, _ = self.client.getBasePositionAndOrientation(self.client.robot)
            base_pos_after = np.array(base_pos_after)
            base_drift = base_pos_after - base_pos_before
            
            # Get current hand positions and compute errors (from actual target, not offset target)
            r_hand_state = self.client.getLinkState(self.client.robot, 
                                                     [link.robot_link_id for link in self.client.links 
                                                      if link.name == "r_hand"][0], 0, 0)
            r_ee_pos = np.array(r_hand_state[0])
            r_error = np.linalg.norm(r_ee_pos - up_pos_right)
            
            l_hand_state = self.client.getLinkState(self.client.robot, 
                                                     [link.robot_link_id for link in self.client.links 
                                                      if link.name == "l_hand"][0], 0, 0)
            l_ee_pos = np.array(l_hand_state[0])
            l_error = np.linalg.norm(l_ee_pos - up_pos_left)
            
            print(f"  Right: error={r_error:.4f}m")
            print(f"  Left:  error={l_error:.4f}m")
            print(f"  Base drift: {np.linalg.norm(base_drift):.4f}m")
            
            # Update targets for next iteration: compensate for base drift, but maintain mirrored offsets
            up_pos_right_adjusted = up_pos_right + np.array([0, -approach_offset, 0]) + base_drift
            up_pos_left_adjusted = up_pos_left + np.array([0, +approach_offset, 0]) + base_drift
            
            # Check convergence
            if r_error < convergence_threshold and l_error < convergence_threshold:
                print(f"\nLift converged after {iteration + 1} iterations!")
                break
        
        print("Both hands lifted")

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
    REPS = 1
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