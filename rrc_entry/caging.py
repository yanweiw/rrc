#!/usr/bin/env python3

"""
To demonstrate reaching a randomly set target point in the arena using torque
control by directly specifying the position of the target only.
"""
import argparse
import numpy as np
import random
import time
from rrc_simulation import sim_finger, visual_objects, sample, collision_objects, camera
from rrc_simulation import finger_types_data, geometric_ik
from rrc_simulation.tasks import move_cube
import random
import time

class Cage:

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--control-mode",
        default="position",
        choices=["position", "torque"],
        help="Specify position or torque as the control mode.",
    )
    argparser.add_argument(
        "--finger-type",
        default="trifingerone",
        choices=finger_types_data.get_valid_finger_types(),
        help="Specify valid finger type.",
    )
    args = argparser.parse_args()
    
    def __init__(self):

        self.time_step = 0.004
        self.finger = sim_finger.SimFinger(finger_type=Cage.args.finger_type,
                                           time_step=self.time_step,
                                           enable_visualization=True,)
        self.num_fingers = self.finger.number_of_fingers
        _kwargs = {"physicsClientId": self.finger._pybullet_client_id}

        if Cage.args.control_mode == "position":
            self.position_goals = visual_objects.Marker(number_of_goals=self.num_fingers)

        self.initial_robot_position=np.array([0.0, np.deg2rad(-60), np.deg2rad(-60)] * 3, dtype=np.float32,)
        self.finger.reset_finger_positions_and_velocities(self.initial_robot_position)
        self.initial_object_pose = move_cube.Pose(position=[0, 0, move_cube._min_height],
                                                  orientation=move_cube.Pose().orientation)
        self.cube = collision_objects.Block(self.initial_object_pose.position, 
                                            self.initial_object_pose.orientation, mass=0.020, **_kwargs)
        self.cube_z = 0.0325
        self.workspace_radius = 0.18
        self.tricamera = camera.TriFingerCameras(**_kwargs)

        self.initial_marker_position = np.array([[ 0.        ,  0.17856407,  self.cube_z],
                                                 [ 0.15464102, -0.08928203,  self.cube_z],
                                                 [-0.15464102, -0.08928203,  self.cube_z]])
        self.position_goals.set_state(self.initial_marker_position)
        self.marker_position = self.initial_marker_position

        self.ik_module = geometric_ik.GeometricIK()

    
    def sample_cube(self):
        random_ori = random.random()
        random_x = random.random() * 0.2 - 0.1
        random_y = random.random() * 0.2 - 0.1
        self.cube.set_state(position=[random_x, random_y, self.cube_z], orientation=[0.0, 0.0, random_ori, 1])
        self.position_goals.set_state(self.initial_marker_position)
        self.marker_position = self.initial_marker_position
        return [random_x, random_y, self.cube_z]


    def cage(self, gain=0.1):
        # calculate triangle center to cube center vector
        cube_position = np.array(self.cube.get_state()[0])
        # print('cube position', cube_position)
        marker_center = self.marker_position.mean(axis=0)
        # print('marker center', marker_center)
        center_to_cube = cube_position - marker_center # center refers to marker center
        # print('center_to_cube', center_to_cube)
        # calculate marker-shrink-towards-center vector
        marker_to_center = marker_center - self.marker_position
        # print('marker_to_center\n', marker_to_center)
        marker_change = (marker_to_center + 5*center_to_cube) * gain
        # print('marker_change\n', marker_change)
        # print('marker_position\n', self.marker_position)
        desired_marker_position = self.marker_position + marker_change

        # desired_marker_position can have some markers outside of workspace
        # print('desired_marker_position\n', desired_marker_position)
        marker_norms = np.linalg.norm(desired_marker_position[:, :2], axis=1)
        # print('marker_norms', marker_norms)
        max_idx = marker_norms.argmax()
        if marker_norms[max_idx] > self.workspace_radius:
            furthest_marker = desired_marker_position[max_idx]
            # print('furthest_marker', furthest_marker)
            cube_to_furthest = furthest_marker - cube_position
            a = np.linalg.norm(cube_to_furthest)**2
            b = 2 * cube_position.dot(cube_to_furthest)
            c = np.linalg.norm(cube_position)**2 - self.workspace_radius**2
            clipping_factor1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
            clipping_factor2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
            # print('clipping_factors', clipping_factor1, clipping_factor2)
            cube_to_marker = desired_marker_position - cube_position
            # print('original cube to desired marker\n', cube_to_marker)
            desired_marker_position = cube_position + cube_to_marker * clipping_factor1
            # print('new desired_marker_position\n', desired_marker_position)

        # recalculate marker change due to clippng, and apply gain
        marker_change = (desired_marker_position - self.marker_position)
        self.marker_position = self.marker_position + marker_change
        self.position_goals.set_state(self.marker_position)

        return self.marker_position # these are the three desired end-effector position


    def demo(self, num):
        for i in range(num):
            _ = self.sample_cube()
            time.sleep(0.5)
            for j in range(15):
                _ = self.cage()
                time.sleep(0.2)