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
        cube_position = np.array(self.cube.get_state()[0])
        marker_center = self.marker_position.mean(axis=0)
        marker_to_center = marker_center - self.marker_position
        center_to_cube = cube_position - marker_center
        marker_change = (marker_to_center + center_to_cube) * gain
        self.marker_position = self.marker_position + marker_change
        self.position_goals.set_state(self.marker_position)


    def demo(self, num):
        for i in range(num):
            _ = self.sample_cube()
            for j in range(15):
                self.cage()
                time.sleep(0.2)