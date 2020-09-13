#!/usr/bin/env python3

"""
To demonstrate reaching a randomly set target point in the arena using torque
control by directly specifying the position of the target only.
"""
import argparse
import numpy as np
import random
import time
from rrc_simulation import sim_finger, visual_objects, sample
from rrc_simulation import finger_types_data
from rrc_simulation import geometric_ik


def main():

    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        "--demo_trajectory",
        default="phase_walking",
        choices=["phase_walking", "circle_drawing", "setpoint"],
        help="Choose demo to run!",
        )

    args = argparser.parse_args()

    time_step = 0.004

    finger = sim_finger.SimFinger(
        finger_type="tri",
        time_step=time_step,
        enable_visualization=True,
    )
    num_fingers = finger.number_of_fingers
    ik_module = geometric_ik.GeometricIK()

    #position_goals = visual_objects.Marker(number_of_goals=num_fingers)

    # First, go to some preconfigured joint trajectory so that we won't run into
    # singularities for the IK module.

    desired_joint_positions = np.array([0.0, -0.5, -0.5, 0.0, -0.5, -0.5, 0.0, -0.5, -0.5])
    finger_action = finger.Action(position=desired_joint_positions)

    # Wait 1 second for the simulation to get there....
    for _ in range(int(1 / time_step)):
        t = finger.append_desired_action(finger_action)
        finger.get_observation(t)
        time.sleep(time_step)

    while True:

        # Time to dance!
        current_time = time.time()

        if (args.demo_trajectory == "phase_walking"):
            desired_ee_trajectory = np.array([0.1 * np.sin(0),
                                              0.1 * np.cos(0),
                                              0.1 + 0.05 * np.sin(current_time), # phase 0
                                              0.1 * np.sin((2./3.) * np.pi),
                                              0.1 * np.cos((2./3.) * np.pi),
                                              0.1 + 0.05 * np.sin(
                                                  current_time + (2./3.) * np.pi),
                                              0.1 * np.sin((4./3.) * np.pi),
                                              0.1 * np.cos((4./3.) * np.pi),
                                              0.1 + 0.05 * np.sin(
                                                  current_time + (4./3.) * np.pi)])
        elif (args.demo_trajectory == "circle_drawing"):
            desired_ee_trajectory = np.array([0.1 * np.sin(0) + 0.05 * np.cos(current_time),
                                              0.1 * np.cos(0) + 0.05 * np.sin(current_time),
                                              0.1,
                                              0.1 * np.sin((2./3.) * np.pi) +
                                                0.05 * np.cos(current_time),
                                              0.1 * np.cos((2./3.) * np.pi) +
                                                0.05 * np.sin(current_time),
                                              0.1,
                                              0.1 * np.sin((4./3.) * np.pi) +
                                                0.05 * np.cos(current_time),
                                              0.1 * np.cos((4./3.) * np.pi) +
                                                0.05 * np.sin(current_time),
                                              0.1])
            
        elif (args.demo_trajectory == "setpoint"):
            desired_ee_trajectory = np.array([0., 0.1, 0.15, 0.1, 0.0, 0.15, -0.1, 0.0, 0.15])
                                              
        desired_joint_positions = ik_module.compute_ik(desired_ee_trajectory)
        finger_action = finger.Action(position=desired_joint_positions)

        t = finger.append_desired_action(finger_action)
        finger.get_observation(t)
        time.sleep(time_step)

if __name__ == "__main__":
    main()
