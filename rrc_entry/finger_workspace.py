import argparse
import numpy as np
import random
import time
from rrc_simulation import sim_finger, visual_objects, sample
from rrc_simulation import finger_types_data
import pybullet

def main():
    '''
    sample joint positions to illustrate end-effector workspace
    '''
    time_step = 0.004

    finger = sim_finger.SimFinger(
        finger_type='fingerone',
        time_step=time_step,
        enable_visualization=True,
    )
    num_fingers = finger.number_of_fingers

    position_goals = visual_objects.Marker(number_of_goals=num_fingers)

    for i in range(500): 
        jpos = np.array(sample.random_joint_positions(number_of_fingers=1)) 
        marker_id = pybullet.createVisualShape(shapeType=pybullet.GEOM_SPHERE, radius=0.015, rgbaColor=[1, 0, 0, 0.5]) 
        goal_id = pybullet.createMultiBody(baseVisualShapeIndex=marker_id, basePosition=[0.18, 0.18, 0.08], baseOrientation=[0, 0, 0, 1]) 
        pybullet.resetBasePositionAndOrientation(goal_id, finger.pinocchio_utils.forward_kinematics(jpos)[0], [0, 0, 0, 1])

if  __name__ == '__main__':
    main()