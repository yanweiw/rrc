#!/usr/bin/env python3
import numpy as np

class GeometricIK():
    """
    Initialize GeometricIK Class.
    TODO(terry-suh): It'd be good if the initializer gets urdfs instead of 
    hard-coded geometry setups here.

    Frame definitions:
      W refers to the world frame (base-link)
      B refers to the base of each fingers 
      T refers to end effector of each fingers. 

    The notation follows Drake's monogram notation.
    """
    def __init__(self):

        # Geometric parameters of the platform.
        # TODO(terry-suh): These should ideally come from urdf file.
        self.L = 0.16 # Length of each finger
        self.R = 0.04 # Radius of tri finger arrangement.
        self.H = 0.34 # Height of the fingers w.r.t. world frame

        # Basic Transformations. Invert them at initialization for speed.
        self.X_B0W = np.linalg.inv(self.get_X_WB(0))
        self.X_B1W = np.linalg.inv(self.get_X_WB(2. * np.pi / 3.))
        self.X_B2W = np.linalg.inv(self.get_X_WB(4. * np.pi / 3.))

        # Store them in a list to make them iterable.
        self.X_BW_lst = [self.X_B0W, self.X_B1W, self.X_B2W]

    def get_X_WB(self, rads):
        X_WB = \
            np.array([[ np.cos(rads), np.sin(rads), 0, self.R * np.sin(rads)],
                      [-np.sin(rads), np.cos(rads), 0, self.R * np.cos(rads)],
                      [            0,            0, 1,                self.H],
                      [            0,            0, 0,                     1]]).astype(np.double)
        return X_WB


    """
    Given 9 dimensional vector x, convert to 9 dimensional joint coordinates q.
    Input: [f0_x, f0_y, f0_z, f1_x, f1_y, f1_z, f2_x, f2_y, f2_z]
    Output: [f0_q0, f0_q1, f0_q2, f1_q0, f1_q1, f1_q2, f2_q0, f2_q1, f2_q2]
    where q0 is the proximal joint and q2 is the distal one.
    """
    def compute_ik(self, x):
        
        assert (len(x) == 9)

        # Parse x into 3 points.
        p_WT_lst = np.split(x, 3)
        q_lst = np.zeros(9)

        
        # Compute IK for each finger
        k = 0
        for i in range(3):
            q = self.compute_individual_ik(p_WT_lst[i], self.X_BW_lst[i])
            q_lst[k] = q[0]
            q_lst[k + 1] = q[1]
            q_lst[k + 2] = q[2]

            # Do this to avoid kinematic singularities.
            # TODO(terry-suh): Let me know if this causes any problems.
            assert (q[2] < 0)
            k += 3
            
        return q_lst
            
    """
    Compute IK for an individual finger. The method here is to do a symbolic
    inverse since this is a textbook IK example.
    """
    def compute_individual_ik(self, p_WT, X_BW):
        # Convert coordinates.
        p_BT = self.undo_homogeneous(X_BW.dot(self.make_homogeneous(p_WT)))

        q0 = -np.arctan2(p_BT[2], p_BT[0]) - np.pi / 2.

        r = np.linalg.norm(p_BT, ord=2)

        q2 = np.arccos((2. * (self.L ** 2.0) - r ** 2.0) /
                        (2. * (self.L ** 2.0))) - np.pi
        
        q1 = -np.arcsin(p_BT[1] / r) + q2 / 2.

        return np.array([q0, q1, q2])

    def make_homogeneous(self, x):
        return np.append(x, 1)

    def undo_homogeneous(self, x):
        return x[0:3]    
        

        
