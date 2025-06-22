import casadi as ca
from scipy.spatial.transform import Rotation as R
import numpy as np

def rotation_vector_difference(rotvec_a, rotvec_b):
    R_a = R.from_rotvec(rotvec_a)
    R_b = R.from_rotvec(rotvec_b)
    R_diff = R_b.inv() * R_a
    return R_diff.as_rotvec()

def pose_difference(pose_a, pose_b):
    pos_diff = pose_a[:3] - pose_b[:3]
    rot_diff = rotation_vector_difference(pose_a[3:], pose_b[3:])
    return np.hstack((pos_diff, rot_diff))

# converts a rotation matrix to a rotation vector
def get_rotvec(rot_matrix):
    rotation = R.from_matrix(rot_matrix)
    return rotation.as_rotvec()

import numpy as np

def block_diag(*arrays):
    arrays = [np.atleast_2d(a) if np.isscalar(a) else np.atleast_2d(a) for a in arrays]

    rows = sum(arr.shape[0] for arr in arrays)
    cols = sum(arr.shape[1] for arr in arrays)
    block_matrix = np.zeros((rows, cols), dtype=arrays[0].dtype)

    current_row = 0
    current_col = 0

    for arr in arrays:
        r, c = arr.shape
        block_matrix[current_row:current_row + r, current_col:current_col + c] = arr
        current_row += r
        current_col += c

    return block_matrix

# solves an unconstrained QP with casadi
class QPSolver:
    def __init__(self, n_dofs):
        self.n_dofs = n_dofs
        self.opti = ca.Opti('conic')
        self.x = self.opti.variable(self.n_dofs)

        # objective function: (1/2) * x.T @ H @ x + F.T @ x
        self.F_ = self.opti.parameter(self.n_dofs)
        self.H_ = self.opti.parameter(self.n_dofs, self.n_dofs)
        objective = 0.5 * ca.mtimes([self.x.T, self.H_, self.x]) + ca.mtimes(self.F_.T, self.x)
        self.opti.minimize(objective)

        # solver options
        p_opts = {'expand': True}
        s_opts = {'max_iter': 1000, 'verbose': False}
        self.opti.solver('osqp', p_opts, s_opts)

    def set_values(self, H, F):
        self.opti.set_value(self.F_, F)
        self.opti.set_value(self.H_, H)

    def solve(self):
        solution = self.opti.solve()
        q_ddot_des = solution.value(self.x)
        return q_ddot_des
    
class LipState:
    def __init__(self,
                 com_position=None,
                 com_velocity=None,
                 com_acceleration=None,
                 zmp_position=None,
                 zmp_velocity=None):
        
        self.com_position     = com_position     if com_position     is not None else np.zeros(3)
        self.com_velocity     = com_velocity     if com_velocity     is not None else np.zeros(3)
        self.com_acceleration = com_acceleration if com_acceleration is not None else np.zeros(3)
        self.zmp_position     = zmp_position     if zmp_position     is not None else np.zeros(3)
        self.zmp_velocity     = zmp_velocity     if zmp_velocity     is not None else np.zeros(3)

class State:
    def __init__(self, ndofs,
                 left_foot_pose=None, 
                 right_foot_pose=None, 
                 com_position=None,
                 torso_orientation=None,
                 base_orientation=None,
                 left_foot_velocity=None, 
                 right_foot_velocity=None, 
                 com_velocity=None,
                 torso_angular_velocity=None,
                 base_angular_velocity=None,
                 left_foot_acceleration=None, 
                 right_foot_acceleration=None, 
                 com_acceleration=None,
                 torso_angular_acceleration=None,
                 base_angular_acceleration=None,
                 joint_position=None, 
                 joint_velocity=None, 
                 joint_acceleration=None,
                 zmp_position=None,
                 zmp_velocity=None):
        
        self.ndofs = ndofs
        self.left_foot_pose             = left_foot_pose              if left_foot_pose             is not None else np.zeros(6)
        self.right_foot_pose            = right_foot_pose             if right_foot_pose            is not None else np.zeros(6)
        self.com_position               = com_position                if com_position               is not None else np.zeros(3)
        self.torso_orientation          = torso_orientation           if torso_orientation          is not None else np.zeros(3)
        self.base_orientation           = base_orientation            if base_orientation           is not None else np.zeros(3)
        self.left_foot_velocity         = left_foot_velocity          if left_foot_velocity         is not None else np.zeros(6)
        self.right_foot_velocity        = right_foot_velocity         if right_foot_velocity        is not None else np.zeros(6)
        self.com_velocity               = com_velocity                if com_velocity               is not None else np.zeros(3)
        self.torso_angular_velocity     = torso_angular_velocity      if torso_angular_velocity     is not None else np.zeros(3)
        self.base_angular_velocity      = base_angular_velocity       if base_angular_velocity      is not None else np.zeros(3)
        self.left_foot_acceleration     = left_foot_acceleration      if left_foot_acceleration     is not None else np.zeros(6)
        self.right_foot_acceleration    = right_foot_acceleration     if right_foot_acceleration    is not None else np.zeros(6)
        self.com_acceleration           = com_acceleration            if com_acceleration           is not None else np.zeros(3)
        self.torso_angular_acceleration = torso_angular_acceleration  if torso_angular_acceleration is not None else np.zeros(3)
        self.base_angular_acceleration  = base_angular_acceleration   if base_angular_acceleration  is not None else np.zeros(3)
        self.joint_position             = joint_position              if joint_position             is not None else np.zeros(self.ndofs)
        self.joint_velocity             = joint_velocity              if joint_velocity             is not None else np.zeros(self.ndofs)
        self.joint_acceleration         = joint_acceleration          if joint_acceleration         is not None else np.zeros(self.ndofs)
        self.zmp_position               = zmp_position                if zmp_position               is not None else np.zeros(3)
        self.zmp_velocity               = zmp_velocity                if zmp_velocity               is not None else np.zeros(3)