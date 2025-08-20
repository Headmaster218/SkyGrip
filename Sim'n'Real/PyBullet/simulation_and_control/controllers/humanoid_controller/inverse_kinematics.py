import dartpy as dart
import numpy as np
from utils import *

class InverseKinematics:
    def __init__(self, robot, redundant_dofs):
        self.robot = robot
        self.dofs = self.robot.getNumDofs()

        # initialize QP solver
        self.qp_solver = QPSolver(self.dofs)

        # selection matrix for redundant dofs
        self.joint_selection = np.zeros((self.dofs, self.dofs))
        for i in range(self.dofs):
            joint_name = self.robot.getDof(i).getName()
            if joint_name in redundant_dofs:
                self.joint_selection[i, i] = 1
    
    def get_joint_accelerations(self, desired, current, supportFoot):
        # self.robot parameters
        lsole = self.robot.getBodyNode('l_sole')
        rsole = self.robot.getBodyNode('r_sole')
        torso = self.robot.getBodyNode('torso')
        base  = self.robot.getBodyNode('body')

        # weights and gains
        tasks = ['lsole', 'rsole', 'com', 'torso', 'base', 'joints']
        weights   = {'lsole':  1., 'rsole':  1., 'com':  1., 'torso': 1. , 'base': 1. , 'joints': 1.e-2}
        pos_gains = {'lsole':  5., 'rsole':  5., 'com':  5., 'torso': 1, 'base': 1, 'joints': 10.   }
        vel_gains = {'lsole': 10., 'rsole': 10., 'com': 10., 'torso': 2, 'base': 2, 'joints': 1.e-1}

        # jacobians
        J = {'lsole' : self.robot.getJacobian(lsole,        inCoordinatesOf=dart.dynamics.Frame.World()),
             'rsole' : self.robot.getJacobian(rsole,        inCoordinatesOf=dart.dynamics.Frame.World()),
             'com'   : self.robot.getCOMLinearJacobian(     inCoordinatesOf=dart.dynamics.Frame.World()),
             'torso' : self.robot.getAngularJacobian(torso, inCoordinatesOf=dart.dynamics.Frame.World()),
             'base'  : self.robot.getAngularJacobian(base,  inCoordinatesOf=dart.dynamics.Frame.World()),
             'joints': self.joint_selection}

        # jacobians derivatives
        Jdot = {'lsole' : self.robot.getJacobianClassicDeriv(lsole, inCoordinatesOf=dart.dynamics.Frame.World()),
                'rsole' : self.robot.getJacobianClassicDeriv(rsole, inCoordinatesOf=dart.dynamics.Frame.World()),
                'com'   : self.robot.getCOMLinearJacobianDeriv(     inCoordinatesOf=dart.dynamics.Frame.World()),
                'torso' : self.robot.getAngularJacobianDeriv(torso, inCoordinatesOf=dart.dynamics.Frame.World()),
                'base'  : self.robot.getAngularJacobianDeriv(base,  inCoordinatesOf=dart.dynamics.Frame.World()),
                'joints': np.zeros((self.dofs, self.dofs))}

        # feedforward terms
        ff = {'lsole' : desired.left_foot_acceleration,
              'rsole' : desired.right_foot_acceleration,
              'com'   : desired.com_acceleration,
              'torso' : desired.torso_angular_acceleration,
              'base'  : desired.base_angular_acceleration,
              'joints': desired.joint_acceleration}

        # error vectors
        pos_error = {'lsole' : pose_difference(desired.left_foot_pose , current.left_foot_pose ),
                     'rsole' : pose_difference(desired.right_foot_pose, current.right_foot_pose),
                     'com'   : desired.com_position - current.com_position,
                     'torso' : rotation_vector_difference(desired.torso_orientation, current.torso_orientation),
                     'base'  : rotation_vector_difference(desired.base_orientation , current.base_orientation ),
                     'joints': desired.joint_position - current.joint_position}

        # velocity error vectors
        vel_error = {'lsole' : desired.left_foot_velocity     - current.left_foot_velocity,
                     'rsole' : desired.right_foot_velocity    - current.right_foot_velocity,
                     'com'   : desired.com_velocity           - current.com_velocity,
                     'torso' : desired.torso_angular_velocity - current.torso_angular_velocity,
                     'base'  : desired.base_angular_velocity  - current.base_angular_velocity,
                     'joints': desired.joint_velocity         - current.joint_velocity}
        
        # cost function
        cost_function_H = np.zeros((self.dofs, self.dofs))
        cost_function_F = np.zeros(self.dofs)
        for task in tasks:
            cost_function_H +=   weights[task] * J[task].T @ J[task]
            cost_function_F += - weights[task] * J[task].T @ (ff[task]
                                                            + vel_gains[task] * vel_error[task]
                                                            + pos_gains[task] * pos_error[task]
                                                            - Jdot[task] @ current.joint_velocity)

        # solve QP using casadi
        self.qp_solver.set_values(cost_function_H, cost_function_F)
        q_ddot_des = self.qp_solver.solve()

        return q_ddot_des[- (self.dofs - 6):]