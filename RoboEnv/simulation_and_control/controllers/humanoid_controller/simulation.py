import numpy as np
import dartpy as dart
import copy
from utils import *
import os
import ismpc
import matplotlib.pyplot as plt
import footstep_planner
import inverse_kinematics as ik
import filter
import foot_trajectory_generator as ftg
import time

class Hrp4Controller(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world, hrp4):
        super(Hrp4Controller, self).__init__(world)
        self.world = world
        self.hrp4 = hrp4
        self.dofs = self.hrp4.getNumDofs()
        world.setTimeStep(0.01)
        self.world_time_step = world.getTimeStep()
        first_swing = 'right'
        self.time = 0

        # robot links
        self.lsole = hrp4.getBodyNode('l_sole')
        self.rsole = hrp4.getBodyNode('r_sole')
        self.torso = hrp4.getBodyNode('torso')
        self.base  = hrp4.getBodyNode('body')

        for i in range(hrp4.getNumJoints()):
            joint = hrp4.getJoint(i)
            dim = joint.getNumDofs()

            # this sets the root joint to passive
            if dim == 6:
                joint.setActuatorType(dart.dynamics.ActuatorType.PASSIVE)

            # this sets the remaining joints as acceleration-controlled
            elif dim == 1:
                print(joint.getName())
                joint.setActuatorType(dart.dynamics.ActuatorType.ACCELERATION)

        # set initial configuration
        initial_configuration = {'CHEST_P': 0., 'CHEST_Y': 0., 'NECK_P': 0., 'NECK_Y': 0., \
                                 'R_HIP_Y': 0., 'R_HIP_R': -3., 'R_HIP_P': -25., 'R_KNEE_P': 50., 'R_ANKLE_P': -25., 'R_ANKLE_R':  3., \
                                 'L_HIP_Y': 0., 'L_HIP_R':  3., 'L_HIP_P': -25., 'L_KNEE_P': 50., 'L_ANKLE_P': -25., 'L_ANKLE_R': -3., \
                                 'R_SHOULDER_P': 4., 'R_SHOULDER_R': -8., 'R_SHOULDER_Y': 0., 'R_ELBOW_P': -25., \
                                 'L_SHOULDER_P': 4., 'L_SHOULDER_R':  8., 'L_SHOULDER_Y': 0., 'L_ELBOW_P': -25.}

        for joint_name, value in initial_configuration.items():
            self.hrp4.setPosition(self.hrp4.getDof(joint_name).getIndexInSkeleton(), value * np.pi / 180.)

        # position the robot on the ground
        lsole_pos = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        rsole_pos = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        self.hrp4.setPosition(5, - (lsole_pos[2] + rsole_pos[2]) / 2.)

        # store initial state
        self.initial = self.retrieve_state()
        self.contact = 'lsole' if first_swing == 'right' else 'rsole' # there is a dummy footstep

        self.desired = copy.deepcopy(self.initial)
        #self.initialize_plot()

        # selection matrix for redundant dofs
        redundant_dofs = [ \
            "NECK_Y", "NECK_P", \
            "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P", \
            "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P"]

        # initialize inverse kinematics
        self.ik = ik.InverseKinematics(self.hrp4, redundant_dofs)

        # initialize footstep planner
        self.footstep_planner = footstep_planner.FootstepPlanner(
            [(0.1, 0., 0.1)] * 20,
            self.initial.left_foot_pose,
            self.initial.right_foot_pose,
            'left' if self.contact == 'lsole' else 'right',
            self.world_time_step
            )

        # initialize MPC controller
        self.mpc = ismpc.Ismpc(self.initial, self.footstep_planner)

        # initialize foot trajectory generator
        self.foot_trajectory_generator = ftg.FootTrajectoryGenerator(self.initial, self.footstep_planner)

        # initialize kalman filter
        A = np.identity(3) + self.world_time_step * self.mpc.A_lip
        B = self.world_time_step * self.mpc.B_lip
        H = np.identity(3)
        Q = block_diag(1., 1., 1.)
        R = block_diag(1e1, 1e2, 1e4)
        P = np.identity(3)
        x = np.array([self.initial.com_position[0], self.initial.com_velocity[0], self.initial.zmp_position[0], \
                      self.initial.com_position[1], self.initial.com_velocity[1], self.initial.zmp_position[1]])
        self.kf = filter.KalmanFilter(block_diag(A, A), \
                                      block_diag(B, B), \
                                      block_diag(H, H), \
                                      block_diag(Q, Q), \
                                      block_diag(R, R), \
                                      block_diag(P, P), \
                                      x)

    def customPreStep(self):
        # create current and desired states
        self.current = self.retrieve_state()

        # update kalman filter
        u = np.array([self.desired.zmp_velocity[0], self.desired.zmp_velocity[1]])
        self.kf.predict(u)
        x_flt, P = self.kf.update(np.array([self.current.com_position[0], self.current.com_velocity[0], self.current.zmp_position[0], \
                                            self.current.com_position[1], self.current.com_velocity[1], self.current.zmp_position[1]]))
        
        # update current state
        self.current.com_position[0] = x_flt[0]
        self.current.com_velocity[0] = x_flt[1]
        self.current.zmp_position[0] = x_flt[2]
        self.current.com_position[1] = x_flt[3]
        self.current.com_velocity[1] = x_flt[4]
        self.current.zmp_position[1] = x_flt[5]

        # get references using MPC
        lip_state, contact = self.mpc.solve(self.current, self.time)
        if contact == 'ds':
            pass
        elif contact == 'ssleft':
            self.contact = 'lsole'
        elif contact == 'ssright':
            self.contact = 'rsole'

        self.desired.com_position     = lip_state.com_position
        self.desired.com_velocity     = lip_state.com_velocity
        self.desired.com_acceleration = lip_state.com_acceleration
        self.desired.zmp_position     = lip_state.zmp_position
        self.desired.zmp_velocity     = lip_state.zmp_velocity

        # get foot trajectories
        feet_trajectories = self.foot_trajectory_generator.generate_feet_trajectories_at_time(self.time)
        self.desired.left_foot_pose          = feet_trajectories['left']['pos']
        self.desired.left_foot_velocity      = feet_trajectories['left']['vel']
        self.desired.left_foot_acceleration  = feet_trajectories['left']['acc']
        self.desired.right_foot_pose         = feet_trajectories['right']['pos']
        self.desired.right_foot_velocity     = feet_trajectories['right']['vel']
        self.desired.right_foot_acceleration = feet_trajectories['right']['acc']

        # set torso and base references to the average of the feet
        self.desired.torso_orientation          = (self.desired.left_foot_pose[:3]         + self.desired.right_foot_pose[:3])         / 2.
        self.desired.torso_angular_velocity     = (self.desired.left_foot_velocity[:3]     + self.desired.right_foot_velocity[:3])     / 2.
        self.desired.torso_angular_acceleration = (self.desired.left_foot_acceleration[:3] + self.desired.right_foot_acceleration[:3]) / 2.
        self.desired.base_orientation           = (self.desired.left_foot_pose[:3]         + self.desired.right_foot_pose[:3])         / 2.
        self.desired.base_angular_velocity      = (self.desired.left_foot_velocity[:3]     + self.desired.right_foot_velocity[:3])     / 2.
        self.desired.base_angular_acceleration  = (self.desired.left_foot_acceleration[:3] + self.desired.right_foot_acceleration[:3]) / 2.

        # get acceleration commands using IK
        commands = self.ik.get_joint_accelerations(self.desired, self.current, self.contact)
        
        # set acceleration commands
        for i in range(self.dofs - 6):
            self.hrp4.setCommand(i + 6, commands[i])

        #self.update_plot()

        self.time += 1

    def retrieve_state(self):
        # com and torso pose (orientation and position)
        com_position = self.hrp4.getCOM()
        torso_orientation = get_rotvec(self.hrp4.getBodyNode('torso').getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).rotation())
        base_orientation  = get_rotvec(self.hrp4.getBodyNode('body' ).getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).rotation())

        # feet poses (orientation and position)
        l_foot_transform = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_orientation = get_rotvec(l_foot_transform.rotation())
        l_foot_position = l_foot_transform.translation()
        left_foot_pose = np.hstack((l_foot_orientation, l_foot_position))
        r_foot_transform = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_orientation = get_rotvec(r_foot_transform.rotation())
        r_foot_position = r_foot_transform.translation()
        right_foot_pose = np.hstack((r_foot_orientation, r_foot_position))

        # velocities
        com_velocity = self.hrp4.getCOMLinearVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        torso_angular_velocity = self.hrp4.getBodyNode('torso').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        base_angular_velocity = self.hrp4.getBodyNode('body').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_spatial_velocity = self.lsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_spatial_velocity = self.rsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())

        zmp_position = self.get_zmp()

        # create State object
        return State(
            ndofs                      = self.dofs,
            left_foot_pose             = left_foot_pose,
            right_foot_pose            = right_foot_pose,
            com_position               = com_position,
            torso_orientation          = torso_orientation,
            base_orientation           = base_orientation,
            left_foot_velocity         = l_foot_spatial_velocity,
            right_foot_velocity        = r_foot_spatial_velocity,
            com_velocity               = com_velocity,
            torso_angular_velocity     = torso_angular_velocity,
            base_angular_velocity      = base_angular_velocity,
            left_foot_acceleration     = np.zeros(6),
            right_foot_acceleration    = np.zeros(6),
            com_acceleration           = np.zeros(3),
            torso_angular_acceleration = np.zeros(3),
            base_angular_acceleration  = np.zeros(3),
            joint_position             = self.hrp4.getPositions(),
            joint_velocity             = self.hrp4.getVelocities(),
            joint_acceleration         = np.zeros(self.dofs),
            zmp_position               = zmp_position,
            zmp_velocity               = np.zeros(3)
        )

    def get_zmp(self):
        total_vertical_force = 0.
        zmp = np.zeros(2)
        for contact in world.getLastCollisionResult().getContacts():
            total_vertical_force += contact.force[2]
            zmp[0] += contact.point[0] * contact.force[2]
            zmp[1] += contact.point[1] * contact.force[2]

        if total_vertical_force <= 0.1: # threshold for when we lose contact
            return np.array([0., 0., 0.]) # FIXME: this should return previous measurement
        else:
            zmp /= total_vertical_force
            # sometimes we get contact points that dont make sense, so we clip the ZMP close to the robot
            midpoint = (self.current.left_foot_pose[3:5] + self.current.right_foot_pose[3:5]) / 2.
            zmp[0] = np.clip(zmp[0], midpoint[0] - 0.3, midpoint[0] + 0.3)
            zmp[1] = np.clip(zmp[1], midpoint[1] - 0.3, midpoint[1] + 0.3)
            return zmp
        
    def initialize_plot(self):
        self.fig, self.ax = plt.subplots(6, 1, figsize=(6, 8))
        self.com_x_data_1, self.com_y_data_1, self.com_z_data_1 = [], [], []
        self.com_x_data_2, self.com_y_data_2, self.com_z_data_2 = [], [], []
        self.foot_x_data_1, self.foot_y_data_1, self.foot_z_data_1 = [], [], []
        self.foot_x_data_2, self.foot_y_data_2, self.foot_z_data_2 = [], [], []
        self.zmp_x_data_1, self.zmp_y_data_1 = [], []
        self.zmp_x_data_2, self.zmp_y_data_2 = [], []

        self.line_com_x_1, = self.ax[0].plot([], [], color='blue')
        self.line_com_y_1, = self.ax[1].plot([], [], color='blue')
        self.line_com_z_1, = self.ax[2].plot([], [], color='blue')
        
        self.line_com_x_2, = self.ax[0].plot([], [], color='red', linestyle='--')
        self.line_com_y_2, = self.ax[1].plot([], [], color='red', linestyle='--')
        self.line_com_z_2, = self.ax[2].plot([], [], color='red', linestyle='--')

        self.line_zmp_1, = self.ax[0].plot([], [], color='green')
        self.line_zmp_1, = self.ax[1].plot([], [], color='green')

        self.line_foot_x_1, = self.ax[3].plot([], [], color='blue')
        self.line_foot_y_1, = self.ax[4].plot([], [], color='blue')
        self.line_foot_z_1, = self.ax[5].plot([], [], color='blue')

        self.line_foot_x_2, = self.ax[3].plot([], [], color='red')
        self.line_foot_y_2, = self.ax[4].plot([], [], color='red')
        self.line_foot_z_2, = self.ax[5].plot([], [], color='red')
        
        self.line_zmp_2, = self.ax[0].plot([], [], color='orange')
        self.line_zmp_2, = self.ax[1].plot([], [], color='orange')

        plt.ion()
        plt.show()

    def update_plot(self):
        # append data
        self.com_x_data_1 .append(self.desired.com_position[0])
        self.com_y_data_1 .append(self.desired.com_position[1])
        self.com_z_data_1 .append(self.desired.com_position[2])
        self.com_x_data_2 .append(self.current.com_position[0])
        self.com_y_data_2 .append(self.current.com_position[1])
        self.com_z_data_2 .append(self.current.com_position[2])
        self.foot_x_data_1.append(self.desired.left_foot_pose[3])
        self.foot_y_data_1.append(self.desired.left_foot_pose[4])
        self.foot_z_data_1.append(self.desired.left_foot_pose[5])
        self.foot_x_data_2.append(self.current.left_foot_pose[3])
        self.foot_y_data_2.append(self.current.left_foot_pose[4])
        self.foot_z_data_2.append(self.current.left_foot_pose[5])
        self.zmp_x_data_1.append(self.desired.zmp_position[0])
        self.zmp_y_data_1.append(self.desired.zmp_position[1])
        self.zmp_x_data_2.append(self.current.zmp_position[0])
        self.zmp_y_data_2.append(self.current.zmp_position[1])

        # update com plots
        self.line_com_x_1.set_data(np.arange(len(self.com_x_data_1)), self.com_x_data_1)
        self.line_com_y_1.set_data(np.arange(len(self.com_y_data_1)), self.com_y_data_1)
        self.line_com_z_1.set_data(np.arange(len(self.com_z_data_1)), self.com_z_data_1)
        self.line_com_x_2.set_data(np.arange(len(self.com_x_data_2)), self.com_x_data_2)
        self.line_com_y_2.set_data(np.arange(len(self.com_y_data_2)), self.com_y_data_2)
        self.line_com_z_2.set_data(np.arange(len(self.com_z_data_2)), self.com_z_data_2)
            
        # update foot plots
        self.line_foot_x_1.set_data(np.arange(len(self.foot_x_data_1)), self.foot_x_data_1)
        self.line_foot_y_1.set_data(np.arange(len(self.foot_y_data_1)), self.foot_y_data_1)
        self.line_foot_z_1.set_data(np.arange(len(self.foot_z_data_1)), self.foot_z_data_1)
        self.line_foot_x_2.set_data(np.arange(len(self.foot_x_data_2)), self.foot_x_data_2)
        self.line_foot_y_2.set_data(np.arange(len(self.foot_y_data_2)), self.foot_y_data_2)
        self.line_foot_z_2.set_data(np.arange(len(self.foot_z_data_2)), self.foot_z_data_2)

        # update zmp
        self.line_zmp_1.set_data(np.arange(len(self.zmp_x_data_1)), self.zmp_x_data_1)
        self.line_zmp_1.set_data(np.arange(len(self.zmp_y_data_1)), self.zmp_y_data_1)
        self.line_zmp_2.set_data(np.arange(len(self.zmp_x_data_2)), self.zmp_x_data_2)
        self.line_zmp_2.set_data(np.arange(len(self.zmp_y_data_2)), self.zmp_y_data_2)

        # set limits
        for i in range(6):
            self.ax[i].relim()
            self.ax[i].autoscale_view()
            
        # redraw the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == "__main__":
    world = dart.simulation.World()

    urdfParser = dart.utils.DartLoader()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hrp4   = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "hrp4.urdf"))
    ground = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "ground.urdf"))
    world.addSkeleton(hrp4)
    world.addSkeleton(ground)
    world.setGravity([0, 0, -9.81])

    # set default inertia
    default_inertia = dart.dynamics.Inertia(1e-8, np.zeros(3), 1e-10 * np.identity(3))
    for body in hrp4.getBodyNodes():
        if body.getMass() == 0.0:
            body.setMass(1e-8)
            body.setInertia(default_inertia)

    node = Hrp4Controller(world, hrp4)

    # create world node and add it to viewer
    viewer = dart.gui.osg.Viewer()
    node.setTargetRealTimeFactor(10) # speed up the visualization by 10x
    viewer.addWorldNode(node)

    #viewer.setUpViewInWindow(0, 0, 1920, 1080)
    viewer.setUpViewInWindow(0, 0, 1280, 720)
    #viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([5., -1., 1.5],
                                 [1.,  0., 0.5],
                                 [0.,  0., 1. ])
    viewer.run()