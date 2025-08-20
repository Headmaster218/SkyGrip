import pinocchio as pin
import json
import os
import numpy as np
from collections import OrderedDict
missing_robot_description = False
try:
    import robot_descriptions as rd
except ImportError:
    missing_robot_description = True
    print("robot_descriptions has not been found")

# global usefull stuff:
# q = [global_base_position, global_base_quaternion, joint_positions]
# v = [local_base_velocity_linear, local_base_velocity_angular, joint_velocities]




# global TODO: add the case for on_rack
#            : add a mechanism to update the robot model base link poistion and orientation when the robot is fixed frame (for floating is not an issue)
#            : managing the case of continuous joints (now they are represented with dimension 2 in pinocchio)
#            : add mechanics to decompose all the dynamic Stuff (M, coriolis and gravity) even more according to control_group_id and control_groups when they are specified

def set_continuous_joint_angle(q, idx, theta):
    q[idx] = np.cos(theta)
    q[idx + 1] = np.sin(theta)

def get_continuous_joint_angle(q, idx):
    cos_theta = q[idx]
    sin_theta = q[idx + 1]
    return np.arctan2(sin_theta, cos_theta)


class ResultsFloatingBaseJoint():
    def __init__(self,base_type):
        self.J = None
        self.J_b = None
        self.J_q = None
        self.M = None
        self.M_bb = None
        self.M_qq = None
        self.M_bq = None
        self.M_qb = None
        self.N = None
        self.N_bb = None
        self.N_qq = None
        self.N_bq = None
        self.N_qb = None
        self.c = None 
        self.c_b = None
        self.c_q = None
        self.g = None
        self.g_b = None
        self.g_q = None
        self.base_type = base_type

    # in all these get functions the default is the full matrix
    def GetJ(self,flag=""):
        if flag =="actuated":
            if self.base_type=="fixed":
                return self.J
            return self.J_q
        if flag =="underactuated":
            if self.base_type=="fixed":
                return None
            return self.J_b
        else:
            return self.J
        
    def GetM(self,flag=""):
        if flag =="actuated":
            if self.base_type=="fixed":
                return self.M
            return self.M_qq, self.M_qb 
        if flag =="underactuated":
            if self.base_type=="fixed":
                return None
            return self.M_bb, self.M_bq
        else:
            return self.M
    # TODO to fix add submatrix out of diagonal N_bq and N_qb 
    def GetN(self,flag=""):
        if flag =="actuated":
            if self.base_type=="fixed":
                return self.N
            return self.N_q
        if flag =="underactuated":
            if self.base_type=="fixed":
                return None
            return self.N_b
        else:
            return self.N
        
    def GetG(self,flag=""):
        if flag =="actuated":
            if self.base_type=="fixed":
                return self.g
            return self.g_q
        if flag =="underactuated":
            if self.base_type=="fixed":
                return None
            return  self.g_b
        else:
            return self.g
        
    def GetC(self,flag=""):
        if flag =="actuated":
            if self.base_type=="fixed":
                return self.c
            return self.c_q
        if flag =="underactuated":
            if self.base_type=="fixed":
                return None
            return self.c_b
        else:
            return self.c

class PinWrapper():
    def __init__(self, conf_file_name, simulator=None, list_link_name_for_reodering = np.empty(0) ,data_source_names=[],  visualizer=False,index=0, conf_file_path_ext=None):
        
        #for index in  range(len(data_source_names)):
        if not isinstance(list_link_name_for_reodering, np.ndarray):
            raise ValueError("list_link_name_for_reodering must be a numpy array with strings and with 2 dimensions, if you have only one list of joint you can instatiate it as list_link_name_for_reodering = np.array([[list_link_name_for_reodering],])")
        # check if the ndarray is bidimensional
        if len(list_link_name_for_reodering.shape) != 2:
            raise ValueError("list_link_name_for_reodering must be a numpy array with 2 dimensions, if you have only one list of joint you can instatiate it as list_link_name_for_reodering = np.array([[list_link_name_for_reodering],])")
        if simulator is None:
            raise ValueError("simulator must be specified")
        else:
            self.simulator = simulator
        
        if conf_file_path_ext: # CHANGE
             conf_file_path = os.path.join(conf_file_path_ext,'configs',conf_file_name) #CHANGE
        else:
            conf_file_path = os.path.join(os.path.dirname(__file__),os.pardir,'configs',conf_file_name)
        with open(conf_file_path) as json_file:
            conf_file_json = json.load(json_file)

        self.conf = conf_file_json
        self.base_type = self.conf['robot_pin']['base_type'][index]    
        self.res = ResultsFloatingBaseJoint(self.base_type)
        self.visualizer = visualizer
        urdf_file = self._UrdfPath(index,conf_file_path_ext)
        self._LoadPinURDF(urdf_file)
        # build the dictionary of feet id and the feet reference frame to standard name
        self.feet_frame_2_standard_name = {}
        # find common elements between the list of contact point and the list of frame associate with each foot 
        # TODO this mechanism should be generalized for any number of contact (now it works only for 4 potential contact points, not flexilbe)
        fl_contact_frame = list(set(self.conf['sim']['feet_contact_names'][index]) & set(self.conf['sim']['FL'][index]))
        if len(fl_contact_frame) != 0:
            self.feet_frame_2_standard_name["FL"] = fl_contact_frame[0]
        fr_contact_frame = list(set(self.conf['sim']['feet_contact_names'][index]) & set(self.conf['sim']['FR'][index]))
        if len(fr_contact_frame) != 0:
            self.feet_frame_2_standard_name["FR"] = fr_contact_frame[0]
        rl_contact_frame = list(set(self.conf['sim']['feet_contact_names'][index]) & set(self.conf['sim']['RL'][index]))
        if len(rl_contact_frame) != 0:
            self.feet_frame_2_standard_name["RL"] = rl_contact_frame[0]
        rr_contact_frame = list(set(self.conf['sim']['feet_contact_names'][index]) & set(self.conf['sim']['RR'][index]))
        if len(rr_contact_frame) != 0:
            self.feet_frame_2_standard_name["RR"] = rr_contact_frame[0]
        # here I need to iterate on the dictionary feet_frame_2_standard_name to get the id of the feet
        self.feet_id = {}
        for n in  self.feet_frame_2_standard_name.values():
            self.feet_id[n] = self.pin_model.getFrameId(n)
            
        # adding mechanism to convert the joint state from the robot to the pinocchio model    
        if self.conf['robot_pin']['joint_state_conversion_active'][index]:
            self.ext2pin = OrderedDict()
            self.pin2ext = OrderedDict()
            if list_link_name_for_reodering.size == 0:
                raise("error, you need to provide a list of link names to reorder the joint state")
            else:
                if list_link_name_for_reodering.shape[0]>1 and (list_link_name_for_reodering.shape[0] != len(data_source_names)):
                    raise("error, more than one list name present, data_source_names need to be specified and list_link_name_for_reodering and data_source_names must have the same length")
                self.CreateIndexJointAssociation(list_link_name_for_reodering.copy(),data_source_names.copy())
        else:
            self.ext2pin = None
            self.pin2ext = None

        # adding control groups to manage different control interfaces in the same robot
        # if control group is not specify do not crash manage exception
        # Using get with a default empty list if 'control_groups' is not found
        control_groups = self.conf.get('robot_pin', {}).get('control_groups', [])
        # Check if the index is valid
        if 0 <= index < len(control_groups):
            self.control_groups = control_groups[index]
        else:
            self.control_groups = None

        # based upon control groups name i should match the joint id with the control group name if it is not empty
        if self.control_groups is not None:
            # here i create an empty dictionary called control_group_id
            self.control_group_id = {}
            # control group is a dictionary
            # first i get all the keys of the dictionary
            control_group_names = list(self.control_groups.keys())
            # for each key i get the list of joint names
            for name in control_group_names:
                joint_names_in_group = self.control_groups[name]
                cur_joint_id = []
                for joint_name in joint_names_in_group:
                    # here i get the index of the joint name in the list of joint names and put in a list
                    cur_joint_id.append(self.pin_model.getJointId(joint_name))
                # here i add the currebt key and the list of joint id to the dictionary
                self.control_group_id[name] = cur_joint_id
        else:
            self.control_group_id = None





    def _UrdfPath(self,index,conf_file_path_ext:str = None):
        global missing_robot_description
        if(self.conf['robot_pin']['robot_description_model'][index] and not missing_robot_description):
            command_line_import = "from robot_descriptions import "+self.conf['robot_pin']['robot_description_model'][index]+"_description"
            exec(command_line_import)
            urdf_path = locals()[self.conf['robot_pin']['robot_description_model'][index]+"_description"].URDF_PATH
        else:
            if conf_file_path_ext: # CHANGE
                urdf_file_path = os.path.join(conf_file_path_ext,'models',self.conf['robot_pin']["urdf_path"][index]) #CHANGE
            else: # CHANGE
                urdf_file_path = os.path.join(os.path.dirname(__file__),os.pardir,'models',self.conf['robot_pin']["urdf_path"][index])
            urdf_path = urdf_file_path
        return urdf_path

    def _LoadPinURDF(self, urdf_file):
        # check if the robot is on rack and update the pinocchio model according
        if self.base_type=="fixed":
            self.pin_model = pin.buildModelFromUrdf(urdf_file)
            self.pin_data = self.pin_model.createData()
            # here i assume that the robot has an actuate and an underactuated part
            # get the number of actuated joints
            self.n_b = 0     # floating base dof (x y z  [0 0 0 1])
            self.n_bdot = 0  # floating base velocity dof (x_dot y_dot z_dot  roll_dot yaw_dot pitch_dot)
            self.n_q = self.pin_model.nq      # joint dof 
            self.n_qdot = self.pin_model.nv   # joint velocity dof
            self.n = self.n_q 
            self.n_dot = self.n_qdot
        else: # TODO add the case for on_rack (only CoM position is fixed orientation is free)
            # Load the urdf model for pinocchio with floating base
            self.pin_model = pin.buildModelFromUrdf(urdf_file,pin.JointModelFreeFlyer())
            self.pin_data = self.pin_model.createData()
            self.n_b = 7     # floating base dof (x y z  [0 0 0 1])
            self.n_bdot = 6  # floating base velocity dof (x_dot y_dot z_dot  omega_x omega_y omega_z) base frame
            self.n_q = self.pin_model.nq - 7      # joint dof 
            self.n_qdot = self.pin_model.nv - 6   # joint velocity dof
            self.n = self.n_q + self.n_b
            self.n_dot = self.n_qdot + self.n_bdot

        # we need to maange the case of continuos joints which are represented with dimension 2 in pinocchio
        # i need to create a mapping that expands the mapping that we already have.

        
        if self.visualizer:
            from pinocchio.visualize import GepettoVisualizer
            package_dir= os.path.dirname(urdf_file)  
            print("package_dir=",package_dir)     
            self.collision_model = pin.buildGeomFromUrdf(self.pin_model,urdf_file,pin.GeometryType.COLLISION,package_dir)
            #self.collision_model = pin.buildGeomFromUrdf(self.pin_model, urdf_file, pin.GeometryType.COLLISION)
            self.visual_model = pin.buildGeomFromUrdf(self.pin_model,urdf_file,pin.GeometryType.VISUAL,package_dir)
            #self.visual_model = pin.buildGeomFromUrdf(self.pin_model, urdf_file, pin.GeometryType.VISUAL)
            self.viz = GepettoVisualizer(self.pin_model, self.collision_model , self.visual_model)
            self.viz.initViewer()
            self.viz.loadViewerModel("pinocchio")
            
    # here the assumption is that name of the joints follow the same order as in the coomputation of the robot dynamics in pinocchio
    # with data_source_names i can create multiple joint index conversion from and to pinocchio 
    # (as long as the ext_list has the same number of elements as data_source_names) 
    def CreateIndexJointAssociation(self,ext_list,data_source_names):

        pin_list = self.getNameActiveJoints()
        # remove the universe joint from pin_list
        pin_list.pop(0)
        # remove root joint is there is one (floating base)
        if "root_joint" in pin_list:
            # remove the root joint from pin_list
            pin_list.pop(0)
        
        # here i assume that ext_list is a list of list of string and i want to get the number of list of string in ext_list
        name_to_index_pin = {name: i for i, name in enumerate(pin_list)}
        for j in range(ext_list.shape[0]):
            # Create dictionaries to map names to indices
            name_to_index_ext = {name: i for i, name in enumerate(ext_list[j])}

            # Build an array of indices for matching names
            matching_indices_ext2pin = []
            ext2pin = []
            pin2ext = []
            for name in pin_list:
                if name in name_to_index_ext:
                    index1 = name_to_index_pin[name]
                    index2 = name_to_index_ext[name]
                    matching_indices_ext2pin.append((index1, index2))
                    ext2pin.append(index2)
            matching_indices_pin2ext = sorted(matching_indices_ext2pin, key=lambda x: x[1])

            for i in range(len(matching_indices_pin2ext)):
                pin2ext.append(matching_indices_pin2ext[i][0])
            # if i have only one data source i do not need specifiy the data source name
            if(ext_list.shape[0]==1):
                self.ext2pin["only_data_source"] = ext2pin
                self.pin2ext["only_data_soruce"] = pin2ext
            else:
                self.ext2pin[data_source_names[j]] = ext2pin
                self.pin2ext[data_source_names[j]] = pin2ext
        
    
    
    def ComputeJacobian(self,q0,frame_name,local_or_global):
        id = self.pin_model.getFrameId(frame_name)
        # empty struct to return 
        # reorder from external to pinocchio
        q0_ = self.ReoderJoints2PinVec(q0,"pos")
        if local_or_global == "global":
            self.res.J = pin.computeFrameJacobian(self.pin_model, self.pin_data, q0_, id, pin.WORLD).copy()
        elif local_or_global == "local":
            self.res.J = pin.computeFrameJacobian(self.pin_model, self.pin_data, q0_, id, pin.LOCAL).copy()
        elif local_or_global == "local_global":
            self.res.J = pin.computeFrameJacobian(self.pin_model, self.pin_data, q0_, id, pin.LOCAL_WORLD_ALIGNED).copy()
        else:
            raise ValueError("local_or_global must be either local or global")
        if self.n_bdot > 0:
            self.res.J_b = self.res.J[self.n_bdot:, :].copy()
            self.res.J_q = self.res.J[:self.n_bdot, :].copy()

        return self.res
    
    def ComputeJacobianFeet(self,q0,feet_name,local_or_global):
        frame_name = self.feet_frame_2_standard_name[feet_name]
        id = self.pin_model.getFrameId(frame_name)
        # empty struct to return 
        # reorder from external to pinocchio
        q0_ = self.ReoderJoints2PinVec(q0,"pos")
        if local_or_global == "global":
            self.res.J = pin.computeFrameJacobian(self.pin_model, self.pin_data, q0_, id, pin.WORLD).copy()
        elif local_or_global == "local":
            self.res.J = pin.computeFrameJacobian(self.pin_model, self.pin_data, q0_, id, pin.LOCAL).copy()
        elif local_or_global == "local_global":
            self.res.J = pin.computeFrameJacobian(self.pin_model, self.pin_data, q0_, id, pin.LOCAL_WORLD_ALIGNED).copy()
        else:
            raise ValueError("local_or_global must be either local or global")
        if self.n_bdot > 0:
            self.res.J_b = self.res.J[:, self.n_bdot:].copy()
            self.res.J_q = self.res.J[:, :self.n_bdot].copy()

        return self.res
    def KinematicIntegration(self,q0,v0,dt):
        # reorder from external to pinocchio
        q0_ = self.ReoderJoints2PinVec(q0,"pos")
        v0_ = self.ReoderJoints2PinVec(v0,"vel")
        q1 = pin.integrate(self.pin_model, q0_, v0_*dt)
        return q1

    # TODO add conversion mechanicsm
    def ComputeFK(self,q,link_name):
        id = self.pin_model.getFrameId(link_name)
        # reorder from external to pinocchio
        q_ = self.ReoderJoints2PinVec(q,"pos")
        pin.forwardKinematics(self.pin_model, self.pin_data, q_)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        return self.pin_data.oMf[id].translation, self.pin_data.oMf[id].rotation
    
    #TODO add this functions to the pin wrapper
    # def GetHipPositionsInBaseFrame(self):
    #     """Get the hip joint positions of the robot within its base frame."""
    #     raise NotImplementedError("Not implemented for Minitaur.")
    #def ComputeMotorAnglesFromFootLocalPosition(self, leg_id,
    #                                            foot_local_position):
    #    pass

    
    #def MapContactForceToJointTorques(self, leg_id, contact_force):
    #    """Maps the foot contact force to the leg joint torques."""
    #    jv = self.ComputeJacobian(leg_id)
    #    motor_torques_list = np.matmul(contact_force, jv)
    #    motor_torques_dict = {}
    #    motors_per_leg = self.p.num_motors // self.p.num_legs
    #    for torque_id, joint_id in enumerate(
    #            range(leg_id * motors_per_leg, (leg_id + 1) * motors_per_leg)):
    #        motor_torques_dict[joint_id] = motor_torques_list[torque_id]
    #    return motor_torques_dict

     #def GetFootPositionsInBaseFrame(self):
    #    pass
        
    #def GetFootPositionsInWorldFrame(self):
    	#pass

    def ComputeMassMatrix(self,q):
        if(q is not np.ndarray):
            q = np.array(q)
        # check if the state is dimensionally consistent
        if q.shape[0] != self.n:
            raise ValueError("q must be of size "+str(self.n))
        
        q_ = self.ReoderJoints2PinVec(q,"pos")
        
        pin.crba(self.pin_model, self.pin_data, q_)
        # it is upper triangular apparently
        self.res.M = self.pin_data.M.copy()
        #self.res.M = self.res.M + self.res.M.T - np.diag(self.res.M.diagonal())
        if self.n_bdot > 0:
            self.res.M_bb = self.res.M[:self.n_bdot, :self.n_bdot]
            self.res.M_bq = self.res.M[:self.n_bdot, self.n_bdot:]
            self.res.M_qq = self.res.M[self.n_bdot:, self.n_bdot:] 
            self.res.M_qb = self.res.M[self.n_bdot:, :self.n_bdot]
        return self.res
    
    def ComputeMassMatrixRNEA(self,x):
        if(x is not np.ndarray):
            x = np.array(x)
        # check if the state is dimensionally consistent
        if x.shape[0] != self.n:
            raise ValueError("x must be of size "+str(self.n))
        
        # reorder from external to pinocchio
        x_ = self.ReoderJoints2PinVec(x,"pos")
        
        M = np.zeros((self.n_dot, self.n_dot))
        xdot_zero    = [0] * self.n_dot
        xdotdot_zero = [0] * self.n_dot
        pin.computeGeneralizedGravity(self.pin_model, self.pin_data, x_)
        g = self.pin_data.g.copy()
        for i in range(self.n_dot):
            cur_accels = np.zeros(self.n_dot)
            cur_accels[i] = 1
            pin.rnea(self.pin_model, self.pin_data, x_, np.array(xdot_zero), cur_accels)
            cur_M_col = self.pin_data.tau  - g
            M[:, i] = cur_M_col

        # full matrix
        self.res.M = M
        #self.res.M = self.res.M + self.res.M.T - np.diag(self.res.M.diagonal())
        if self.n_bdot > 0:
            self.res.M_b = self.res.M[:self.n_bdot, :]
            self.res.M_q = self.res.M[self.n_bdot:, :]
        return self.res

    def ComputeCoriolisMatrix(self,q,qdot):
        if(q is not np.ndarray):
            q = np.array(q)
        if(qdot is not np.ndarray):
            qdot = np.array(qdot)

        # check if the state is dimensionally consistent
        if q.shape[0] != self.n:
            raise ValueError("q must be of size "+str(self.n))
        if qdot.shape[0] != self.n_dot:
            raise ValueError("qdot must be of size "+str(self.n_dot))
        
        # reorder from external to pinocchio
        q_ = self.ReoderJoints2PinVec(q,"pos")
        qdot_ = self.ReoderJoints2PinVec(qdot,"vel")
    
        pin.computeCoriolisMatrix(self.pin_model, self.pin_data, q_, qdot_)
        self.res.N = self.pin_data.C.copy()
        if self.n_bdot > 0:
            self.res.N_bb = self.res.N[:self.n_bdot, :self.n_bdot]
            self.res.N_bq = self.res.N[:self.n_bdot, self.n_bdot:]
            self.res.N_qq = self.res.N[self.n_bdot:, self.n_bdot:] 
            self.res.N_qb = self.res.N[self.n_bdot:, :self.n_bdot]
        return self.res
    
    def ComputeCoriolis(self,q,qdot):
        if(q is not np.ndarray):
            q = np.array(q)
        if(qdot is not np.ndarray):
            qdot = np.array(qdot)

        # check if the state is dimensionally consistent
        if q.shape[0] != self.n:
            raise ValueError("q must be of size "+str(self.n))
        if qdot.shape[0] != self.n_dot:
            raise ValueError("qdot must be of size "+str(self.n_dot))
        
        # reorder from external to pinocchio
        q_ = self.ReoderJoints2PinVec(q,"pos")
        qdot_ = self.ReoderJoints2PinVec(qdot,"vel")

        xdotdot_zero = [0] * self.n_dot
        pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q_)
        g = self.pin_data.g.copy()
        pin.rnea(self.pin_model, self.pin_data, q_, qdot_, np.array(xdotdot_zero))
        self.res.c = self.pin_data.tau.copy() - g
    
        #pin.computeCoriolisMatrix(self.pin_model, self.pin_data, q, qdot)
        #self.res.c = self.pin_data.C @ qdot

        if self.n_bdot > 0:
            self.res.c_b =  self.res.c[:self.n_bdot,]
            self.res.c_q =  self.res.c[self.n_bdot:,]
        return self.res

    def ComputeGravity(self,q):
        if(q is not np.ndarray):
            q = np.array(q)
        # check if the state is dimensionally consistent
        if q.shape[0] != self.n:
            raise ValueError("q must be of size "+str(self.n))
        # reorder from external to pinocchio
        q_ = self.ReoderJoints2PinVec(q,"pos")
        
        pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q_)
        self.res.g = self.pin_data.g.copy()
        if self.n_bdot > 0:
            self.res.g_b = self.res.g[:self.n_bdot]
            self.res.g_q = self.res.g[self.n_bdot:] 
        return self.res

    def ComputeAllTerms(self,q,qdot):
        self.ComputeGravity(q)
        self.ComputeCoriolis(q,qdot)
        self.ComputeMassMatrix(q)
        self.ComputeCoriolisMatrix(q,qdot)

    def DirectDynamicsActuatedZeroTorqueNoContact(self,q,qdot):
        # we need to reorder the input to get it in the right order for pinocchio
        if(q is not np.ndarray):
            q = np.array(q)
        if(qdot is not np.ndarray):
            qdot = np.array(qdot)

        # check if the state is dimensionally consistent
        if q.shape[0] != self.n:
            raise ValueError("q must be of size "+str(self.n))
        if qdot.shape[0] != self.n_dot:
            raise ValueError("qdot must be of size "+str(self.n_dot))
        
        # update all the terms (i do  not need to reorder because it si done inside each function called in ComputeAllTerms)
        self.ComputeAllTerms(q,qdot)
        if(self.base_type =="floating"):
            # coriolis + gravity actuated
            n_q = self.res.c_q + self.res.g_q
            # coriolis + gravity underactuated
            n_b = self.res.c_b + self.res.g_b
            Mass_matrix_actuated = self.res.M_qq - self.res.M_qb @ np.linalg.inv(self.res.M_bb) @ self.res.M_bq
            # DEBUG 
            # detM_qq = np.linalg.det(self.res.M_qq)
            # detM_bb = np.linalg.det(self.res.M_bb)
            # detM_qbM_bbM_bq = np.linalg.det(self.res.M_qb @ np.linalg.inv(self.res.M_bb) @ self.res.M_bq)
            # determinant_mass_matrix_actuated = np.linalg.det(Mass_matrix_actuated)
            # detertminant_full_M = np.linalg.det(self.res.M)
            # print("detM_qq=",detM_qq,"detM_bb=",detM_bb,"detM_qbM_bbM_bq=",detM_qbM_bbM_bq)
            # print("determinant_mass_matrix actuated = ",determinant_mass_matrix_actuated)
            # print("determinant_full_M = ",detertminant_full_M)
            # # checking strong inertial coupling for the matrix Mbq
            # rank = np.linalg.matrix_rank(self.res.M_bq)
            # if rank != self.n_bdot:
            #     print("Mbq is not full rank")
            # print("M_bq rank=",rank)
            # rank_M_bar = np.linalg.matrix_rank(Mass_matrix_actuated)
            # print("M_bar rank=",rank_M_bar)
            # M_bar_trace = np.trace(Mass_matrix_actuated)
            # print("M_bar trace=",M_bar_trace)
            # M_bar_eigenvalues = np.linalg.eigvals(Mass_matrix_actuated)
            # print("M_bar eigenvalues=",M_bar_eigenvalues)

            Coriolis_actuated = -n_q + self.res.M_qb @ np.linalg.inv(self.res.M_bb) @ n_b
            acc =  np.linalg.inv(Mass_matrix_actuated) @ Coriolis_actuated
        elif(self.base_type =="fixed"):
            #DEBUG
            detertminant_full_M = np.linalg.det(self.res.M)
            print("determinant_full_M = ",detertminant_full_M)
            rank_Mass_matrix_actuated = np.linalg.matrix_rank(Mass_matrix_actuated)
            print("M rank=",rank_Mass_matrix_actuated)
            # coriolis + gravity 
            n = self.res.c + self.res.g
            acc =  np.linalg.inv(self.res.M) @ (-n)
        elif(self.base_type =="on_rack"):
            raise ValueError("not implemented yet")
        return acc
    
    # here i compute inverse dynamics with rnea to test the correctness of the computation inverse dynamics actuated part no contact
    def FullInverseDynamicsNoContact(self, x, xdot, xdotdot):
        x_ = self.ReoderJoints2PinVec(x,"pos")
        xdot_ = self.ReoderJoints2PinVec(xdot,"vel")
        xdotdot_ = self.ReoderJoints2PinVec(xdotdot,"vel")
        pin.rnea(self.pin_model, self.pin_data, x_, xdot_, xdotdot_)
        tau_pin = self.pin_data.tau.copy()
        return tau_pin
    
    # here i assume only forces no torques (point contact)
    def FullInverseDynamicsWithContact(self, x, xdot, xdotdot, feet_contact_map, local_or_global):
        x_ = self.ReoderJoints2PinVec(x,"pos")
        xdot_ = self.ReoderJoints2PinVec(xdot,"vel")
        xdotdot_ = self.ReoderJoints2PinVec(xdotdot,"vel")

        tau_contact = np.zeros((self.pin_model.nv,))
        for foot_name, ext_force  in enumerate(feet_contact_map):
            # here i need to pass the non reordere state to ComputeJacobianFeet because i will reorder it inside the function
            res = self.ComputeJacobianFeet(x,foot_name,local_or_global)
            cur_contact_jacobian =res.J[:3,:].T.copy()
            if foot_name == "FL":
                cur_contact_torques = np.dot(cur_contact_jacobian,ext_force)
            elif foot_name == "FR":
                cur_contact_torques = np.dot(cur_contact_jacobian,ext_force)
            elif foot_name == "RL":
                cur_contact_torques = np.dot(cur_contact_jacobian,ext_force)
            elif foot_name == "RR":
                cur_contact_torques = np.dot(cur_contact_jacobian,ext_force)
     
            tau_contact = tau_contact + cur_contact_torques


        pin.rnea(self.pin_model, self.pin_data, x_, xdot_, xdotdot_)
        tau_FL = self.pin_data.tau.copy()
        tau_pin = tau_FL - tau_contact
        return tau_pin, tau_contact, tau_FL

    def InverseDynamicsActuatedPartNoContact(self, x_prev, xdot_prev, xdotdot_prev):
        self.ComputeAllTerms(x_prev,xdot_prev)
        # added reordering of the joints to get the right order for pinocchio
        #x_prev_ = self.ReoderJoints2PinVec(x_prev,"pos")
        #xdot_prev_ = self.ReoderJoints2PinVec(xdot_prev,"vel")
        xdotdot_prev_ = self.ReoderJoints2PinVec(xdotdot_prev,"vel")
        if(self.base_type =="floating"):
            xdotdot_prev_b = xdotdot_prev_[:self.n_bdot]
            xdotdot_prev_q = xdotdot_prev_[self.n_bdot:]
            n_q = self.res.c_q + self.res.g_q
            tau_pin = self.res.M_qq @ xdotdot_prev_q + self.res.M_qb @ xdotdot_prev_b + n_q
            return tau_pin
        elif(self.base_type =="fixed"):
            n = self.res.c + self.res.g
            tau_pin = self.res.M @ xdotdot_prev_ + n
            return tau_pin
        elif(self.base_type =="on_rack"):
            raise ValueError("not implemented yet")
        
    def ABA(self,q,qdot,tau):
        # reorder from external to pinocchio
        q_ = self.ReoderJoints2PinVec(q,"pos")
        qdot_ = self.ReoderJoints2PinVec(qdot,"vel")
        # here since the tau is only the actuated part I need just to reorder that one
        tau_ = self._FromExtToPinVec(tau)
        if(self.base_type =="fixed"):
            acc = pin.aba(self.pin_model, self.pin_data, q_, qdot_, tau_)
        else:
            # here I assume that the floating base is always not actuated
            base_torque = np.zeros((6,))
            tau_ = np.concatenate((base_torque,tau_))
            acc = pin.aba(self.pin_model, self.pin_data, q_, qdot_, tau_)
        return acc

    def GetTotalMassFromUrdf(self):
        mass = []
        pin.computeTotalMass(self.pin_model, self.pin_data)
        mass = self.pin_data.mass[0]
        return mass
    
    def GetMassLink(self,link):
        mass = 0
        pin.computeSubtreeMasses(self.pin_model, self.pin_data)
        # get the id of the link
        id = self.pin_model.getFrameId(link)
        mass = self.pin_data.mass[id]
        return mass
    
    def getDynamicsInfo(self):
        for i,name in enumerate(self.pin_model.names):
            print(name,"=",self.pin_model.inertias[i])
    
    def getNameActiveJoints(self):
        name_list = []
        for i in range(len(self.pin_model.names)):
            name_list.append(self.pin_model.names[i])
        return name_list.copy()
    
    def getNumberofActuatedJoints(self):
        return self.n_q
    
    def DisplayModel(self,q):
        if(self.visualizer):
            q_ = self.ReoderJoints2PinVec(q,"pos")
            self.viz.display(q_)
        else:
            print("Visualizer not initialized")

    # this functions reorganization are applied only at the joints level
    # i assume that the if no source_name is specified i always take the first element of the dictionary
    def _FromExtToPinVec(self,x,source_name=[]):
        new_x = np.zeros((len(x),))
        # assign value from x to new_x using a list of indices self.reorder
        if self.conf['robot_pin']['joint_state_conversion_active']:
            if not source_name:
                # i get the value associated to the first element of the dictionary
                _,index_list =  next(iter(self.ext2pin.items())) 
            else:
                index_list = self.ext2pin[source_name]
            for i, index in enumerate(index_list):
                new_x[i] = x[index]
        else:
            new_x = x.copy()
        return new_x
    
     # this functions reorganization are applied only at the joints level
    # i assume that the if no source_name is specified i always take the first element of the dictionary
    def _FromPinToExtVec(self,x,source_name=[]):
        new_x = np.zeros((len(x),))
        # assign value from x to new_x using a list of indices self.reorder
        if self.conf['robot_pin']['joint_state_conversion_active']:
            if not source_name:
                # i get the value associated to the first element of the dictionary
                _,index_list =  next(iter(self.pin2ext.items()))
            else:
                index_list = self.pin2ext[source_name]
            for i, index in enumerate(index_list):
                new_x[i] = x[index]
        else:
            new_x = x.copy()

        return new_x
    
    # the reodering for a matrix happens first rowwise than column wise
    # if a source_name is not specified i take the first element of the dictionary
    def _FromPinToExtMat(self,X,source_name=[]):
        new_X = np.zeros((X.shape[0],X.shape[1]))
        new_X2= np.zeros((X.shape[0],X.shape[1]))
        # assign value from x to new_x using a list of indices self.reorder
        if self.conf['robot_pin']['joint_state_conversion_active']:
            if not source_name:
                # i get the value associated to the first element of the dictionary
                _,index_list =  next(iter(self.pin2ext.items()))
            else:
                index_list = self.pin2ext[source_name]
            for i, index in enumerate(index_list):
                new_X[i,:] = X[index,:].copy()
            for i, index in enumerate(index_list):
                new_X2[:,i] = new_X[:,index].copy()
        else:
            new_X2 = X.copy()

        return new_X2
    
    # flag can be pos or vel
    # when it is pos it means that we are considering the position with the floating base (so the joints are after 7 positions pos + quaternion)
    # when it is vel it means that we are considering the velocity with the floating base (so the joints are after 6 velocities vel + angular velocity)
    # you need to use vel also for acceleration/torque contributions (because the acceleration has the same dimension of the velocity)
    def _ExtractJointsVec(self,x,flag="pos"):
        if flag == "pos":
            ind = self.n_b
        elif flag == "vel":
            ind = self.n_bdot
        else:
            raise ValueError("flag must be either pos or vel")
        
        if ind > 0:
            x_q = x[ind:].copy()
        else:
            x_q = x.copy()

        return x_q
    
    def _ExtractJointsMat(self,X,flag="pos"):
        if flag == "pos":
            ind = self.n_b
        elif flag == "vel":
            ind = self.n_bdot
        else:
            raise ValueError("flag must be either pos or vel")
        if ind > 0:
            X_q = X[ind:,:].copy()
        else:
            X_q = X.copy()

        return X_q
    
    # this function copy back the joints state in the right position in the vector
    def _CopyJointsVec(self, x_dest, x_q, flag="pos"):

        # i need to do this in order to prevent side effect on x_dest
        x_dest_new = x_dest.copy() 

        if flag == "pos":
            ind = self.n_b
        elif flag == "vel":
            ind = self.n_bdot
        else:
            raise ValueError("flag must be either pos or vel")
        if ind > 0:
            x_dest_new[ind:] = x_q.copy()
        else:
            x_dest_new = x_q.copy()

        return x_dest_new
    
    # this function copy back the joints state in the right position in the matrix
    def _CopyJointsMat(self, X_dest, X_q, flag="pos"):

        X_dest_new = X_dest.copy()

        if flag == "pos":
            ind = self.n_b
        elif flag == "vel":
            ind = self.n_bdot
        else:
            raise ValueError("flag must be either pos or vel")
        if ind > 0:
            X_dest_new[ind:,:] = X_q.copy()
        else:
            X_dest_new = X_q.copy()

        return X_dest_new
        
    def ReoderJoints2PinVec(self,x,pos_or_vel,source_name=[]):
        x_q = self._ExtractJointsVec(x,pos_or_vel)
        # here we reorder the joint state to match the pinocchio model
        x_q_reordered = self._FromExtToPinVec(x_q,source_name)
        # here we copy the q_joints_new into the q vector to be able to compute the coriolis
        x_ = self._CopyJointsVec(x,x_q_reordered,pos_or_vel)
        return x_
    
    def ReoderJoints2ExtVec(self,x,pos_or_vel,source_name=[]):
        x_q = self._ExtractJointsVec(x,pos_or_vel)
        x_q_reodered = self._FromPinToExtVec(x_q,source_name)
        x_ = self._CopyJointsVec(x, x_q_reodered,pos_or_vel)
        return x_
    
    def ReoderJoints2ExMat(self,X,pos_or_vel,source_name=[]):
        X_q = self._ExtractJointsMat(X,pos_or_vel)
        X_q_reodered = self._FromPinToExtMat(X_q,source_name)
        X_ = self._CopyJointsMat(X, X_q_reodered,pos_or_vel)
        return X_
    
    def GetConfigurationVariable(self,param):
        first_param = "robot_"+self.simulator
        return self.conf[first_param][param]
    
    
    def ComputeDynamicRegressor(self,q,qd,qdd):
        # reorder from external to pinocchio
        q_ = self.ReoderJoints2PinVec(q,"pos")
        qd_ = self.ReoderJoints2PinVec(qd,"vel")
        qdd_ = self.ReoderJoints2PinVec(qdd,"vel")
        
        
        R = pin.computeJointTorqueRegressor(self.pin_model, self.pin_data, q_, qd_, qdd_)
        return R