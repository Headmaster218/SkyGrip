# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# global stuff to remember:
# - in pybullet the mass attached to the trunk are ignored when we have a floating base while pinocchio keep them into account, in order to have 
#   the same behaviour we need to reduce a lot the mass and inertia of links attached to the robot floating-base doing merge fixed links with
#   pybullet_client.URDF_MERGE_FIXED_LINKS in the loadURDF function does not work really well and makes the simulation less precise with respect to 
#   the pinocchio one.  

# contact forces measurements:
# for the contact i treat the feet contact differently (because i assume they are special contact points) while i treat all the other
# contact points in a separate way so i will have measures that is specific for the feet contact and all the other cointact forces will 
# be cosidereded in a different structure  

# GLOBAL TODO:
# I need to add the possibility to not have a specified CoM position and orientation and use only the one provided by the URDF


# reminder for delay in the simulation:
# all the functions affected by delay are:
# - GetMotorAngles
# - GetMotorVelocities
# - GetMotorAccelerationTMinusOne
# - GetBasePosition
# - GetBaseOrientation
# - GetBaseLinVelocity
# - GetBaseAngVelocity
# - GetBaseLinVelocityBodyFrame
# - GetBaseAngVelocityBodyFrame
# - GetBaseVelocitiesBodyFrame
# - GetSystemState


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect
import time
from typing import Optional, Any, Tuple

import collections
import copy
import math
import re
from collections import deque
import numpy as np
import pinocchio as pin
import json
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client
from ..controllers.servo_motor import MotorCommands, ServoMotorModel ## CHANGE changed for the package structure
from ..utils import adjust_value 
# this is only for checking if the module is installed (TODO change this)
missing_robot_description = False
try:
    import robot_descriptions as rd
except ImportError:
    missing_robot_description = True
    print("robot_descriptions has not been found")

TWO_PI = 2 * math.pi

class EmptyObj():
  pass


def MapToMinusPiToPi(angles):
    """Maps a list of angles to [-pi, pi].

  Args:
    angles: A list of angles in rad.

  Returns:
    A list of angle mapped to [-pi, pi].
  """
    mapped_angles = copy.deepcopy(angles)
    for i in range(len(angles)):
        mapped_angles[i] = math.fmod(angles[i], TWO_PI)
        if mapped_angles[i] >= math.pi:
            mapped_angles[i] -= TWO_PI
        elif mapped_angles[i] < -math.pi:
            mapped_angles[i] += TWO_PI
    return mapped_angles

    

# in this class we create a pinocchio object and we use it to compute the kinematics and dynamics
# and we keep a copy of the state of the robot

# one new config parameters is called "base_type"="fixed" "floating" "on_rack"
class SimRobot():
    def __init__(self,
                 pybullet_client,
                 conf_file_json, # the conf_file_json is a json file that contains all the parameters of the robot
                 index,
                 config_file_path_ext: str = None, # if it is not none th config file is not in the local config_file folder
                 ):
       
        self.conf = conf_file_json
        
        self.base_type = self.conf['robot_pybullet']['base_type'][index]
        self.base_name = self.conf['robot_pybullet']["floating_base_name"][index]
        # this variable is instatiated in self._buildLinkNameToId()
        self.base_link_name = None
        self.base_link_2_com_pos_offset = None
        self.base_link_2_com_ori_offset = None

        self.self_collision_enabled = self.conf['robot_pybullet']['collision_enable'][index]
        self.foot_link_ids={}
        self.foot_link_sensors_ids={}
        print("file path ext: ",config_file_path_ext)
        urdf_path = self._UrdfPath(index,config_file_path_ext)
        # here i create the robot object inside pybullet sim and i save it under bot_pybullet
        self._LoadPybulletURDF(urdf_path, pybullet_client,index) #(self.conf['robot']["urdf_path"], pybullet_client)

        self.num_motors = len(self.active_joint_ids)
        # here i set the desired foot friction and restitution
        self.SetFootFriction(pybullet_client, self.conf['robot_pybullet']['foot_friction'][index])
        self.SetFootRestitution(pybullet_client, self.conf['robot_pybullet']['foot_restitution'][index])
        
        if len(self.conf['robot_pybullet']["init_link_base_orientation"][index]) == 0:  
            self.link_floating_base_pos, self.link_floating_base_ori=pybullet_client.getBasePositionAndOrientation(self.bot_pybullet)
            # TODO to check
            _, self.init_orientation_inv = pybullet_client.invertTransform(position=[0, 0, 0], orientation=self._GetDefaultInitOrientation())
        else:
            self.link_floating_base_ori = self.conf['robot_pybullet']["init_link_base_orientation"][index]
            self.link_floating_base_pos = self.conf['robot_pybullet']["init_link_base_position"][index]
             # TODO to check
            _, self.init_orientation_inv = pybullet_client.invertTransform(position=[0, 0, 0], orientation=self._GetDefaultInitOrientation())

        if len(self.conf['robot_pybullet']["motor_offset"][index]) == 0:
            self.motor_offset = np.zeros(len(self.active_joint_ids))
        else:
            self.motor_offset = self.conf['robot_pybullet']["motor_offset"][index]

        if len(self.conf['robot_pybullet']["motor_direction"][index]) == 0:
            self.motor_direction = np.ones(len(self.active_joint_ids))
        else:
            self.motor_direction = self.conf['robot_pybullet']["motor_direction"][index]

        self.applied_motor_commands = np.zeros(len(self.active_joint_ids))

        # self.servoPositionGain = (self.conf['robot']["servo_pos_gains"]*np.ones((1, len(self.active_joint_ids)))).tolist()[0]
        # self.servoVelocityGain = (self.conf['robot']["servo_vel_gains"]*np.ones((1, len(self.active_joint_ids)))).tolist()[0]

        self.init_joint_angles = self.conf['robot_pybullet']["init_motor_angles"][index]
        # if self.conf['robot_pybullet']["init_motor_vel"] is not empty we use otherwise is zero
        if(self.conf['robot_pybullet']["init_motor_vel"][index]):
            self.init_joint_vel = self.conf['robot_pybullet']["init_motor_vel"][index]
        else:
            self.init_joint_vel = list(np.zeros((len(self.active_joint_ids),)))

        self.servo_motor_model = ServoMotorModel(len(self.active_joint_ids), self.conf['robot_pybullet']["servo_pos_gains"][index], self.conf['robot_pybullet']["servo_vel_gains"][index],
                                                 friction_torque=self.conf['robot_pybullet']['motor_damping'][index], friction_coefficient=self.conf['robot_pybullet']['motor_damping_coeff'][index],
                                                 elastic_torque=self.conf['robot_pybullet']['motor_elastic_torque'][index],elastic_coefficient=self.conf['robot_pybullet']['motor_elastic_coeff'][index], 
                                                 motor_load=self.conf['robot_pybullet']['motor_inertia'][index], motor_load_coefficient=self.conf['robot_pybullet']['motor_inertia_coeff'][index])
        

        # adding measures noise to the simulation
        self.noise_flag = self.conf['robot_pybullet']['noise_flag'][index]
        self.robot_noise = self.conf['robot_pybullet']['robot_noise'][index]
        # adusting values for each entry of self.robot_noise it id a dictionary
        for key in self.robot_noise.keys():
            if key in ("joint_cov", "joint_vel_cov", "joint_acc_cov", "joint_torque_cov"):
                self.robot_noise[key] = adjust_value(self.noise_flag, self.robot_noise[key], self.num_motors, key)
            if key in ("base_pos_cov", "base_lin_vel_cov", "base_ang_vel_cov"):
                self.robot_noise[key] = adjust_value(self.noise_flag, self.robot_noise[key], 3, key)
            if key in ("base_ori_cov"):
                self.robot_noise[key] = adjust_value(self.noise_flag, self.robot_noise[key], 3, key)

        #self.noise_cov = adjust_value(self.noise_flag, self.conf['robot_pybullet']['noise_covariance'][index], self.active_joint_ids, 'noise_covariance')
            
        # adding delay measures in the simulation
        # i want to be sure that i have a delay only if the flag is up
        self.delay_measure_flag = self.conf['robot_pybullet']['delay_measure_flag'][index]
        if self.delay_measure_flag:
            # Define the delay in terms of steps
            self.delay_steps = self.conf['robot_pybullet']['delay_measure_steps'][index]   # i do minus one because i will always have at least the previous state
        else:
            self.delay_steps = 0
        self.state_buffer = collections.deque(maxlen=self.delay_steps + 1) # +1 to include the current state


    def _UrdfPath(self,index,config_file_path_ext: str = None): #CHANGE
        global missing_robot_description
        if(self.conf['robot_pybullet']['robot_description_model'][index] and not missing_robot_description):
            command_line_import = "from robot_descriptions import "+self.conf['robot_pybullet']['robot_description_model'][index]+"_description"
            exec(command_line_import)
            urdf_path = locals()[self.conf['robot_pybullet']['robot_description_model'][index]+"_description"].URDF_PATH
        else:
            if config_file_path_ext: #CHANGE
                urdf_file_path = os.path.join(config_file_path_ext,'models',self.conf['robot_pybullet']["urdf_path"][index]) #CHANGE
            else: #CHANGE
                urdf_file_path = os.path.join(os.path.dirname(__file__),os.pardir,os.pardir,'models',self.conf['robot_pybullet']["urdf_path"][index])
            urdf_path = urdf_file_path
        return urdf_path

    def _LoadPybulletURDF(self, urdf_file, pybullet_client,index):

        # TODO I could consider add this flag to get a closer behaviour to pinocchio URDF_MERGE_FIXED_LINKS, not really it does not work well
        #Loads the URDF file for the robot.#
        if self.self_collision_enabled:
            flags = pybullet_client.URDF_USE_INERTIA_FROM_FILE | pybullet_client.URDF_USE_SELF_COLLISION
            if(self.base_type=="fixed"):
                self.bot_pybullet = pybullet_client.loadURDF(
                    urdf_file,
                    self._GetDefaultInitPosition(),
                    self._GetDefaultInitOrientation(),
                    useFixedBase=True,
                    flags=flags)
            else:
                self.bot_pybullet = pybullet_client.loadURDF(
                    urdf_file,
                    self._GetDefaultInitPosition(),
                    self._GetDefaultInitOrientation(),
                    useFixedBase=False,
                    flags=flags)
        else:
            if(self.base_type=="fixed"):
                self.bot_pybullet = pybullet_client.loadURDF(
                    urdf_file, self._GetDefaultInitPosition(),
                    self._GetDefaultInitOrientation(),
                    useFixedBase=True,
                    flags=pybullet_client.URDF_USE_INERTIA_FROM_FILE)
               
            else:
                self.bot_pybullet = pybullet_client.loadURDF(
                    urdf_file, self._GetDefaultInitPosition(),
                    self._GetDefaultInitOrientation(),
                    useFixedBase=False,
                    flags=pybullet_client.URDF_USE_INERTIA_FROM_FILE)

                
        if self.base_type=="on_rack":
            self.rack_constraint = (self._CreateRackConstraint(
                self._GetDefaultInitPosition(), self._GetDefaultInitOrientation(), pybullet_client))
            
        self._BuildJointNameToIdAndActiveJoint(pybullet_client)
        # remove linear damping of the link (0.04 by default) and angular damping (0.04 by default)
        self._RemoveDefaultJointDamping(pybullet_client)
        # remove joint damping and friction
        self._RemoveURDFJointDampingAndFriction(pybullet_client)
        self._BuildFeetJointIDAndForceSensors(pybullet_client,index)
        self._buildLinkNameToId(pybullet_client)


    def _BuildJointNameToIdAndActiveJoint(self, pybullet_client):
            num_joints = pybullet_client.getNumJoints(self.bot_pybullet)
            self.joint_name_to_id = {}
            self.active_joint_ids = []
            for i in range(num_joints):
                joint_info = pybullet_client.getJointInfo(self.bot_pybullet, i)
                #print(joint_info)
                self.joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

                if joint_info[2] != pybullet_client.JOINT_FIXED:
                    self.active_joint_ids.append(joint_info[0])

    # TODO check because apparently ti does not remove the dampign and friction under <joint><dynamic> tag
    def _RemoveDefaultJointDamping(self, pybullet_client):
        num_joints = pybullet_client.getNumJoints(self.bot_pybullet)
        for i in range(num_joints):
            joint_info = pybullet_client.getJointInfo(self.bot_pybullet, i)
            pybullet_client.changeDynamics(joint_info[0],
                                        -1,
                                        linearDamping=0.0,
                                        angularDamping=0.0)
            
    # TODO check because apparently ti does not remove the dampign and friction under <joint><dynamic> tag
    # proobably i need to remove the friction as well
    def _RemoveURDFJointDampingAndFriction(self, pybullet_client):
        num_joints = pybullet_client.getNumJoints(self.bot_pybullet)
        for i in range(num_joints):
            joint_info = pybullet_client.getJointInfo(self.bot_pybullet, i)
            pybullet_client.changeDynamics(joint_info[0],
                                        -1,
                                        jointDamping=0)
            
    def _BuildFeetJointIDAndForceSensors(self, pybullet_client,index):

        # check if self.conf['enable_feet_joint_force_sensors']['feet_contact_names'] is not empty
        if(self.conf['robot_pybullet']['enable_feet_joint_force_sensors'][index]):
            # build the dictionary of feet id and the feet reference frame to stadanrd name
            self.feet_force_sensor_frame_2_id = {}
            # find common elements between the list of contact point and the list of frame associate with each foot 
            fl_contact_frame = list(set(self.conf['robot_pybullet']['enable_feet_joint_force_sensors'][index]) & set(self.conf['sim']['FL'][index]))
            if(not fl_contact_frame):
                raise ValueError("FL sensor frame not found in the list of contact frames, check enable_feet_joint_force_sensors under robot_pybullet and fl under sim in the config file")
            else:
                fl_contact_frame = fl_contact_frame[0]
            fr_contact_frame = list(set(self.conf['robot_pybullet']['enable_feet_joint_force_sensors'][index]) & set(self.conf['sim']['FR'][index]))
            if(not fr_contact_frame):
                raise ValueError("FR sensor frame not found in the list of contact frames, check enable_feet_joint_force_sensors under robot_pybullet and fl under sim in the config file")
            else:
                fr_contact_frame = fr_contact_frame[0]
            rl_contact_frame = list(set(self.conf['robot_pybullet']['enable_feet_joint_force_sensors'][index]) & set(self.conf['sim']['RL'][index]))
            if(not rl_contact_frame):
                raise ValueError("RL sensor frame not found in the list of contact frames, check enable_feet_joint_force_sensors under robot_pybullet and rl under sim in the config file")
            else:
                rl_contact_frame = rl_contact_frame[0]
            rr_contact_frame = list(set(self.conf['robot_pybullet']['enable_feet_joint_force_sensors'][index]) & set(self.conf['sim']['RR'][index]))
            if(not rr_contact_frame):
                raise ValueError("RR sensor frame not found in the list of contact frames, check enable_feet_joint_force_sensors under robot_pybullet and rr under sim in the config file")
            else:
                rr_contact_frame = rr_contact_frame[0]
            
        for _id in range(pybullet_client.getNumJoints(self.bot_pybullet)):
            _link_name = pybullet_client.getJointInfo(self.bot_pybullet, _id)[12].decode('UTF-8')
            _joint_name= pybullet_client.getJointInfo(self.bot_pybullet, _id)[1].decode("utf-8")
            if _link_name in self.conf['sim']['feet_contact_names'][index]:
                self.foot_link_ids[_link_name] = _id
            #here i build the structure of the feet force sensors that i will use to measure the force in the feet 
            # and i will associate this force to the FL FR RL RR label that generalize the feet contact and it is indepent from urdf
            if(self.conf['robot_pybullet']['enable_feet_joint_force_sensors'][index]):
                if _joint_name == fl_contact_frame:
                    self.foot_link_sensors_ids["FL"] = _id
                if _joint_name == fr_contact_frame:
                    self.foot_link_sensors_ids["FR"] = _id
                if _joint_name == rl_contact_frame:
                    self.foot_link_sensors_ids["RL"] = _id
                if _joint_name == rr_contact_frame:
                    self.foot_link_sensors_ids["RR"] = _id    

                # here I enable the joint torque sensors for the feet
                for id in  self.foot_link_sensors_ids.values():
                    pybullet_client.enableJointForceTorqueSensor(self.bot_pybullet, id, True) 

    def _buildLinkNameToId(self,pybullet_client):
        num_joints = pybullet_client.getNumJoints(self.bot_pybullet)
        self.link_name_to_id = {}
        for i in range(num_joints):
            joint_info = pybullet_client.getJointInfo(self.bot_pybullet, i)
            #here when the floating link is found i save the corresponding floating link
            if(joint_info[1].decode("UTF-8")==self.base_name):
                self.base_link_name = joint_info[12].decode("UTF-8")
                link_info = pybullet_client.getLinkState(self.bot_pybullet, joint_info[0], computeLinkVelocity=0, computeForwardKinematics=1)
                # local position offset of inertial frame (center of mass) expressed in the URDF link frame
                self.base_link_2_com_pos_offset = np.asarray(link_info[2])
                #local orientation (quaternion [x,y,z,w]) offset of the inertial frame expressed in URDF link frame.
                self.base_link_2_com_ori_offset = np.asarray(link_info[3])
            self.link_name_to_id[joint_info[12].decode("UTF-8")] = joint_info[0]

    def get_link_id_from_name(self, link_name: str):
        return self.link_name_to_id[link_name]
    
    def get_pybullet_bot_index(self):
        return self.bot_pybullet

    def SetFootFriction(self, pybullet_client, foot_friction):
        """Set the lateral friction of the feet.

    Args:
      foot_friction: The lateral friction coefficient of the foot. This value is
        shared by all four feet.
    """
        for key in self.foot_link_ids.keys():
            link_id = self.foot_link_ids[key]
            pybullet_client.changeDynamics(self.bot_pybullet,
                                           link_id,
                                           lateralFriction=foot_friction)
    
    def SetFootRestitution(self, pybullet_client, foot_restitution):
        """Set the restitution of the feet."""
        for key in self.foot_link_ids.keys():
            link_id = self.foot_link_ids[key]
            pybullet_client.changeDynamics(self.bot_pybullet,
                                           link_id,
                                           restitution=foot_restitution)

        
        
    def _CreateRackConstraint(self, init_position, init_orientation, pybullet_client):
            """Create a constraint that keeps the chassis at a fixed frame.

        This frame is defined by init_position and init_orientation.

        Args:
        init_position: initial position of the fixed frame.
        init_orientation: initial orientation of the fixed frame in quaternion
            format [x,y,z,w].

        Returns:
        Return the constraint id.
        """
            fixed_constraint = pybullet_client.createConstraint(
                parentBodyUniqueId=self.bot_pybullet,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=pybullet_client.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=init_position,
                childFrameOrientation=init_orientation)
            return fixed_constraint

    #TODO check how to process this input from json file and convert it in a list of doubles
    
    def _GetDefaultInitPosition(self):
        return [0.0 , 0.0 , 0.0]
    def _GetDefaultInitOrientation(self):
        return [0.0, 0.0, 0.0, 1.0]
  


    def _GetLinkIdByName(self, pybullet_client, link_name):
        
        # Get the number of joints (links) in the robot
        num_joints = pybullet_client.getNumJoints(self.bot_pybullet)

        # Search for the link with the specified name and get its ID
        link_id = -1  # Default value if the link is not found

        for i in range(num_joints):
            joint_info = pybullet_client.getJointInfo(self.bot_pybullet, i)
            link_name_in_joint_info = joint_info[12].decode("UTF-8")  # Decode the byte string
            if link_name_in_joint_info == link_name:
                link_id = i
                break

        if link_id != -1:
            print("Link " + link_name + " has ID " + str(link_id))
        else:
            print("Link " + link_name + " not found")
        
        return link_id
    
    def getNameActiveJoints(self,pybullet_client):
        active_joint_names = []
        for joint_index in self.active_joint_ids:
            joint_info = pybullet_client.getJointInfo(self.bot_pybullet, joint_index)
            active_joint_names.append(joint_info[1].decode("utf-8"))  # Joint name is stored as bytes, so decode to get a string
        return active_joint_names

class SimInterface():
    """This class provide an interface to the pybullet simulator."""

    def __init__(self,
                 conf_file_name: str, # here i assume that the conf file is in the  config_file folder
                 conf_file_path_ext: str = None, # if it is not none th config file is not in the local config_file folder
                 use_gui: bool = True):
        """Constructs the robot and the environment in pybullet. 

    Args:
      
      conf_file_path: 
      time_step: The time step of the simulation.
      use_gui: whether to bring up the pybullet gui, or to use direct mode without a gui
    """
        
        # reading json file and instatiate some variables
        if conf_file_path_ext: #CHANGE
            conf_file_path = os.path.join(conf_file_path_ext,'configs',conf_file_name) #CHANGE
        else: #CHANGE
            conf_file_path = os.path.join(os.path.dirname(__file__),os.pardir,os.pardir,'configs',conf_file_name)
        with open(conf_file_path) as json_file:
             conf_file_json = json.load(json_file)
             
        # here i save all the parameters that are necessary for the environment
        self.time_step = conf_file_json["sim"]["time_step"]
        # here i need to create the environment and get the robot object
        self.use_gui = use_gui
        connection_mode = None
        if self.use_gui:
            connection_mode = pybullet.GUI
        else:
            connection_mode = pybullet.DIRECT
        self.pybullet_client = bullet_client.BulletClient(connection_mode=connection_mode)
        self.pybullet_client.setPhysicsEngineParameter(numSolverIterations=30)
        self.pybullet_client.setTimeStep(self.time_step)
        self.pybullet_client.setGravity(0, 0, -9.81)
        self.pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
        self.pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground_body = self.pybullet_client.loadURDF("plane.urdf")
        # I create this variable to store information about the environment objects
        self.env=EmptyObj()
        
        # here I load the environment script if is specified in the json file
        if(conf_file_json['env_pybullet']['env_script_name']):
            # here I build the path to the script which is in the env
            if conf_file_path_ext: #CHANGE
                path_to_env = os.path.join(conf_file_path_ext,'models','scenes',conf_file_json['env_pybullet']['env_script_name']) #CHANGE
            else: #CHANGE
                path_to_env = os.path.join(os.path.dirname(__file__),os.pardir,'models','scenes',conf_file_json['env_pybullet']['env_script_name'])
            self.LoadEnv(path_to_env)

        # here I create the robot object lists
        self.bot=[]
        if isinstance(conf_file_json['robot_pybullet']['urdf_path'], list):
            for i in  range(len(conf_file_json['robot_pybullet']['urdf_path'])):
                bot = SimRobot(self.pybullet_client,conf_file_json,i,conf_file_path_ext) #CHANGE
                self.bot.append(bot)
        else:
            raise TypeError("Expected 'urdf_path' to be a list but got {}".format(type(conf_file_json['robot_pybullet']['urdf_path']).__name__))

         #DEBUG --------------------
        self.GetFootFriction()
        self.GetFootRestitution()
         # -------------------------

        # here i reset the robot
        self.ResetPose()
        # gravity direction
        self._com_grav_vector_world_frame = [0, 0, -1]

        # variable to manage the simulator behaviour
        self.step_counter = 0


        #self.Reset(reset_time=reset_time)
        self.observation_history=collections.deque(maxlen=100)
        # get the first observation
        self.ReceiveObservation()

    ## load env function ----------------------------------------------------------------------------
    # here the assumption is that the script already is written in  way that is accesing self.pybullet_client
    def LoadEnv(self, env_script_name):
        # Open the file in read mode ('r')
        with open(env_script_name, 'r') as file:
            # Iterate through each line in the file
            for line in file:
                print("cur_line= ",line)
                exec(line.strip())

    ## simulation functions -------------------------------------------------------------------------
    # reading data from the simulator
    #TODO omogenize using the historical state to manage current state previous and any state with a delay 
    def ReceiveObservation(self):
        for j in  range(len(self.bot)):
            """Receive the observation from sensors.

            This function is called once per step. The observations are only updated
            when this function is called.
            """
            # update previous state
            if self.step_counter > 0:    
                #prev motor angles
                
                self.bot[j].prev_motor_angles = self.bot[j].motor_angles # self.GetMotorAngles(j)
                # prev motor velocities
                self.bot[j].prev_motor_vel =  self.bot[j].motor_velocities #self.GetMotorVelocities(j)
                # prev base position and orientation
                self.bot[j].prev_base_position = self.bot[j].base_position #self.GetBasePosition(j)
                self.bot[j].prev_base_orientation = self.bot[j].base_orientation #self.GetBaseOrientation(j)
                self.bot[j].prev_base_lin_vel = self.bot[j].base_lin_vel  #self.GetBaseLinVelocity(j)
                self.bot[j].prev_base_ang_vel = self.bot[j].base_ang_vel  #self.GetBaseAngVelocity(j)
                # prev base velocities in base frame
                self.bot[j].prev_base_lin_vel_base_frame = self.bot[j].base_lin_vel_body_frame  #self.GetBaseLinVelocityBodyFrame(j)
                self.bot[j].prev_base_ang_vel_base_frame = self.bot[j].base_ang_vel_body_frame  #self.GetBaseAngVelocityBodyFrame(j)
            else:
                self.bot[j].prev_motor_angles = self.bot[j].init_joint_angles
                # prev motor velocities
                self.bot[j].prev_motor_vel = self.bot[j].init_joint_vel
                self.bot[j].prev_motor_accelerations = np.zeros(len(self.bot[j].active_joint_ids))
                # prev base position and orientation
                self.bot[j].prev_base_position = self.bot[j].link_floating_base_pos
                self.bot[j].prev_base_orientation = self.bot[j].link_floating_base_ori
                if self.bot[j].conf['robot_pybullet']['init_link_base_vel'] and self.bot[j].conf['robot_pybullet']['init_link_base_ang_vel']:
                    self.bot[j].prev_base_lin_vel = self.bot[j].conf['robot_pybullet']['init_link_base_vel']
                    self.bot[j].prev_base_ang_vel = self.bot[j].conf['robot_pybullet']['init_link_base_ang_vel']
                # prev base velocities in base frame
                #self.bot[j].prev_base_lin_vel_base_frame = self.GetBaseLinVelocityBodyFrame(j)
                #self.bot[j].prev_base_ang_vel_base_frame = self.GetBaseAngVelocityBodyFrame(j)

            # update of the current state    
            self.bot[j].motor_angles = self.ExtractMotorAngles(j)
            self.bot[j].motor_velocities = self.ExtractMotorVelocities(j)
            self.bot[j].motor_accelerations = self.ComputeMotorAccelerationTMinusOne(j)
            #  TODO DEBUG TOCHECK apparently this function return position and orientation of the center of mass of the robot
            # not the floating joint position and orientation (which i guess is the one used by the pinocchio model)
            self.bot[j].base_position, self.bot[j].base_orientation = (self.pybullet_client.getBasePositionAndOrientation(self.bot[j].bot_pybullet))
            # DEBUG for checking the postion of the floating joint on the flkoating link between pinocchio and pybullet
            #base_link_frame_pos, base_link_frame_ori = self.GetFloatingBaseLinkPositionAndOrientation()

            self.bot[j].base_position = np.asarray(self.bot[j].base_position)
            self.bot[j].base_orientation = np.asarray(self.bot[j].base_orientation)
            # Computes the relative orientation relative to the robot's
            # initial_orientation. TODO (they appear to be indentical (orientation and self.bot.base_orientation) why is it necessary??? to check)
            # _, self.bot.base_orientation = self.pybullet_client.multiplyTransforms(
            #     positionA=[0, 0, 0],
            #     orientationA=orientation,
            #     positionB=[0, 0, 0],
            #     orientationB=self.bot.init_orientation_inv)
            # body velocities world frame
            self.bot[j].base_lin_vel, self.bot[j].base_ang_vel = self.pybullet_client.getBaseVelocity(self.bot[j].bot_pybullet)
            self.bot[j].base_lin_vel = np.asarray(self.bot[j].base_lin_vel)
            self.bot[j].base_ang_vel = np.asarray(self.bot[j].base_ang_vel)
            # body velocities base frame
            self.bot[j].base_lin_vel_body_frame, self.bot[j].base_ang_vel_body_frame = self.ComputeBaseVelocitiesBodyFrame(j)
            
            #### UPDATING CURRENT STATE FOR MANAGING DELAY IN THE SIMULATION #### ------------------------------
           
            current_state = {
            'motor_angles':self.bot[j].motor_angles,
            'motor_velocities':self.bot[j].motor_velocities,
            'motor_accelerations':self.bot[j].motor_accelerations,
            'base_position': self.bot[j].base_position,
            'base_orientation': self.bot[j].base_orientation,
            'base_lin_vel': self.bot[j].base_lin_vel,
            'base_ang_vel': self.bot[j].base_ang_vel,
            'base_lin_vel_body_frame': self.bot[j].base_lin_vel_body_frame,
            'base_ang_vel_body_frame': self.bot[j].base_ang_vel_body_frame
            }
            self.bot[j].state_buffer.append(current_state)  # Update the buffer
        
        # get all ideal observation (no noise no delay) take all the observation for all the robots   
        self.observation_history.appendleft(self.GetAllObservationIdeal())
            
    # sending command to the simulator
    # function to advance the simulation every time step
    # added unseless_control_mode for retrocompatibility for now
    def Step(self, action, unseless_control_mode=None):
        """Steps simulation."""
        
        #proc_action = self.ProcessAction(action, i)
        proc_action = action
        self._StepInternal(proc_action)
        self.step_counter += 1
        self.last_action = action

    # function that is actually making the simulation step after applying the control actions
    def _StepInternal(self, action):
        # here i apply the action to the robot
        self.ApplyAction(action)
        # the simulation step is singular for all the robots
        self.pybullet_client.stepSimulation()
        self.ReceiveObservation()

    # function that compute the applied torque (even if is not a torque control) and send the torque to the simulated
    # robot
    def ApplyAction(self, cmd):
        """Apply the motor commands using the motor model.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
      motor_control_mode: A MotorControlMode enum.
    """
        self.last_action_time = self.step_counter * self.time_step
        # motor_commands = np.asarray(motor_commands)
         # we apply the action for each robot
        for index in  range(len(self.bot)):
            #motor_commands = self.bot[index].servo_motor_model.compute_torque(cmd[index], self.GetMotorAngles(index), self.GetMotorVelocities(index), motor_control_mode[index])
            # Transform into the motor space when applying the torque.
            #self.bot[index].applied_motor_commands = np.multiply(motor_commands,
            #                                        self.bot[index].motor_direction)
            
            # all the joints are controlled by torque
            #self._SetMotorTorqueByIds( motor_control_mode[index], self.bot[index].active_joint_ids, self.bot[index].applied_motor_commands,index)
            # Determine command and control mode based on the number of robots
            current_cmd = cmd[index] if len(self.bot) > 1 else cmd
            # now control mode is inside current_cmd
            #current_control_mode = motor_control_mode[index] if len(self.bot) > 1 else motor_control_mode

            # Compute the torque using the appropriate command and control mode
            if self.bot[index].base_type=="fixed":
                cur_M = self.ComputeMassMatrix(previous_state=False, index=index)
            if self.bot[index].base_type=="floating":
                M_big = self.ComputeMassMatrix(previous_state=False, index=index)
                m_big_base, m_big_joints, m_big_joint_to_base, m_big_base_to_joint=self.ComputeSplitMassMatrix(M_big)
                cur_M = m_big_joints

            motor_commands = self.bot[index].servo_motor_model.compute_torque(current_cmd, self.GetMotorAngles(index), self.GetMotorVelocities(index),self.ComputeMotorAccelerationTMinusOne(index), cur_M)
            # Transform into the motor space when applying the torque
            self.bot[index].applied_motor_commands = np.multiply(motor_commands, self.bot[index].motor_direction)
            
            # All the joints are controlled by torque
            self._SetMotorTorqueByIds(self.bot[index].active_joint_ids, self.bot[index].applied_motor_commands, index)

    # function that apply the motor torque to the simulation (here one joint)
    # we assume that the commands is always in torque (more stable in pybullet)
    # the different control mode are managed by the servo_motor_model
    def _SetMotorTorqueById(self, motor_id, commands, index=0):

        self.pybullet_client.setJointMotorControl2(
            bodyIndex=self.bot[index].bot_pybullet,
            jointIndex=motor_id,
            controlMode=self.pybullet_client.TORQUE_CONTROL,
            force=commands)

    # function that apply the motor command to the simulation (here many joints) 
    # we assume that the commands is always in torque (more stable in pybullet)
    # the different control mode are managed by the servo_motor_model
    # HERE I CAN ADD THE CONTROL SWITCH USING THE COMMAND VARIABLE
    # FOR NOW EVERYTHIN IS TORQUE CONTROL
    def _SetMotorTorqueByIds(self, motor_ids, commands, index=0):

        # here i check if the commands is non empty and then i apply the torque to the joints
        if(commands.size>0): #CHANGE
            # check if commands is multidimensional
            if len(commands.shape) > 1: #CHANGE
                raise ValueError("Expected 'commands' to be a 1D array but got a {}D array".format(len(commands.shape))) #CHANGE
            control_mode=self.pybullet_client.TORQUE_CONTROL
            self.pybullet_client.setJointMotorControlArray(
            bodyIndex=self.bot[index].bot_pybullet,
            jointIndices=motor_ids,
            controlMode=control_mode,
            forces=commands.tolist()) #CHANGE

    
    def _SetDesiredMotorAngleByName(self, motor_name, desired_angle, index=0):
        self._SetDesiredMotorAngleById(self.bot[index].joint_name_to_id[motor_name],
                                       desired_angle)
    
    # # reset functions (run at the beginning to initialize all the structure and make the robot stand up) -------------------------------
    # TODO here we should the set of the base velocity and joint velocities  
    def ResetPose(self):
        i  = 0
        for j in  range(len(self.bot)):
            i=0
            for id in self.bot[j].active_joint_ids:
                # very important! this allows for the disabling of the motor at the joint level
                # this is necessary to avoid the motor to fight against any kind of motion
                # it acts as a joint with friction
                # for more details check this link: https://github.com/bulletphysics/bullet3/issues/2463
                self.pybullet_client.setJointMotorControl2(bodyIndex=self.bot[j].bot_pybullet,
                                                        jointIndex=(id),
                                                        controlMode=self.pybullet_client.VELOCITY_CONTROL,
                                                        targetVelocity=0,
                                                        force=0)
                # here i set the joint state to the initial position 
                # self.pybullet_client.resetJointState(self.bot.bot_pybullet,
                #                                     id,
                #                                     self.bot.init_joint_angles[i],
                #                                     targetVelocity=0)
                self.pybullet_client.resetJointState(self.bot[j].bot_pybullet,
                                                    id,
                                                    self.bot[j].init_joint_angles[i],
                                                    targetVelocity=self.bot[j].init_joint_vel[i])
                i = i + 1
                #print(i)
       
            # reset position and orientation to the one specified in the configuration file
            # TODO check if it works for the fixed base as well 
            self.pybullet_client.resetBasePositionAndOrientation(self.bot[j].bot_pybullet, self.bot[j].link_floating_base_pos, self.bot[j].link_floating_base_ori)
            # here I update the the init_orientation_inv with the actual initial com orientation of the floating body 
            _, self.bot[j].init_orientation_inv = self.pybullet_client.invertTransform(position=[0, 0, 0], orientation=self.bot[j].link_floating_base_ori)
            
            if(self.bot[j].base_type=="floating"):
                # setting initil com velocity of the robot (if specified in the config file)
                if self.bot[j].conf['robot_pybullet']['init_link_base_vel'] and self.bot[j].conf['robot_pybullet']['init_link_base_ang_vel']:
                    self.pybullet_client.resetBaseVelocity(self.bot[j].bot_pybullet, self.bot[j].conf['robot_pybullet']['init_link_base_vel'], self.bot[j].conf['robot_pybullet']['init_link_base_ang_vel'])
                    #self.bot.prev_base_pos_vel = self.bot.conf['robot_pybullet']['init_link_base_vel']
                    #self.bot.prev_base_pos_ang_vel = self.bot.conf['robot_pybullet']['init_link_base_ang_vel']
                else:
                    print("WARNING: the initial velocity of the robot is not specified in the config file, setting zero velocity for the base instead")
                    #self.bot.prev_base_pos_vel = np.array([0, 0, 0])
                    #self.bot.prev_base_pos_ang_vel = np.array([0, 0, 0])
            
    # TODO functions that has not be updated yet (not sure it is necessary)
    # TODO all these function are executed for resetting the robot in the original code
    # def Reset(self, reload_urdf=True, default_motor_angles=None, reset_time=3.0):
    #     """Reset the minitaur to its initial states.

    # Args:
    #   reload_urdf: Whether to reload the urdf file. If not, Reset() just place
    #     the minitaur back to its starting position.
    #   default_motor_angles: The default motor angles. If it is None, minitaur
    #     will hold a default pose (motor angle math.pi / 2) for 100 steps. In
    #     torque control mode, the phase of holding the default pose is skipped.
    #   reset_time: The duration (in seconds) to hold the default motor angles. If
    #     reset_time <= 0 or in torque control mode, the phase of holding the
    #     default pose is skipped.
    # """
    #     if reload_urdf:
            
    #         self._RecordMassInfoFromURDF()
    #         self._RecordInertiaInfoFromURDF()
    #         self.ResetPose(add_constraint=True)
    #     else:
    #         self.p.pybullet_client.resetBasePositionAndOrientation(
    #             self.bot.bot_pybullet, self._GetDefaultInitPosition(),
    #             self._GetDefaultInitOrientation())
    #         self.p.pybullet_client.resetBaseVelocity(self.bot.bot_pybullet, [0, 0, 0],
    #                                                  [0, 0, 0])
    #         self.ResetPose(add_constraint=False)

    #     self.observation_history.clear()
    #     self.step_counter = 0
    #     self.state_action_counter = 0
    #     self.last_action = np.zeros(self.p.num_motors*5)
    #     self.last_action_torques = np.zeros(self.p.num_motors)
    #     self._SettleDownForReset(default_motor_angles, reset_time)
    #     # at the end of the reset we set a true reset_done to communicate that we can proceed with the control part
    #     self.reset_done = True

    # def _SettleDownForReset(self, default_motor_angles, reset_time):
    #     """Sets the default motor angles and waits for the robot to settle down.

    # The reset is skipped is reset_time is less than zero.

    # Args:
    #   default_motor_angles: A list of motor angles that the robot will achieve
    #     at the end of the reset phase.
    #   reset_time: The time duration for the reset phase.
    # """
    #     if reset_time <= 0:
    #         return
    #     # Important to fill the observation buffer.
    #     self.ReceiveObservation()
    #     for _ in range(500):
    #         self._StepInternal(
    #             self.rd.INIT_MOTOR_ANGLES,
    #             motor_control_mode=robot_config.MotorControlMode.POSITION)
    #         # Don't continue to reset if a safety error has occurred.
    #         if not self.p.is_safe:
    #             return

    #     if default_motor_angles is None:
    #         return

    #     num_steps_to_reset = int(reset_time / self.time_step)
    #     for _ in range(num_steps_to_reset):
    #         self._StepInternal(
    #             default_motor_angles,
    #             motor_control_mode=robot_config.MotorControlMode.POSITION)
    #         # Don't continue to reset if a safety error has occurred.
    #         if not self.p.is_safe:
    #             return

    def GetFootContacts(self):
        BODY_B_FIELD_NUMBER = 2
        LINK_A_FIELD_NUMBER = 3
        all_contacts = self.pybullet_client.getContactPoints(bodyA=self.bot.bot_pybullet)
        contacts = [False, False, False, False]
        for contact in all_contacts:
            # Ignore self contacts
            if contact[BODY_B_FIELD_NUMBER] == self.bot.bot_pybullet:
                continue
            try:
                toe_link_index = self.bot.foot_link_ids.index(
                    contact[LINK_A_FIELD_NUMBER])
                contacts[toe_link_index] = True
            except ValueError:
                continue

        return contacts
    
     # dynamics methods --------------------------------------------------------------------------------
    def ComputeMassMatrixRNEA(self,previous_state=False,index=0):
        if(self.bot.base_type=="fixed"):
            if(previous_state):
                x,xdot = self.GetSystemPreviousStateInternal(True)
            else:
                x,xdot = self.GetSystemStateInternal(True,index)
            M = np.zeros((len(xdot), len(xdot)))
            xdot_zero    = [0] * len(xdot)
            xdotdot_zero = [0] * len(xdot)
            gravity = np.asarray(self.pybullet_client.calculateInverseDynamics(self.bot[index].bot_pybullet, x.tolist(), xdot_zero, xdotdot_zero))
            gravity = np.delete(gravity, 6, 0)
            for i in range(len(xdot)):
                cur_accels = np.zeros(len(xdot))
                cur_accels[i] = 1
                cur_M_col = np.asarray(self.pybullet_client.calculateInverseDynamics(self.bot[index].bot_pybullet, x.tolist(), xdot_zero, cur_accels.tolist())) - gravity
                M[:, i] = cur_M_col
            return M

        else:
            print("it is not possible to compute the mass matrix with the floating base using the RNEA method in pybullet")
            # x,xdot = self.GetSystemState()
            # """Computes the mass matrix of the robot."""
            # M = np.zeros((len(xdot)+1, len(xdot)+1))
            # xdot_zero    = [0] * (len(xdot)+1)
            # xdotdot_zero = [0] * (len(xdot)+1)
            # gravity = np.asarray(self.pybullet_client.calculateInverseDynamics(self.bot.bot_pybullet, x.tolist(), xdot_zero, xdotdot_zero,flags=1))
            # for i in range(len(xdot)+1):
            #     cur_accels = np.zeros(len(xdot)+1)
            #     cur_accels[i] = 1
            #     cur_M_col = np.asarray(self.pybullet_client.calculateInverseDynamics(self.bot.bot_pybullet, x.tolist(), xdot_zero, cur_accels.tolist(),flags=1)) - gravity
            #     M[:, i] = cur_M_col
            # M = np.delete(M, 6, 0)
            # M = np.delete(M, 6, 1)
            # return M
    
    def ComputeMassMatrix(self,previous_state=False,index=0):
        if(self.bot[index].base_type=="fixed"):
            if(previous_state):
                x,xdot = self.GetSystemPreviousStateInternal(True)
            else:
                x,xdot = self.GetSystemStateInternal(True,index)
            M = self.pybullet_client.calculateMassMatrix(self.bot[index].bot_pybullet,x.tolist())
            return np.asarray(M)
        else:
            if(previous_state):
                x,xdot = self.GetSystemPreviousStateInternal()
            else:
                x,xdot = self.GetSystemStateInternal()
            """Computes the mass matrix of the robot."""
            M = self.pybullet_client.calculateMassMatrix(self.bot[index].bot_pybullet,x.tolist(),flags=1)
            # remove the 7th row because it is zero (that's how pybullet works with flag=1)
            M = np.delete(M, 6, 0)
            # remove the 7th column because it is zero (that's how pybullet works with flag=1)
            M = np.delete(M, 6, 1)
            # wrong version kept only for update in the future
            #M = self.pybullet_client.calculateMassMatrix(self.bot.bot_pybullet,x.tolist())
            return np.asarray(M)
        
    def ComputeSplitMassMatrix(self,M_big):
        """Splits the mass matrix into the base mass matrix and the leg mass matrix."""
        m_big_base = M_big[:6, :6]
        m_big_joints = M_big[6:, 6:]
        m_big_joint_to_base = M_big[:6, 6:]
        m_big_base_to_joint = M_big[6:, :6]
        return m_big_base, m_big_joints, m_big_joint_to_base, m_big_base_to_joint

    
    def ComputeCoriolisAndGravityForces(self,previous_state=False,index=0):
        """Computes the Coriolis and gravity forces."""
        if(self.bot[index].base_type=="fixed"):
            if(previous_state):
                x,xdot = self.GetSystemPreviousStateInternal(True)
            else:
                x,xdot = self.GetSystemStateInternal(True)
            return np.asarray(self.pybullet_client.calculateInverseDynamics(self.bot[index].bot_pybullet, x.tolist(), xdot.tolist(), xdotdot_zero))
        else:
            if(previous_state):
                x,xdot = self.GetSystemPreviousStateInternal()
            else:
                x,xdot = self.GetSystemStateInternal()
        xdotdot_zero = [0] * len(xdot) + 1
        return np.asarray(self.pybullet_client.calculateInverseDynamics(self.bot[index].bot_pybullet, x.tolist(), xdot.tolist(), xdotdot_zero,flags=1))
    
    # with this flag base_velocity_in_base_frame we can compute the gravity action on the base but in the base frame
    def ComputeGravity(self,base_velocity_in_base_frame=False,previous_state=False,index=0):
        grav = []
        if(self.bot[index].base_type=="fixed"):
            if(previous_state):
                x,xdot = self.GetSystemPreviousStateInternal(True)
            else:
                x,xdot = self.GetSystemStateInternal(True)
            xdot_zero    = [0] * len(xdot)
            xdotdot_zero = [0] * len(xdot)
            return np.asarray(self.pybullet_client.calculateInverseDynamics(self.bot[index].bot_pybullet, x.tolist(), xdot_zero, xdotdot_zero))
        else:
            if(previous_state):
                x,xdot = self.GetSystemPreviousStateInternal()
            else:
                x,xdot = self.GetSystemStateInternal()
            xdot_zero    = [0] * (len(xdot) + 1)
            xdotdot_zero = [0] * (len(xdot) + 1) 
            grav = np.asarray(self.pybullet_client.calculateInverseDynamics(self.bot.bot_pybullet, x.tolist(), xdot_zero, xdotdot_zero, flags=1))
            # remove the 7th row because it is zero (that's how pybullet works with flag=1)
            grav = np.delete(grav, 6, 0)
            # wrong version kept only for update in the future
            # xdot_zero    = [0] * (len(xdot))
            # xdotdot_zero = [0] * (len(xdot)) 
            # grav = np.asarray(self.pybullet_client.calculateInverseDynamics(self.bot.bot_pybullet, x.tolist(), xdot_zero, xdotdot_zero))

            #TODO check if this one is ok for the rotational part
            if(base_velocity_in_base_frame):
                sim_base_gravity_pos = grav[:3,]
                sim_base_gravity_pos = self.TransformWorld2Body(sim_base_gravity_pos,index)
                sim_base_gravity_ori = grav[3:6,]
                sim_base_gravity_ori = self.TransformAngularVelocityToLocalFrame(sim_base_gravity_ori)
                # here we reassemble the gravity vector in the body frame
                sim_base_gravity = np.concatenate((sim_base_gravity_pos,sim_base_gravity_ori),axis=0)
                grav[:6,] = sim_base_gravity
            return grav
        
    def ComputeCoriolis(self,base_velocity_in_base_frame=False,previous_state=False,index=0):
        """Computes the Coriolis forces."""
        if(self.bot[index].base_type=="fixed"):
            if(previous_state):
                x,xdot = self.GetSystemPreviousStateInternal(True)
            else:
                x,xdot = self.GetSystemStateInternal(True)
            xdot_zero    = [0] * (len(xdot))
            xdotdot_zero = [0] * (len(xdot))
            gravity = np.asarray(self.pybullet_client.calculateInverseDynamics(self.bot[index].bot_pybullet, x.tolist(), xdot_zero, xdotdot_zero))
            coriolis = np.asarray(self.pybullet_client.calculateInverseDynamics(self.bot[index].bot_pybullet, x.tolist(), xdot.tolist(), xdotdot_zero)) - gravity
            return coriolis
        else:
            x,xdot = self.GetSystemStateInternal()
            # I need to add a zero to the velocity vector because pybullet calculateInverseDynamics expect a vector of 7 elements for the base velocity 
            xdot_new = np.zeros((len(xdot) + 1))
            xdot_new[:6,] = xdot[:6,]
            xdot_new[7:,] = xdot[6:,]
            xdot_zero    = [0] * (len(xdot) + 1)
            xdotdot_zero = [0] * (len(xdot) + 1)
            gravity = np.asarray(self.pybullet_client.calculateInverseDynamics(self.bot[index].bot_pybullet, x.tolist(), xdot_zero, xdotdot_zero,flags=1))
            coriolis = np.asarray(self.pybullet_client.calculateInverseDynamics(self.bot[index].bot_pybullet, x.tolist(), xdot_new.tolist(), xdotdot_zero,flags=1)) - gravity
            coriolis = np.delete(coriolis, 6, 0)
            # wrong version kept only just in case the code is updated in the future
            # xdot_zero    = [0] * (len(xdot))
            # xdotdot_zero = [0] * (len(xdot))
            # gravity = np.asarray(self.pybullet_client.calculateInverseDynamics(self.bot.bot_pybullet, x.tolist(), xdot_zero, xdotdot_zero))
            # coriolis = np.asarray(self.pybullet_client.calculateInverseDynamics(self.bot.bot_pybullet, x.tolist(), xdot.tolist(), xdotdot_zero)) - gravity
            
            #TODO check if this one is right for the velocity part 
            if(base_velocity_in_base_frame): 
                sim_base_coriolis_pos = coriolis[:3,]
                sim_base_coriolis_pos = self.TransformWorld2Body(sim_base_coriolis_pos,index)
                sim_base_coriolis_ori = coriolis[3:6,]
                sim_base_coriolis_ori = self.TransformAngularVelocityToLocalFrame(sim_base_coriolis_ori)
                # here we reassemble the gravity vector in the body frame
                sim_base_coriolis = np.concatenate((sim_base_coriolis_pos,sim_base_coriolis_ori),axis=0)
                coriolis[:6,] = sim_base_coriolis
            
            return coriolis
    
    def DirectDynamicsActuatedNoContact(self,tau,previous_state=False, index=0):
        if(self.bot[index].base_type=="fixed"):
            c = self.ComputeCoriolis(previous_state=previous_state,index=index)
            g = self.ComputeGravity(previous_state=previous_state,index=index)
            M = self.ComputeMassMatrix(previous_state=previous_state,index=index)
            # coriolis + centrifugal + gravity
            n = c + g
            acc =  np.linalg.inv(M) @ (tau-n)
        elif(self.bot[index].base_type=="floating"):
            n_bdot = 6
            c = self.ComputeCoriolis(previous_state=previous_state,index=index)
            c_b =  c[:n_bdot,]
            c_q =  c[n_bdot:,]
            g = self.ComputeGravity(previous_state=previous_state,index=index)
            g_b =  g[:n_bdot,]
            g_q =  g[n_bdot:,]
            M = self.ComputeMassMatrix(previous_state=previous_state,index=index)
            M_bb = M[:n_bdot, :n_bdot]
            M_bq = M[:n_bdot,  n_bdot:]
            M_qq = M[n_bdot:,  n_bdot:] 
            M_qb = M[n_bdot:, :n_bdot]
            # coriolis + gravity on the base
            n_b = c_b + g_b
            # coriolis + gravity on the joints
            n_q = c_q + g_q
            Mass_matrix_actuated = M_qq - M_qb @ np.linalg.inv(M_bb) @ M_bq
            Coriolis_actuated = -n_q + M_qb @ np.linalg.inv(M_bb) @ n_b
            acc =  np.linalg.inv(Mass_matrix_actuated) @ Coriolis_actuated
        return acc

    # utilities function -------------------------------------------------------------------------------
    
    def SkewSymmetric(self, vector):
        """Return the skew symmetric matrix of a vector."""
        return np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]],
                        [-vector[1], vector[0], 0]])
    
    # TODO to fix with the rotation from world to body frame
    # def RotateTwistWorld2Base(self, rigid_lin_vel_world, rigid_ang_vel_world, world_body_position, world_body_orientation_quat):
    #     """Rotate a twist by a roto traslation."""
    #     # Convert the body orientation to a rotation matrix.
    #     world_twist = np.concatenate((np.asarray(rigid_ang_vel_world), np.asarray(rigid_lin_vel_world)))
    #     world_body_orientation = self.pybullet_client.getMatrixFromQuaternion(world_body_orientation_quat)
    #     world_body_orientation = np.array(world_body_orientation).reshape(3, 3)

    #     body_body_orientation = np.transpose(world_body_orientation)
    #     body_body_position = -np.dot(body_body_orientation, world_body_position)
        
    #     adjoint_matrix = np.zeros((6, 6))
    #     adjoint_matrix[:3, :3] = body_body_orientation
    #     adjoint_matrix[3:, 3:] = body_body_orientation
    #     adjoint_matrix[3:, :3] = np.dot(self.SkewSymmetric(body_body_position), body_body_orientation)
    #     body_twist = np.dot(adjoint_matrix, world_twist)
    #     # first is the body linear velocity and then body rotational velocity
    #     return  body_twist[3:], body_twist[:3]
    
    
    def TransformWorld2Body(self, world_value,index=0):
        """Transform the value from world frame to body frame.
        Args:
          world_vector: The value in world frame.
        Returns:
          The value in body frame.
        """
        
        #base_orientation = self.GetBaseOrientation(j)
        base_orientation = self.bot[index].base_orientation

        _, inverse_rotation = self.pybullet_client.invertTransform(
        (0, 0, 0), base_orientation)

        pos, _ = self.pybullet_client.multiplyTransforms((0, 0, 0), inverse_rotation, world_value, (0, 0, 0, 1))

        return np.array(pos)
        
    def TransformBody2World(self, body_value,index=0):
        """Transform the value from world frame to body frame.
        Args:
          world_vector: The value in world frame.
        Returns:
          The value in body frame.
        """
        
        #base_orientation = self.GetBaseOrientation(j)
        base_orientation = self.bot[index].base_orientation

        pos, _ = self.pybullet_client.multiplyTransforms((0, 0, 0), base_orientation, body_value, (0, 0, 0, 1))

        return np.array(pos)

    def TransformAngularVelocityToLocalFrame(self, angular_velocity,
                                             orientation):
        """Transform the angular velocity from world frame to robot's frame.

    Args:
      angular_velocity: Angular velocity of the robot in world frame.
      orientation: Orientation of the robot represented as a quaternion.

    Returns:
      angular velocity of based on the given orientation.
    """
        # Treat angular velocity as a position vector, then transform based on the
        # orientation given by dividing (or multiplying with inverse).
        # Get inverse quaternion assuming the vector is at 0,0,0 origin.
        _, orientation_inversed = self.pybullet_client.invertTransform(
            [0, 0, 0], orientation)
        # Transform the angular_velocity at neutral orientation using a neutral
        # translation and reverse of the given orientation.
        relative_velocity, _ = self.pybullet_client.multiplyTransforms(
            [0, 0, 0], orientation_inversed, angular_velocity,
            self.pybullet_client.getQuaternionFromEuler([0, 0, 0]))
        return np.asarray(relative_velocity)

    # the resulting rotation corresponds to have a point or a vector in space and you want to rotate it first by Q2 and then by Q1, 
    # you would use the quaternion product Q1 * Q2 to represent this combined rotation.
    # when it says first it is equivalent to do with rotation matrix 1^R_3=1^R_2 * 2^R_3 
    # in order to find 1^R_3. if we define 1^R_2 = Q1 and 2^R_3 = Q2 then 1^R_3 = Q1 * Q2 where in this case 1^R_3 is a quaternion
    # first Q2 thant Q1
    def quaternion_multiply(Q1, Q2):
        # Extracting individual components from the quaternions
        x1, y1, z1, w1 = Q1
        x2, y2, z2, w2 = Q2

        # Applying the quaternion multiplication formula
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2

        return [x, y, z, w]

    # # get and set functions ---------------------------------------------------------------------------------------
    
    #### IMPORTANT #### anything that is get under here refer to functions that provide values outside the simulator (for the controller)
    # also function that are get and depends on other get are still valid (for example getstate())
    def GetPyBulletClient(self):
        return self.pybullet_client
    
    def GetTimeStep(self):
        return self.time_step
    

    def GetTimeSinceReset(self):
        return self.step_counter * self.time_step

    #with this function i can get the position and orientation of any robot link in world frame
    # link_or_com = allows to choose if the position and orientation of the joint which the link is attached to
    #  or the link CoM is returned (world frame)
    def GetLinkPositionAndOrientation(self, link_name, joint_or_com, index=0):
        # here i check if the link name is in the list of the links of the robot
        if link_name not in self.bot[index].link_name_to_id.keys():
            print("ERROR: the link name is not in the list of the links of the robot")
            return [], []
        else:
            link_id =self.bot[index].link_name_to_id[link_name]
            
            result = self.pybullet_client.getLinkState(self.bot[index].bot_pybullet, link_id, computeLinkVelocity=0, computeForwardKinematics=1)
            if(joint_or_com=="joint"):
                # result[4]= position of the joint in world frame
                # result[5]= orientation of the joint in world frame
                return result[4], result[5]
            elif(joint_or_com=="com"):
                # result[0]= position of the link CoM in world frame
                # result[1]= orientation of the link CoM in world frame
                return result[0], result[1]

    # this function allows for computing position and orientation of the reference frame located in the link frame (urdf) instead of the CoM
    # I wrote this function to verify if when i measure the floating base i get the floating base center of mass or the floating base link
    # from this i can confirm i get the floating base link
    # TODO now we need to ensure that the position of the link frame is the same with the one in pinocchio
    def GetFloatingBaseLinkPositionAndOrientation(self, index=0):
        # here I'm using a weird name just to avoid collision with the common names used in the urdf for bases
        if self.bot[index].base_type == "fixed":
            print("the robot has a fixed base, it is not possible to get the position and orientation of the floating base link")
            return [], []
        else:
            link_name =self.bot[index].base_link_name
            link_pos_floating_base_world, link_ori_floating_base_world = self.GetLinkPositionAndOrientation(link_name, "joint")
            return link_pos_floating_base_world, link_ori_floating_base_world

    # here we assume the initial position is the link position and not the com position
    # I need to do the convesion from the floating base of the link to the one of the com
    def GetConfInitPosition(self,index):
            #link_floating_base_pos = self.bot[index].conf['robot_pybullet']["init_link_base_position"][index]
            # i need to find the com position correspoding to the current link floating base position
            #pos, _ = self.pybullet_client.multiplyTransforms((0, 0, 0), self.bot.conf['robot_pybullet']["init_link_base_orientation"], self.bot.base_link_2_com_pos_offset, (0, 0, 0, 1))
            #com_floating_base_pos = link_floating_base_pos + pos
            return self.bot[index].link_floating_base_pos
   
    
    # here we assume the initial position is the link position and not the com position
    # I need to do the conversion from the floating base of the link to the one of the com
    def GetConfInitOrientation(self,index):
            #link_floating_base_ori = self.bot[index].conf['robot_pybullet']["init_link_base_orientation"][index]
            # here i need to compose the rotation to get the orientation of the frame attached to the CoM (i think the rotation offset between link and com is always the identity)
            #com_floating_base_ori = self.quaternionProduct(link_floating_base_ori,self.bot.base_link_2_com_ori_offset)
            return self.bot[index].link_floating_base_ori
    
    def  GetInitMotorAngles(self,index=0):
        return self.bot[index].init_joint_angles
    
    def GetInitMotorVelocities(self,index=0):
        return self.bot[index].init_joint_vel


    def ExtractMotorAngles(self, index):
        motor_angles = []
        cur_joint_states = self.pybullet_client.getJointStates(self.bot[index].bot_pybullet, self.bot[index].active_joint_ids)
        # if self.bot.joint_states is not empty
        if cur_joint_states:
            motor_angles = [state[0] for state in cur_joint_states]
            motor_angles = np.multiply(
                np.asarray(motor_angles) - np.asarray(self.bot[index].motor_offset),
                self.bot[index].motor_direction)
        
        return motor_angles
    # affected by delay measurement if true
    def GetMotorAngles(self,index):
        """Gets the motor angles at the current moment

    Returns:
      Motor angles
    """ 
        if self.bot[index].delay_measure_flag:
            if len(self.bot[index].state_buffer) > 0:
                delayed_state = self.bot[index].state_buffer[0]
                motor_angles = delayed_state['motor_angles'].copy()
            else:
                print("buffer is empty")
        else:
            motor_angles = self.bot[index].motor_angles.copy()

        # Add noise if the noise flag is set
        if self.bot[index].noise_flag:
            noise = np.random.normal(0, self.bot[index].robot_noise["joint_cov"], size=motor_angles.shape)
            motor_angles += noise

        return motor_angles
    
    def ExtractMotorVelocities(self,index):
        motor_velocities = []
        cur_joint_states = self.pybullet_client.getJointStates(self.bot[index].bot_pybullet, self.bot[index].active_joint_ids)
        if cur_joint_states:
            motor_velocities = [state[1] for state in cur_joint_states]
            motor_velocities = np.multiply(motor_velocities, self.bot[index].motor_direction)
        return motor_velocities
    # affected by delay measurement if true
    def GetMotorVelocities(self,index):
        """Get the velocity of all  motors.

        Returns:
        Velocities of all motors.
        """
        if self.bot[index].delay_measure_flag:
            if len(self.bot[index].state_buffer) > 0:
                delayed_state = self.bot[index].state_buffer[0]
                motor_velocities = delayed_state['motor_velocities'].copy()
            else:
                print("buffer is empty")
        else:
            motor_velocities = self.bot[index].motor_velocities.copy()
        
        # Add noise if the noise flag is set
        if self.bot[index].noise_flag:
            noise = np.random.normal(0, self.bot[index].robot_noise["joint_vel_cov"], size=motor_velocities.shape)
            motor_velocities += noise

        return motor_velocities

    def ComputeMotorAccelerationTMinusOne(self,index):
        """Get the acceleration of all the motors. at the previous time step"""
        if self.bot[index].num_motors == 0:
            return np.zeros(self.bot[index].num_motors)
        else:
            motor_acceleration = (self.ExtractMotorVelocities(index) - self.bot[index].prev_motor_vel) / self.time_step
            return np.squeeze(motor_acceleration)
        
    # affected by delay measurement if true
    def GetMotorAccelerationTMinusOne(self,index):
        if self.bot[index].delay_measure_flag:
            if len(self.bot[index].state_buffer) > 0:
                delayed_state = self.bot[index].state_buffer[0]
                motor_accel = delayed_state['motor_acceleration'].copy()
        else:
            motor_accel = self.bot[index].motor_acceleration.copy()

        # Add noise if the noise flag is set
        if self.bot[index].noise_flag:
            noise = np.random.normal(0, self.bot[index].robot_noise["joint_acc_cov"], size=motor_accel.shape)
            motor_accel += noise
        
        return motor_accel
    
    
    # affected by delay measurement if true
    def GetBasePosition(self,index=0):
        """Get the position of minitaur's base.

    Returns:
      The position of the robot's base.
    """ 
        if self.bot[index].delay_measure_flag:
            if len(self.bot[index].state_buffer) > 0:
                delayed_state = self.bot[index].state_buffer[0]
                base_pos = delayed_state['base_position'].copy()
            else:
                print("buffer is empty")
        else:
            base_pos = self.bot[index].base_position.copy()

        # Add noise if the noise flag is set
        if self.bot[index].noise_flag:
            noise = np.random.normal(0, self.bot[index].robot_noise["base_pos_cov"], size=base_pos.shape)
            base_pos += noise

        return base_pos
    
    # affected by delay measurement if true
    def GetBaseOrientation(self,index=0):
        """Get the orientation of minitaur's base, represented as quaternion.

        Returns:
        The orientation of minitaur's base.
        """
        if self.bot[index].delay_measure_flag:
           if len(self.bot[index].state_buffer) > 0:
                delayed_state = self.bot[index].state_buffer[0]
                base_ori = delayed_state['base_orientation'].copy()
           else:
                print("buffer is empty")
        else:
            base_ori = self.bot[index].base_orientation.copy()

        # Add noise if the noise flag is set
        if self.bot[index].noise_flag:
            noise_roll = np.random.normal(0, self.bot[index].robot_noise["base_ori_cov"][0],)
            noise_pitch = np.random.normal(0, self.bot[index].robot_noise["base_ori_cov"][1])
            noise_yaw = np.random.normal(0, self.bot[index].robot_noise["base_ori_cov"][2])
    
            # Convert Euler angles to quaternion
            noise_quaternion = self.pybullet_client.getQuaternionFromEuler([noise_roll, noise_pitch, noise_yaw])
            base_ori = self.pybullet_client.multiplyTransforms([0,0,0], noise_quaternion, [0,0,0], base_ori)[1]
            # normalize quaternion
            norm = np.linalg.norm(base_ori)
            if norm < 1e-8:
                raise ValueError("Cannot normalize a zero-length quaternion")
            base_ori = [component / norm for component in base_ori]
            #noise = np.random.normal(0, self.bot[index].robot_noise["base_ori_cov"], size=base_ori.shape)
            #base_ori += noise
            # here i need to ensure that the orientation is a quaternion that is unitary 


        return base_ori
        

    # the base linear velocity is filtered (world frame)
    # affected by delay measurement if true
    def GetBaseLinVelocity(self,index=0):
        """Get the linear velocity of quadruped base.

    Returns:
      The velocity of the robot's base.
    """
        if self.bot[index].delay_measure_flag:
            if len(self.bot[index].state_buffer) > 0:
                delayed_state = self.bot[index].state_buffer[0]
                base_lin_vel = delayed_state['base_lin_vel'].copy()
            else:
                print("buffer is empty")
        else:
            base_lin_vel = self.bot[index].base_lin_vel 

        # Add noise if the noise flag is set
        if self.bot[index].noise_flag:
            noise = np.random.normal(0, self.bot[index].robot_noise["base_lin_vel_cov"], size=base_lin_vel.shape)
            base_lin_vel += noise
        
        return base_lin_vel
    
    def ComputePdot(self,index=0):
        """Get the linear velocity of quadruped base."""
        #pDot = (np.asarray(self.GetBasePosition()) - np.asarray(self.bot[index].prev_base_position.copy())) / self.time_step
        pDot = (np.asarray(self.bot[index].base_position.copy()) - np.asarray(self.bot[index].prev_base_position.copy())) / self.time_step
        return pDot

        
    def ComputeBaseLinAccelerationTMinusOne(self,index=0):
        """Get the linear acceleration of quadruped base."""
        #base_acc = (np.asarray(self.GetBaseLinVelocity()) - np.asarray(self.bot.prev_base_lin_vel.copy())) / self.time_step
        base_acc = (np.asarray(self.bot[index].base_lin_vel) - np.asarray(self.bot[index].prev_base_lin_vel.copy())) / self.time_step
        return base_acc
    
    # the base linear velocity is filtered (body frame) 
    # affected by delay measurement if true
    def GetBaseLinVelocityBodyFrame(self,index=0):
        """Get the linear velocity of robot's base. in body frame
    Returns:
      The velocity of robot's base.
    """
       # velocity = np.array(self.GetBaseVelocity())
       # com_velocity_body_frame = self.TransformWorld2Body(velocity)
        if self.bot[index].delay_measure_flag:
            if len(self.bot[index].state_buffer) > 0:
                delayed_state = self.bot[index].state_buffer[0]
                base_lin_vel_body = delayed_state['base_lin_vel_body_frame'].copy()
            else:
                print("buffer is empty")
        else:
            base_lin_vel_body = self.bot[index].base_lin_vel_body_frame.copy()

        if self.bot[index].noise_flag:
            noise = np.random.normal(0, self.self.bot[index].robot_noise["base_lin_vel_cov"], size=base_lin_vel_body.shape)
            base_lin_vel_body += noise
        
        return base_lin_vel_body
    
    def ComputeBaseLinAccelerationBodyFrameTMinusOne(self,index=0):
        """Get the linear acceleration of quadruped base."""
        base_acc = (np.asarray(self.bot[index].base_lin_vel_body_frame) - np.asarray(self.bot[index].prev_base_lin_vel_base_frame.copy())) / self.time_step
        return base_acc
        
    # affected by delay measurement if true
    def GetBaseAngVelocity(self,index=0):
        """Get the linear velocity of robot's base.

    Returns:
      The velocity of robot's base.
    """
        if self.bot[index].delay_measure_flag:
            if len(self.bot[index].state_buffer) > 0:
                delayed_state = self.bot[index].state_buffer[0]
                base_ang_vel = delayed_state['base_ang_vel'].copy()
            else:
                print("buffer is empty")
        else:
            base_ang_vel = self.bot[index].base_ang_vel.copy()

        if self.bot[index].noise_flag:
            noise = np.random.normal(0, self.bot[index].robot_noise["base_ang_vel_cov"], size=base_ang_vel.shape)
            base_ang_vel += noise

        return base_ang_vel

    def ComputeBaseAngAccelerationTMinusOne(self,index=0):
        """Get the ang acceleration of quadruped base."""
        #base_ang_acc = (np.asarray(self.GetBaseAngVelocity()) - np.asarray(self.bot[index].prev_base_ang_vel.copy())) / self.time_step
        base_ang_acc = (np.asarray(self.bot[index].base_ang_vel.copy()) - np.asarray(self.bot[index].prev_base_ang_vel.copy())) / self.time_step
        return base_ang_acc

    # affected by delay measurement if true
    def GetBaseAngVelocityBodyFrame(self,index=0):

        #ang_velocity = np.array(self.GetBaseAngVelocity())
        #orientation = self.GetBaseOrientation()
        #return self.TransformAngularVelocityToLocalFrame(ang_velocity,
        #                                                 orientation)
        if self.bot[index].delay_measure_flag:
            if len(self.bot[index].state_buffer) > 0:
                delayed_state = self.bot[index].state_buffer[0]
                base_ang_vel_body = delayed_state['base_ang_vel_body_frame'].copy()
            else:
                print("buffer is empty")
        else:
            base_ang_vel_body = self.bot[index].base_ang_vel_body_frame.copy()
        
        if self.bot[index].noise_flag:
            noise = np.random.normal(0, self.bot[index].robot_noise["base_ang_vel_cov"], size=base_ang_vel_body.shape)
            base_ang_vel_body += noise

        return base_ang_vel_body
    
    def ComputeBaseAngAccelerationBodyFrameTMinusOne(self,index=0):
        """Get the ang acceleration of quadruped base."""
        
        base_ang_acc_body_frame = (np.asarray(self.bot[index].base_ang_vel_body_frame) - np.asarray(self.bot[index].prev_base_ang_vel_base_frame.copy())) / self.time_step
        return base_ang_acc_body_frame
    
    # TODO if the linear velocity in world frame corrspoend to the derivative of the position (pdot in park modern robotics page 99) we can just rotate it to get the velocity in body frame for the spatial velocity
    def ComputeBaseVelocitiesBodyFrame(self,index=0):
        """Get the velocity of robot's base."""
        base_lin_vel_world = self.bot[index].base_lin_vel
        base_ang_vel_world = self.bot[index].base_ang_vel

        base_lin_vel_body = self.TransformWorld2Body(base_lin_vel_world,index)
        base_ang_vel_body = self.TransformWorld2Body(base_ang_vel_world,index)

        #base_lin_vel_body, base_ang_vel_body = self.RotateTwistWorld2Base(base_lin_vel_world, base_ang_vel_world, self.GetBasePosition(), self.GetBaseOrientation())
        return base_lin_vel_body, base_ang_vel_body
    
    # affected by delay measurement if true
    def GetBaseVelocitiesBodyFrame(self,index=0):
        if self.bot[index].delay_measure_flag:
            if len(self.bot[index].state_buffer) > 0:
                # always get the oldest state in the buffer
                delayed_state = self.bot[index].state_buffer[0]
                base_lin_vel_body = delayed_state['base_lin_vel_body_frame'].copy() , 
                base_ang_vel_body = delayed_state['base_ang_vel_body_frame'].copy()
            else:
                print("buffer is empty")
        else:
            base_lin_vel_body = self.bot[index].base_lin_vel_body_frame.copy()
            base_ang_vel_body = self.bot[index].base_ang_vel_body_frame.copy()

        if self.bot[index].noise_flag:
            noise = np.random.normal(0, self.bot[index].robot_noise["base_lin_vel_cov"], size=base_lin_vel_body.shape)
            base_lin_vel_body += noise

            noise = np.random.normal(0, self.bot[index].robot_noise["base_ang_vel_cov"], size=base_ang_vel_body.shape)
            base_ang_vel_body += noise

        return base_lin_vel_body, base_ang_vel_body
    
    # def GetSystemStateAccelerationTMinusOne(self,base_frame=True):
    #     if(self.bot.base_type=="fixed"):
    #             return self.getMotorAccelerationTMinusOne()
    #     elif(self.bot.base_type=="floating"):
    #         if(base_frame):
    #             base_acc = self.GetBaseLinAccelerationBodyFrameTMinusOne()
    #             base_ang_acc = self.GetBaseAngAccelerationBodyFrameTMinusOne()
    #         else:
    #             base_acc = self.GetBaseLinAccelerationTMinusOne()
    #             base_ang_acc = self.GetBaseAngAccelerationTMinusOne()
            
    #             qddot = self.getMotorAccelerationTMinusOne()
    #             return np.concatenate((base_acc,base_ang_acc,qddot))
            
    def ComputeSystemStateAccelerationTMinusOne(self,base_frame=True,index=0):
        if(self.bot.base_type=="floating"):
            if(base_frame):
                base_acc = self.ComputeBaseLinAccelerationBodyFrameTMinusOne(index)
                base_ang_acc = self.ComputeBaseAngAccelerationBodyFrameTMinusOne(index)
            else:
                base_acc = self.ComputeBaseLinAccelerationTMinusOne(index)
                base_ang_acc = self.ComputeBaseAngAccelerationTMinusOne(index)
        else:
            # warning this is not correct for the fixed base
            print("it is not possible to provide base acceleration for the fixed base")
            
        #qddot = self.getMotorAccelerationTMinusOne()
        qddot = self.bot[index].motor_acceleration.copy()
        if(self.bot.base_type=="floating"):
            return np.concatenate((base_acc,base_ang_acc,qddot))
        else:
            return qddot
            

    def GetGravVecBodyFrame(self,index=0):
        com_grav_vector_body_frame = self.TransformWorld2Body(self._com_grav_vector_world_frame,index)

        return com_grav_vector_body_frame

    def ComputeBaseRollPitchYaw(self,index=0):
        """Get robot's base orientation in euler angle in the world frame.

    Returns:
      A tuple (roll, pitch, yaw) of the base in world frame.
    """
        orientation = self.bot[index].base_orientation
        roll_pitch_yaw = self.pybullet_client.getEulerFromQuaternion(orientation)
        return np.asarray(roll_pitch_yaw)

    def GetMotorTorques(self,index=0):
            """Get the amount of torque the motors are exerting.

        Returns:
        Motor torques of all eight motors.
        """
            cur_torque = self.bot[index].applied_motor_commands.copy()
            # Add noise if the noise flag is set
            if self.bot[index].noise_flag:
                noise = np.random.normal(0, self.bot[index].robot_noise["joint_torque_cov"], size=cur_torque.shape)
                cur_torque += noise
            return cur_torque


    # def GetBaseRollPitchYawRate(self):
    #     """Get the rate of orientation change of the minitaur's base in euler angle.

    # Returns:
    #   rate of (roll, pitch, yaw) change of the minitaur's base.
    # """
    #     angular_velocity = self.pybullet_client.getBaseVelocity(self.bot.bot_pybullet)[1]
    #     orientation = self.GetBaseOrientation()
    #     return self.TransformAngularVelocityToLocalFrame(angular_velocity,
    #                                                      orientation)

    # IMPORTANT only for use outside the class
    # this function return the full system state floating base + joints and their velocities 
    # TODO check directly insside bot if the robot is fixed base or floating base
    # affected by delay measurement if true (because of all the get functions used inside)
    def GetSystemState(self, fixed_base=False,base_vel_base_frame=False,index=0):
        pos_b = self.GetBasePosition(index)
        ori_b = self.GetBaseOrientation(index)

        if(base_vel_base_frame):
            # velocity in body frame
            vel_b = self.GetBaseLinVelocityBodyFrame(index)
            ang_vel_b = self.GetBaseAngVelocityBodyFrame(index)
        else:
            # velocity in world frame
            vel_b = self.GetBaseLinVelocity(index)
            ang_vel_b = self.GetBaseAngVelocity(index)
            
        q = self.GetMotorAngles(index)
        qdot = self.GetMotorVelocities(index)
        
        if(fixed_base):
            x = q.squeeze()
            xdot = qdot.squeeze()
        else: 
            if(len(q)!=0): # here i check if the robot has motors
                x = np.concatenate((np.asarray(pos_b), np.asarray(ori_b), q.squeeze()))
            else:
                x = np.concatenate((np.asarray(pos_b), np.asarray(ori_b)))
            if(len(qdot)!=0): # here i check if the robot has motors
                xdot = np.concatenate((np.asarray(vel_b), np.asarray(ang_vel_b), qdot.squeeze()))
            else:  
                xdot = np.concatenate((np.asarray(vel_b), np.asarray(ang_vel_b)))
        return x, xdot
    
    def GetSystemStateInternal(self, fixed_base=False,base_vel_base_frame=False,index=0):
        pos_b = self.bot[index].base_position.copy()
        ori_b = self.bot[index].base_orientation.copy()

        if(base_vel_base_frame):
            # velocity in body frame
            #vel_b = self.GetBaseLinVelocityBodyFrame(index)
            vel_b = self.bot[index].base_lin_vel_body_frame.copy()
            #ang_vel_b = self.GetBaseAngVelocityBodyFrame(index)
            ang_vel_b = self.bot[index].base_ang_vel_body_frame.copy()
        else:
            # velocity in world frame
            #vel_b = self.GetBaseLinVelocity(index)
            vel_b = self.bot[index].base_lin_vel.copy()
           #ang_vel_b = self.GetBaseAngVelocity(index)
            ang_vel_b = self.bot[index].base_ang_vel.copy()
            
        #q = self.GetMotorAngles(index)
        q = self.bot[index].motor_angles.copy()
        #qdot = self.GetMotorVelocities(index)
        qdot = self.bot[index].motor_velocities.copy()
        
        if(fixed_base):
            x = q.squeeze()
            xdot = qdot.squeeze()
        else: 
            if(len(q)!=0): # here i check if the robot has motors
                x = np.concatenate((np.asarray(pos_b), np.asarray(ori_b), q.squeeze()))
            else:
                x = np.concatenate((np.asarray(pos_b), np.asarray(ori_b)))
            if(len(qdot)!=0): # here i check if the robot has motors
                xdot = np.concatenate((np.asarray(vel_b), np.asarray(ang_vel_b), qdot.squeeze()))
            else:  
                xdot = np.concatenate((np.asarray(vel_b), np.asarray(ang_vel_b)))
        return x, xdot
    
    def GetSystemPreviousStateInternal(self, fixed_base=False,base_vel_base_frame=False, index=0):
        pos_b = self.bot[index].prev_base_position
        ori_b = self.bot[index].prev_base_orientation

        if(base_vel_base_frame):
            # velocity in body frame
            vel_b = self.bot[index].prev_base_lin_vel_base_frame.copy()
            ang_vel_b = self.bot[index].prev_base_ang_vel_base_frame.copy()
        else:
            # velocity in world frame
            vel_b = self.bot[index].prev_base_lin_vel.copy()
            ang_vel_b = self.bot[index].prev_base_ang_vel.copy()

        q = self.bot[index].prev_motor_angles
        qdot = self.bot[index].prev_motor_vel
        
        
        if(fixed_base):
            prev_x = q.squeeze()
            prev_xdot = qdot.squeeze()
        else: 
            if(len(q)!=0): # here i check if the robot has motors
                prev_x = np.concatenate((np.asarray(pos_b), np.asarray(ori_b), q.squeeze()))
            else:
                prev_x = np.concatenate((np.asarray(pos_b), np.asarray(ori_b)))
            if(len(qdot)!=0): # here i check if the robot has motors
                prev_xdot = np.concatenate((np.asarray(vel_b), np.asarray(ang_vel_b), qdot.squeeze()))
            else:
                prev_xdot = np.concatenate((np.asarray(vel_b), np.asarray(ang_vel_b)))
        return prev_x, prev_xdot
    
    # this function return the full system state floating base + joints and their velocities
    # without delay and noise 
    def GetAllObservationIdeal(self):
        bot_observation = []
        for j in  range(len(self.bot)):
            observation = []
            observation.extend(self.bot[j].motor_angles)
            observation.extend(self.bot[j].motor_velocities)
            observation.extend(self.GetMotorTorques(j))
            observation.extend(self.bot[j].base_position)
            observation.extend(self.bot[j].base_orientation)
            observation.extend(self.bot[j].base_lin_vel)
            observation.extend(self.bot[j].base_ang_vel)
            observation.extend(self.bot[j].base_lin_vel_body_frame)
            observation.extend(self.bot[j].base_ang_vel_body_frame)
            bot_observation.append(observation)
        return bot_observation  

    # return 
    def GetAllObservation(self):
        bot_observation = []
        for j in  range(len(self.bot)):
            observation = []
            observation.extend(self.GetMotorAngles(j))
            observation.extend(self.GetMotorVelocities(j))
            observation.extend(self.GetMotorTorques(j))
            observation.extend(self.GetBasePosition(j))
            observation.extend(self.GetBaseOrientation(j))
            observation.extend(self.GetBaseLinVelocity(j))
            observation.extend(self.GetBaseAngVelocity(j))
            observation.extend(self.GetBaseLinVelocityBodyFrame(j))
            observation.extend(self.GetBaseAngVelocityBodyFrame(j))
            bot_observation.append(observation)
        return bot_observation  


    def GetActionDimension(self, index=0):
        """Get the length of the action list.

    Returns:
      The length of the action list.
    """
        return self.bot[index].num_motors

    def GetMassLink(self,link_name, index=0):
        link_id = self.bot[index]._GetLinkIdByName(self.pybullet_client, link_name)
        return self.pybullet_client.getDynamicsInfo(self.bot[index].bot_pybullet, link_id)[0]
    
    def GetTotalMassFromUrdf(self, index=0):
        num_links = self.pybullet_client.getNumJoints(self.bot[index].bot_pybullet)
        # Get the list of masses for each link
        tot_mass = 0
        print("start")
        for link_index in range(num_links):
            link_info = self.pybullet_client.getDynamicsInfo(self.bot[index].bot_pybullet, link_index)
            print(link_info[0])
            tot_mass = tot_mass + link_info[0]
        return tot_mass  
        
    def GetInertiaLink(self,link_name, index=0):
        link_id = self.bot[index]._GetLinkIdByName(self.pybullet_client, link_name)
        return self.pybullet_client.getDynamicsInfo(self.bot[index].bot_pybullet, link_id)[2]

    def SetMassLink(self, link_name, mass, index=0):
        link_id = self.bot[index]._GetLinkIdByName(self.pybullet_client, link_name)
        self.pybullet_client.changeDynamics(self.bot[index].bot_pybullet,
                                                   link_id,
                                                   mass=mass)
    def SetDiffMassLink(self, link_name, dif_mass, index=0):
        link_id = self.bot[index]._GetLinkIdByName(self.pybullet_client, link_name)

        new_mass = self.pybullet_client.getDynamicsInfo(self.bot[index].bot_pybullet, link_id)[0]+dif_mass
        if new_mass <= 0:
            print("mass cannot be negative")
            print("current mass is=",self.pybullet_client.getDynamicsInfo(self.bot[index].bot_pybullet, link_id)[0])
        else:
            self.pybullet_client.changeDynamics(self.bot[index].bot_pybullet,
                                                   link_id,
                                                   mass=self.pybullet_client.getDynamicsInfo(self.bot[index].bot_pybullet, link_id)[0]+dif_mass)

    def SetInertiaLink(self, link_name, inertia, index=0):
        link_id = self.bot[index]._GetLinkIdByName(self.pybullet_client, link_name)
        for inertia_value in inertia:
            if (np.asarray(inertia_value) < 0).any():
                raise ValueError("Values in inertia matrix should be non-negative.")
            
        self.pybullet_client.changeDynamics(self.bot[index].bot_pybullet,
                                                   link_id,
                                                   localInertiaDiagonal=inertia)

    def GetFootLinkIDs(self):
        """Get list of IDs for all foot links."""
        return self.bot.foot_link_ids
    
    # this function returna dictionary where each element is the full GRF vector (6x1) for every feet Independtly if the feet is in contact or not (local frame)
    def getFeetGRFLocal(self):
        return self.bot.feet_grf_local
    # this function the full GRF vector (6x1) for one foot Independtly if the foot is in contact or not (local frame)
    def GetFootGRFLocal(self,foot):
        return self.bot.feet_grf_local[foot]
    
    # this function returna dictionary where each element is the full GRF vector (6x1) for every feet Independtly if the feet is in contact or not (world frame)
    def getFeetGRFWolrd(self):
        return self.bot.feet_grf_world
    # this function the full GRF vector (6x1) for one foot Independtly if the foot is in contact or not (world frame)
    def GetFootGRFWolrd(self,foot):
        return self.bot.feet_grf_world[foot]
    
    # TODO add this fucntion to 
    def ComputeFootGRF(self):
        GRF ={}
        for key, value in self.foot_link_sensors_ids.items():
            GRF[key] = self.pybullet_client.getJointState(self.bot.bot_pybullet, value)[2]
        return GRF
    

    def GetFootFriction(self,index=0):
        """Get the lateral friction coefficient of the feet."""
        for key in self.bot[index].foot_link_ids.keys():
            link_id = self.bot[index].foot_link_ids[key]
            print("foot= ", key, " lateral friction =", self.pybullet_client.getDynamicsInfo(self.bot[index].bot_pybullet, link_id)[1])


    def SetFootFriction(self, foot_friction, index=0):
        """Set the lateral friction of the feet.

    Args:
      foot_friction: The lateral friction coefficient of the foot. This value is
        shared by all four feet.
    """
        for key in self.bot[index].foot_link_ids.keys():
            link_id = self.bot[index].foot_link_ids[key]
            self.pybullet_client.changeDynamics(self.bot[index].bot_pybullet,
                                                  link_id,
                                                  lateralFriction=foot_friction)
    def SetFloorFriction(self, floor_friction):
        self.pybullet_client.changeDynamics(self.ground_body, -1, lateralFriction=1.0)


    def GetFootRestitution(self,index=0):
        """Get the coefficient of restitution at the feet."""
        
        for key in self.bot[index].foot_link_ids.keys():
            link_id = self.bot[index].foot_link_ids[key]
            print("foot= ", key, " restitution =", self.pybullet_client.getDynamicsInfo(self.bot[index].bot_pybullet, link_id)[5])

    def SetFootRestitution(self, foot_restitution, index=0):
        """Set the coefficient of restitution at the feet.

    Args:
      foot_restitution: The coefficient of restitution (bounciness) of the feet.
        This value is shared by all four feet.
    """
        for key in self.bot[index].foot_link_ids.keys():
            link_id = self.bot[index].foot_link_ids[key]
            self.pybullet_client.changeDynamics(self.bot[index].bot_pybullet,
                                                  link_id,
                                                  restitution=foot_restitution)

    def SetJointFriction(self, joint_frictions, index=0):
        for knee_joint_id, friction in zip(self.bot.foot_link_ids, joint_frictions):
            self.p.pybullet_client.setJointMotorControl2(
                bodyIndex=self.bot[index].bot_pybullet,
                jointIndex=knee_joint_id,
                controlMode=self.p.pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=friction)

    def GetNumKneeJoints(self, index=0):
        return len(self.bot[index].foot_link_ids)

    def _AddSensorNoise(self, sensor_values, noise_stdev):
        if noise_stdev <= 0:
            return sensor_values
        observation = sensor_values + np.random.normal(scale=noise_stdev,
                                                       size=sensor_values.shape)
        return observation

    def SetMotorGains(self, kp, kd, index=0):
        """Set the gains of all motors.

    # These gains are PD gains for motor positional control. kp is the
    # proportional gain and kd is the derivative gain.

    # Args:
    #   kp: proportional gain(s) of the motors.
    #   kd: derivative gain(s) of the motors.
    # """
    #     if isinstance(kp, (collections.Sequence, np.ndarray)):
    #         self.p.motor_kps = np.asarray(kp)
    #     else:
    #         self.p.motor_kps = np.full(self.p.num_motors, kp)

    #     if isinstance(kd, (collections.Sequence, np.ndarray)):
    #         self.p.motor_kds = np.asarray(kd)
    #     else:
    #         self.p.motor_kds = np.full(self.p.num_motors, kd)

    #     
        self.bot[index].servo_motor_model.set_motor_gains(kp, kd)

    def GetMotorGains(self, index=0):
        """Get the gains of the motor.

    Returns:
      The proportional gain.
      The derivative gain.
    """
        return self.bot[index].servo_motor_model.get_motor_gains()

    def SetTimeSteps(self, simulation_step):
        """Set the time steps of the control and simulation.

    Args:
      action_repeat: The number of simulation steps that the same action is
        repeated.
      simulation_step: The simulation time step.
    """
        self.time_step = simulation_step
        #self.action_repeat = action_repeat

    # def _GetMotorNames(self):
    #     return self.bot.

    def getNameActiveJoints(self, index=0):
        
        return self.bot[index].getNameActiveJoints(self.pybullet_client)
    

    def getDynamicsInfo(self, body_id, link_id=-1):
        """
        Get dynamic information about the mass, center of mass, friction and other properties of the base and links.

        Args:
            body_id (int): body/object unique id.
            link_id (int): link/joint index or -1 for the base.

        Returns:
            float: mass in kg
            float: lateral friction coefficient
            np.array[float[3]]: local inertia diagonal. Note that links and base are centered around the center of
                mass and aligned with the principal axes of inertia.
            np.array[float[3]]: position of inertial frame in local coordinates of the joint frame
            np.array[float[4]]: orientation of inertial frame in local coordinates of joint frame
            float: coefficient of restitution
            float: rolling friction coefficient orthogonal to contact normal
            float: spinning friction coefficient around contact normal
            float: damping of contact constraints. -1 if not available.
            float: stiffness of contact constraints. -1 if not available.
            int: body type 1=rigid body, 2 = multi body, 3 = soft body
            float: collision margin, internal parameters non consistent
        """
        info = list(self.pybullet_client.getDynamicsInfo(body_id, link_id))
        for i in range(2, 5):
            info[i] = np.asarray(info[i])
        return info
    
    def GetBotDynamicsInfo(self,index=0):
    
        joint_info = self.pybullet_client.getDynamicsInfo(self.bot[index].bot_pybullet, -1)
        print(joint_info)
        num_joints = self.pybullet_client.getNumJoints(self.bot[index].bot_pybullet)
        for i in range(num_joints):
            joint_info = self.pybullet_client.getDynamicsInfo(self.bot[index].bot_pybullet, i)
            print(joint_info)
        
    def GetJointInfo(self, body_id, joint_id):
        """
        Return information about the given joint about the specified body.

        Note that this method returns a lot of information, so specific methods have been implemented that return
        only the desired information. Also, note that we do not convert the data here.

        Args:
            body_id (int): unique body id.
            joint_id (int): joint id is included in [0..`num_joints(body_id)`].

        Returns:
            [0] int:        the same joint id as the input parameter
            [1] str:        name of the joint (as specified in the URDF/SDF/etc file)
            [2] int:        type of the joint which implie the number of position and velocity variables.
                            The types include JOINT_REVOLUTE (=0), JOINT_PRISMATIC (=1), JOINT_SPHERICAL (=2),
                            JOINT_PLANAR (=3), and JOINT_FIXED (=4).
            [3] int:        q index - the first position index in the positional state variables for this body
            [4] int:        dq index - the first velocity index in the velocity state variables for this body
            [5] int:        flags (reserved)
            [6] float:      the joint damping value (as specified in the URDF file)
            [7] float:      the joint friction value (as specified in the URDF file)
            [8] float:      the positional lower limit for slider and revolute joints
            [9] float:      the positional upper limit for slider and revolute joints
            [10] float:     maximum force specified in URDF. Note that this value is not automatically used.
                            You can use maxForce in 'setJointMotorControl2'.
            [11] float:     maximum velocity specified in URDF. Note that this value is not used in actual
                            motor control commands at the moment.
            [12] str:       name of the link (as specified in the URDF/SDF/etc file)
            [13] np.array[float[3]]:  joint axis in local frame (ignored for JOINT_FIXED)
            [14] np.array[float[3]]:  joint position in parent frame
            [15] np.array[float[4]]:  joint orientation in parent frame
            [16] int:       parent link index, -1 for base
        """
        info = list(self.pybullet_client.getJointInfo(body_id, joint_id))
        info[1] = info[1] if isinstance(info[1], str) else info[1].decode("utf-8")  # bytes vs str (Py2 vs Py3)
        info[12] = info[12] if isinstance(info[12], str) else info[12].decode("utf-8")
        info[13] = np.asarray(info[13])
        info[14] = np.asarray(info[14])
        info[15] = np.asarray(info[15])
        return info

    def GetBotJointsInfo(self,index=0):
        
        num_joints = self.pybullet_client.getNumJoints(self.bot[index].bot_pybullet)
        for i in range(num_joints):
            joint_info = self.pybullet_client.getJointInfo(self.bot[index].bot_pybullet, i)
            print(joint_info)

    def GetBotJointsLimit(self,index=0):
        lower_limits = []
        upper_limits = []
        num_joints = self.pybullet_client.getNumJoints(self.bot[index].bot_pybullet)
        for i in range(num_joints):
            joint_info = self.pybullet_client.getJointInfo(self.bot[index].bot_pybullet, i)
            if(joint_info[2] != 4):
                lower_limits.append(joint_info[8])
                upper_limits.append(joint_info[9])
        return lower_limits, upper_limits
    
    def GetBotJointsVelLimit(self,index=0):
        vel_limits = []
        num_joints = self.pybullet_client.getNumJoints(self.bot[index].bot_pybullet)
        for i in range(num_joints):
            joint_info = self.pybullet_client.getJointInfo(self.bot[index].bot_pybullet, i)
            if(joint_info[2] != 4):
                vel_limits.append(joint_info[11])
        return vel_limits
    
    def GetBotJointsTorqueLimit(self, index=0):
        torque_limits = []
        num_joints = self.pybullet_client.getNumJoints(self.bot[index].bot_pybullet)
        for i in range(num_joints):
            joint_info = self.pybullet_client.getJointInfo(self.bot[index].bot_pybullet, i)
            if(joint_info[2] != 4):
                torque_limits.append(joint_info[10])
        return torque_limits
    
    # this function allows to move the robot without physics (it is purely kinematic)
    def SetjointPosition(self, position, index=0):
        """Set the position of all motors."""
        for i, id in enumerate(self.bot[index].active_joint_ids):
            self.pybullet_client.resetJointState(self.bot[index].bot_pybullet,
                                                id,
                                                position[i],
                                                targetVelocity=self.bot[index].init_joint_vel[i])
    
    # here when i do that im resetting the center of mass of the floating base 
    # TODO reset the reference frame position rather than the CoM
    def SetfloatingBasePositionAndOrientation(self, position, orientation, index=0):
        """Set the position of all motors."""
        self.pybullet_client.resetBasePositionAndOrientation(self.bot[index].bot_pybullet,
                                                position,
                                                orientation)


    
    # visualizing routine for kinematic position of the robot 
    def KinematicVisualizer(self, q_res, dyn_model, visual_delays=0):
    # key to sto the simulation      
        sKey = ord('s')
        # key to get cartesian position of a link
        pKey = ord('p')
        #initialize previous_tau with the first torque command
        print("replay trajectory...")
        while True:
            counter = 0
            for q_val in q_res.T:   # q_res.T interpolated value of the joint, solution["q"] is the value of the joint at the nodes
                cur_base_pos = q_val[:3].copy().tolist()
                cur_base_ori = q_val[3:7].copy().tolist()
                cur_action = q_val[7:].copy().tolist()

                # here the hypothesis is that horizon and pinocchio are alligned and in need only to convert to pybullet order for the visualization
                cur_action = dyn_model._FromPinToExtVec(cur_action)
                
                self.SetfloatingBasePositionAndOrientation(cur_base_pos,cur_base_ori)
                self.SetjointPosition(cur_action)

                keys = self.GetPyBulletClient().getKeyboardEvents()
                if sKey in keys and keys[sKey] and self.GetPyBulletClient().KEY_WAS_TRIGGERED:
                    return 
                if pKey in keys and keys[pKey] and self.GetPyBulletClient().KEY_WAS_TRIGGERED:
                    print("insert the name of the link you want to get the position of: ")
                    reference_frame = input()
                    print("press 0 or 1 to get the world position of the com or the joint attached to the link: ")
                    selector =input()
                    if selector == "0":
                        print(self.GetLinkPositionAndOrientation(reference_frame,"com"))
                    elif selector == "1":
                        print(self.GetLinkPositionAndOrientation(reference_frame,"joint"))
                    else:
                        print("wrong input")
                    keys={}
                print("current step: ", counter)
                counter += 1
                if visual_delays > 0:
                    time.sleep(visual_delays)
                
    

    # utilities functions single robot---------------------------------------------------------------------------------------
    
    def DynamicSanityCheck1(self,pin_dynamic_model,index=0):
        """Checks if the pinocchio and pybullet dynamic models are the same."""
        if(self.bot.base_type=="fixed"):
            x,xdot = self.GetSystemStateInternal(fixed_base=True,index=index)
        else:
            x,xdot = self.GetSystemStateInternal(fixed_base=False,base_vel_base_frame=True,index=index)

        res = pin_dynamic_model.ComputeCoriolis(x, xdot)
        pin_coriolis = res.GetC()

        # # convert the xdot in the pinocchio order
        # xdot_joint = pin_dynamic_model.ExtractJointsVec(xdot,"vel")
        # # here we reorder the joint state vel to match the pinocchio model
        # xdot_joint_new = pin_dynamic_model.FromExtToPinVec(xdot_joint)
        # # here we copy the qdot_joint_new into the qdot vector to be able to compute the coriolis
        # xdot = pin_dynamic_model.CopyJointsVec(xdot,xdot_joint_new,"vel")

        # pin_coriolis = pin_coriolis_matrix @ xdot
        pin_coriolis = pin_dynamic_model.ReoderJoints2ExtVec(pin_coriolis,"vel")
        # TODO to check i should use a twist adjoint maybe not because it is only for velocities so no chance to get it comparable
        if(self.bot.base_type=="floating"):
            pin_coriolis_pos = pin_coriolis[:3,]
            pin_coriolis_pos = self.TransformBody2World(pin_coriolis_pos,index)
            pin_coriolis_ori = pin_coriolis[3:6,]
            # TODO check if using this it is gonna work for the angular acceleration contribution
            pin_coriolis_ori = self.TransformAngularVelocityToLocalFrame(pin_coriolis_ori)
            # here we reassemble the gravity vector in the world frame
            pin_coriolis_body = np.concatenate((pin_coriolis_pos,pin_coriolis_ori),axis=0)
            # we copy pin_gravity_body into pin_gravity
            pin_coriolis[:6,] = pin_coriolis_body
        
        sim_coriolis = self.ComputeCoriolis()

        res_coriolis = pin_coriolis - sim_coriolis
        error_coriolis = np.linalg.norm(res_coriolis)

        res = pin_dynamic_model.ComputeMassMatrixRNEA(x)
        pin_mass_matrix = res.GetM()
        pin_mass_matrix = pin_dynamic_model.ReoderJoints2ExMat(pin_mass_matrix,"vel")

        sim_mass_matrix = self.ComputeMassMatrix()

        # res_mass_matrix is a matrix 
        res_mass_matrix = pin_mass_matrix - sim_mass_matrix
        error_mass_matrix = np.linalg.norm(res_mass_matrix)

        res = pin_dynamic_model.ComputeGravity(x)
        pin_gravity = res.GetG()
        pin_gravity = pin_dynamic_model.ReoderJoints2ExtVec(pin_gravity,"vel")
        # TODO verify if this is correct, maybe we need to use the adjoint for the velocity? maybe not, probably not 
        if(self.bot.base_type=="floating"):
            pin_gravity_pos = self.TransformBody2World(pin_gravity_pos,index)
            pin_gravity_ori = pin_gravity[3:6,]
            # TODO check if using this it is gonna work for the angular acceleration contribution
            pin_gravity_ori = self.TransformAngularVelocityToLocalFrame(pin_gravity_ori)
            # here we reassemble the gravity vector in the world frame
            pin_gravity_body = np.concatenate((pin_gravity_pos,pin_gravity_ori),axis=0)
            # we copy pin_gravity_body into pin_gravity
            pin_gravity[:6,] = pin_gravity_body

        sim_gravity = self.ComputeGravity()

        res_gravity = pin_gravity - sim_gravity
        error_gravity = np.linalg.norm(res_gravity)

        #print the results
        #print("pin mass matrix = ", pin_mass_matrix)
        #print("sim mass matrix = ", sim_mass_matrix)
        print("error_mass_matrix=",error_mass_matrix)
        print("residual error_mass_matrix=",res_mass_matrix)
        #print("pin coriolis = ", pin_coriolis)
        #print("sim coriolis = ", sim_coriolis)
        print("error_coriolis=",error_coriolis)
        print("residual error_coriolis=",res_coriolis)
        #print("pin gravity = ", pin_gravity)
        #print("sim gravity = ", sim_gravity)
        print("error_gravity=",error_gravity)
        print("residual error_gravity=",res_gravity)


    
    def DynamicSanityCheck2(self,pin_dynamic_model,tau,index=0):
        if(self.bot.base_type=="fixed"):
            x,xdot = self.GetSystemStateInternal(fixed_base=True,index=index)
        else:
            x,xdot = self.GetSystemStateInternal(fixed_base=False,base_vel_base_frame=True,index=index)

        if(self.step_counter >1):
            # we need to that because the numerical acceleration is always computed for the previous time step
            acc_sim_numerical_t_minus_one = self.getMotorAccelerationTMinusOne()
            res_acc_vs_pin = self.acc_pin_t - acc_sim_numerical_t_minus_one
            res_acc_vs_sim = self.acc_sim_t - acc_sim_numerical_t_minus_one
            error_acc_pin = np.linalg.norm(res_acc_vs_pin)
            error_acc_sim = np.linalg.norm(res_acc_vs_sim)
            # print the results
            print("acc_pin_t at previous time step = ", self.acc_pin_t)
            print("acc_with_direct_dynamics_t at previous time step = ", self.acc_with_direct_dynamics_t)
            print("acc_sim_t at previous time step = ", self.acc_sim_t)
            print("acc_sim_numerical_t_minus_one= ", acc_sim_numerical_t_minus_one)
            print("error acc pin =",error_acc_pin)
            print("residual error acc VS pin=",res_acc_vs_pin)
            print("error acc sim =",error_acc_sim)
            print("residual error acc VS sim=",res_acc_vs_sim)
            

        # update pinocchio acceleration
        self.acc_pin_t=pin_dynamic_model.DirectDynamicsActuatedZeroTorqueNoContact(x,xdot)
        # reorder output for comparison with the simulator
        self.acc_pin_t = pin_dynamic_model._FromPinToExtVec(self.acc_pin_t)
        self.acc_with_direct_dynamics_t = pin_dynamic_model.ABA(x,xdot,tau)
        # reorder output for comparison with the simulator
        self.acc_with_direct_dynamics_t = pin_dynamic_model.ReoderJoints2ExtVec(self.acc_with_direct_dynamics_t,'vel')
        # update simulator acceleration
        self.acc_sim_t = self.DirectDynamicsActuatedNoContact(tau)
        
    # def DynamicSanityCheck3(self,pin_dynamic_model, previous_tau):
        
    #     print("------------------------------------------------------------------------------------")
    #     if(self.step_counter >1):
    #         if(self.bot.base_type=="fixed"):
    #             prev_x,prev_xdot = self.GetSystemPreviousState(fixed_base=True)
    #         else:
    #             prev_x,prev_xdot = self.GetSystemPreviousState(fixed_base=False,base_vel_base_frame=True)

    #         previous_tau_pin = pin_dynamic_model.ReoderJoints2PinVec(previous_tau,"vel")
    #         xdotdot_aba_pin = pin_dynamic_model.ABA(prev_x,prev_xdot,previous_tau_pin)
    #         xdotdot_prev_sim_measured = self.GetSystemStateAccelerationTMinusOne(base_frame=True)
    #         xdotdot_prev_sim_measured_reordered = pin_dynamic_model.ReoderJoints2PinVec(xdotdot_prev_sim_measured,"vel")
    #         # print the results
    #         xdotdot_prev_sim_computed = self.DirectDynamicsActuatedNoContact(previous_tau,previous_state=True)
    #         xdotdot_prev_sim_computed = pin_dynamic_model.ReoderJoints2PinVec(xdotdot_prev_sim_computed,"vel")
    #         print("acceleration pinocchio aba = ", xdotdot_aba_pin)
    #         print("acceleration previous step base frame sim computed=",xdotdot_prev_sim_computed)
    #         print("acceleration previous step base frame sim measured=",xdotdot_prev_sim_measured_reordered)
    #         pin_torques = pin_dynamic_model.InverseDynamicsActuatedPartNoContact(prev_x,prev_xdot,xdotdot_prev_sim_measured)
    #         torques_res = pin_torques - previous_tau_pin
    #         error_torques = np.linalg.norm(torques_res)
    #         print("pin torques t-1 = ", pin_torques)
    #         print("commanded torques t-1 = ", previous_tau_pin)
    #         print("error torques t-1 = ", error_torques)
    #         print("residual torques t-1 = ", torques_res)


    def DynamicSanityCheck3(self,pin_dynamic_model, previous_tau, index=0):
        
        if(self.step_counter >1):
            if(self.bot.base_type=="fixed"):
                prev_x,prev_xdot = self.GetSystemPreviousStateInternal(fixed_base=True)
                cur_pos,cur_vel = self.GetSystemStateInternal(fixed_base=True,base_vel_base_frame=True,index=index)
            else:
                prev_x,prev_xdot = self.GetSystemPreviousStateInternal(fixed_base=False,base_vel_base_frame=True)
                _,prev_xdot_world = self.GetSystemPreviousStateInternal(fixed_base=False,base_vel_base_frame=False)
                cur_pos,cur_vel = self.GetSystemStateInternal(fixed_base=False,base_vel_base_frame=True,index=index)

            print("------------------------------------------------------------------------------------")
            
           
            
            # inside the function perfom the reordering of the joints to the pinocchio model
            xdotdot_aba_pin = pin_dynamic_model.ABA(cur_pos,cur_vel,previous_tau)
            xdotdot_prev_base = self.ComputeSystemStateAccelerationTMinusOne(base_frame=True,index=index)
            xdotdot_prev_base_reordered = pin_dynamic_model.ReoderJoints2PinVec(xdotdot_prev_base,"vel")
            xdotdot_prev_world = self.ComputeSystemStateAccelerationTMinusOne(base_frame=False,index=index)
            xdotdot_prev_world_reordered = pin_dynamic_model.ReoderJoints2PinVec(xdotdot_prev_world,"vel")

            print("acceleration pinocchio aba base frame                = ", xdotdot_aba_pin)
            print("acceleration previous step base frame  pybullet (reord) = ", xdotdot_prev_base_reordered)
            print("acceleration previous step world frame pybullet (reord) = ", xdotdot_prev_world_reordered)
            # i need to change the order of the acceleration to match the one of mujoco because the reordering of each input is done inside the function InverseDynamicsActuatedPartNoContact
            xdotdot_aba_pin_external_order = pin_dynamic_model.ReoderJoints2ExtVec(xdotdot_aba_pin,"vel")
            pin_torques = pin_dynamic_model.InverseDynamicsActuatedPartNoContact(cur_pos,cur_vel,xdotdot_aba_pin_external_order)
            pin_torques_full_rnea = pin_dynamic_model.FullInverseDynamicsNoContact(cur_pos, cur_vel, xdotdot_aba_pin_external_order)
            print("pin torques from rnea (full inverse dynamics)", pin_torques_full_rnea)
            torques_res = pin_torques - previous_tau
            error_torques = np.linalg.norm(torques_res)
            #print("pin torques t-1 = ", pin_torques)
            #print("commanded torques t-1 = ", previous_tau)
            print("error torques t-1 = ", error_torques)
            print("residual torques t-1 = ", torques_res)

    def KinematicSanityCheck(self,index=0):
        # here i compute the velocity of the system to see if the linear velocity needs to be read as the velocity of the entire robot applied in its center of mass (in world frame) $\dot{p}$
        # or if it is the velocity of the base frame expressed in the world frame for the spatial velocity case $v_w = \dot{p} - \omega \times (p)$
        
        linear_velocity_world_frame = self.bot[index].base_lin_vel.copy()
        print("------------------------------------------------------------------------------")
        print("body_lin_velocity_world_frame=",linear_velocity_world_frame)
        if(self.step_counter >1):                                     #self.GetBasePosition()
            v_s = self.ComputePdot(index) - np.cross(self.bot[index].base_ang_vel,self.bot[index].base_position)
            print("Pdot=",self.ComputePdot(index))
            # print velocity world frame (v_s) (park modern robotics page 99)
            print("body_lin_velocity_world_frame_v_s=",v_s)
            
         
        linear_velocity_body_frame_with_quat = self.bot[index].base_lin_vel_body_frame.copy() 
        
        ang_velocity_world_frame = self.bot[index].base_ang_vel.copy()

        
        # # trunk lin vel
        # print("trunk_lin_vel=",trunk_lin_vel)
        # # trunk ang vel
        # print("trunk_ang_vel=",trunk_ang_vel)
        # # print ang velocity world frame
        # print("trunk_ang_vel_body_local_frame=",trunk_ang_vel_local_frame)
        # print("trunk_lin_vel_body_local_frame=",trunk_lin_vel_local_frame)
        print("ang_velocity_world_frame=",ang_velocity_world_frame)
        # print velocity body frame with quat
        print("body_lin_velocity_local_frame=",linear_velocity_body_frame_with_quat)
        # print velocity world frame


    def calc_inverse_kinematics(self, bot_index: int, target_position, euler_angles_radians: Optional = None,
                     target_frame: Optional[str] = None) -> Any:
        """calculate what angles the robot joints should have to reach the targets. Targets in world coordinates. Takes angles in radians.

        Args:
            bot_index: int, what internal robot index to use to do calcs
            target_position (npt.NDArray[float]): x, y, z
            euler_angles_radians (Optional[npt.NDArray[float]], optional): rx, ry, rz, or None. Easier to solve if None. Defaults to None.
            target_frame (Optional[str], optional): what frame to use for the calcs. Defaults to None.

        Returns:
            npt.NDArray[float]: the joint angles
        """

        ori_des_quat = None
        if euler_angles_radians is not None:
            ori_des_quat = pybullet.getQuaternionFromEuler(euler_angles_radians)
        if target_frame is None:
            target_frame = "panda_link8"

        link_id = self.bot[bot_index].get_link_id_from_name(target_frame)
        lower_limits, upper_limits = self.GetBotJointsLimit(bot_index)
        pybullet_robot_index = self.bot[bot_index].get_pybullet_bot_index()

        joint_poses_pybullet = self.pybullet_client.calculateInverseKinematics(pybullet_robot_index,
                                                                               link_id,
                                                                               target_position,
                                                                               ori_des_quat,
                                                                               lowerLimits=lower_limits,
                                                                               upperLimits=upper_limits)
        joint_poses_pybullet = np.array(joint_poses_pybullet)
        return joint_poses_pybullet

    def get_pose(self, bot_index: int, target_frame: str = "panda_link8") -> Tuple[Any, Any]:
        """Helper function to get the current pose of the robot from the current angles.

        Args:
            bot_index (int): What is the index of the robot to calculate?
            target_frame (str, optional): What frame should the pose be in, from frames in the URDF. Defaults to "gripper".. Defaults to "gripper".

        Returns:
            Tuple[npt.NDArray[float], npt.NDArray[float]]: position, orientation where orientation is euler angles in radians
        """

        link_id = self.bot[bot_index].get_link_id_from_name(target_frame)
        pybullet_robot_index = self.bot[bot_index].get_pybullet_bot_index()

        link_state = self.pybullet_client.getLinkState(
            pybullet_robot_index, link_id, computeForwardKinematics=True)

        linkWorldPosition = link_state[0]
        linkWorldOrientation = link_state[1]

        link_world_orientation = pybullet.getEulerFromQuaternion(
            linkWorldOrientation)
        link_world_orientation = np.array(link_world_orientation)
        return linkWorldPosition, link_world_orientation