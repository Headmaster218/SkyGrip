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

"""Servo Motor model."""
import os
import inspect


import collections
import numpy as np
from ..utils import adjust_value 


#class MotorCommands(object):
#    def __init__(self, ctrl_value=np.array([]), control_list=np.array([])):
#      self.ctrl_cmd = ctrl_value
#      self.control_list = control_list

class MotorCommands(object):
    def __init__(self, ctrl_value=np.array([]), control_list=[]):
        self.control_list = control_list
        
        # Determine the desired length for ctrl_cmd
        desired_length = len(self.control_list)
        
        # Check if ctrl_value has only one element
        if np.isscalar(ctrl_value) or (isinstance(ctrl_value, np.ndarray) and ctrl_value.size == 1):
            # Extend ctrl_value to match the desired length
            self.ctrl_cmd = np.full(desired_length, ctrl_value)
        else:
            # Ensure ctrl_value has the correct length
            if len(ctrl_value) != desired_length:
                raise ValueError(f"ctrl_value must have length {desired_length}, got {len(ctrl_value)}")
            self.ctrl_cmd = ctrl_value      

    def SetControlCmd(self,ctrl_value,control_list):
        self.control_list = control_list
        
        # Determine the desired length for ctrl_cmd
        desired_length = len(self.control_list)
        
        # Check if ctrl_value has only one element
        if np.isscalar(ctrl_value) or (isinstance(ctrl_value, np.ndarray) and ctrl_value.size == 1):
            # Extend ctrl_value to match the desired length
            self.ctrl_cmd = np.full(desired_length, ctrl_value)
        else:
            # Ensure ctrl_value has the correct length
            if len(ctrl_value) != desired_length:
                raise ValueError(f"ctrl_value must have length {desired_length}, got {len(ctrl_value)}")
            self.ctrl_cmd = ctrl_value

# create an abstract class for motor_model
class ServoMotorModel(object):
    """A simple motor model for Laikago.

    When in POSITION mode, the torque is calculated according to the difference
    between current and desired joint angle, as well as the joint velocity.
    For more information about PD control, please refer to:
    https://en.wikipedia.org/wiki/PID_controller.

    The model supports a HYBRID mode in which each motor command can be a tuple
    (desired_motor_angle, position_gain, desired_motor_velocity, velocity_gain,
    torque).

  """

    def __init__(self,
             n_motors,
             kp=60,
             kd=1,
             motor_control_mod="position",
             torque_limits=None,
             friction_torque=False,
             friction_coefficient=0.0,  # Added missing comma
             elastic_torque=False,
             elastic_coefficient=0.0,  # Added missing comma
             motor_load=False,
             motor_load_coefficient=0.0):
    
        self.n_motors = n_motors
        self._kp = kp
        self._kd = kd
        self._torque_limits = torque_limits
        self.friction_torque = friction_torque
        self.friction_coeff = adjust_value(friction_torque, friction_coefficient, n_motors, "friction_coefficient")
        #self.friction_coeff = friction_coefficient
        self.elastic_torque = elastic_torque
        #self.elastic_coefficient = elastic_coefficient
        self.elastic_coefficient = adjust_value(elastic_torque, elastic_coefficient, n_motors, "elastic_coefficient")
        self.motor_load = motor_load
        #self.motor_load_coefficient = motor_load_coefficient
        self.motor_load_coefficient = adjust_value(motor_load, motor_load_coefficient, n_motors, "motor_load_coefficient")
        
        
        # Handling torque limits initialization
        if torque_limits is not None:
            if isinstance(torque_limits, (collections.Sequence, np.ndarray)):
                self._torque_limits = np.asarray(torque_limits)
            else:
                self._torque_limits = np.full(self.n_motors, torque_limits)
        
        self._motor_control_mode = motor_control_mod
        self._strength_ratios = np.full(self.n_motors, 1)  # Strength ratio set to

    def set_strength_ratios(self, ratios):
        """Set the strength of each motors relative to the default value.

    Args:
      ratios: The relative strength of motor output. A numpy array ranging from
        0.0 to 1.0.
    """
        self._strength_ratios = ratios

    def set_motor_gains(self, kp, kd):
        """Set the gains of all motors.

    These gains are PD gains for motor positional control. kp is the
    proportional gain and kd is the derivative gain.

    Args:
      kp: proportional gain of the motors.
      kd: derivative gain of the motors.
    """
        self._kp = kp
        self._kd = kd

    def get_motor_gains(self):
        """Set the gains of all motors.

    These gains are PD gains for motor positional control. kp is the
    proportional gain and kd is the derivative gain.

    Args:
      kp: proportional gain of the motors.
      kd: derivative gain of the motors.
    """
        return self._kp, self._kd

    # def set_voltage(self, voltage):
    #   pass
    #
    # def get_voltage(self):
    #   return 0.0
    #
    # def set_viscous_damping(self, viscous_damping):
    #   pass
    #
    # def get_viscous_dampling(self):
    #   return 0.0

    def compute_torque(self,
                          motor_commands,
                          cur_q,
                          cur_qdot,
                          prev_qdotdot,
                          M):
        """Convert the commands (position control or torque control) to torque.

    Args:
      motor_commands: The desired motor angle if the motor is in position
        control mode. The pwm signal if the motor is in torque control mode.
      cur_q: The motor angle observed at the current time step. It is
        actually the true motor angle observed a few milliseconds ago (pd
        latency).
      cur_qdot: The motor velocity observed at the current time step, it
        is actually the true motor velocity a few milliseconds ago (pd latency).
      true_motor_velocity: The true motor velocity. The true velocity is used to
        compute back EMF voltage and viscous damping.
      motor_control_mode: A MotorControlMode enum.

    Returns:
      motor_torques: The torque that needs to be applied to the motor.
    """
        if not isinstance(motor_commands,MotorCommands):
          print(" the motor command has to be an instance of the class MotorCommands")
          exit()

        # NOW control interface in control commands
        #if not motor_control_mode:
        #    motor_control_mode = self._motor_control_mode

        additional_torques = np.full(self.n_motors, 0.0)
        
        # todo adding other friction model as stribeck and dry friction
        if self.friction_torque:
            # build vector of torque sign and compute friction torque
            #torque_sign = np.sign(motor_commands.tau_cmd.squeeze())
            #additional_torques = self.friction_coeff * (np.abs(cur_qdot) * torque_sign)
            # element wise multiplication
            additional_torques = - self.friction_coeff * cur_qdot
       
        if self.elastic_torque:
            # element wise multiplication
            additional_torques += - self.elastic_coefficient * cur_q
        
        # computing additional torques by using the M matrix
        additional_torques = M @ additional_torques


        # this should go with the acceleration of the motor but we will use the previous acceleration as an estimation for the current one
        #if self.motor_load:
        #    additional_torques += - self.motor_load_coefficient * prev_qdotdot

        # OLD CODE    
        # # No processing for motor torques
        # if motor_control_mode == "torque":
        #     assert len(motor_commands.tau_cmd.squeeze()) == self.n_motors
        #     motor_torques = self._strength_ratios * motor_commands.tau_cmd + M @ additional_torques
        #     return motor_torques

        # desired_motor_angles = np.full(self.n_motors, 0)
        # desired_motor_velocities = np.full(self.n_motors, 0)
        # kp = None
        # kd = None
        
        # if motor_control_mode == "position":
        #     assert len(motor_commands.pos_cmd.squeeze()) == self.n_motors
        #     kp = self._kp
        #     kd = self._kd
        #     if(len(motor_commands.pos_cmd)):
        #       desired_motor_angles = motor_commands.pos_cmd.squeeze()
        #     if (len(motor_commands.vel_cmd.squeeze())):
        #       desired_motor_velocities = motor_commands.vel_cmd
        # else:
        #     print("Undefined motor_control_mode=", motor_control_mode)
        #     exit()
        # motor_torques = -1 * (kp * (cur_q - desired_motor_angles)) - kd * (
        #         cur_qdot - desired_motor_velocities)
        # motor_torques = (self._strength_ratios * motor_torques).squeeze() + additional_torques
        # if self._torque_limits is not None:
        #     if len(self._torque_limits) != len(motor_torques):
        #         raise ValueError(
        #             "Torque limits dimension does not match the number of motors.")
        #     motor_torques = np.clip(motor_torques, -1.0 * self._torque_limits,
        #                             self._torque_limits)

        # return motor_torques
        # OLD CODE END

        # Loop over each motor
        motor_torques = np.zeros(self.n_motors)
        for i in range(self.n_motors):
            mode = motor_commands.control_list[i]
            
            if mode == "torque":
                # Ensure that tau_cmd is available and has the correct length
               
                # Compute motor torque directly
                motor_torque = self._strength_ratios[i] * motor_commands.ctrl_cmd[i] + additional_torques[i] 
                motor_torques[i] = motor_torque
                
            elif mode == "position":
                # Retrieve desired angle and velocity for the motor
                desired_motor_angle = motor_commands.ctrl_cmd[i, 0]
                desired_motor_velocity = motor_commands.ctrl_cmd[i,1]
                
                # Retrieve kp and kd (can be scalar or array)
                kp = self._kp[i] if hasattr(self._kp, '__iter__') else self._kp
                kd = self._kd[i] if hasattr(self._kd, '__iter__') else self._kd
                
                # Compute motor torque using PD control
                motor_torque = - (kp * (cur_q[i] - desired_motor_angle)) - kd * (cur_qdot[i] - desired_motor_velocity)
                motor_torque = self._strength_ratios[i] * motor_torque + additional_torques[i]
                motor_torques[i] = motor_torque
                
            elif mode == "velocity":
                # Retrieve desired angle and velocity for the motor
                desired_motor_velocity = motor_commands.ctrl_cmd[i]
                
                # Retrieve kp and kd (can be scalar or array)
                kd = self._kd[i] if hasattr(self._kd, '__iter__') else self._kd
                
                # Compute motor torque using PD control
                motor_torque =  - kd * (cur_qdot[i] - desired_motor_velocity)
                motor_torque = self._strength_ratios[i] * motor_torque + additional_torques[i]
                motor_torques[i] = motor_torque
                
            else:
                raise ValueError(f"Undefined motor_control_mode for motor {i}: {mode}")

        # Apply torque limits if necessary
        if self._torque_limits is not None:
            if len(self._torque_limits) != len(motor_torques):
                raise ValueError("Torque limits dimension does not match the number of motors.")
            motor_torques = np.clip(motor_torques, -1.0 * self._torque_limits, self._torque_limits)

        return motor_torques