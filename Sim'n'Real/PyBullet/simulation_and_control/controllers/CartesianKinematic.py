
import numpy as np
import pinocchio as pin


def applyJointVelSaturation(qd_des,joint_vel_saturation):
    """
    Apply joint velocity saturation to the desired joint velocity.
    
    Parameters:
        qd_des (np.array): The desired joint velocity.
        joint_vel_saturation (float): The joint velocity saturation value.
    
    Returns:
        np.array: The desired joint velocity with saturation applied.
    """
    # Apply joint velocity saturation
    qd_des_clipped = np.clip(qd_des, -joint_vel_saturation, joint_vel_saturation)
    # Check if clipping has occurred
    clip_happened = not np.array_equal(qd_des, qd_des_clipped)
    if clip_happened:
        print("Joint velocity saturation occurred! check the desired joint velocity")
    return qd_des_clipped

# dead zone function
def apply_dead_zone(velocity,thresh):
  """
  Apply a dead zone to the velocity vector.
  
  Parameters:
      velocity (np.array): The velocity vector.
      threshold (float): The threshold below which velocities are set to zero.
      
  Returns:
      np.array: The velocity vector with the dead zone applied.
  """
  # Apply dead zone to each component of the velocity
  dead_zone_velocity = np.where(np.abs(velocity) < thresh, 0, velocity)
  return dead_zone_velocity


def CartesianDiffKin(dyn_model,controlled_frame_name,cur_q, p_des, pd_des,ori_des, ori_vel_des, delta_t, ori_pos_both,  kp_pos, kp_ori, joint_vel_saturation):
    # here i compute the jacobian and the jacobian time derivative
    
    dead_zone_thresh_joints = 0.01
    
    P_pos = np.eye(3) * kp_pos
    P_ori = np.eye(3) * kp_ori
    
    # this is the vertical stack of P_pos and P_ori
    P_tot = np.vstack((P_pos,P_ori))
    
    
    cur_cartesian_pos,Cur_R = dyn_model.ComputeFK(cur_q, controlled_frame_name)
    # DEBUG (print)
    #print("cur_cartesian_pos = ", cur_cartesian_pos)

    dyn_model.ComputeJacobian(cur_q,controlled_frame_name,"local_global")
    # DEBUG (print)
    #print("cur_jacobian",self.dyn_model.res.J)

    # Since we are only interested in linear velocity, we select the first three rows of the Jacobian
    J_linear = dyn_model.res.J[:3, :]
    J_angular = dyn_model.res.J[3:, :]

    # here i check if the desired orientation (ori_des) 
    if ori_des is None:
        # i set the angle error to zero
        angle_error_base_frame = np.zeros(3)
    else:
        # compute ori des in quaternion
        #Cur_R =pin.rpy.rpyToMatrix(cur_cartesian_ori)
        cur_quat = pin.Quaternion(Cur_R)
        cur_quat = cur_quat.normalize()
        ori_des_quat = pin.Quaternion(ori_des)
        ori_des_quat = ori_des_quat.normalize()

        # Ensure quaternion is in the same hemisphere as the desired orientation
        cur_quat_coeff = cur_quat.coeffs()
        ori_des_quat_coeff = ori_des_quat.coeffs()
        if np.dot(cur_quat_coeff, ori_des_quat_coeff) < 0.0:
            cur_quat_coeff = cur_quat_coeff * -1.0
            cur_quat = pin.Quaternion(cur_quat_coeff)

        # Compute the "difference" quaternion (assuming orientation_d is also a pin.Quaternion object)
        angle_error_quat = cur_quat.inverse() * ori_des_quat
        # extract coefficient x y z from the quaternion
        angle_error = angle_error_quat.coeffs()
        angle_error = angle_error[:3]
        
        # rotate the angle error in the base frame
        angle_error_base_frame = Cur_R@angle_error


    # computing position error
    pos_error = p_des - cur_cartesian_pos

    if(ori_pos_both=="pos"):
      cur_J = J_linear
      cur_P = P_pos  
      cur_error = pos_error 
      cur_d_des = pd_des   
    if (ori_pos_both=="ori"):
      cur_J = J_angular
      cur_P = P_ori
      cur_error = angle_error_base_frame
      cur_d_des = np.zeros(3)
    if (ori_pos_both=="both"):
      cur_J = dyn_model.res.J
      cur_P = P_tot
      cur_error = np.concatenate((pos_error,angle_error_base_frame),axis=0)
      cur_d_des = np.concatenate((pd_des,np.zeros(3)),axis=0)      

    # here i compute the desired joint velocity
    qd_des= np.linalg.pinv(cur_J) @ (cur_P@(cur_error) + cur_d_des)

    qd_des_dead_zone = apply_dead_zone(qd_des,dead_zone_thresh_joints)
    
    qd_des_clip = applyJointVelSaturation(qd_des_dead_zone,joint_vel_saturation)

    # DEBUG (zeoring any reference to make it not moving)
    #qd_des_clip = np.zeros(7)

    # here i compute the desired joint position
    q_des = dyn_model.KinematicIntegration(cur_q, qd_des_clip, delta_t)

    # DEBUG (print)
    #print("-------------------------------------------------------------------------")
    # Calculate the determinant
    det_J = np.linalg.det(cur_J@cur_J.T)

    # Check if the matrix is singular
    #if np.isclose(det_J, 0):
    #    print("The Jacobian is singular.")
    #else:
    #   print("The Jacobian is not singular.")
    #print("cur_q",cur_q)
    #print("cur_cartesian_pos",cur_cartesian_pos)
    #print("des_cartesian_pos",p_des)
    #print("des_ori",ori_des)
    #print("cur_cartesian_ori",Cur_R)
    #print("error=",cur_error)
    #print("cur_d_des=",cur_d_des)
    #print("current desired joint velocity=",qd_des)
    #print("current desired joint velocity with deadzone=",qd_des_dead_zone)
    
    
    return q_des, qd_des_clip