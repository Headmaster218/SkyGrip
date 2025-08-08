import numpy as np

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def velocity_to_wheel_angular_velocity(desired_linear_velocity,desired_angular_velocity, wheel_base_width, wheel_radius, number_of_wheels=4):
    # Compute wheel velocities
    v_left = desired_linear_velocity - (desired_angular_velocity * wheel_base_width / 2)
    v_right = desired_linear_velocity + (desired_angular_velocity * wheel_base_width / 2)

    # Convert to wheel angular velocities
    left_wheel_velocity = v_left / wheel_radius
    right_wheel_velocity = v_right / wheel_radius

    if number_of_wheels == 2:
        print("2 wheels: returning the velocity for each wheel")
        return left_wheel_velocity, right_wheel_velocity
    elif number_of_wheels == 4:
        print("4 wheels: returning half the velocity for each wheel")
        return left_wheel_velocity/2, right_wheel_velocity/2
    else:
        # warning message
        print("number of wheels different from 2 or 4, adjsut the wheel velocity accordingly")
        return left_wheel_velocity, right_wheel_velocity

def differential_drive_regulation_controller(
    current_position,
    current_orientation,
    desired_position,
    desired_orientation,
    wheel_radius,
    wheel_base_width,
    kp_pos, 
    kp_ori,
    number_of_wheels=4,
    max_linear_velocity=100,
    max_angular_velocity=100
):
    """
    Computes the wheel velocities to regulate the robot to the desired position and orientation.

    Parameters:
    - current_position: tuple (x, y)
        The current position of the robot.
    - current_orientation: float
        The current orientation (theta) of the robot in radians.
    - desired_position: tuple (x_d, y_d)
        The desired position for the robot.
    - desired_orientation: float
        The desired orientation (theta_d) of the robot in radians.
    - wheel_radius: float
        The radius of the wheels.
    - wheel_base_width: float
        The distance between the two wheels.
    - max_linear_velocity: float
        The maximum linear speed of the robot.
    - max_angular_velocity: float
        The maximum angular speed of the robot.

    Returns:
    - left_wheel_velocity: float
        The angular velocity for the left wheel (rad/s).
    - right_wheel_velocity: float
        The angular velocity for the right wheel (rad/s).
    """
    # Compute position errors
    error_x = desired_position[0] - current_position[0]
    error_y = desired_position[1] - current_position[1]
    distance_error = np.hypot(error_x, error_y)

    # Compute the angle to the desired position
    angle_to_goal = np.arctan2(error_y, error_x)

    # Compute orientation errors
    orientation_error = desired_orientation - current_orientation
    orientation_error = np.arctan2(np.sin(orientation_error), np.cos(orientation_error))  # Normalize to [-pi, pi]

    # Compute heading error
    heading_error = angle_to_goal - current_orientation
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    # Controller gains (tunable parameters)
    Kp_linear = kp_pos    # Proportional gain for linear velocity
    Kp_angular = kp_ori  # Proportional gain for angular velocity

    # Compute the desired linear and angular velocities
    # Adjust linear velocity based on heading error to slow down when not facing the goal
    adjusted_linear_velocity = Kp_linear * distance_error * np.cos(heading_error)
    adjusted_angular_velocity = Kp_angular * heading_error + Kp_angular * 0.5 * orientation_error

    # Limit velocities to maximum values
    desired_linear_velocity = np.clip(adjusted_linear_velocity, -max_linear_velocity, max_linear_velocity)
    desired_angular_velocity = np.clip(adjusted_angular_velocity, -max_angular_velocity, max_angular_velocity)

    # Compute wheel velocities
    left_wheel_velocity, right_wheel_velocity = velocity_to_wheel_angular_velocity(desired_linear_velocity,desired_angular_velocity, wheel_base_width, wheel_radius, number_of_wheels)

    return left_wheel_velocity, right_wheel_velocity


import numpy as np

def differential_drive_controller_adjusting_bearing(
    current_position,
    current_orientation,
    desired_position,
    desired_orientation,
    wheel_radius,
    wheel_base_width,
    kp_pos,
    kp_ori,
    number_of_wheels=4,
    position_tolerance=0.05,
    orientation_tolerance=0.05,
    max_linear_velocity=100.0,
    max_angular_velocity=100.0
):
    """
    Computes the wheel velocities to navigate the robot to the desired position and then adjust orientation.

    Parameters:
    - current_position: tuple (x, y)
        The current position of the robot.
    - current_orientation: float
        The current orientation (theta) of the robot in radians.
    - desired_position: tuple (x_d, y_d)
        The desired position for the robot.
    - desired_orientation: float
        The desired orientation (theta_d) of the robot in radians.
    - wheel_radius: float
        The radius of the wheels.
    - wheel_base_width: float
        The distance between the two wheels.
    - kp_pos: float
        Proportional gain for position control.
    - kp_ori: float
        Proportional gain for orientation control.
    - position_tolerance: float
        Distance threshold to consider the position reached.
    - orientation_tolerance: float
        Angle threshold to consider the orientation achieved.
    - max_linear_velocity: float
        Maximum linear speed of the robot.
    - max_angular_velocity: float
        Maximum angular speed of the robot.

    Returns:
    - left_wheel_velocity: float
    - right_wheel_velocity: float
    - at_goal: bool
        Indicates whether the robot has reached the goal position and orientation.
    """
    # Compute position error
    error_x = desired_position[0] - current_position[0]
    error_y = desired_position[1] - current_position[1]
    distance_error = np.hypot(error_x, error_y)

    # Compute angle to goal
    angle_to_goal = np.arctan2(error_y, error_x)

    # Compute heading error
    heading_error = angle_to_goal - current_orientation
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    # Initialize control actions
    desired_linear_velocity = 0.0
    desired_angular_velocity = 0.0

    # Check if the robot is close enough to the goal position
    if distance_error > position_tolerance:
        # Position control phase
        # Adjust linear velocity based on heading error
        desired_linear_velocity = kp_pos * distance_error
        desired_linear_velocity = np.clip(desired_linear_velocity, -max_linear_velocity, max_linear_velocity)

        # Adjust angular velocity to minimize heading error
        desired_angular_velocity = kp_ori * heading_error
        desired_angular_velocity = np.clip(desired_angular_velocity, -max_angular_velocity, max_angular_velocity)
    else:
        # Orientation adjustment phase
        # Compute orientation error
        orientation_error = desired_orientation - current_orientation
        orientation_error = np.arctan2(np.sin(orientation_error), np.cos(orientation_error))

        # If orientation error is significant, adjust angular velocity
        if abs(orientation_error) > orientation_tolerance:
            desired_linear_velocity = 0.0  # Stop moving forward
            desired_angular_velocity = kp_ori * orientation_error
            desired_angular_velocity = np.clip(desired_angular_velocity, -max_angular_velocity, max_angular_velocity)
        else:
            # Goal reached
            at_goal = True
            return 0.0, 0.0, at_goal

    # Compute wheel velocities
    left_wheel_velocity, right_wheel_velocity = velocity_to_wheel_angular_velocity(desired_linear_velocity,desired_angular_velocity, wheel_base_width, wheel_radius, number_of_wheels)

    at_goal = False
    return left_wheel_velocity, right_wheel_velocity, at_goal




def regulation_polar_coordinates(x, y, theta, xg, yg, thetag,wheel_radius,
    wheel_base_width, k_rho, k_alpha, k_beta,number_of_wheels=4):
    # Calculate the position of the goal in robot's local frame
    dx = xg - x
    dy = yg - y
    rho = np.sqrt(dx**2 + dy**2)
    alpha = wrap_angle(-theta + np.arctan2(dy, dx))

    thetag_local = wrap_angle(thetag - theta)

    #beta = -theta - alpha + thetag
    beta = wrap_angle(thetag_local - alpha)

    # Control laws for linear and angular velocities
    desired_linear_velocity = k_rho * rho
    desired_angular_velocity = k_alpha * alpha + k_beta * beta
     # Compute wheel velocities
    v_left = desired_linear_velocity - (desired_angular_velocity * wheel_base_width / 2)
    v_right = desired_linear_velocity + (desired_angular_velocity * wheel_base_width / 2)

    # Convert linear velocities to angular velocities (rad/s)
    left_wheel_velocity = v_left / wheel_radius
    right_wheel_velocity = v_right / wheel_radius

    return left_wheel_velocity, right_wheel_velocity

def euler_to_quaternion(theta):
    # Convert a 2D Euler angle to a quaternion
    w = np.cos(theta / 2)
    z = np.sin(theta / 2)
    return np.array([w, 0, 0, z])  # [w, x, y, z]

def quaternion_conjugate(q):
    # Compute the conjugate of a quaternion
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    # Quaternion multiplication (Hamilton product)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quaternion_to_euler(q):
    # Convert a quaternion back to a 2D Euler angle
    w, x, y, z = q
    return 2 * np.arctan2(z, w)

def regulation_polar_coordinate_quat(x, y, theta, xg, yg, thetag,wheel_radius,
                                    wheel_base_width, k_rho, k_alpha, k_beta,number_of_wheels=4):
                                            
    # Compute position difference
    dx = xg - x
    dy = yg - y

    # Compute rho (distance to goal)
    rho = np.hypot(dx, dy)

    # Compute the angle to the goal
    angle_to_goal = np.arctan2(dy, dx)

    # Convert orientations to quaternions
    q_robot = euler_to_quaternion(theta)
    q_goal_direction = euler_to_quaternion(angle_to_goal)
    q_goal_orientation = euler_to_quaternion(thetag)

    # Compute alpha (relative angle to goal direction)
    q_alpha = quaternion_multiply(quaternion_conjugate(q_robot), q_goal_direction)
    alpha = quaternion_to_euler(q_alpha)
    alpha = wrap_angle(alpha)

    # Compute beta (relative orientation error)
    q_beta = quaternion_multiply(quaternion_conjugate(q_robot), q_goal_orientation)
    beta = quaternion_to_euler(q_beta)
    beta = wrap_angle(beta - alpha)

    # Control laws
    desired_linear_velocity = k_rho * rho
    desired_angular_velocity = k_alpha * alpha + k_beta * beta

    # Compute wheel velocities
    left_wheel_velocity, right_wheel_velocity = velocity_to_wheel_angular_velocity(desired_linear_velocity,desired_angular_velocity, wheel_base_width, wheel_radius, number_of_wheels)

    return left_wheel_velocity, right_wheel_velocity

     
