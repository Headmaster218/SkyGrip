# Python Classes Documentation

## Part 1: ResultsFloatingBaseJoint Class
This class handles the results and matrices related to dynamics computations of a robot with a floating base.

### Constructor: `ResultsFloatingBaseJoint(base_type)`
- **Parameters**:
  - `base_type`: The type of the robot base (fixed or floating).

### Method: `GetJ()`
- Returns the full Jacobian matrix or submatrices depending on the base type and flag.

### Method: `GetM()`
- Returns the full Mass matrix or submatrices depending on the base type and flag.

### Method: `GetN()`
- Returns the full Coriolis matrix or submatrices depending on the base type and flag.

### Method: `GetG()`
- Returns the full gravity vector or subvectors depending on the base type and flag.

### Method: `GetC()`
- Returns the full Coriolis vector or subvectors depending on the base type and flag.

## Part 2: PinWrapper Class
This class wraps around the Pinocchio library functionalities to provide robot dynamics computations.

### Constructor: `PinWrapper(conf_file_name, simulator, list_link_name_for_reordering, data_source_names, visualizer, index)`
- **Parameters**:
  - `conf_file_name`: Configuration file name.
  - `simulator`: Simulator name.
  - `list_link_name_for_reordering`: Ordered list of link names for joint state reordering.
  - `data_source_names`: Names of the data sources for state reordering.
  - `visualizer`: Boolean to indicate if visualization is enabled.
  - `index`: Index for configurations that support multiple instances or configurations.

### Method: `ComputeJacobian(q0, frame_name, local_or_global)`
- Computes the Jacobian matrix for a given frame.

### Method: `ComputeMassMatrix(q)`
- Computes the mass matrix at a given configuration.

### Method: `ComputeCoriolisMatrix(q, qdot)`
- Computes the Coriolis matrix for the given state.

### Method: `ComputeGravity(q)`
- Computes the gravity vector for the given configuration.

### Method: `DirectDynamicsActuatedZeroTorqueNoContact(q, qdot)`
- Computes the direct dynamics of the robot assuming no contact and zero torque input.

### Utility Functions:
- **`GetTotalMassFromUrdf()`**: Returns the total mass of the robot as defined in the URDF file.
- **`GetMassLink(link)`**: Returns the mass of a specific link.
- **`getNameActiveJoints()`**: Returns the names of active joints in the robot model.
- **`getDisplayModel(q)`**: Visualizes the robot configuration if a visualizer is available.

### Reordering and Conversion Functions:
These functions handle the conversion and reordering of joint states and matrices between external formats and the internal Pinocchio format.

### Advanced Dynamics Computations:
Includes functions for computing inverse dynamics, forward dynamics, and handling dynamics with contact forces.

## Part 3: Jacobian and Kinematic Computations

### Method: `ComputeJacobianFeet(q0, feet_name, local_or_global)`
- **Parameters**:
  - `q0`: The joint configuration vector.
  - `feet_name`: The name of the foot for which the Jacobian is computed.
  - `local_or_global`: Specifies if the Jacobian should be computed in the local or global frame.
- **Description**: Computes the Jacobian matrix for a specified foot.

### Method: `KinematicIntegration(q0, v0, dt)`
- **Parameters**:
  - `q0`: Initial configuration vector.
  - `v0`: Velocity vector.
  - `dt`: Time step for integration.
- **Returns**: The configuration after integrating the velocity over the time step.
- **Description**: Performs kinematic integration to compute the next configuration from the current configuration and velocity.

### Method: `ComputeFK(q, link_name)`
- **Parameters**:
  - `q`: Joint configuration vector.
  - `link_name`: Name of the link for which the forward kinematics is computed.
- **Returns**: Position and orientation of the specified link.
- **Description**: Computes the forward kinematics for a given link.

## Part 4: Dynamics Computations

### Method: `ComputeMassMatrixRNEA(x)`
- **Parameters**:
  - `x`: The configuration vector.
- **Returns**: The mass matrix computed using the Recursive Newton-Euler Algorithm (RNEA).
- **Description**: Computes the mass matrix using the RNEA, which is an alternative method to `ComputeMassMatrix`.

### Method: `ComputeCoriolis(q, qdot)`
- **Parameters**:
  - `q`: The configuration vector.
  - `qdot`: The velocity vector.
- **Returns**: The Coriolis forces vector.
- **Description**: Computes the Coriolis forces based on the current configuration and velocity.

### Method: `ComputeAllTerms(q, qdot)`
- **Parameters**:
  - `q`: The configuration vector.
  - `qdot`: The velocity vector.
- **Description**: Computes all the dynamic terms (mass matrix, Coriolis matrix, gravity vector) for the current configuration and velocity.

### Method: `InverseDynamicsActuatedPartNoContact(x_prev, xdot_prev, xdotdot_prev)`
- **Parameters**:
  - `x_prev`: Previous configuration vector.
  - `xdot_prev`: Previous velocity vector.
  - `xdotdot_prev`: Previous acceleration vector.
- **Returns**: The torques for the actuated part of the robot.
- **Description**: Computes the inverse dynamics for the actuated part of the robot without considering contact forces.

### Method: `ABA(q, qdot, tau)`
- **Parameters**:
  - `q`: Configuration vector.
  - `qdot`: Velocity vector.
  - `tau`: Torque vector.
- **Returns**: The acceleration vector after applying the Articulated Body Algorithm (ABA).
- **Description**: Computes the forward dynamics using the ABA, which is a fast algorithm for computing accelerations given torques.

## Part 5: Utilities and Helper Functions

### Method: `GetTotalMassFromUrdf()`
- **Returns**: The total mass of the robot from the URDF.
- **Description**: Computes and returns the total mass of the robot using the URDF file.

### Method: `GetMassLink(link)`
- **Parameters**:
  - `link`: The name of the link.
- **Returns**: The mass of the specified link.
- **Description**: Retrieves the mass of a particular link within the robot.

### Method: `getDynamicsInfo()`
- **Description**: Prints detailed information about the dynamics of each link in the robot, such as inertia and mass.

### Method: `getNameActiveJoints()`
- **Returns**: A list of the names of active joints in the robot.
- **Description**: Retrieves the names of all active joints, which are joints that can be actuated or measured.

### Method: `getNumberofActuatedJoints()`
- **Returns**: The number of actuated joints in the robot.
- **Description**: Returns the total count of joints that can be controlled.

### Method: `DisplayModel(q)`
- **Parameters**:
  - `q`: The joint configuration vector.
- **Description**: Displays the robot model using a visualizer, if one is initialized.

## Part 6: Reordering and Conversion

### Method: `_FromExtToPinVec(x, source_name=[])`
- **Parameters**:
  - `x`: The input vector in external format.
  - `source_name`: Optional source name for data conversion.
- **Returns**: A reordered vector in the Pinocchio format.
- **Description**: Converts a vector from an external format to the internal Pinocchio format.

### Method: `_FromPinToExtVec(x, source_name=[])`
- **Parameters**:
  - `x`: The input vector in Pinocchio format.
  - `source_name`: Optional source name for data conversion.
- **Returns**: A reordered vector in the external format.
- **Description**: Converts a vector from the internal Pinocchio format to an external format.

### Method: `ReorderJoints2PinVec(x, pos_or_vel, source_name=[])`
- **Parameters**:
  - `x`: The input vector (position or velocity).
  - `pos_or_vel`: Specifies if the input vector is a position or velocity vector.
  - `source_name`: Optional source name for reordering.
- **Returns**: A reordered vector in the Pinocchio format.
- **Description**: Reorders joint states for compatibility with Pinocchio's expected format.

### Method: `ReorderJoints2ExtVec(x, pos_or_vel, source_name=[])`
- **Parameters**:
  - `x`: The input vector (position or velocity).
  - `pos_or_vel`: Specifies if the input vector is a position or velocity vector.
  - `source_name`: Optional source name for reordering.
- **Returns**: A reordered vector in the external format.
- **Description**: Reorders joint states from Pinocchio's format to an external format.

## Part 7: Configuration Management

### Method: `GetConfigurationVariable(param)`
- **Parameters**:
  - `param`: The configuration parameter name.
- **Returns**: The value of the specified configuration parameter.
- **Description**: Retrieves configuration parameters related to the robot's dynamics and kinematics setup.

## Part 8: Additional Utility Functions and Helpers

### Method: `_FromPinToExtMat(X, source_name=[])`
- **Parameters**:
  - `X`: The input matrix in Pinocchio format.
  - `source_name`: Optional source name for data conversion.
- **Returns**: A reordered matrix in the external format.
- **Description**: Converts a matrix from the Pinocchio format to an external format, useful for tasks requiring matrix data manipulation.

### Method: `_ExtractJointsVec(x, flag="pos")`
- **Parameters**:
  - `x`: The input vector.
  - `flag`: Specifies whether the vector is for position (`"pos"`) or velocity (`"vel"`).
- **Returns**: Extracted joint vector from the input vector.
- **Description**: Extracts the joint states from the input vector based on whether it represents positions or velocities.

### Method: `_ExtractJointsMat(X, flag="pos")`
- **Parameters**:
  - `X`: The input matrix.
  - `flag`: Specifies whether the matrix is for position (`"pos"`) or velocity (`"vel"`).
- **Returns**: Extracted joint matrix from the input matrix.
- **Description**: Extracts the joint matrix from the input matrix based on the type of data (positions or velocities).

### Method: `_CopyJointsVec(x_dest, x_q, flag="pos")`
- **Parameters**:
  - `x_dest`: The destination vector to copy into.
  - `x_q`: The joint vector to copy.
  - `flag`: Specifies whether the vector is for position (`"pos"`) or velocity (`"vel"`).
- **Returns**: A new vector with the joint data copied into the correct positions.
- **Description**: Copies the joint states into the correct locations within the destination vector based on the specified type (positions or velocities).

### Method: `_CopyJointsMat(X_dest, X_q, flag="pos")`
- **Parameters**:
  - `X_dest`: The destination matrix to copy into.
  - `X_q`: The joint matrix to copy.
  - `flag`: Specifies whether the matrix is for position (`"pos"`) or velocity (`"vel"`).
- **Returns**: A new matrix with the joint data copied into the correct positions.
- **Description**: Copies the joint matrix into the correct locations within the destination matrix based on the specified type (positions or velocities).

### Method: `ReorderJoints2ExMat(X, pos_or_vel, source_name=[])`
- **Parameters**:
  - `X`: The input matrix (position or velocity).
  - `pos_or_vel`: Specifies if the input matrix is for position or velocity.
  - `source_name`: Optional source name for reordering.
- **Returns**: A reordered matrix in the external format.
- **Description**: Reorders joint matrices from Pinocchio's format to an external format, aiding in compatibility with other data sources or simulations.

## Part 9: Configuration and Initialization

### Method: `__init__(self, conf_file_name, simulator=None, list_link_name_for_reodering=np.empty(0), data_source_names=[], visualizer=False, index=0)`
- **Parameters**:
  - `conf_file_name`: The name of the configuration file.
  - `simulator`: The simulator type being used.
  - `list_link_name_for_reodering`: An optional list of link names for joint reordering.
  - `data_source_names`: Names of data sources for joint conversion.
  - `visualizer`: A boolean flag indicating whether to initialize a visualizer.
  - `index`: Index of the robot configuration to load.
- **Description**: Initializes the `PinWrapper` class, setting up the configuration, loading the URDF, and initializing variables for dynamics and visualization.

### Method: `_UrdfPath(index)`
- **Parameters**:
  - `index`: The index of the robot configuration to use.
- **Returns**: The path to the URDF file for the robot model.
- **Description**: Constructs the file path to the URDF based on the configuration and index, handling different robot models and paths.

### Method: `_LoadPinURDF(urdf_file)`
- **Parameters**:
  - `urdf_file`: The path to the URDF file to load.
- **Description**: Loads the URDF file into the Pinocchio model, initializing the model data and setting up the dynamics properties based on whether the robot has a floating base or is fixed.

## Part 10: Error Handling and Assertions

### Method: `CreateIndexJointAssociation(self, ext_list, data_source_names)`
- **Parameters**:
  - `ext_list`: A list of joint names for external sources.
  - `data_source_names`: Names of the data sources corresponding to the joint lists.
- **Description**: Creates an association between joint indices for different data sources and the Pinocchio model, allowing for consistent joint ordering across different formats.

### Error Handling in Constructor
- **Error Checks**:
  - Ensures that `list_link_name_for_reodering` is a 2D numpy array.
  - Validates that `simulator` is specified and not `None`.
  - Checks compatibility between the number of lists in `list_link_name_for_reodering` and `data_source_names`.
- **Description**: The constructor and methods include multiple checks to ensure that inputs and configurations are valid, preventing common errors related to file paths, data dimensions, and missing configurations.


