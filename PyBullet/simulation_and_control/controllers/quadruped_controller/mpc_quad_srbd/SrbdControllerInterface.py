import numpy as np
# TODO replace this with_pin_wrapper
from gym_quadruped.utils.quadruped_utils import LegsAttr
from centroidal_nmpc_nominal import Acados_NMPC_Nominal

#from quadruped_pympc import config as cfg

class SRBDControllerInterface:
    """This is an interface for a controller that uses the SRBD method to optimize the gait"""


    def __init__(self, ):
        """ Constructor for the SRBD controller interface """
        
        self.type = cfg.mpc_params['type']
        self.mpc_dt = cfg.mpc_params['dt']
        self.horizon = cfg.mpc_params['horizon']
        


        self.previous_contact_mpc = np.array([1, 1, 1, 1])
        
        self.controller = Acados_NMPC_Nominal()

            
    def compute_control(self, 
                        state_current: dict,
                        ref_state: dict,
                        contact_sequence: np.ndarray,
                        inertia: np.ndarray,
                        pgg_phase_signal: np.ndarray,
                        pgg_step_freq: float,
                        optimize_swing: int) -> [LegsAttr, LegsAttr, LegsAttr, LegsAttr, LegsAttr, float]:
        """Compute the control using the SRBD method

        Args:
            state_current (dict): The current state of the robot
            ref_state (dict): The reference state of the robot
            contact_sequence (np.ndarray): The contact sequence of the robot
            inertia (np.ndarray): The inertia of the robot
            pgg_phase_signal (np.ndarray): The periodic gait generator phase signal of the legs (from 0 to 1)
            pgg_step_freq (float): The step frequency of the periodic gait generator
            optimize_swing (int): The flag to optimize the swing

        Returns:
            tuple: The GRFs and the feet positions in world frame, 
                   and the best sample frequency (only if the controller is sampling)
        """
    

        current_contact = np.array([contact_sequence[0][0],
                                    contact_sequence[1][0],
                                    contact_sequence[2][0],
                                    contact_sequence[3][0]])
       
       
        # use Gradient-Based MPC
        
        nmpc_GRFs, \
        nmpc_footholds, \
        _, \
        _ = self.controller.compute_control(state_current,
                                            ref_state,
                                            contact_sequence,
                                            inertia=inertia)
        
        nmpc_joints_pos = None
        nmpc_joints_vel = None
        nmpc_joints_acc = None


        nmpc_footholds = LegsAttr(FL=nmpc_footholds[0],
                                    FR=nmpc_footholds[1],
                                    RL=nmpc_footholds[2],
                                    RR=nmpc_footholds[3])
        
        
        # If the controller is using RTI, we need to linearize the mpc after its computation
        # this helps to minize the delay between new state->control, but only in a real case.
        # Here we are in simulation and does not make any difference for now
        if (self.controller.use_RTI):
            # preparation phase
            self.controller.acados_ocp_solver.options_set('rti_phase', 1)
            self.controller.acados_ocp_solver.solve()
            # print("preparation phase time: ", controller.acados_ocp_solver.get_stats('time_tot'))


        best_sample_freq = pgg_step_freq


        # TODO: Indexing should not be hardcoded. Env should provide indexing of leg actuator dimensions.
        nmpc_GRFs = LegsAttr(FL=nmpc_GRFs[0:3] * current_contact[0],
                                FR=nmpc_GRFs[3:6] * current_contact[1],
                                RL=nmpc_GRFs[6:9] * current_contact[2],
                                RR=nmpc_GRFs[9:12] * current_contact[3])
            

        return nmpc_GRFs, nmpc_footholds, nmpc_joints_pos, nmpc_joints_vel, nmpc_joints_acc, best_sample_freq
        