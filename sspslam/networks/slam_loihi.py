import numpy as np
import nengo
from . import PathIntegration, CircularConvolution, Product

# import os
# import sys
# sys.path.append("../")
# from sspspace import HexagonalSSPSpace

class SLAMLoihiNetwork(nengo.network.Network):
    r"""  Nengo network that performs simultaneous localization and mapping (SLAM), modified for nengo-loihi
    
    
    Parameters
    ----------
        ssp_space : SSPSpace
            Specifies the SSP representation that is being used.
            
        view_rad : float
            The agent's view radius.
            
        n_landmarks : int
            The number of landmarks that the agent may encounter. Only used for constructing 
            landmark_sps, which can optional be supplied.
            
        pi_n_neurons : int
            Number of neurons per VCO population in the Path Integrator network.
            
        mem_n_neurons : int
            Number of neurons in environment memory population.
            
        circonv_n_neurons : int
            Number of neurons per dim in populations that compute circular convolution.
            
        dotprod_n_neurons : int
            Number of neurons per dim in populations that compute a dot product.
            
        velocity_input : Node
            Should supply the agent's velocity. Included as optionally arg to avoid pass-through nodes.
            Default is None. 
            
        landmark_vecssp_input : Node
           Should supply SSP representation of vectors to landmarks in view.
           Included as optionally arg to avoid pass-through nodes
           Default is None. 
           
        landmark_sp_input : Node
            Should supply Semantic Pointers of landmarks in view. 
            ncluded as optionally arg to avoid pass-through nodes
            Default is None.
            
        no_landmark_in_view : Node
            Should indicate if landmark is in view (0 yes, 10 no).
            Included as optionally arg to avoid pass-through nodes
            Default is None.
            
        landmark_sps : np.ndarray
            A (num of landmarks x SSP dim) array. The Semantic Pointer representation 
            of the landmark identities. 
            
        tau : flaot
            Synaptic constant used for most connections. Default is 0.01.
            
        tau_pi : flaot
            Synaptic constant for recurrent connections in the Path Integrator network.
            Default is 0.05.
            
        update_thres : float
            How similar the PI output and position estimate from memory must be to 
            perform correction to the PI network. Revalent range is (0, 1).
            Smaller = more corrections including potentially erroneous corrections.
            Default is 0.2.
            
        vel_scaling_factor : float
            Scales the dynamics of the PI network. Used when velocity input has been normalized
            Default is 1.
            
        rad_scaling_factor : float
            Scales the radius of the VCOs in the PI network.
            Default is 1.
            
        shift_rate : float
            Scales the corecctions to PI network. Larger means faster corrections
            but potentially stability issuses. Revalent range is (0, 1). Default is 0.1.
            
        pes_learning_rate : float
             Learning rate of PES rule on decoders of env memory population. Default is 1e-3 .
             
        encoders : array
            A (mem_n_neurons x ssp_dim) array. Encoders of env memory population.
            'Good' encoders will be computed if encoders=None. Default is None.
            
        solver : nengo.Solver
            The solver to use for CircularConvolution populations. A sparse
            solver may help the loihi network compile without weight discretization errors.
            
        pi_solver_weights : bool
            If the solver on the recurrent connections in the PI net should use weights (True)
            or decoders (False). Default is False. 
            
        seed : int
            Default is 0.
            
    Attributes
    ----------
        landmark_sps : np.ndarray
            The landmark Semantic Pointers used
            
        velocity_input : Node
            Should supply the agent's velocity. *Requried input.*
            
        landmark_vecssp_input : Node
           Should supply SSP representation of vectors to landmarks in view. *Requried input.*
           
        landmark_sp_input : Node
            Should supply Semantic Pointers of landmarks in view. *Requried input.*
            
        no_landmark_in_view : Node
            Should indicate if landmark is in view (0 yes, 10 no). *Requried input.*
            
        pathintegrator : Network
            Network created with `.PathIntegration` to maintai an SSP estimate
            of self-position (in pathintegrator.output) by integrating a velocity signal.  
            pathintegrator.oscillators are the recurrent VCO population that  
            compute the dyanmics. 
            
        landmark_ssp_ens : Network
            Network created with `.CircularConvolution` to bind the SSP landmark
            vectors with output of gridcells. landmark_ssp_ens.output is an
            estimate of landmarks global locations (as SSPs) given current self-position estimate
            
        assomemory : Network
            Network that learns to associatelandmark Semantic Pointers to their location (as an SSP). 
            Landmark SPs are encoded by assomemory.memory and their SSP locations
            in assomemory.recall. Use assomemory.conn_out to examine pes error/ decoders
            and assomemory.conn_in for encoders (if voja is used)
            
        position_estimate : Network
            Network created with `.CircularConvolution` to bind the landmark SSP 
            locations from memory with the inverse of the SSP vector to landmarks.
            The result (position_estimate.output) is an estimate of self-position
            used to correct the PI network.
            
        output : Node
            Self-position estimate. Output of pathintegrator 

    Examples
    --------

    A basic example where data on agent's velocity and vectors to landmarks are
    given

       from sspslam.networks import SLAMLoihiNetwork, get_slam_input_functions
       
       # Get data
       domain_dim = ... # dim of space agent moves in 
       initial_agent_position = ...
       velocity_data = ...
       vec_to_landmarks_data = ...
       view_rad = ...
       
       # Construct SSPSpace
       ssp_space = HexagonalSSPSpace(...)
       d = ssp_space.ssp_dim
       
       # Construct SP space for discrete landmark representations
       lm_space = SPSpace(n_landmarks, d)
       
       # Convert data arrays to functions for nodes
       velocity_func, vel_scaling_factor, is_landmark_in_view, _, landmark_sp_func, _, landmark_vecssp_func = get_slam_input_functions(ssp_space,lm_space, velocity_data, vec_to_landmarks_data, view_rad)

       with nengo.Network():
           # If running agent online instead, these will be output from other networks
           velocty = nengo.Node(velocity_func)
           init_state = nengo.Node(lambda t: ssp_space.encode(initial_agent_position) if t<0.05 else np.zeros(d))
           landmark_vecssp = nengo.Node(landmark_vecssp_func, size_out = d)
           landmark_sp = nengo.Node(landmark_sp_func, size_out=d)
           no_landmark_in_view = nengo.Node(is_landmark_in_view, size_out=1)
           #
           
           slam = SLAMLoihiNetwork(ssp_space, lm_space, view_rad, n_landmarks,
                      		500, 500, 50, 50, 
                              velocty, landmark_vecssp, landmark_sp, no_landmark_in_view,
                              vel_scaling_factor=vel_scaling_factor)
           
           nengo.Connection(init_state, slam.pathintegrator.input, synapse=None)
           
           slam_output_p = nengo.Probe(slam.pathintegrator.output, synapse=0.05)


    """
    def __init__(self, ssp_space, lm_space,  view_rad, n_landmarks,
        		pi_n_neurons, mem_n_neurons, circonv_n_neurons, dotprod_n_neurons,
                velocity_input=None, landmark_vecssp_input=None, landmark_sp_input=None, no_landmark_in_view=None,
                tau=0.01, tau_pi = 0.05,
                update_thres=0.2, vel_scaling_factor =1.0, rad_scaling_factor=1,
                shift_rate=0.1, pes_learning_rate=1e-2, 
                encoders=None, solver=nengo.Default, pi_solver_weights=False, seed=0):
        super().__init__()
        
       	d=ssp_space.ssp_dim
        domain_dim = ssp_space.domain_dim
        
        landmark_sps = lm_space.vectors
        rng = np.random.RandomState(seed=seed)
        if encoders is None:
            encoders = landmark_sps[rng.randint(n_landmarks,size=mem_n_neurons),:]

        intercept = (np.dot(landmark_sps, landmark_sps.T) - np.eye(n_landmarks)).flatten().max()

            
        
        with self:
            self.velocity_input = nengo.Node(size_in=domain_dim, label='vel_input') if velocity_input is None else velocity_input
            self.landmark_vecssp_input = nengo.Node(size_in=d, label='lm_vecssp_input') if landmark_vecssp_input is None else landmark_vecssp_input
            self.landmark_sp_input = nengo.Node(size_in=d, label='lm_sp_input') if landmark_sp_input is None else landmark_sp_input
            self.no_landmark_in_view = nengo.Node(size_in=1, label='lm_in_view_input') if no_landmark_in_view is None else no_landmark_in_view
            #
            
             
            # Path Integrator network
            self.pathintegrator = PathIntegration(ssp_space, pi_n_neurons, tau_pi, max_radius=rad_scaling_factor,
                          scaling_factor=vel_scaling_factor, stable=True, with_gcs=False, solver_weights=pi_solver_weights, label='pathint')
            nengo.Connection(self.velocity_input, self.pathintegrator.velocity_input, synapse=None)
            self.output = self.pathintegrator.output
            #
            
            # Landmark perception
            self.landmark_ssp_ens = CircularConvolution(circonv_n_neurons, dimensions=d,
                                                      solver=solver, label='landmark_circonv')
            nengo.Connection(self.pathintegrator.output, self.landmark_ssp_ens.input_a, synapse=tau)
            nengo.Connection(self.landmark_vecssp_input, self.landmark_ssp_ens.input_b, synapse=0)
            #
            
                        
            # Env. map module
            self.assomemory = nengo.Network(seed=seed)
            self.assomemory.memory = nengo.Ensemble(mem_n_neurons, d, intercepts=[intercept] * mem_n_neurons,
                                        encoders=encoders, radius=1, label='memory')
            self.assomemory.recall = nengo.Ensemble(mem_n_neurons, d, label='memory_recall')
            nengo.Connection(self.landmark_sp_input, self.assomemory.memory, synapse=None, label='map_conn_in')
            self.assomemory.conn_out = nengo.Connection(self.assomemory.memory, self.assomemory.recall,
                                learning_rule_type=nengo.PES(pes_learning_rate), label='map_conn_pes',
                                function=lambda x: np.zeros(d) )
            
            mem_error = nengo.Ensemble(mem_n_neurons, d, label='memory_pes_error')
            nengo.Connection(self.no_landmark_in_view, mem_error.neurons, transform=[[-2.5]] * mem_n_neurons, synapse=None)
            nengo.Connection(self.landmark_ssp_ens.output, mem_error, transform=-1, synapse=tau)
            nengo.Connection(self.assomemory.recall, mem_error, synapse=tau)
            nengo.Connection(mem_error, self.assomemory.conn_out.learning_rule, synapse=tau)
            #
                
            # Estimate position using env map
            self.position_estimate = CircularConvolution(circonv_n_neurons, d, input_magnitude=1,
                                                 invert_a=True,solver=solver, label='newpos_circonv')
            nengo.Connection(self.landmark_vecssp_input, self.position_estimate.input_a, synapse=None)
            nengo.Connection(self.assomemory.recall, self.position_estimate.input_b, synapse=tau)
            #
            
            # Correction = new pos. estimate - pi estimate
            self.correction = nengo.Ensemble(mem_n_neurons, d, label='correction_ens')
            nengo.Connection(self.position_estimate.output, self.correction, synapse=tau, transform = 1)
            nengo.Connection(self.pathintegrator.output, self.correction, synapse=tau, transform = -1)
            nengo.Connection(self.correction, self.pathintegrator.input, synapse=0.1, transform = shift_rate) # long synapse
            #
            
            # Gate correction: requries dot product
            bias = nengo.Node(1,label='threshold_bias')
            self.threshold = nengo.Ensemble(circonv_n_neurons, 1,
                                            intercepts=nengo.dists.Choice([update_thres]),
                                            encoders=np.ones((circonv_n_neurons,1)), label='threshold')
            nengo.Connection(bias, self.threshold, synapse=None)
            nengo.Connection(self.no_landmark_in_view, self.threshold,  synapse=None)
            nengo.Connection(self.threshold, self.correction.neurons, transform=[[-5]] * mem_n_neurons, synapse=0.05)
            
            sq1 = nengo.networks.EnsembleArray(
                max(1, dotprod_n_neurons // 2), n_ensembles=d, ens_dimensions=1,
                radius= 1*np.sqrt(2), label='dotprod_sq1')#np.sqrt(2))
            sq2 = nengo.networks.EnsembleArray(
                max(1, dotprod_n_neurons // 2), n_ensembles=d, ens_dimensions=1,
                radius= 1*np.sqrt(2), label='dotprod_sq2')#np.sqrt(2))
            tr = 1. / np.sqrt(2.)
            nengo.Connection(
                self.position_estimate.output, sq1.input, transform=tr, synapse=tau)
            nengo.Connection(
                self.pathintegrator.output, sq1.input, transform=tr, synapse=tau)
            nengo.Connection(
                self.position_estimate.output, sq2.input, transform=tr, synapse=tau)
            nengo.Connection(
                self.pathintegrator.output, sq2.input, transform=-tr, synapse=tau)
            for i in range(d):
                nengo.Connection(sq1.ea_ensembles[i], self.threshold, function=lambda x: -.5*x**2, synapse=tau) #signs flipped
                nengo.Connection(sq2.ea_ensembles[i], self.threshold, function=lambda x: .5*x**2, synapse=tau)
            #

  

  
