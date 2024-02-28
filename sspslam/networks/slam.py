import numpy as np
import nengo
from . import PathIntegration, AssociativeMemory, CircularConvolution, Product



class SLAMNetwork(nengo.network.Network):
    r"""  Nengo network that performs simultaneous localization and mapping (SLAM)
    
    Parameters
    ----------
        ssp_space : SSPSpace
            Specifies the SSP representation that is being used for encoding location.
            
        lm_space : SPSpace
            Specifies the SP representation that is being used for encoding landmarks.
            
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
            Number of neurons per dim in populations that perform circular convolution.
            
        tau : flaot
            Synaptic constant used for most connections. Default is 0.01.
            
        tau_pi : flaot
            Synaptic constant for recurrent connections in the Path Integrator network.
            Default is 0.05.
            
        update_thres : float
            How similar the PI output and position estimate from memory must be to 
            perform correction to the PI network. Revalent range is (0, 1).
            Smaller = more corrections including potentially erroneous corrections.
            Default is 0.2
            
        vel_scaling_factor : float
            Scales the dynamics of the PI network. Used when velocity input has been normalized.
            Default is 1.
            
        rad_scaling_factor : float
            Scales the radius of the VCOs in the PI network.
            Default is 1.
            
        shift_rate : float
            Scales the corecctions to PI network. Larger means faster corrections
            but potentially stability issuses. Revalent range is (0, 1). Default is 0.1.
            
        voja_learning_rate : float
            Learning rate of Voja rule on encoders of env memory population. Default is 5e-4.
            
        pes_learning_rate : float
             Learning rate of PES rule on decoders of env memory population. Default is 1e-3 .
             
        clean_up_method : None, 'grid' ,'direct-optim', 'network', 'network-optim'
            Method for 'cleaning-up' output of PI network. Default is grid
            'grid' is the most stable but computational expensive (especially in high dim spaces);
            'direct-optim' is accurate but even slower/ computational expensive; 
            'network' is fastest at simulation time but less stable and may require time at initialization to train a network
            'network-optim' is similar to network but a bit more accuracte and slower
            None means no clean-up
            
        gc_n_neurons : int
            Number of neurons used to encode the result of the clean-up.
            If <=0, output of clean-up stored in a Node.
            If >0, will use grid cells (requries ssp_space to be HexagonalSSPSpace).
            Default is 0.
            
        encoders : np.ndarray
            A (mem_n_neurons x ssp_dim) array. Initial encoders of env memory population.
            Will default to nengo.Default.
            
        voja : bool
            Whether to use Voja rule. If not used and encoders not given, good
            encoders (randomly selected landmark SPs) will be computed and used.
            
        seed : int
            Default is 0.
            
    Attributes
    ----------
        velocity_input : Node
            Should supply the agent's velocity. *Requried input.*
            
        landmark_vec_input : Node
           Should supply vectors to landmarks in view. *Requried input.*
           
        landmark_id_input : Node
            Needs to receive index of landmark in view. Outputs Semantic Pointers. *Requried input.*
            
        landmark_vec_ssp : Node
            Will return the SSP rep of output from landmark_vec_input
            
        no_item_in_view : Node
            Will return signal indicating if landmark is in view
            
        pathintegrator : Network
            Network created with `.PathIntegration` to maintai an SSP estimate
            of self-position (in pathintegrator.output) by integrating a velocity signal.  
            pathintegrator.oscillators are the recurrent VCO population that  
            compute the dyanmics. 
            
        gridcells : Node or Ensemble
            Encodes output of PI network post-clean-up
            
        landmark_ssp_ens : Network
            Network created with `.CircularConvolution` to bind the SSP landmark
            vectors with output of gridcells. landmark_ssp_ens.output is an
            estimate of landmarks global locations (as SSPs) given current self-position estimate
            
        assomemory : Network
            Network created with `.AssociativeMemory`. Learns to associate 
            landmark Semantic Pointers to their location (as an SSP). 
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

       from sspslam.networks import SLAMNetwork, get_slam_input_functions
       
       # Get data
       domain_dim = ... # dim of space agent moves in 
       initial_agent_position = ...
       velocity_data = ...
       vec_to_landmarks_data = ...
       n_landmarks = ...
       view_rad = ...
       
       # Construct SSPSpace
       ssp_space = HexagonalSSPSpace(...)
       d = ssp_space.ssp_dim
       
       # Construct SP space for discrete landmark representations
       lm_space = SPSpace(n_landmarks, d)
       
       # Convert data arrays to functions for nodes
       velocity_func, vel_scaling_factor, is_landmark_in_view, landmark_id_func, _, landmark_vec_func, _ = get_slam_input_functions(ssp_space,lm_space, velocity_data, vec_to_landmarks_data, view_rad)

       with nengo.Network():
           # If running agent online instead, these will be output from other networks
           velocty = nengo.Node(velocity_func)
           init_state = nengo.Node(lambda t: ssp_space.encode(initial_agent_position) if t<0.05 else np.zeros(d))
           landmark_vec = nengo.Node(landmark_vec_func, size_out = domain_dim)
           #
           
           slam = SLAMNetwork(ssp_space, lm_space, view_rad, n_landmarks,
                      		500, 500, 50, vel_scaling_factor=vel_scaling_factor)
           
           nengo.Connection(velocty, slam.velocity_input, synapse=0.01) 
           nengo.Connection(init_state, slam.pathintegrator.input, synapse=None)
           nengo.Connection(landmark_vec, slam.landmark_vec_input, synapse=None)
           
           slam_output_p = nengo.Probe(slam.pathintegrator.output, synapse=0.05)


    """
    def __init__(self, ssp_space, lm_space, view_rad, n_landmarks,
        		pi_n_neurons, mem_n_neurons, circonv_n_neurons, 
                tau=0.01, tau_pi = 0.05,
                update_thres=0.2, vel_scaling_factor =1.0,rad_scaling_factor=1.0, shift_rate=0.1,
                voja_learning_rate=5e-4, pes_learning_rate=1e-2,
                clean_up_method='grid', gc_n_neurons = 0, encoders=None, voja=True,seed=0):
        super().__init__()
        
        domain_dim = ssp_space.domain_dim
       	d=ssp_space.ssp_dim
               
        rng = np.random.RandomState(seed=seed)
        landmark_sps = lm_space.vectors
        # if landmark_sps is None:
        #     landmark_sps = nengo.dists.UniformHypersphere(surface=True).sample(n_landmarks, d, rng=rng)
        if (not voja) and (encoders is None):
            encoders = landmark_sps[rng.randint(n_landmarks,size=mem_n_neurons),:]
        intercept = (np.dot(landmark_sps, landmark_sps.T) - np.eye(n_landmarks)).flatten().max()
        

        if clean_up_method=='grid':
            sample_ssps,sample_points = ssp_space.get_sample_pts_and_ssps(100)
            def clean_up_fun(x):
                sims =  sample_ssps @ x
                return sample_ssps[np.argmax(sims),:]
            self.sample_ssps = sample_ssps
            self.sample_points = sample_points
        else:
            if (clean_up_method=='network' ) | (clean_up_method=='network-optim') :
                ssp_space.train_decoder_net(n_training_pts=200000, n_hidden_units = 8,
                    learning_rate=1e-3, n_epochs = 20, load_file=True, save_file=True)
            def clean_up_fun(x):
                return ssp_space.clean_up(np.atleast_2d(x),method=clean_up_method).reshape(-1)
            
        self.clean_up_fun = clean_up_fun
   
    
        
        def is_landmark_in_view(t, x):
            if np.linalg.norm(x) <= view_rad:
                return 0
            else:
                return 1
        
        def update_state_func(t,x):
            if ( np.allclose(x[-1],0,atol=1e-3) & (np.sum(x[:d]*x[d:-1]) > update_thres)): #(t<init_t) | 
                return shift_rate*(x[:d] - x[d:-1])
            else:
                return np.zeros(d)
            
        
        with self:
            self.velocity_input = nengo.Node(size_in=domain_dim, label='vel_input')
            self.landmark_vec_input = nengo.Node(size_in=domain_dim, label='lm_vec_input')
            self.landmark_id_input = nengo.Node(lambda t,x: landmark_sps[int(x)] if x>0
                                                else np.zeros(d), size_in=1, label='lm_id_input')
            
            self.landmark_vec_ssp = nengo.Node(lambda t,x: ssp_space.encode(x).flatten(), size_in=domain_dim, label='lm_vecssp_input')
            nengo.Connection(self.landmark_vec_input, self.landmark_vec_ssp, synapse=None)
            self.no_landmark_in_view = nengo.Node(is_landmark_in_view, size_in=domain_dim, label='lm_in_view_input')
            nengo.Connection(self.landmark_vec_input, self.no_landmark_in_view, synapse=None)
            
            self.update_state = nengo.Node(update_state_func,size_in=2*d + 1)
            nengo.Connection(self.no_landmark_in_view, self.update_state[-1], synapse=None)
             
            # PI network
            self.pathintegrator = PathIntegration(ssp_space, pi_n_neurons, tau_pi, max_radius=rad_scaling_factor,
                          scaling_factor=vel_scaling_factor, stable=True, label='pathint')
            self.output = self.pathintegrator.output
            nengo.Connection(self.velocity_input,self.pathintegrator.velocity_input, synapse=None)
            nengo.Connection(self.update_state,self.pathintegrator.input, synapse=None)
            
            # Object vec cells
            # self.ovc_ens = nengo.Ensemble(ovc_n_neurons, d)#, encoders= OVC_encoders)
            # nengo.Connection(self.landmark_vec_ssp,self.ovc_ens, synapse=None)
            
            self.landmark_ssp_ens = CircularConvolution(circonv_n_neurons, dimensions=d, label='landmark_circonv')
            nengo.Connection(self.landmark_vec_ssp, self.landmark_ssp_ens.input_b, synapse=None)
            
            # Clean-up
            if gc_n_neurons is None:
                nengo.Connection(self.pathintegrator.output, self.landmark_ssp_ens.input_a, synapse=tau)
            elif gc_n_neurons>0:
                gc_encoders = ssp_space.sample_grid_encoders(gc_n_neurons)
                cleanup = nengo.Node(lambda t,x: clean_up_fun(x), size_in=d)
                self.gridcells = nengo.Ensemble(gc_n_neurons,d, encoders = gc_encoders)
                nengo.Connection(self.pathintegrator.output, cleanup, synapse=tau)
                nengo.Connection(cleanup, self.gridcells, synapse=tau)
                nengo.Connection(self.gridcells, self.landmark_ssp_ens.input_a, synapse=tau)
            else:
                self.gridcells = nengo.Node(lambda t,x: clean_up_fun(x), size_in=d)
                nengo.Connection(self.pathintegrator.output, self.gridcells, synapse=tau)
                nengo.Connection(self.gridcells, self.landmark_ssp_ens.input_a, synapse=None)
            
                        
            # Env map
            self.assomemory = AssociativeMemory(mem_n_neurons, d, d, intercept,
                                           voja_learning_rate=voja_learning_rate,
                                           pes_learning_rate=pes_learning_rate,
                                           voja=voja,encoders=encoders)

            nengo.Connection(self.landmark_id_input, self.assomemory.key_input, synapse=None)
            nengo.Connection(self.landmark_ssp_ens.output, self.assomemory.value_input, synapse=tau)
            nengo.Connection(self.no_landmark_in_view, self.assomemory.learning, synapse=None)
            
            # Estimate position using env map
            self.position_estimate = CircularConvolution(circonv_n_neurons, d, invert_a=True, label='newpos_circonv')
            nengo.Connection(self.landmark_vec_ssp, self.position_estimate.input_a, synapse=tau)
            # self.assomemory.recall.output=lambda t, x: x
            nengo.Connection(self.assomemory.recall, self.position_estimate.input_b, 
                             synapse=tau, function=lambda x: ssp_space.make_unitary(x))
            
            # Gate input. Only updates PI using env map if update isn't 'too far off' and object is actually in view
            nengo.Connection(self.position_estimate.output, self.update_state[:d], synapse=tau)
            nengo.Connection(self.pathintegrator.output, self.update_state[d:-1], synapse=tau)

  
  
 
def get_slam_input_functions(ssp_space, lm_space, velocity_data, vec_to_landmarks_data, view_rad, dt=0.001):
    r""" A helper function. 
    
    The SLAM model requries a velocity signal, the SSP representation of 
    vectors to objects/landmarks in view (exact or apporximated from sensor data),
    identities of landmarks in view, and input that says whether or not a landmark is in view.
   
    In some applications, these will be computed on the fly by other networks that must be linked up. 
    Other times we run agent simulations seperately and collect this data and perform SLAM after the fact to test the algorithm.
    This function converts such data into several functions that can be put in nodes to supply input to the SLAM model.
    
    Parameters
    ----------
        ssp_space : SSPSpace
            Specifies the SSP representation that is being used
            
        velocity_data : np.ndarray
            With shape (length of path x dim of space). 
            velocity_data[i,:] is the velocity of the agent at timestep j
            
        vec_to_landmarks_data : np.ndarray
            With shape (num of landmarks x length of path x dim of space). 
            vec_to_landmarks_data[i,j,:] is the vector from the agent to landmark i at timestep j
            
        view_rad : float
            The agent's view radius.
            
        dt : flaot
            Timestep of simulation from which vec_to_landmarks was obtained. 
            Default is 0.001 (nengo's default)
                              
        landmark_sps : np.ndarray
            A (num of landmarks x SSP dim) array. The Semantic Pointer representation 
            of the landmark identities. If not given, will be generated
            
        seed : int
            Default is 0
            
    Output
    ----------
        velocity_func : callable
            Takes input t, the current time of the simulation. 
            Returns agent's velcoity at time t.
            
        vel_scaling_factor : float
            Needed for PathIntegration network
            
        is_item_in_view : callable
            Takes input t, the current time of the simulation. 
            Returns 0 if a landmark is in view, otherwise returns 10. 
            Used for inhibiting neural populations & turning off learning
            
        landmark_id_func :  callable
            Takes input t, the current time of the simulation. 
            Returns the index (int between 0 and n_landmarks-1) of the landmark 
            in view at time t. Returns index of closest landmark if multiple are
            in view. Returns -1 if none in view.  Only used for constructing the other functions
            
        landmark_sp_func :  callable
            Takes input t, the current time of the simulation. 
            Returns the Semantic Pointer repreesntation of the landmark 
            in view at time t. Returns vector of zeros if none in view. 
            
        landmark_vec_func : callable
            Takes input t, the current time of the simulation. 
            Returns the vector to the landmark in view at time t. Returns 
            vector of zeros if none in view.  Only used for constructing landmark_vecssp_func
            
        landmark_vecssp_func : callable
            Takes input t, the current time of the simulation. 
            Returns the SSP representation of the vector to the landmark in 
            view at time t. Returns vector of zeros if none in view.
    """
    n_landmarks = vec_to_landmarks_data.shape[1]
    pathlen = vec_to_landmarks_data.shape[0]
    domain_dim = vec_to_landmarks_data.shape[2]
    d = ssp_space.ssp_dim
    
    landmark_sps = lm_space.vectors
       
    real_freqs = (ssp_space.phase_matrix @ velocity_data.T)
    vel_scaling_factor = 1/np.max(np.abs(real_freqs))
    vels_scaled = velocity_data*vel_scaling_factor
    velocity_func = lambda t: vels_scaled[int(np.minimum(np.floor(t/dt), pathlen-2))]
    
    # At time t, if a landmark(s) is in view return the index of the landmark, else return -1
    # Only used for constructing the later functions
    def landmark_id_func(t):
        current_vecs = vec_to_landmarks_data[int(np.minimum(np.floor(t/dt), pathlen-2)),:,:]
        dists = np.linalg.norm(current_vecs, axis=1)
        if np.all(dists > view_rad):
            return -1
        else:
            return np.argmin(dists)
        
    # At time t, if a landmark is in view return the vector to the landmark (from the input data)
    def landmark_vec_func(t):
        cur_id = landmark_id_func(t)
        if cur_id<0:
            return np.zeros(domain_dim)
        else:
            return vec_to_landmarks_data[int(np.minimum(np.floor(t/dt), pathlen-2)),cur_id, :]
        
   # At time t, if a landmark is in view return the Semantic Pointer representation of the landmark
    def landmark_sp_func(t):
        cur_id = landmark_id_func(t)
        if cur_id<0:
            return np.zeros(d)
        else:
            return landmark_sps[cur_id]
            
    # At time t, if a landmark is in view return the SSP representation of the vector to the landmark (from the input data)  
    def landmark_vecssp_func(t):
        cur_id = landmark_id_func(t)
        if cur_id<0:
            return np.zeros(d)
        else:
            return ssp_space.encode(vec_to_landmarks_data[int(np.minimum(np.floor(t/dt), pathlen-2)), cur_id, :]).flatten()

    # Is an item in view at time t? if no return 10 else return 0. Used for inhibiting neural populations
    def is_landmark_in_view(t):
        cur_id = landmark_id_func(t)
        if cur_id<0:
            return 10
        else:
            return 0
        
    return velocity_func, vel_scaling_factor, is_landmark_in_view, landmark_id_func, landmark_sp_func, landmark_vec_func, landmark_vecssp_func
