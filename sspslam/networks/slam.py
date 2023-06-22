import numpy as np
import nengo
import nengo_spa as spa
from . import PathIntegration, ObjectVectorCells, InputGatedNetwork, AssociativeMemory
from sspslam.sspspace import HexagonalSSPSpace

# A version of the SLAM network useful for experiments
class SLAMNetwork(nengo.network.Network):
    def __init__(self, ssp_space, item_locations, path, T, view_rad,
        		pi_n_neurons, ovc_n_neurons, other_n_neurons, grid_cell_neurons, dt=0.001, tau=0.05, tau_pi = 0.05,
                update_thres=0.2, scale_fac =1.0, item_sps = None,
                wall_boundaries=None, wall_fssps=None, clean_up_method='grid', shift_rate=0.1,
                voja_learning_rate=5e-4, pes_learning_rate=1e-2):
        super().__init__()
        
        domain_dim = ssp_space.domain_dim
       	d=ssp_space.ssp_dim
       
       	timesteps = np.arange(0, T, dt)
        pathlen = len(timesteps)
        
        # Generate object location SSPs
        item_ssps = ssp_space.encode(item_locations)
        n_items = item_locations.shape[0]
        self.item_ssps = item_ssps
        item_fssps = ssp_space.encode_fourier(item_locations)
        rng = np.random.RandomState(seed=0)
        if item_sps is None:
            item_sps = nengo.dists.UniformHypersphere(surface=True).sample(n_items, d, rng=rng)
        intercept = (np.dot(item_sps, item_sps.T) - np.eye(n_items)).flatten().max()
        self.item_sps = item_sps
        
        if wall_boundaries is not None:
            n_walls = wall_boundaries.shape[0]
            wall_sps = nengo.dists.UniformHypersphere(surface=True).sample(n_walls, d, rng=np.random.RandomState(seed=100))
        else:
            wall_sps = None
        self.wall_sps = wall_sps
        
        
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
        
        def is_item(t):
            for i in np.arange(n_items):
                if np.linalg.norm(path[int(t/dt)-1,:] - item_locations[i,:]) <= view_rad:
                    return 0
            return 1
        
        if type(ssp_space) == HexagonalSSPSpace:
            encoders = ssp_space.sample_grid_encoders(grid_cell_neurons)
        else:
            encoders = ssp_space.get_sample_pts_and_ssps(grid_cell_neurons, method='sobol')
        
        def update_state_func(t,x):
            if ( (not is_item(t)) & (np.sum(x[:d]*x[d:]) > 0.2)):
                return shift_rate*(x[:d] - x[d:])
            else:
                return np.zeros(d)
        
        with self:
            self.vel_input = nengo.Node(size_in=domain_dim)
            self.update_state = nengo.Node(update_state_func,size_in=2*d)
             
            # PI network
            self.pathintegrator = PathIntegration(ssp_space, pi_n_neurons, tau_pi, 
                          scaling_factor=scale_fac, stable=True)
            nengo.Connection(self.vel_input,self.pathintegrator.velocity_input, synapse=None)
            nengo.Connection(self.update_state,self.pathintegrator.input, synapse=None)
          
            
            # Object vec cells, store vector to objects in view & more
            self.ovcs = ObjectVectorCells(ssp_space, ovc_n_neurons, other_n_neurons,
                            path, view_rad,
                            item_locations,item_fssps, item_sps, dt=dt,
                            wall_boundaries=wall_boundaries, wall_fssps=wall_fssps, wall_sps=wall_sps) 
            
            self.gridcells = nengo.Ensemble(grid_cell_neurons, d, encoders=encoders)
            self.pathintegrator.output.output = lambda t, x: x
            nengo.Connection(self.pathintegrator.output, self.gridcells, synapse=None, function=clean_up_fun)
            nengo.Connection(self.gridcells, self.ovcs.position_input, synapse=tau)
            
            # Env map
            self.assomemory = AssociativeMemory(other_n_neurons, d, d, intercept,
                                           voja_learning_rate=voja_learning_rate,
                                           pes_learning_rate=pes_learning_rate)
            nengo.Connection(self.ovcs.vision_sp_input, self.assomemory.key_input, synapse=None)
            nengo.Connection(self.ovcs.obj_ssp_ens.output, self.assomemory.value_input, synapse=tau)
            nengo.Connection(self.ovcs.is_item, self.assomemory.learning, synapse=None)
            
            # Estimate position using env map
            self.position_estimate = nengo.networks.CircularConvolution(other_n_neurons, d, invert_a=True)
            nengo.Connection(self.ovcs.ovc_ens, self.position_estimate.input_a, synapse=tau, function=lambda x: ssp_space.make_unitary(x))
            self.assomemory.recall.output=lambda t, x: x
            nengo.Connection(self.assomemory.recall, self.position_estimate.input_b, synapse=tau,function=lambda x: ssp_space.make_unitary(x))
            
            # Gate input. Only updates PI using env map if update isn't 'too far off' and object is actually in view
            nengo.Connection(self.position_estimate.output, self.update_state[:d], synapse=tau)
            nengo.Connection(self.pathintegrator.output, self.update_state[d:], synapse=tau)


  
  
