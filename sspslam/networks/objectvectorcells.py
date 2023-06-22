import nengo
import numpy as np
from sspslam.utils import ScatteredHypersphere


# This network creates a population of object/border vector cells (called ovc_ens) --
# they represent vectors to objects/walls in view. This is used along with current
# position input (via node position_input) to compute object's global location (in obj_ssp_ens)
# and OBJ_SSP*OBJ_SP in obj_ens
# You must pass in ssp_space, n_OVCs, n_neurons,
# and either (1)  path, view_rad and (item_locations, item_fssps, item_sps) AND/OR
#               (all_boundaries, wall_fssps, wall_sps) so that vision input
#               about vectors to objects etc. can be created.
#   or (2) pass in your own custom vision input functions 
#       vision_funcs=(vision_ssp_fun, vision_sp_fun, is_item)
#       where vision_ssp_fun(t) gives the sum of vectors to objects/walls in view at time t
#           vision_sp_fun gives sum of object semantic pointers (e.g. 'TREE', 'GRREN * BALL', etc.) in view at t
#           is_item returns 0 if something is in view and zero otherwise
class ObjectVectorCells(nengo.network.Network):
    def __init__(self, ssp_space, n_OVCs, n_neurons,
                 path=None, view_rad=None,
                 item_locations=None, item_fssps=None, item_sps=None, 
                 wall_boundaries=None, wall_fssps=None, wall_sps=None,
                 vision_funcs=None, tau=0.05,dt=0.001,
                 **kwargs):
        d = ssp_space.ssp_dim
        
        OVC_vectors = view_rad*ScatteredHypersphere(surface=False).sample(n_OVCs, ssp_space.domain_dim)
        OVC_encoders = ssp_space.encode(OVC_vectors)
        
        if vision_funcs is None:
            fssp_path = ssp_space.encode_fourier(path)
            
            if item_locations is None:
                num_items=0
            else:
                num_items = len(item_sps)
            if wall_boundaries is None:
                num_walls=0
            else:
                num_walls = len(wall_sps)
            def vision_ssp_fun(t):
                deltaS = np.zeros(d)
                S = fssp_path[int(t/dt)-1,:]
                anyobjs = False
                for i in range(num_items):
                    vec = item_locations[i,:] - path[int(t/dt)-1,:]
                    if np.linalg.norm(vec) <= view_rad:
                        deltaS = deltaS + ssp_space.encode(vec)
                        anyobjs = True
                for i in range(num_walls):
                    closestX = np.clip(path[int(t/dt)-1,0],wall_boundaries[i,0,0],wall_boundaries[i,0,1])
                    closestY = np.clip(path[int(t/dt)-1,1],wall_boundaries[i,1,0],wall_boundaries[i,1,1])
                    dist_to_wall = np.sqrt((closestX - path[int(t/dt)-1,0])**2 + (closestY - path[int(t/dt)-1,1])**2)
                    if dist_to_wall <= view_rad:
                        deltaS = deltaS + np.fft.ifft( ssp_space.make_unitary_fourier(wall_fssps[i]) / S)
                        anyobjs = True
                if anyobjs:
                    deltaS = ssp_space.normalize(deltaS)
                return deltaS.reshape(-1)
            
            def vision_sp_fun(t):
                deltaS = np.zeros(d)
                for i in np.arange(num_items):
                    if np.linalg.norm(path[int(t/dt)-1,:] - item_locations[i,:]) <= view_rad:
                        deltaS = deltaS + item_sps[i] 
                for i in np.arange(num_walls):
                    closestX = np.clip(path[int(t/dt)-1,0],wall_boundaries[i,0,0],wall_boundaries[i,0,1])
                    closestY = np.clip(path[int(t/dt)-1,1],wall_boundaries[i,1,0],wall_boundaries[i,1,1])
                    dist_to_wall = np.sqrt((closestX - path[int(t/dt)-1,0])**2 + (closestY - path[int(t/dt)-1,1])**2)
                    if dist_to_wall <= view_rad:
                        deltaS = deltaS + wall_sps[i]
                return deltaS.reshape(-1)
            
            def is_item(t):
                for i in range(num_items):
                    if np.linalg.norm(path[int(t/dt)-1,:] - item_locations[i,:]) <= view_rad:
                        return 0
                for i in range(num_walls):
                    closestX = np.clip(path[int(t/dt)-1,0],wall_boundaries[i,0,0],wall_boundaries[i,0,1])
                    closestY = np.clip(path[int(t/dt)-1,1],wall_boundaries[i,1,0],wall_boundaries[i,1,1])
                    dist_to_wall = np.sqrt((closestX - path[int(t/dt)-1,0])**2 + (closestY - path[int(t/dt)-1,1])**2)
                    if dist_to_wall <= view_rad:
                        return 0
                return 1
        else:
                vision_ssp_fun, vision_sp_fun, is_item = vision_funcs
            
        
        super().__init__(**kwargs)
        with self:
            self.position_input = nengo.Node(size_in=d)
            
            # From vision
            self.vision_ssp_input = nengo.Node(vision_ssp_fun, label='obj_Dssp_input')
            self.vision_sp_input = nengo.Node(vision_sp_fun, label='obj_sp_input')
            self.is_item = nengo.Node(is_item)
            
            self.ovc_ens = nengo.Ensemble(n_OVCs, d, encoders= OVC_encoders, label='ovc')
            nengo.Connection(self.vision_ssp_input,self.ovc_ens, synapse=None)
            
            self.obj_ssp_ens = nengo.networks.CircularConvolution(n_neurons=n_neurons, dimensions=d, label='obj_ssp')
            nengo.Connection(self.position_input, self.obj_ssp_ens.input_a, synapse=0)
            nengo.Connection(self.ovc_ens, self.obj_ssp_ens.input_b, synapse=tau)
            
            
