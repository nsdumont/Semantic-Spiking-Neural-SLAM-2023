import nengo
import numpy as np
import matplotlib.pyplot as plt
import sspslam

import pytry
from argparse import ArgumentParser
import os
import os.path
import random
import pickle



class SLAMTrial(pytry.Trial):
    def params(self):
        self.param('ssp_dim', ssp_dim=151)
        self.param('domain_dim', domain_dim=2)
        self.param('sim time', time=120)
        self.param('path limit', limit=0.08)
        self.param('ovc_n_neurons', ovc_n_neurons=800)
        self.param('pi_n_neurons',  pi_n_neurons=700)
        self.param('other_n_neurons', other_n_neurons=700)
        self.param('grid_cell_neurons', grid_cell_neurons=1000)
    
    def evaluate(self, p):
        domain_dim = p.domain_dim
        bounds = np.tile([-1,1],(domain_dim,1))
        ssp_space = sspslam.sspspace.HexagonalSSPSpace(domain_dim,ssp_dim=p.ssp_dim, 
                         domain_bounds=1.1*bounds, length_scale=0.1)
        d = ssp_space.ssp_dim
        
        T = p.time
        dt = 0.001
        timesteps = np.arange(0, T, dt)
        radius=1
        path = np.hstack([nengo.processes.WhiteSignal(T, high=p.limit, seed=p.seed+i).run(T,dt=dt) for i in range(domain_dim)])
        shift_fun = lambda x, new_min, new_max: (new_max - new_min)*(x - np.min(x))/(np.max(x) - np.min(x))  + new_min
        for i in range(path.shape[1]):
            path[:,i] = shift_fun(path[:,i], -0.9*radius,0.9*radius)
  
        
        pathlen = path.shape[0]
        vels = (1/dt)*( path[(np.minimum(np.floor(timesteps/dt) + 1, pathlen-1)).astype(int),:] -
                       path[(np.minimum(np.floor(timesteps/dt), pathlen-2)).astype(int),:])
        real_freqs = (ssp_space.phase_matrix @ vels.T)
        scale_fac = 1/np.max(np.abs(real_freqs))
        vels_scaled = vels*scale_fac
        
        real_ssp = ssp_space.encode(path)
        
        
        n_items = 10
        view_rad = 0.2
        item_locations = 0.9*radius*2*(sspslam.utils.Rd_sampling(n_items, domain_dim,seed=p.seed) - 0.5)
        
        pi_n_neurons = p.pi_n_neurons
        ovc_n_neurons = p.ovc_n_neurons
        other_n_neurons = p.other_n_neurons
        tau = 0.05
        model = nengo.Network(seed=p.seed)
        with model:
            vel_input = nengo.Node(lambda t: vels_scaled[int(t/dt)-1] )#+ noise[int(t/dt)-1]
            init_state = nengo.Node(lambda t: real_ssp[int(t/dt)-1] if t<0.05 else np.zeros(d))
            
            # SLAM
            slam = sspslam.networks.SLAMNetwork(ssp_space, item_locations, path, T, view_rad,
        		pi_n_neurons, ovc_n_neurons, other_n_neurons, grid_cell_neurons, dt, tau, 
                update_thres=0.2, scale_fac = scale_fac, clean_up_method='grid')
            
            nengo.Connection(vel_input, slam.vel_input, synapse=0.01)
            nengo.Connection(init_state, slam.pathintegrator.input, synapse=None)
            
            # vs. PI only
            pathintegrator = sspslam.networks.PathIntegration(ssp_space, pi_n_neurons, tau, 
                          scaling_factor=scale_fac, stable=True)
            nengo.Connection(vel_input,pathintegrator.velocity_input, synapse=0.01)
            nengo.Connection(init_state, pathintegrator.input, synapse=None)

            
            ssp_slam_p  = nengo.Probe(slam.pathintegrator.output, synapse=0.05)
            ssp_pi_p  = nengo.Probe(pathintegrator.output, synapse=0.05)
            
        sim = nengo.Simulator(model)
        sim.run(T)
        
        def get_path_info(ssp_path):
            sims = ssp_path @ slam.sample_ssps.T
            max_sims = np.max(sims,axis=1)
            arg_max_sim = np.argmax(sims,axis=1)
            sim_path_est = slam.sample_points[arg_max_sim,:]
            return max_sims, sim_path_est
        
        pi_sim_to_exact = np.sum(sim.data[ssp_pi_p]*real_ssp, axis=1)
        pi_sim_to_closest, pi_sim_path  = get_path_info(sim.data[ssp_pi_p])
        slam_sim_to_exact = np.sum(sim.data[ssp_slam_p]*real_ssp, axis=1)
        slam_sim_to_closest, slam_sim_path  = get_path_info(sim.data[ssp_slam_p])
        
        return dict(
             sim_pi_ssp = sim.data[ssp_pi_p],
             sim_slam_ssp = sim.data[ssp_slam_p],
             path=path,
             ts= sim.trange(),
             ssp_space=ssp_space,
             scale_fac=scale_fac,
             pi_sim_to_exact = pi_sim_to_exact,
             pi_sim_path = pi_sim_path,
             pi_sim_to_closest= pi_sim_to_closest,
             slam_sim_to_exact = slam_sim_to_exact,
             slam_sim_path = slam_sim_path,
             slam_sim_to_closest= slam_sim_to_closest,
             item_locations=item_locations
        )




if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument('--ssp-dim', dest='ssp_dim', type=int, default=151)
    parser.add_argument('--domain-dim', dest='domain_dim', type=int, default=2)
    parser.add_argument('--pi-n-neurons', dest='pi_n_neurons', type=int, default=700)
    parser.add_argument('--trial-time', dest='trial_time', type=float, default=120)
    parser.add_argument('--path-gen-param', dest='limit', type=float, default=0.08)
    parser.add_argument('--num-trials', dest='num_trials', type=int, default=5)
    parser.add_argument('--data-dir', dest='data_dir', type=str,
                        default='./Github/Semantic-Spiking-Neural-SLAM-2023/data/2d_trials')

    
    args = parser.parse_args()

    random.seed(1)
    seeds = [random.randint(1,100000) for _ in range(args.num_trials)]

    data_path = os.path.join(args.data_dir)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    for seed in seeds:
        params = {'pi_n_neurons':args.pi_n_neurons,
                  'data_format':'npz',
                  'data_dir':data_path,
                  'seed':seed, 
                  'ssp_dim':args.ssp_dim,
                  'domain_dim':args.domain_dim
                  }
        r = SLAMTrial().run(**params)

