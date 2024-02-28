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
        self.param('mem_n_neurons', mem_n_neurons=800)
        self.param('pi_n_neurons',  pi_n_neurons=400)
        self.param('circonv_n_neurons', circonv_n_neurons=100)
        self.param('gc_n_neurons', gc_n_neurons=1000)
    
    def evaluate(self, p):
        domain_dim = p.domain_dim
        bounds = np.tile([-1,1],(domain_dim,1))
        ssp_space = sspslam.HexagonalSSPSpace(domain_dim,ssp_dim=p.ssp_dim, 
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
        
        n_items = 10
        view_rad = 0.2
	obj_locs = 0.9*radius*2*(sspslam.utils.Rd_sampling(n_landmarks, domain_dim, seed=args.seed) - 0.5)
        vec_to_landmarks = obj_locs.reshape(n_landmarks, 1, domain_dim) - path.reshape(1, pathlen, domain_dim)

	real_ssp = ssp_space.encode(path) 
	landmark_ssps = ssp_space.encode(obj_locs)
	lm_space = sspslam.SPSpace(n_landmarks, d, seed=p.seed)

	velocity_func, vel_scaling_factor, is_landmark_in_view, landmark_id_func, landmark_sp_func, landmark_vec_func, landmark_vecssp_func = get_slam_input_functions(ssp_space, lm_space, vels, vec_to_landmarks, view_rad)

	clean_up_method = 'grid'
        
        pi_n_neurons = p.pi_n_neurons
        mem_n_neurons = p.mem_n_neurons
        other_n_neurons = p.other_n_neurons
	gc_n_neurons = p.gc_n_neurons
        tau = 0.05
        model = nengo.Network(seed=p.seed)
        with model:
            vel_input = nengo.Node(velocity_func, label='vel_input')
    	    init_state = nengo.Node(lambda t: real_ssp[int(np.minimum(np.floor(t/dt), n_timesteps-1))] if t<0.05 else np.zeros(d), label='init_state')
            
            # SLAM
            landmark_vec = nengo.Node(landmark_vec_func)
    	    landmark_id = nengo.Node(landmark_id_func)
    	    slam = sspslam.networks.SLAMNetwork(ssp_space, lm_space, view_rad, n_landmarks,
            		pi_n_neurons, mem_n_neurons, circonv_n_neurons, 
                    tau_pi = tau,update_thres=0.2, vel_scaling_factor =vel_scaling_factor,
                    shift_rate=0.1,voja_learning_rate=5e-4, pes_learning_rate=1e-3,
                    clean_up_method=clean_up_method, gc_n_neurons = gc_n_neurons, encoders=None, voja=True, seed=p.seed)
    	    nengo.Connection(landmark_vec, slam.landmark_vec_input, synapse=None)
    	    nengo.Connection(landmark_id, slam.landmark_id_input, synapse=None)
    	    nengo.Connection(vel_input,slam.velocity_input, synapse=None) 
    	    nengo.Connection(init_state, slam.pathintegrator.input, synapse=None)
            
            # vs. PI only
            pathintegrator = sspslam.networks.PathIntegration(ssp_space, pi_n_neurons, tau, 
                  scaling_factor=scale_fac, stable=True)
    	    nengo.Connection(vel_input,pathintegrator.velocity_input, synapse=None)
    	    nengo.Connection(init_state, pathintegrator.input, synapse=None)

            
            slam_output_p  = nengo.Probe(slam.pathintegrator.output, synapse=0.05)
            pi_output_p  = nengo.Probe(pathintegrator.output, synapse=0.05)
            
        sim = nengo.Simulator(model)
    	with sim:
    	    sim.run(T)
        
        slam_sim_path  = ssp_space.decode(sim.data[slam_output_p], 'from-set','grid', 100)
	pi_sim_path  = ssp_space.decode(sim.data[pi_output_p], 'from-set','grid', 100)

	slam_sim = np.sum(sim.data[slam_output_p]*real_ssp,axis=1)/np.linalg.norm(sim.data[slam_output_p],axis=1)
	pi_sim = np.sum(sim.data[pi_output_p]*real_ssp,axis=1)/np.linalg.norm(sim.data[pi_output_p],axis=1)
        
        return dict(
             path = path,
             obj_locs=obj_locs,
             ts= sim.trange(),
             ssp_space=ssp_space,
             slam_output_p = sim.data[slam_output_p],
             pi_output_p = sim.data[pi_output_p],
             pi_sim = pi_sim_to_exact,
             pi_sim_path = pi_sim_path,
             slam_sim = slam_sim_to_exact,
             slam_sim_path = slam_sim_path,
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

