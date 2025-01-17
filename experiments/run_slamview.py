import numpy as np
import time
import nengo
import sys, os
import argparse
sys.path.insert(1, os.path.dirname(os.getcwd()))
#os.chdir("..")
import sspslam
from sspslam.networks import get_slamview_input_functions
import sspslam.utils as utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--backend", default="ocl", type=str, help="can be [cpu|ocl|loihi-sim|loihi]")
parser.add_argument("--domain-dim", default=2, type=int,
                    help="Dim of path to generate")
parser.add_argument("--limit", default=0.1, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--T", default=200, type=float, help='The total simulation time in seconds. ')
parser.add_argument("--n-landmarks", default=100, type=int, help='The number of landmarks in the env. ')
parser.add_argument("--update-thres", default=0.2, type=float, help='')
parser.add_argument("--view-rad", default=0.2, type=float, help='')
parser.add_argument("--shift-rate", default=0.02, type=float, help='')

parser.add_argument("--pi-n-neurons", default=800, type=int, help="Number of neurons per VCO population in the PI net. Around 400-800 works alright usually, depending on the desired accuracy.")
parser.add_argument("--mem-n-neurons", default=970, type=int, help='Number of neurons in the memory, memory error, and correction populations. These are ensembles representing a ssp_dim-D vector. Having this >= ~10*ssp_dim works well.')
parser.add_argument("--circonv-n-neurons", default=50, type=int, help='Number of neurons per dim in the circular convolution (and product) networks. Around 50-100 works well.')
parser.add_argument("--ssp-dim", default=97, type=int, help='The dim of the SSP and SP vectors. Note that with HexSSPs only certain dims are allowed , 2*(domain_dim+1)*n + 1 for integer n>=1. If you specify a ssp-dim is given that does not satisfy this, then a correct value that is close to the given input will be used' )
parser.add_argument("--length-scale", default=0.3, type=float, help='The length scale of the SSPs.')
parser.add_argument("--n-scales", default=4, type=int)
parser.add_argument("--n-rotates", default=4, type=int)

parser.add_argument("--save-dir", default='data/slamview_results/')
parser.add_argument("--use-rand", action='store_true')

parser.add_argument("--save", action='store_true', help='Whether to save the simulation data (needed for the make_plots scripts)')
parser.add_argument("--no-voja", action='store_true',  help="Whether to use Voja.")
parser.add_argument("--no-cleanup", action='store_true', 
                    help="Whether to have a cleanup post-path int net. ")
parser.add_argument("--plot", action='store_true',help='Whether to plot the output upon completion.')
parser.add_argument("--save-plot", action='store_true', help='Whether to save figures. Only works if --plot True.')
parser.add_argument("--save-name-extra", default='')

parser.add_argument("--approx-vel", action='store_true')
parser.add_argument("--single-obj", action='store_true')
parser.add_argument("--vel-noise", default=0., type=float)
parser.add_argument("--vel-n-neurons", default=500, type=int)
args = parser.parse_args()

pi_n_neurons = args.pi_n_neurons
mem_n_neurons = args.mem_n_neurons
circonv_n_neurons=args.circonv_n_neurons



backend = args.backend
if backend=='ocl':
    os.environ['PYOPENCL_CTX'] = '0'
    import nengo_ocl



tau=0.05
dt = 0.001
T= args.T
timesteps = np.arange(0, T, dt)
radius = 1
domain_dim = args.domain_dim
path = np.hstack([nengo.processes.WhiteSignal(T, high=args.limit, seed=args.seed+i).run(T,dt=dt) for i in range(domain_dim)])
shift_fun = lambda x, new_min, new_max: (new_max - new_min)*(x - np.min(x))/(np.max(x) - np.min(x))  + new_min
for i in range(path.shape[1]):
    path[:,i] = shift_fun(path[:,i], -0.9*radius,0.9*radius)    
pathlen = path.shape[0]
vels = (1/dt)*( path[(np.minimum(np.floor(timesteps/dt) + 1, pathlen-1)).astype(int),:] -
                path[(np.minimum(np.floor(timesteps/dt), pathlen-2)).astype(int),:])

view_rad = args.view_rad
n_landmarks = args.n_landmarks
obj_locs = 0.9*radius*2*(sspslam.utils.Rd_sampling(n_landmarks, domain_dim, seed=args.seed) - 0.5)
vec_to_landmarks = obj_locs[None,:,:] - path[:,None,:]


# Create SSP space
bounds = np.vstack([np.min(path,axis=0)*1.5, np.max(path,axis=0)*1.5]).T
if args.use_rand:
    ssp_space = sspslam.RandomSSPSpace(domain_dim,ssp_dim=args.ssp_dim,
                     domain_bounds=bounds, length_scale=args.length_scale, seed=args.seed)
else:
    if args.n_scales > 0:
        ssp_space = sspslam.HexagonalSSPSpace(domain_dim,n_scales=args.n_scales,n_rotates=args.n_rotates,
                         domain_bounds=bounds, length_scale=args.length_scale, seed=args.seed)
    else:
        ssp_space = sspslam.HexagonalSSPSpace(domain_dim,ssp_dim=args.ssp_dim,
                         domain_bounds=bounds, length_scale=args.length_scale, seed=args.seed)
d = ssp_space.ssp_dim
# Targets
real_ssp = ssp_space.encode(path) 
landmark_ssps = ssp_space.encode(obj_locs)

# Create SP landmark space
lm_space = sspslam.SPSpace(n_landmarks, d, seed=args.seed)

# Get input functions for nodes
velocity_func, vel_scaling_factor, is_landmark_in_view, landmark_func = get_slamview_input_functions(ssp_space, lm_space, vels, vec_to_landmarks, view_rad)

clean_up_method = 'grid' #if args.cleanup else None

model = nengo.Network(seed=args.seed)
with model:
    if args.approx_vel:
        vel_syn = 0.01
        _vel_input = nengo.Node(velocity_func, label='vel_input')
        vel_input = nengo.Ensemble(args.vel_n_neurons,domain_dim)#,noise=vel_noise_process)
        nengo.Connection(_vel_input, vel_input, synapse=None)
        vel_p = nengo.Probe(vel_input, synapse=vel_syn)
        _vel_p = nengo.Probe(_vel_input, synapse=None)
    else:
        vel_syn=None
        vel_input = nengo.Node(velocity_func, label='vel_input')
    
    
    init_state = nengo.Node(lambda t: real_ssp[int((t-dt)/dt)] if t<0.05 else np.zeros(d), label='init_state')

    landmark_input = nengo.Node(landmark_func)
    landmark_inview = nengo.Node(is_landmark_in_view)
    slam = sspslam.networks.SLAMViewNetwork(ssp_space, lm_space, view_rad, n_landmarks,
            		pi_n_neurons, mem_n_neurons, circonv_n_neurons, 
                    tau_pi = tau,update_thres=args.update_thres, vel_scaling_factor =vel_scaling_factor,
                    shift_rate=args.shift_rate,voja_learning_rate=5e-4, pes_learning_rate=1e-3,
                    clean_up_method=clean_up_method, gc_n_neurons = 0, encoders=None, voja=not args.no_voja, seed=args.seed)
    nengo.Connection(landmark_input, slam.view_input, synapse=None)
    nengo.Connection(landmark_inview, slam.no_landmark_in_view, synapse=None)
    nengo.Connection(vel_input,slam.velocity_input, synapse=None) 
    nengo.Connection(init_state, slam.pathintegrator.input, synapse=None)
    

    slam_output_p = nengo.Probe(slam.pathintegrator.output, synapse=0.05)
    if args.save:
        mem_weights = nengo.Probe(slam.assomemory.conn_out, "weights", sample_every=T)
    if args.plot:
        recall_p = nengo.Probe(slam.assomemory.recall, synapse=0.05)
    # posest_p = nengo.Probe(slam.position_estimate.output, synapse=0.05)
    # correct_p = nengo.Probe(slam.correction, synapse=0.05)
    # thres_p = nengo.Probe(slam.threshold, synapse=0.05)
    
     
   
nengo.rc['progress']['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'
if backend=='cpu':
    sim = nengo.Simulator(model)
elif backend=='ocl':
    import nengo_ocl
    sim = nengo_ocl.Simulator(model)


start = time.thread_time()
start2 = time.time()
with sim:
    sim.run(T)
elapsed_thread_time = time.thread_time() - start
elapsed_time = time.time() - start2


#
      
if path.shape[0]>100000:
    skip = 100
    sim_path_est  = ssp_space.decode(sim.data[slam_output_p][::skip,:], 'from-set','grid', 100 if domain_dim<3 else 30)
    ts = sim.trange()[::skip]
    path = path[::skip]
    slam_sim_out = sim.data[slam_output_p][::skip]
    real_ssp = real_ssp[::skip]
    slam_sims = np.sum(slam_sim_out*real_ssp,axis=1)/np.maximum(1e-6,np.linalg.norm(slam_sim_out,axis=1))
    slam_error = np.sqrt(np.sum((path - sim_path_est)**2,axis=1))
else:
    sim_path_est  = ssp_space.decode(sim.data[slam_output_p], 'from-set','grid', 100 if domain_dim<3 else 30)
    ts = sim.trange()
    slam_sim_out = sim.data[slam_output_p]
    real_ssp = real_ssp
    slam_sims = np.sum(slam_sim_out*real_ssp,axis=1)/np.maximum(1e-6,np.linalg.norm(slam_sim_out,axis=1))
    slam_error = np.sqrt(np.sum((path - sim_path_est)**2,axis=1))


if args.save:
    if args.approx_vel:
        sig_to_noise_ratio = 10 * np.log10(np.var(sim.data[_vel_p]) / np.var(sim.data[_vel_p]-sim.data[vel_p])) #in decibels (dB)
    else:
        sig_to_noise_ratio=0
    if  ('loihi' not in backend):
        decoders = sim.data[mem_weights][-1].T
        activites  = nengo.builder.ensemble.get_activities(sim.data[slam.assomemory.memory], slam.assomemory.memory, lm_space.vectors)
        landmark_ssps_est = np.dot(activites, decoders)
        landmark_loc_est = ssp_space.decode(landmark_ssps_est, 'from-set', 'grid', 100 if domain_dim<3 else 30)
    else:
        landmark_ssps_est = None
        landmark_loc_est = None
        
    extra_name = args.save_name_extra
    if args.domain_dim != 2:
        extra_name = '_dim_'+  str(args.domain_dim)
    if backend != 'cpu':
        extra_name = '_backend_' + backend + extra_name
    if args.approx_vel:
        extra_name = extra_name + f'_velnneurons_{args.vel_n_neurons}'
    slam_filename = f'slamview_{extra_name}_sspdim_{d}_pinneurons_{pi_n_neurons}_memnneurons_{mem_n_neurons}_ccnneurons_{circonv_n_neurons}_T_{int(T)}_limit_{args.limit}_seed_{args.seed}.npz'
    np.savez(os.path.join(os.getcwd(),args.save_dir + slam_filename), 
                 timesteps=timesteps,
                 ts = ts, path=path,real_ssp=real_ssp,
                  obj_locs=obj_locs,view_rad=view_rad,
                  slam_sim_out = slam_sim_out, 
                  slam_sims = slam_sims,
                  slam_path = sim_path_est,
                  slam_error = slam_error,
                  landmark_ssps_est = landmark_ssps_est,
                  landmark_loc_est = landmark_loc_est,
                  elapsed_time=elapsed_time,elapsed_thread_time=elapsed_thread_time,
                  args=args,sig_to_noise_ratio=sig_to_noise_ratio)
        
if args.plot:
    import matplotlib.pyplot as plt

    from scipy.spatial.distance import cosine

    fig = plt.figure(figsize=(5.5, 4))
    spec = fig.add_gridspec(3, 2)
    ax0 = fig.add_subplot(spec[0, :])
    ax0.plot(ts, [cosine(slam_sim_out[t],real_ssp[t]) for t in range(len(ts))])
    ax0.set_ylabel("Cosine Error")
    ax0.set_xlabel("Time (s)")
    ax0.set_xlim([0,T])
    ax1 = fig.add_subplot(spec[1, :])
    ax1.plot(ts, slam_error)
    ax1.set_ylabel("Distance Error")
    ax1.set_xlabel("Time (s)")
    ax1.set_xlim([0,T])
    
    
    ax10 = fig.add_subplot(spec[2, 0])
    ax10.plot(ts, path[:,0],color='gray')
    ax10.plot(ts, sim_path_est[:,0],'--',color='k')
    ax10.set_xlim([0,T])
    ax10.set_xlabel('Time (s)')
    ax10.set_ylabel('x')
    ax11 = fig.add_subplot(spec[2, 1])
    ax11.plot(ts, path[:,1],color='gray')
    ax11.plot(ts, sim_path_est[:,1],'--',color='k')
    ax11.set_xlim([0,T])
    ax11.set_xlabel('Time (s)')
    ax11.set_ylabel('y')
    fig.suptitle('SLAM output')
    plt.show()
    if args.save_plot:
        plt.savefig(os.path.join(os.getcwd(),'figures/slamview_' + backend + '.png'))


