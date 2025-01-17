import numpy as np
import time
import nengo
import sys, os
import argparse
sys.path.insert(1, os.path.dirname(os.getcwd()))
#os.chdir("..")
import sspslam
from sspslam.networks import get_slam_input_functions, get_slam_input_functions2


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--backend", default="ocl", type=str, help="can be [cpu|ocl|loihi-sim|loihi]")
parser.add_argument("--domain-dim", default=2, type=int,
                    help="Dim of path to generate")

parser.add_argument("--path-data", default=None, type=str, help='The path and name to path data. Should be a .npy file with a n-timesteps by domain-dim array. If None (default), a random path will be generated using the seed, T, and limit' )
parser.add_argument("--data-dt", default=0.001, type=float, help='The timestep of the path-data. If greater than the simulation dt (0.001s), interpolation will be used.'  )


parser.add_argument("--limit", default=0.1, type=float, help='If using a random path, this will determine the max. freq. content of the path. See nengo.processes.WhiteSignal for details')
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--T", default=200, type=float, help='The total simulation time in seconds. ')
parser.add_argument("--n-landmarks", default=50, type=int, help='The number of landmarks in the env. ')
parser.add_argument("--view-rad", default=0.2, type=float, help='The view radius of the agent. In this script the agent can see in all directions')
parser.add_argument("--update-thres", default=0.2, type=float, help='The similarity threshold needed for loop closure. If the self-location estimate obtained from the memory network has a similarity greater than update-thres with the PI output, a correction happens.')
parser.add_argument("--shift-rate", default=0.2, type=float, help='The speed of correction/the shift to memory network output during loop closure. Works well around 0.1-0.5.')

parser.add_argument("--pi-n-neurons", default=800, type=int, help="Number of neurons per VCO population in the PI net. Around 500-1500 works alright usually, depending on the desired accuracy.")
parser.add_argument("--mem-n-neurons", default=970, type=int, help='Number of neurons in the memory, memory error, and correction populations. These are ensembles representing a ssp_dim-D vector. Having this >= ~10*ssp_dim works well.')
parser.add_argument("--circonv-n-neurons", default=100, type=int, help='Number of neurons per dim in the circular convolution (and product) networks. Around 50-100 works well.')
parser.add_argument("--gc-n-neurons", default=0, type=int, help="Number of grid cells. If None there will be no grid cells or clean-up step. If 0 there will be clean-up bit it will be stored in a node, not with grid cells. If >use-0, this will add another neural population to the model.")

parser.add_argument("--ssp-dim", default=97, type=int, help='The dim of the SSP and SP vectors. Not used if n-scales is > 0 and use-rand is off. Note that with HexSSPs only certain dims are allowed , 2*(domain_dim+1)*n + 1 for integer n>=1. If you specify a ssp_dim is given that does not satisfy this, then a correct value that is close to the given input will be used' )
parser.add_argument("--n-scales", default=0, type=int, help="The number of scales used to generate HexSSP mapping. Will determine the SSP dim if not zero. Not used when use-rand is selected")
parser.add_argument("--n-rotates", default=3, type=int, help="The number of rotations used to generate HexSSP mapping. Not used when use-rand is selected or if n-scales is 0")

parser.add_argument("--use-rand", action='store_true', help="Toggles if RandSSPs are used, otherwise the script uses HexSSPs")

parser.add_argument("--length-scale", default=0.2, type=float, help='The length scale of the SSPs.')
parser.add_argument("--save", action='store_true', help='Whether to save the simulation data (needed for the make_plots scripts)')
parser.add_argument("--no-voja", action='store_true',  help="Whether to use Voja.")
parser.add_argument("--no-cleanup", action='store_true', 
                    help="Whether to have a cleanup post-path int net. ")
parser.add_argument("--plot", action='store_true',help='Whether to plot the output upon completion.')
parser.add_argument("--save-plot", action='store_true', help='Whether to save figures. Only works if --plot True.')
parser.add_argument("--save-dir", default='data', help="Where to save output")
parser.add_argument("--save-name-extra", default='', help="Any extra text to put in save file (eg to note certain options used)")

parser.add_argument("--single-obj", action='store_true', help="Toggles if the model is constrained to only see a single object at a time")
parser.add_argument("--approx-vel", action='store_true', help="Toggles if velocity is represented via neural population, which adds noise to integration")
parser.add_argument("--vel-n-neurons", default=500, type=int, help="Number of neurons in velocity population, if one is used (see approx-vel)")


args = parser.parse_args()
# 'slam_backend_ocl_sspdim_97_pinneurons_800_memnneurons_970_ccnneurons_100_T_500_seed_{}.npz',

# args.single_obj = True # off worked

pi_n_neurons = args.pi_n_neurons
mem_n_neurons = args.mem_n_neurons
circonv_n_neurons=args.circonv_n_neurons
dotprod_n_neurons=args.circonv_n_neurons

backend = args.backend
if backend=='ocl':
    import nengo_ocl
elif 'loihi' in backend:
    import nengo_loihi
    from nengo_loihi.neurons import LoihiSpikingRectifiedLinear,LoihiLIF
    nengo_loihi.set_defaults()
if backend=='loihi':
    import nxsdk
    import matplotlib as mpl
    mpl.rcParams["backend"] = "Agg"
else:
    import sspslam.utils as utils


def stretch_trajectory(traj, original_dt=0.02, new_dt=0.001):
    n_steps = traj.shape[0]
    total_time = n_steps * original_dt
    n_timesteps = int(total_time / new_dt)
    original_times = np.linspace(0, total_time, n_steps)
    new_times = np.linspace(0, total_time, n_timesteps)
    new_traj = np.zeros((n_timesteps, 2))
    new_traj[:, 0] = np.interp(new_times, original_times, traj[:, 0])
    new_traj[:, 1] = np.interp(new_times, original_times, traj[:, 1])
    return new_traj


tau = 0.05
dt = 0.001
radius = 1
if args.path_data is None:
    T = args.T
    domain_dim = args.domain_dim
    path = np.hstack(
        [nengo.processes.WhiteSignal(T, high=args.limit, seed=args.seed + i).run(T, dt=dt) for i in range(domain_dim)])
else:
    path = np.load(os.path.join(os.getcwd(),args.path_data))[:99999, :]
    if args.data_dt != dt:
        path = stretch_trajectory(path, original_dt=args.data_dt, new_dt=dt)

    T = path.shape[0] * dt
    domain_dim = path.shape[1]


timesteps = np.arange(0, T, dt)
shift_fun = lambda x, new_min, new_max: (new_max - new_min)*(x - np.min(x))/(np.max(x) - np.min(x))  + new_min
for i in range(path.shape[1]):
    path[:,i] = shift_fun(path[:,i], -0.9*radius,0.9*radius)    
pathlen = path.shape[0]
vels = (1/dt)*np.diff(path, axis=0, prepend=path[0,:].reshape(1,-1))


view_rad = args.view_rad
n_landmarks = args.n_landmarks
obj_locs = 0.9*radius*2*(sspslam.utils.Rd_sampling(n_landmarks, domain_dim, seed=args.seed) - 0.5)
vec_to_landmarks = obj_locs[None,:,:] - path[:,None,:]


# Create SSP space
bounds = radius*np.tile([-1,1],(domain_dim,1))
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
if args.single_obj:
    velocity_func, vel_scaling_factor, is_landmark_in_view, landmark_id_func, landmark_sp_func, landmark_vec_func, landmark_vecssp_func = get_slam_input_functions(ssp_space, lm_space, vels, vec_to_landmarks, view_rad)
else:
    velocity_func, vel_scaling_factor, is_landmark_in_view, landmark_id_func, landmark_sp_func, landmark_vec_func, landmark_vecssp_func = get_slam_input_functions2(ssp_space, lm_space, vels, vec_to_landmarks, view_rad)

clean_up_method = None if args.no_cleanup else 'grid'

model = nengo.Network(seed=args.seed)
if 'loihi' in backend:
    model.config[nengo.Ensemble].neuron_type = LoihiLIF()
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

    landmark_vec = nengo.Node(landmark_vecssp_func, label='lm_vecssp_input')
    landmark_id = nengo.Node(landmark_sp_func, label='lm_sp_input')
    is_landmark = nengo.Node(is_landmark_in_view, label='lm_in_view_input')
    
    if 'loihi' in backend:
        slam = sspslam.networks.SLAMLoihiNetwork(ssp_space,lm_space,  view_rad, n_landmarks,
            		pi_n_neurons, mem_n_neurons, circonv_n_neurons, dotprod_n_neurons,
                    vel_input, landmark_vec, landmark_id, is_landmark,
                    tau_pi = tau, update_thres=args.update_thres, vel_scaling_factor = vel_scaling_factor, 
                    shift_rate=0.1, pes_learning_rate=1e-3, 
                    encoders=None, seed=args.seed) #solver=default_solver, 

    else:
        slam = sspslam.networks.SLAMNetwork(ssp_space, lm_space, view_rad, n_landmarks,
                		pi_n_neurons, mem_n_neurons, circonv_n_neurons, 
                        tau_pi = tau,update_thres=args.update_thres, vel_scaling_factor =vel_scaling_factor,
                        shift_rate=args.shift_rate,voja_learning_rate=1e-4,#5e-4 #TODO: have the two learning rates as args
                        pes_learning_rate=5e-3, intercept=0.1, #1e-3 # 0.2
                        clean_up_method=clean_up_method, gc_n_neurons = args.gc_n_neurons, encoders=None, voja=not args.no_voja, seed=args.seed)
        nengo.Connection(landmark_vec, slam.landmark_vec_ssp, synapse=None)
        nengo.Connection(landmark_id, slam.landmark_id_input, synapse=None)
        nengo.Connection(is_landmark, slam.no_landmark_in_view, synapse=None)
        nengo.Connection(vel_input,slam.velocity_input, synapse=vel_syn) 
    nengo.Connection(init_state, slam.pathintegrator.input, synapse=None)
    

    slam_output_p = nengo.Probe(slam.pathintegrator.output, synapse=0.05)
    if args.save and ('loihi' not in backend): # its hard to get learned weights out of loihi model. TODO: figure that out
        mem_weights = nengo.Probe(slam.assomemory.conn_out, "weights", sample_every=T)

nengo.rc['progress']['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'
if backend=='cpu':
    sim = nengo.Simulator(model)
elif backend=='ocl':
    sim = nengo_ocl.Simulator(model)
elif backend=='loihi-sim':
    sim = nengo_loihi.Simulator(model,remove_passthrough=False, target='sim')
elif backend=='loihi':
    sim = nengo_loihi.Simulator(model,remove_passthrough=False, target='loihi', precompute=False,
                               hardware_options={
                            "snip_max_spikes_per_step": 300,
                            "allocator": nengo_loihi.hardware.allocators.Greedy(),
                            "n_chips": 15
                            })  

if 'loihi' in backend:
    util_sum = sim.model.utilization_summary()
    print("Number of LoihiBlocks: " + str(len(util_sum)))
    np.save('data/slam_loihi_utilization_summary.npy', util_sum)  

    all_block_info = np.zeros((len(util_sum), 4))
    for i, s in enumerate(util_sum):
        all_block_info[i,0] = s[s.find(': ')+len(': '):s.rfind('% compartments')]
        all_block_info[i,1] = s[s.find('compartments, ')+len('compartments, '):s.rfind('% in-axons')]
        all_block_info[i,2] = s[s.find('in-axons, ')+len('in-axons, '):s.rfind('% out-axons')]
        all_block_info[i,3] = s[s.find('out-axons, ')+len('out-axons, '):s.rfind('% synapses')]
        
    print("Avg. compartment utilization: {:.2f}%".format(np.mean(all_block_info[:,0])) )
    print("Avg. in-axons utilization: {:.2f}%".format(np.mean(all_block_info[:,1])) )
    print("Avg. out-axons utilization: {:.2f}%".format(np.mean(all_block_info[:,2])) )
    print("Avg. synapses utilization: {:.2f}%".format(np.mean(all_block_info[:,3])) )


start = time.thread_time()
start2 = time.time()
with sim:
    sim.run(T)
elapsed_thread_time = time.thread_time() - start
elapsed_time = time.time() - start2



# # Decode position estimate
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
    if args.approx_vel: # save info about the noise in the velocity signal
        sig_to_noise_ratio = 10 * np.log10(np.var(sim.data[_vel_p]) /  np.var(sim.data[_vel_p]-sim.data[vel_p]))  #in decibels (dB)
    else:
        sig_to_noise_ratio=0
    if  ('loihi' not in backend):
    #TODO: need to get final encoders as well, this wont be correct
        decoders = sim.data[mem_weights][-1].T
        activites  = nengo.builder.ensemble.get_activities(sim.data[slam.assomemory.memory], slam.assomemory.memory, lm_space.vectors)
        landmark_ssps_est = np.dot(activites, decoders)
        landmark_loc_est = ssp_space.decode(landmark_ssps_est, 'from-set', 'grid', 100 if domain_dim<3 else 30)
    else:
        landmark_ssps_est = None
        landmark_loc_est = None
        
    extra_name =  args.save_name_extra
    if args.domain_dim != 2:
        extra_name = '_dim_'+  str(args.domain_dim)
    if backend != 'cpu':
        extra_name = '_backend_' + backend + extra_name
    if args.approx_vel:
        extra_name = extra_name + f'_velnneurons_{args.vel_n_neurons}'
    slam_filename = f'slam_{extra_name}_sspdim_{d}_pinneurons_{pi_n_neurons}_memnneurons_{mem_n_neurons}_ccnneurons_{circonv_n_neurons}_T_{int(T)}_limit_{args.limit}_seed_{args.seed}.npz'
        
    np.savez(os.path.join(os.getcwd(), args.save_dir, slam_filename), 
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
                  args=args, sig_to_noise_ratio=sig_to_noise_ratio)
        

        
if args.plot:
    # Plot estimate
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
        utils.save(fig, os.path.join(os.getcwd(), 'figures/slam_' + backend + '.png'))

