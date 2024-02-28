import numpy as np
import time
import nengo
import matplotlib.pyplot as plt
import tables
import sys, os
import argparse
sys.path.insert(1, os.path.dirname(os.getcwd()))
os.chdir("..")
import sspslam
from sspslam.networks import get_slam_input_functions
import sspslam.utils as utils

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--backend", default="cpu", type=str, help="can be [cpu|ocl|loihi-sim|loihi]")
parser.add_argument("--domain-dim", default=2, type=int,
                    help="Dim of path to generate")
parser.add_argument("--limit", default=0.08, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--T", default=10, type=float, help='The total simulation time in seconds. ')
parser.add_argument("--n-landmarks", default=10, type=int, help='The number of landmarks in the env. ')

parser.add_argument("--pi_n_neurons", default=500, type=int, help="Number of neurons per VCO population in the PI net. Around 400-800 works alright usually, depending on the desired accuracy.")
parser.add_argument("--mem_n_neurons", default=550, type=int, help='Number of neurons in the memory, memory error, and correction populations. These are ensembles representing a ssp_dim-D vector. Having this >= ~10*ssp_dim works well.')
parser.add_argument("--circonv_n_neurons", default=50, type=int, help='Number of neurons per dim in the circular convolution (and product) networks. Around 50-100 works well.')
# parser.add_argument("--dotprod_n_neurons", default=50, type=int)
parser.add_argument("--ssp_dim", default=55, type=int, help='The dim of the SSP and SP vectors. Note that with HexSSPs only certain dims are allowed , 2*(domain_dim+1)*n + 1 for integer n>=1. If you specify a ssp_dim is given that does not satisfy this, then a correct value that is close to the given input will be used' )
parser.add_argument("--length_scale", default=0.1, type=float, help='The length scale of the SSPs.')
parser.add_argument("--save", default=False, type=bool, help='Whether to save the simulation data (needed for the make_plots scripts)')
parser.add_argument("--voja", default=True, type=bool,
                    help="Whether to use Voja.")
parser.add_argument("--cleanup", default=True, type=bool,
                    help="Whether to have a cleanup post-path int net. ")
parser.add_argument("--plot", default=True, type=bool, help='Whether to plot the output upon completion.')
parser.add_argument("--save_plot", default=False, type=bool, help='Whether to save figures. Only works if --plot True.')



args = parser.parse_args()

pi_n_neurons = args.pi_n_neurons
mem_n_neurons = args.mem_n_neurons
circonv_n_neurons=args.circonv_n_neurons
dotprod_n_neurons=args.circonv_n_neurons

backend = args.backend
if backend=='ocl':
    import nengo_ocl



tau=0.05
dt = 0.001
T= args.T
radius = 1
domain_dim = args.domain_dim
path = np.hstack([nengo.processes.WhiteSignal(T, high=args.limit, seed=args.seed+i).run(T,dt=dt) for i in range(domain_dim)])
shift_fun = lambda x, new_min, new_max: (new_max - new_min)*(x - np.min(x))/(np.max(x) - np.min(x))  + new_min
for i in range(path.shape[1]):
    path[:,i] = shift_fun(path[:,i], -0.9*radius,0.9*radius)    
pathlen = path.shape[0]
vels = (1/dt)*( path[(np.minimum(np.floor(timesteps/dt) + 1, pathlen-1)).astype(int),:] -
                path[(np.minimum(np.floor(timesteps/dt), pathlen-2)).astype(int),:])

view_rad = 0.2
n_landmarks = args.n_landmarks
obj_locs = 0.9*radius*2*(sspslam.utils.Rd_sampling(n_landmarks, domain_dim, seed=args.seed) - 0.5)
vec_to_landmarks = obj_locs.reshape(n_landmarks, 1, domain_dim) - path.reshape(1, pathlen, domain_dim)

# Create SSP space
domain_dim = 2 # 2d x space
bounds = np.vstack([np.min(path,axis=0)*1.5, np.max(path,axis=0)*1.5]).T
ssp_space = sspslam.HexagonalSSPSpace(domain_dim,ssp_dim=args.ssp_dim,
                 domain_bounds=bounds, length_scale=args.length_scale, seed=args.seed)
d = ssp_space.ssp_dim
# Targets
real_ssp = ssp_space.encode(path) 
landmark_ssps = ssp_space.encode(obj_locs)

# Create SP landmark space
lm_space = sspslam.SPSpace(n_landmarks, d, seed=args.seed)

# Get input functions for nodes
velocity_func, vel_scaling_factor, is_landmark_in_view, landmark_id_func, landmark_sp_func, landmark_vec_func, landmark_vecssp_func = get_slam_input_functions(ssp_space, lm_space, vels, vec_to_landmarks, view_rad)

clean_up_method = 'grid' if args.cleanup else None

model = nengo.Network(seed=args.seed)
with model:
    vel_input = nengo.Node(velocity_func, label='vel_input')
    init_state = nengo.Node(lambda t: real_ssp[int(np.minimum(np.floor(t/dt), n_timesteps-1))] if t<0.05 else np.zeros(d), label='init_state')

    landmark_vec = nengo.Node(landmark_vec_func)
    landmark_id = nengo.Node(landmark_id_func)
    slam = sspslam.networks.SLAMNetwork(ssp_space, lm_space, view_rad, n_landmarks,
            		pi_n_neurons, mem_n_neurons, circonv_n_neurons, 
                    tau_pi = tau,update_thres=0.2, vel_scaling_factor =vel_scaling_factor,
                    shift_rate=0.1,voja_learning_rate=5e-4, pes_learning_rate=1e-3,
                    clean_up_method=clean_up_method, gc_n_neurons = 0, encoders=None, voja=args.voja, seed=args.seed)
    nengo.Connection(landmark_vec, slam.landmark_vec_input, synapse=None)
    nengo.Connection(landmark_id, slam.landmark_id_input, synapse=None)
    nengo.Connection(vel_input,slam.velocity_input, synapse=None) 
    nengo.Connection(init_state, slam.pathintegrator.input, synapse=None)
    

    slam_output_p = nengo.Probe(slam.pathintegrator.output, synapse=0.05)
    if args.save:
        mem_weights = nengo.Probe(slam.assomemory.conn_out, "weights", sample_every=T)
    if args.plot:
        landmark_p =  nengo.Probe(slam.landmark_ssp_ens.output, synapse=0.05)
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


# # Decode position estimate
sim_path_est  = ssp_space.decode(sim.data[slam_output_p], 'from-set','grid', 100)


if args.save:
    slam_filename = 'slam_' + backend + '_sspdim_' + str(d) + '_pinneurons_' + str(pi_n_neurons) + '_memnneurons_' + str(mem_n_neurons) + '_ccnneurons_' + str(circonv_n_neurons) + '_T_' + str(int(T)) + '_seed_' + str(args.seed) + '.npz'
    np.savez("data/" + slam_filename, ts = sim.trange(),path=path,real_ssp=real_ssp,
                  obj_locs=obj_locs,view_rad=view_rad,
                  slam_sim_out = sim.data[slam_output_p], 
                  slam_sims = np.sum(sim.data[slam_output_p]*real_ssp,axis=1)/np.linalg.norm(sim.data[slam_output_p],axis=1),
                  slam_path = sim_path_est,
                  slam_error =np.sqrt(np.sum((path - sim_path_est)**2,axis=1)),
                  elapsed_time=elapsed_time,elapsed_thread_time=elapsed_thread_time)
        

        
if args.plot:
    # Plot estimate
    fig = plt.figure(figsize=(5.5, 3.5))
    spec = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(spec[0, :])
    ax0.plot(sim.trange(), np.sum(sim.data[slam_output_p]*real_ssp,axis=1)/(np.linalg.norm(sim.data[slam_output_p],axis=1)*np.linalg.norm(real_ssp,axis=1)))
    ax0.set_ylabel("Similarity")
    ax0.set_xlabel("Time (s)")
    ax0.set_xlim([0,T])
    plt.legend()
    ax10 = fig.add_subplot(spec[1, 0])
    ax10.plot(sim.trange(), path[:int(T/dt),0],color='gray')
    ax10.plot(sim.trange(), sim_path_est[:,0],'--',color='k')
    ax10.set_xlim([0,T])
    ax10.set_xlabel('Time (s)')
    ax10.set_ylabel('x')
    ax11 = fig.add_subplot(spec[1, 1])
    ax11.plot(sim.trange(), path[:int(T/dt),1],color='gray')
    ax11.plot(sim.trange(), sim_path_est[:,1],'--',color='k')
    ax11.set_xlim([0,T])
    ax11.set_xlabel('Time (s)')
    ax11.set_ylabel('y')
    fig.suptitle('SLAM output')
    if args.save_plot:
        plt.savefig('figures/slam_' + backend + '.png')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    p = []
    simss = []
    for j in range(n_landmarks):
        simseries = np.sum(sim.data[landmark_p]  * landmark_ssps[j,:].reshape(1,-1), axis=1)
        ax.plot(sim.trange(), simseries, label=str(j),linewidth=1,alpha=0.8)
    #ax.legend()
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel("Similarity to true location", fontsize=8)
    ax.set_xlim(0,T)
    ax.set_title('Obj location from OVCs')
    if args.save_plot:
        plt.savefig('figures/slam_landmark_' + backend + '.png')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    p = []
    simss = []
    for j in range(n_landmarks):
        simseries = np.sum(sim.data[recall_p]  * landmark_ssps[j,:].reshape(1,-1), axis=1)
        ax.plot(sim.trange(), simseries, label=str(j),linewidth=1,alpha=0.8)
    #ax.legend()
    ax.set_xlabel('Time (s)', fontsize=8)
    ax.set_ylabel("Similarity to true location", fontsize=8)
    ax.set_xlim(0,T)
    ax.set_title('Obj location from memory recall')
    if args.save_plot:
        plt.savefig('figures/slam_mem_' + backend + '.png')


