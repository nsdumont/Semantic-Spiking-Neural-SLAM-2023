import numpy as np
import time
import nengo
import matplotlib.pyplot as plt
import tables
import sys,os
sys.path.insert(1, os.path.dirname(os.getcwd()))
os.chdir("..")
import sspslam
import argparse
import sspslam.utils as utils

# Parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--backend", default="loihi-sim", type=str, help="can be [cpu|ocl]")
parser.add_argument("--domain-dim", default=2, type=int,
                    help="Dim of path to generate")
parser.add_argument("--limit", default=0.08, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--T", default=10, type=float, help='The total simulation time in seconds. ')

parser.add_argument("--pi_n_neurons", default=500, type=int, help="Number of neurons per VCO population in the PI net. Around 400-800 works alright usually, depending on the desired accuracy.")
parser.add_argument("--ssp_dim", default=55, type=int, help='The dim of the SSP and SP vectors. Note that with HexSSPs only certain dims are allowed , 2*(domain_dim+1)*n + 1 for integer n>=1. If you specify a ssp_dim is given that does not satisfy this, then a correct value that is close to the given input will be used' )
parser.add_argument("--length_scale", default=0.3, type=float, help='The length scale of the SSPs.')
parser.add_argument("--save", default=False, type=bool, help='Whether to save the simulation data (needed for the make_plots scripts)')
parser.add_argument("--plot", default=True, type=bool, help='Whether to plot the output upon completion.')
parser.add_argument("--save_plot", default=False, type=bool, help='Whether to save figures. Only works if --plot True.')

args = parser.parse_args()

backend = args.backend
if backend=='ocl':
    import nengo_ocl

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


# Create SSP space
domain_dim = 2 # 2d x space
bounds = np.vstack([np.min(path,axis=0)*1.5, np.max(path,axis=0)*1.5]).T
ssp_space = sspslam.HexagonalSSPSpace(domain_dim,ssp_dim=args.ssp_dim,
                 domain_bounds=bounds, length_scale=args.length_scale, seed=args.seed)
d = ssp_space.ssp_dim
real_ssp = ssp_space.encode(path) 

scale_fac = 1/np.max(np.abs(ssp_space.phase_matrix @ vels.T))
vels_scaled = vels*scale_fac


pi_n_neurons = args.pi_n_neurons
tau = 0.05
model = nengo.Network(seed=args.seed)
with model:
    vel_input = nengo.Node(lambda t: vels_scaled[int(np.minimum(np.floor(t/dt), n_timesteps-1))])
    init_state = nengo.Node(lambda t: real_ssp[int(np.minimum(np.floor(t/dt), n_timesteps-1))] if t<0.05 else np.zeros(d))
    
    pathintegrator = sspslam.networks.PathIntegration(ssp_space, pi_n_neurons, tau, 
                  scaling_factor=scale_fac, stable=True)
    nengo.Connection(vel_input,pathintegrator.velocity_input, synapse=None)
    nengo.Connection(init_state, pathintegrator.input, synapse=None)
    
    ssp_p = nengo.Probe(pathintegrator.output, synapse=0.05)
   
nengo.rc['progress']['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'
if backend=='cpu':
    sim = nengo.Simulator(model)
elif backend=='ocl':
    sim = nengo_ocl.Simulator(model) 
  
start = time.thread_time()
start2 = time.time()
with sim:
    sim.run(T)
elapsed_thread_time = time.thread_time() - start
elapsed_time = time.time() - start2


# Decode position estimate
sim_path_est  = ssp_space.decode(sim.data[ssp_p], 'from-set','grid', 100)

if args.save:
    filename = 'pi_' + backend + '_sspdim_' + str(d) + '_pinneurons_' + str(pi_n_neurons) + '_T_' + str(int(T)) + '_seed_' + str(args.seed) + '.npz'
    np.savez("data/" + filename, ts = sim.trange(),path=path,real_ssp=real_ssp,
              pi_sim_out = sim.data[ssp_p],
              pi_sims = np.sum(sim.data[ssp_p]*real_ssp,axis=1)/np.linalg.norm(sim.data[ssp_p],axis=1),
              pi_path = sim_path_est, 
              pi_error =np.sqrt(np.sum((path - sim_path_est)**2,axis=1)),
              elapsed_time=elapsed_time,elapsed_thread_time=elapsed_thread_time)


if args.plot:
    # Plot estimate
    fig = plt.figure(figsize=(5.5, 3.5),dpi=200)
    spec = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(spec[0, :])
    ax0.plot(sim.trange(), np.sqrt(np.sum(sim.data[ssp_p]*real_ssp,axis=1))/np.linalg.norm(sim.data[ssp_p],axis=1))
    ax0.set_ylabel("Similarity")
    ax0.set_xlabel("Time (s)")
    ax0.set_xlim([0,T])
    plt.legend()
    ax10 = fig.add_subplot(spec[1, 0])
    ax10.plot(sim.trange(), path[:,0],color='gray')
    ax10.plot(sim.trange(), sim_path_est[:,0],'--',color='k')
    ax10.set_xlim([0,T])
    ax10.set_xlabel('Time (s)')
    ax10.set_ylabel('x')
    ax11 = fig.add_subplot(spec[1, 1])
    ax11.plot(sim.trange(), path[:,1],color='gray')
    ax11.plot(sim.trange(), sim_path_est[:,1],'--',color='k')
    ax11.set_xlim([0,T])
    ax11.set_xlabel('Time (s)')
    ax11.set_ylabel('y')
    fig.suptitle('PI output')
    if args.save_plot:
        plt.savefig('figures/pi_' + backend + '.png')
    
    plt.figure(dpi=200)
    plt.title('Rover trajectory in environment')
    plt.plot(path[:,0], path[:,1],color='gray', label='Ground truth')
    plt.plot(sim_path_est[10:-1000,0], sim_path_est[10:-1000,1],'--', label='Output of PI', lw=2)
    plt.legend()
    if args.save_plot:
        plt.savefig('figures/pi_traj_' + backend + '.png')



