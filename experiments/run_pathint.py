import numpy as np
import time
import nengo

import sys, os
sys.path.insert(1, os.path.dirname(os.getcwd()))
#os.chdir("..")
import sspslam
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--backend", default="ocl", type=str, help="can be [cpu|ocl|loihi-sim|loihi]")
parser.add_argument("--path-data", default=None, type=str, help='The path and name to path data.' )
parser.add_argument("--data-dt", default=0.001, type=float  )
parser.add_argument("--domain-dim", default=2, type=int,
                    help="Dim of path to generate")
parser.add_argument("--limit", default=0.1, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--T", default=20, type=float, help='The total simulation time in seconds. ')

parser.add_argument("--pi-n-neurons", default=800, type=int, help="Number of neurons per VCO population in the PI net. Around 500-1500 works alright usually, depending on the desired accuracy, limit, and T.")
parser.add_argument("--ssp-dim", default=97, type=int, help='The dim of the SSP and SP vectors. Not used if n-scales is > 0 and use-rand is off. Note that with HexSSPs only certain dims are allowed , 2*(domain_dim+1)*n + 1 for integer n>=1. If you specify a ssp_dim is given that does not satisfy this, then a correct value that is close to the given input will be used' )
parser.add_argument("--n-scales", default=0, type=int, help="The number of scales used to generate HexSSP mapping. Will determine the SSP dim if not zero. Not used when use-rand is selected")
parser.add_argument("--n-rotates", default=3, type=int, help="The number of rotations used to generate HexSSP mapping. Not used when use-rand is selected or if n-scales is 0")

parser.add_argument("--length-scale", default=0.2, type=float, help='The length scale of the SSPs')
parser.add_argument("--save", action='store_true', help='Toggles whether to save the simulation data')
parser.add_argument("--plot", action='store_true', help='Toggles whether to plot the output upon completion.')
parser.add_argument("--save-plot", action='store_true', help='Toggles whether to save figures, only works if --plot True.')
parser.add_argument("--use-rand", action='store_true', help="Toggles if RandSSPs are used, otherwise the script uses HexSSPs")
parser.add_argument("--neuron-type", default='lif', help="The neuron model used. Options are lif, lifrate, relu")

parser.add_argument("--save-dir", default='data', help="Where to save output")
parser.add_argument("--save-name-extra", default='', help="Any extra text to put in save file (eg to note certain options used)")

parser.add_argument("--approx-vel", action='store_true', help="Toggles if velocity is represented via neural population, which adds noise to integration")
parser.add_argument("--vel-n-neurons", default=500, type=int, help="Number of neurons in velocity population, if one is used (see approx-vel)")

args = parser.parse_args()

backend = args.backend
if backend == 'ocl':
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
    import sspslam.utils as utils # the plotting code will not work on inrc

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


tau=0.05
dt = 0.001
radius = 1
if args.path_data is None:
    T= args.T
    domain_dim = args.domain_dim
    path = np.hstack([nengo.processes.WhiteSignal(T, high=args.limit, seed=args.seed+i).run(T,dt=dt) for i in range(domain_dim)])
else:
    path = np.load(os.path.join(os.getcwd(), args.path_data))[:49999,:]
    if args.data_dt != dt:
        path = stretch_trajectory(path, original_dt=args.data_dt, new_dt=dt)
    
    T = path.shape[0]*dt
    domain_dim = path.shape[1]

timesteps = np.arange(0, T, dt)
shift_fun = lambda x, new_min, new_max: (new_max - new_min)*(x - np.min(x))/(np.max(x) - np.min(x))  + new_min
for i in range(path.shape[1]):
    path[:,i] = shift_fun(path[:,i], -0.9*radius,0.9*radius)    
pathlen = path.shape[0]
vels = (1/dt)*np.diff(path, axis=0, prepend=path[0,:].reshape(1,-1))


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
real_ssp = ssp_space.encode(path) 

# scale_fac = 1/np.max(np.linalg.norm(vels,axis=1))
scale_fac = 1/np.max(np.abs(ssp_space.phase_matrix @ vels.T))
vels_scaled = vels * scale_fac

if args.neuron_type=='lifrate':
    neuron_type=nengo.LIFRate()
elif args.neuron_type=='lif':
    neuron_type=nengo.LIF()
elif args.neuron_type=='relu':
    neuron_type=nengo.RectifiedLinear()
    
pi_n_neurons = args.pi_n_neurons
tau = 0.05
model = nengo.Network(seed=args.seed)
model.config[nengo.Ensemble].neuron_type = neuron_type
if 'loihi' in backend:
    model.config[nengo.Ensemble].neuron_type = LoihiLIF()
    # scale_fac=0.5*scale_fac
with model:
    if args.approx_vel:
        vel_syn = 0.01
        _vel_input = nengo.Node(lambda t: vels_scaled[int((t-dt)/dt)], label='vel_input')
        vel_input = nengo.Ensemble(args.vel_n_neurons,domain_dim)
        nengo.Connection(_vel_input, vel_input, synapse=None)
        vel_p = nengo.Probe(vel_input, synapse=vel_syn)
    else:
        vel_syn=None
        vel_input = nengo.Node(lambda t: vels_scaled[int((t-dt)/dt)], label='vel_input')
        
    init_state = nengo.Node(lambda t: real_ssp[int((t-dt)/dt)] if t<0.05 else np.zeros(d))
    pathintegrator = sspslam.networks.PathIntegration(ssp_space, pi_n_neurons, tau,
                      scaling_factor=scale_fac, stable=True,solver_weights=False)
    
    nengo.Connection(vel_input,pathintegrator.velocity_input, synapse=None)
    nengo.Connection(init_state, pathintegrator.input, synapse=None)
    
    ssp_p = nengo.Probe(pathintegrator.output, synapse=0.05)


nengo.rc['progress']['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'
if backend=='cpu':
    sim = nengo.Simulator(model)
elif backend=='ocl':
    sim = nengo_ocl.Simulator(model) 
elif backend=='loihi-sim':
    sim = nengo_loihi.Simulator(model, target='sim')
elif backend=='loihi':
    sim = nengo_loihi.Simulator(model, target='loihi', precompute=True,
                               hardware_options={
                            "snip_max_spikes_per_step": 1000,
                            "allocator": nengo_loihi.hardware.allocators.Greedy()
                        })   
    
start = time.thread_time()
start2 = time.time()
with sim:
    sim.run(T)
elapsed_thread_time = time.thread_time() - start
elapsed_time = time.time() - start2
#sim.data[pathintegrator.recur_conns[1]].weights

# Decode position estimate
if path.shape[0]>100000:
    skip = 100
    sim_path_est = ssp_space.decode(sim.data[ssp_p][::skip,:], 'from-set','grid', 100 if domain_dim<3 else 50)
    ts = sim.trange()[::skip]
    path = path[::skip]
    pi_sim_out = sim.data[ssp_p][::skip]
    real_ssp = real_ssp[::skip]
    pi_sims = np.sum(pi_sim_out*real_ssp,axis=1)/np.linalg.norm(pi_sim_out,axis=1)
    pi_error = np.sqrt(np.sum((path - sim_path_est)**2,axis=1))
else:
    sim_path_est  = ssp_space.decode(sim.data[ssp_p], 'from-set','grid', 100 if domain_dim<3 else 50)
    ts = sim.trange()
    pi_sim_out = sim.data[ssp_p]
    real_ssp = real_ssp
    pi_sims = np.sum(pi_sim_out*real_ssp,axis=1)/np.linalg.norm(pi_sim_out,axis=1)
    pi_error = slam_error = np.sqrt(np.sum((path - sim_path_est)**2,axis=1))

if args.save:
    if args.approx_vel:
        sig_to_noise_ratio = 10 * np.log10(np.var(vels_scaled) / np.var(vels_scaled-sim.data[vel_p]))
    else:
        sig_to_noise_ratio = np.nan
    
    extra_name =   args.save_name_extra 
    if args.domain_dim != 2:
        extra_name = '_dim_'+  str(args.domain_dim)
    if backend != 'cpu':
        extra_name = '_backend_' + backend + extra_name
    if args.approx_vel:
        extra_name = extra_name + f'_velnneurons_{args.vel_n_neurons}'
    filename = f'pi{extra_name}_sspdim_{d}_pinneurons_{pi_n_neurons}_T_{int(T)}_limit_{args.limit}_seed_{args.seed}.npz'
    
    np.savez(os.path.join(os.getcwd(),args.save_dir, filename), ts = ts, path=path,real_ssp=real_ssp,
              pi_sim_out =pi_sim_out,
              pi_sims = pi_sims,
              pi_path = sim_path_est, 
              pi_error =pi_error,
              elapsed_time=elapsed_time,elapsed_thread_time=elapsed_thread_time,
              args=args, sig_to_noise_ratio=sig_to_noise_ratio)


if args.plot:
    import matplotlib.pyplot as plt

    from scipy.spatial.distance import cosine

    fig = plt.figure(figsize=(5.5, 4))
    spec = fig.add_gridspec(3, 2)
    ax0 = fig.add_subplot(spec[0, :])
    ax0.plot(ts, [cosine(pi_sim_out[t],real_ssp[t]) for t in range(len(ts))])
    ax0.set_ylabel("Cosine Error")
    ax0.set_xlabel("Time (s)")
    ax0.set_xlim([0,T])
    ax1 = fig.add_subplot(spec[1, :])
    ax1.plot(ts, pi_error)
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
    fig.suptitle('PI output')
    fig.show()


    if args.save_plot:
        plt.savefig(os.path.join(os.getcwd(),'figures/pi_' + str(args.seed) + '.png'))
    else:
        plt.show()

