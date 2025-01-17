import numpy as np
import time
import nengo

import sys, os
sys.path.insert(1, os.path.dirname(os.getcwd()))
#os.chdir("..")
import sspslam
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from nengo_extras.plot_spikes import (
    cluster,
    merge,
    plot_spikes,
    preprocess_spikes,
    sample_by_variance,
)


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

parser.add_argument("--pi-n-neurons", default=800, type=int, help="Number of neurons per VCO population in the PI net. Around 400-800 works alright usually, depending on the desired accuracy.")
parser.add_argument("--ssp-dim", default=97, type=int, help='The dim of the SSP and SP vectors. Note that with HexSSPs only certain dims are allowed , 2*(domain_dim+1)*n + 1 for integer n>=1. If you specify a ssp_dim is given that does not satisfy this, then a correct value that is close to the given input will be used' )
parser.add_argument("--n-scales", default=3, type=int)
parser.add_argument("--n-rotates", default=3, type=int)

parser.add_argument("--length-scale", default=0.2, type=float, help='The length scale of the SSPs.')
parser.add_argument("--save", action='store_true', help='Whether to save the simulation data (needed for the make_plots scripts)')
parser.add_argument("--use-rand", action='store_true')
parser.add_argument("--neuron-type", default='lif')

parser.add_argument("--save-dir", default='data')
parser.add_argument("--save-name-extra", default='')

parser.add_argument("--approx-vel", action='store_true')
parser.add_argument("--vel-noise", default=0., type=float)
parser.add_argument("--vel-n-neurons", default=500, type=int)

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
# The gif made later will plot every skip # of sim timesteps -- so we only need weights info every skip*sdt
skip = 100
model = nengo.Network(seed=args.seed)
model.config[nengo.Ensemble].neuron_type = neuron_type
if 'loihi' in backend:
    model.config[nengo.Ensemble].neuron_type = LoihiLIF()
    # scale_fac=0.5*scale_fac
with model:
    if args.approx_vel:
        vel_syn = 0.01
        _vel_input = nengo.Node(lambda t: vels_scaled[int((t-dt)/dt)], label='vel_input')
        vel_input = nengo.Ensemble(args.vel_n_neurons,domain_dim)#,noise=vel_noise_process)
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
    vco_p = nengo.Probe(pathintegrator.oscillators.output[3:12], synapse=0.05)
    vco_n1 = nengo.Probe(pathintegrator.oscillators.ea_ensembles[1].neurons[:500], synapse=None, sample_every=skip*dt)
    vco_n2 = nengo.Probe(pathintegrator.oscillators.ea_ensembles[2].neurons[:500], synapse=None, sample_every=skip*dt)
    vco_n3 = nengo.Probe(pathintegrator.oscillators.ea_ensembles[3].neurons[:500], synapse=None, sample_every=skip*dt)



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

###### GIF creation begins

# Plot will start at sim timestep start 
start = 100
sim_ssps =sim.data[ssp_p][start::skip]
sample_ssps, sample_points = ssp_space.get_sample_pts_and_ssps(num_points_per_dim =100)
sim_grid = (sample_ssps.reshape(100,100,-1) @ sim_ssps.T)**2 # The square makes the sim proptional to probability 
X = sample_points[:,0].reshape(100,100)
Y = sample_points[:, 1].reshape(100, 100)
blues_custom = LinearSegmentedColormap.from_list('BluesCustom', ['#ffffff', '#08306b']) # want while as the background color

# Use nengo functions to sort spike trains, makes it look a bit nicer
_, spks1 = merge(*cluster(*sample_by_variance(sim.trange(), sim.data[vco_n1],
                             num=200, filter_width=0.02),
                              filter_width=0.002), num=200)
_, spks2 = merge(*cluster(*sample_by_variance(sim.trange(), sim.data[vco_n2],
                             num=200, filter_width=0.02),
                              filter_width=0.002), num=200)
_, spks3 = merge(*cluster(*sample_by_variance(sim.trange(), sim.data[vco_n3],
                             num=200, filter_width=0.02),
                              filter_width=0.002), num=200)
                              
# The plot of spikes will be sliding, spk_T is the size of the x-axis
spk_T = 10
spk_stop = int(9/(dt*skip)) # only max 9 seconds of spikes shown at once
spk_shape = (int(spk_T/(dt*skip)),spks1.shape[1])

n_ndim=20 
n_vcos = 3 # number of vcos to plot, this is a variable but three is hardcoded other places so work is needed to change this

# create fig and subplots
fig = plt.figure(figsize=(6,(3/6)*6), dpi=300)
fig.patch.set_facecolor('white')
gs = GridSpec(n_vcos, 3, width_ratios=[3,1,2], hspace=0.3)
ax = fig.add_subplot(gs[:,0])
ax.set_facecolor('white')
axs = []
axs2 = []
# set up apperance of subplots 
for i in range(n_vcos):
    _ax = fig.add_subplot(gs[i,1])
    _ax.set_facecolor('white')
    _ax.spines['left'].set_position('zero')
    _ax.spines['bottom'].set_position('zero')
    _ax.spines['right'].set_color('none')
    _ax.spines['top'].set_color('none')
    _ax.tick_params(labelbottom=False, labelleft=False)
    _ax.set_xlim(-1.1,1.1)
    _ax.set_ylim(-1.1, 1.1)
    axs.append(_ax)
    _ax2 = fig.add_subplot(gs[i, 2])
    if i!=n_vcos-1:
        _ax2.tick_params(
            axis='y', which='both', bottom=True,
            top=False, left=False, right=False,
            labelbottom=True, labelleft=False)
        _ax2.set_xticklabels([])
    else:
        _ax2.tick_params(
            axis='y', which='both', bottom=True,
            top=False, left=False, right=False,
            labelbottom=False, labelleft=False)
        _ax2.set_xlabel('Time [s]')

    _ax2.set_xlim(0, spk_T)
    _ax2.set_ylim(0, 200)
    axs2.append(_ax2)

# Initialize plot elements: lines, spike arrays, contours (set to None)
line1, = ax.plot([], [], color='gray', zorder=2) # ground truth path
line2, = ax.plot([], [], '--', color='k', zorder=3) # ssp-pi MLE path
lines = []
lines2 = []
ims = []
cols = [utils.oranges[0], utils.reds[0], utils.purples[0]]
for i in range(n_vcos):
    _line, = axs[i].plot([], [], color=cols[i]) # vco output
    lines.append(_line)
    _line, = axs[i].plot([], [], color=cols[i], alpha=0.2) # vco output a bit in the past (to get fading effect)
    lines2.append(_line)
    _im = axs2[i].imshow(np.zeros(spk_shape).T, cmap='binary', # spike plot, set to all zeros (no spikes) to start
                         vmin=0, vmax=1000,aspect='auto',
                         interpolation='none',
                         extent=[0, spk_T, 0, 200]) # 200 neurons is from the nengo spike sort/cluster code earlier
    ims.append(_im)
ax.set_axis_off()
contour = None


def update(t):
    """Update function for animation."""
    global contour
    if contour:
        for c in contour.collections:
            c.remove()  # Remove old contours to avoid overlap
    contour = ax.contourf(X, Y, sim_grid[:, :, t],
                          cmap=blues_custom, levels=20, zorder=1)
    line1.set_data(path[start:t*skip, 0], path[start:t*skip, 1])
    line2.set_data(sim_path_est[start:t * skip, 0], sim_path_est[start:t * skip, 1])
    for i in range(n_vcos):
        lines[i].set_data(sim.data[vco_p][np.maximum(start,t*skip - 2000):t*skip, 3*(i)],
                          sim.data[vco_p][np.maximum(start,t*skip - 2000):t*skip, 1+3*(i)])
        lines2[i].set_data(sim.data[vco_p][np.maximum(start, t * skip - 3000):t * skip, 3 * (i )],
                          sim.data[vco_p][np.maximum(start, t * skip - 3000):t * skip, 1 + 3 * (i )])

        ims[i].set_extent([max(0, (t - spk_stop)*dt*skip) , max(0, (t - spk_stop)*dt*skip) + spk_T, 0, 200])
        axs2[i].set_xlim(max(0, (t - spk_stop)*dt*skip), max(0, (t - spk_stop)*dt*skip) + spk_T)

    _spk_arr = np.zeros(spk_shape)
    _spk_arr[:min(spk_stop,t)] = spks1[max(0, t - spk_stop):t]
    ims[0].set_array(_spk_arr.T)
    _spk_arr = np.zeros(spk_shape)
    _spk_arr[:min(spk_stop,t)] = spks2[max(0, t - spk_stop):t]
    ims[1].set_array(_spk_arr.T)
    _spk_arr = np.zeros(spk_shape)
    _spk_arr[:min(spk_stop,t)] = spks3[max(0, t - spk_stop):t]
    ims[2].set_array(_spk_arr.T)

# Uncomment and comment out ani and writer lines to just see a plot of output at sim timestep 10 
# update(10)
# fig.show()
# ax.set_title(f"Time: {t*dt}")


# Set up the animation
frames = sim_grid.shape[2]
ani = FuncAnimation(fig, update, frames=frames, interval=50)  # Adjust interval for playback speed

# Save the animation as GIF
writer = PillowWriter(fps=20)  # Adjust fps as needed
ani.save("ssp_pi_output.gif", writer=writer)
plt.close()


