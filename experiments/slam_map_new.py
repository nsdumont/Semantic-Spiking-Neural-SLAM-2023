import numpy as np
import nengo
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys, os
sys.path.insert(1, os.path.dirname(os.getcwd()))
os.chdir("..")
import sspslam
import nengo_ocl
from sspslam.networks import get_slam_input_functions
import sspslam.utils as utils
import matplotlib.gridspec as gridspec

seed=0
# Create SSP space
domain_dim = 2 # 2d x space
radius = 1 # radius of x space (need for path generation and decoding)
bounds = radius*np.tile([-1,1],(domain_dim,1))
ssp_space = sspslam.HexagonalSSPSpace(domain_dim,n_scales=8,n_rotates=5,
                 domain_bounds=1.2*bounds, length_scale=0.3)

d = ssp_space.ssp_dim

# Generate test path
T = 60
dt = 0.001
timesteps = np.arange(0, T, dt)
path = np.hstack([nengo.processes.WhiteSignal(T, high=.05, seed=0).run(T,dt=dt),
                  nengo.processes.WhiteSignal(T, high=.05, seed=1).run(T,dt=dt)])


 
shift_fun = lambda x, new_min, new_max: (new_max - new_min)*(x - np.min(x))/(np.max(x) - np.min(x))  + new_min
path[:,0] = shift_fun(path[:,0], -0.9*radius,0.9*radius)
path[:,1] = shift_fun(path[:,1], -0.9*radius,0.9*radius)
real_ssp = ssp_space.encode(path) 
real_inv_ssp = ssp_space.invert(real_ssp)


# Generate radnom object locations
n_items = 4#5
view_rad = 0.3
#item_locations = 0.75*2*(sspslam.utils.Rd_sampling(n_items, domain_dim,seed=0) - 0.5)
item_locations = np.array([[-0.8,-0.5],
                           [ 0.0, -0.6],
                           [-0.2 ,  0.2], [ 0.6,  0.2]])
item_ssps = ssp_space.encode(item_locations)
vec_to_landmarks = item_locations[None,:,:] - path[:,None,:]

item_shapes = ['^','^','s','s']
shape_values, shape_counts = np.unique(item_shapes, return_counts=True)
n_shapes = len(shape_counts)
item_cols = [utils.blues[1],utils.oranges[1],utils.blues[1],utils.oranges[1]]
col_values, col_counts = np.unique(item_cols, return_counts=True)
n_cols = len(col_counts)

shape_sps = nengo.dists.UniformHypersphere(surface=True).sample(n_shapes, d, rng=np.random.RandomState(seed=0))
col_sps = nengo.dists.UniformHypersphere(surface=True).sample(n_cols, d, rng=np.random.RandomState(seed=10))

shape_idx = [np.where(shape_values==v)[0].item() for v in item_shapes]
col_idx = [np.where(col_values==v)[0].item() for v in item_cols]
item_sps = ssp_space.bind(shape_sps[shape_idx,:], col_sps[col_idx,:])

# wall_boundaries = np.array([[[-1.1,-0.95],[0.5,1.1]], [[-1,-0.4],[0.95,1.1]]])
wall_boundaries = np.array([[[-1.1,-0.95],[0.2,1.1]],
                            [[-0.95,-0.0],[0.95,1.1]]])
n_walls = wall_boundaries.shape[0]
wall_sps = nengo.dists.UniformHypersphere(surface=True).sample(n_walls, d, rng=np.random.RandomState(seed=20))


from scipy.integrate import dblquad

wall_ssps = np.zeros((n_walls,d))
for j in range(n_walls):
    for i in range(d):
        wall_ssps[j,i]=dblquad(lambda y,x: ssp_space.encode(np.array([x,y]))[0,i], 
                          wall_boundaries[j,0,0],wall_boundaries[j,0,1],
                          wall_boundaries[j,1,0], wall_boundaries[j,1,1], epsabs= 1e-4)[0]
    
wall_ssps = ssp_space.normalize(wall_ssps)
wall_names = ['WALL1','WALL2']
# im=ssp_space.similarity_plot(wall_ssps[0,:]+wall_ssps[1,:]);
# plt.colorbar(im)


# Plot env
plt.figure(figsize=(2.5,2.5))
ax = plt.gca()
plt.plot(path[:,0],path[:,1],linewidth=2,color='k'); plt.xlim([-1.1,1.1]);plt.ylim([-1.1,1.1])
#syms = ['.','*','^','s','X','D','p','H','>','<','d','|','+',]
for i in range(n_items):
    plt.plot(item_locations[i,0],item_locations[i,1], item_shapes[i],markersize=6, 
             markeredgewidth=0.5, markerfacecolor=item_cols[i])
sspslam.utils.circles(item_locations[:,0],item_locations[:,1], view_rad, '#525252', alpha=0.4, lw=5, edgecolor='none',linewidth=0)
for i in np.arange(n_walls):
    # plt.plot(wall_boundaries[i,0,:],wall_boundaries[i,1,0]*np.ones(2), "k")
    # plt.plot(wall_boundaries[i,0,:],wall_boundaries[i,1,1]*np.ones(2), "k")
    # plt.plot(wall_boundaries[i,0,0]*np.ones(2),wall_boundaries[i,1,:], "k")
    # plt.plot(wall_boundaries[i,0,1]*np.ones(2),wall_boundaries[i,1,:], "k")
    sspslam.utils.circles(np.tile(wall_boundaries[i][0,:],2),
                          np.repeat(wall_boundaries[i][1,:],2), view_rad, '#B9B9B9', alpha=1, lw=5, edgecolor='none',linewidth=0)
    ax.add_patch(Rectangle((wall_boundaries[i,0,0] - view_rad,wall_boundaries[i,1,0]),
                       wall_boundaries[i,0,1]-wall_boundaries[i,0,0] + 2*view_rad,
                       wall_boundaries[i,1,1]-wall_boundaries[i,1,0],
                       linewidth=0,edgecolor='#B9B9B9',facecolor='#B9B9B9'))
    ax.add_patch(Rectangle((wall_boundaries[i,0,0] ,wall_boundaries[i,1,0]- view_rad),
                       wall_boundaries[i,0,1]-wall_boundaries[i,0,0] ,
                       wall_boundaries[i,1,1]-wall_boundaries[i,1,0]+ 2*view_rad,
                       linewidth=0,edgecolor='#B9B9B9',facecolor='#B9B9B9'))
for i in np.arange(n_walls):
    ax.add_patch(Rectangle((wall_boundaries[i,0,0],wall_boundaries[i,1,0]),
                       wall_boundaries[i,0,1]-wall_boundaries[i,0,0],
                       wall_boundaries[i,1,1]-wall_boundaries[i,1,0],
                       linewidth=0,edgecolor='k',facecolor='k'))
plt.xlim([-1.2,1.2])
plt.ylim([-1.2,1.2])
plt.title("SLAM env")
plt.show()
#plt.savefig('a_slam_env.pdf')

pathlen = path.shape[0]
vels = (1/dt)*np.diff(path, axis=0, prepend=path[0,:].reshape(1,-1))

real_ssp = ssp_space.encode(path) 
lm_space = sspslam.SPSpace(n_items + n_walls, d, seed=seed, vectors=np.vstack([item_sps,wall_sps]))

# velocity_func, vel_scaling_factor, is_landmark_in_view, landmark_id_func, landmark_sp_func, landmark_vec_func, landmark_vecssp_func = get_slam_input_functions(ssp_space, lm_space, vels, vec_to_landmarks, view_rad)
clean_up_method = 'grid'

landmark_sps = lm_space.vectors
vel_scaling_factor = 1 / np.max(np.abs(ssp_space.phase_matrix @ vels.T))
vels_scaled = vels * vel_scaling_factor
velocity_func = lambda t: vels_scaled[int(np.minimum(np.floor(t / dt), pathlen - 2))]


def construct_vec_to_walls_vectorized(path, wall_boundaries):
    n_timesteps, _ = path.shape
    n_walls, _, _ = wall_boundaries.shape

    # Expand dimensions for broadcasting
    path_expanded = np.expand_dims(path, axis=1)  # Shape: (n_timesteps, 1, 2)
    walls_expanded = np.expand_dims(wall_boundaries, axis=0)  # Shape: (1, n_walls, 2, 2)

    # Extract wall boundaries
    x1 = walls_expanded[:, :, 0, 0]  # Shape: (1, n_walls)
    x2 = walls_expanded[:, :, 0, 1]  # Shape: (1, n_walls)
    y1 = walls_expanded[:, :, 1, 0]  # Shape: (1, n_walls)
    y2 = walls_expanded[:, :, 1, 1]  # Shape: (1, n_walls)

    # Clamp x and y coordinates
    clamped_x = np.clip(path_expanded[:, :, 0], x1, x2)  # Shape: (n_timesteps, n_walls)
    clamped_y = np.clip(path_expanded[:, :, 1], y1, y2)  # Shape: (n_timesteps, n_walls)

    # Combine clamped coordinates
    closest_points = np.stack((clamped_x, clamped_y), axis=-1)  # Shape: (n_timesteps, n_walls, 2)

    # Compute vectors to walls
    vec_to_walls = closest_points - path_expanded  # Shape: (n_timesteps, n_walls, 2)

    return vec_to_walls

vec_to_walls = construct_vec_to_walls_vectorized(path, wall_boundaries)

def landmark_id_func(t):
    current_vecs = vec_to_landmarks[int((t-dt)/dt), :, :]
    current_vecs_to_walls = vec_to_walls[int((t-dt)/dt), :, :]
    current_vecs = np.vstack([current_vecs, current_vecs_to_walls])
    dists = np.linalg.norm(current_vecs, axis=1)
    if np.all(dists > view_rad):
        return -1
    else:
        return np.argmin(dists)
def landmark_sp_func(t):
    cur_id = landmark_id_func(t)
    if cur_id < 0:
        return np.zeros(d)
    else:
        return landmark_sps[cur_id]


# At time t, if a landmark is in view return the SSP representation of the vector to the landmark (from the input data)
def landmark_vecssp_func(t):
    cur_id = landmark_id_func(t)
    if cur_id < 0:
        return np.zeros(d)
    elif cur_id < n_items:
        return ssp_space.encode(
            vec_to_landmarks[int((t-dt)/dt), cur_id, :]).flatten()
    else:
        return ssp_space.bind(real_inv_ssp[int((t-dt)/dt), :],
                                wall_ssps[cur_id-n_items,:]).flatten()



# Is an item in view at time t? if no return 10 else return 0. Used for inhibiting neural populations
def is_landmark_in_view(t):
    cur_id = landmark_id_func(t)
    if cur_id < 0:
        return 10
    else:
        return 0

pi_n_neurons = 250
mem_n_neurons = 10*d
circonv_n_neurons = 100
intercept = (np.dot(item_ssps, item_ssps.T) - np.eye(n_items)).flatten().max()
model = nengo.Network(seed=seed)
with model:
    vel_input = nengo.Node(velocity_func, label='vel_input')
    init_state = nengo.Node(lambda t: real_ssp[int((t - dt) / dt)] if t < 0.05 else np.zeros(d), label='init_state')

    landmark_vec = nengo.Node(landmark_vecssp_func)
    landmark_id = nengo.Node(landmark_sp_func)
    is_landmark = nengo.Node(is_landmark_in_view)
    slam = sspslam.networks.SLAMNetwork(ssp_space, lm_space, view_rad, n_items + n_walls,
                                        pi_n_neurons, mem_n_neurons, circonv_n_neurons,
                                        tau_pi=0.05, update_thres=0.2, vel_scaling_factor=vel_scaling_factor,
                                        shift_rate=0.1, voja_learning_rate=5e-4,
                                        pes_learning_rate=5e-3,
                                        clean_up_method=clean_up_method,
                                        gc_n_neurons=0, encoders=None, voja=True,
                                        seed=seed)
    # slam = sspslam.networks.SLAMNetwork(ssp_space, lm_space, view_rad, n_items + n_walls,
    #                                     pi_n_neurons, mem_n_neurons, circonv_n_neurons,
    #                                     tau_pi=0.05, update_thres=0.2, vel_scaling_factor=vel_scaling_factor,
    #                                     shift_rate=0.1, voja_learning_rate=5e-4, pes_learning_rate=1e-3,
    #                                     clean_up_method=clean_up_method, gc_n_neurons=1000, encoders=None, voja=True,
    #                                     seed=seed)

    nengo.Connection(landmark_vec, slam.landmark_vec_ssp, synapse=None)
    nengo.Connection(landmark_id, slam.landmark_id_input, synapse=None)
    nengo.Connection(is_landmark, slam.no_landmark_in_view, synapse=None)
    nengo.Connection(vel_input, slam.velocity_input, synapse=None)
    nengo.Connection(init_state, slam.pathintegrator.input, synapse=None)

    pathintegrator = sspslam.networks.PathIntegration(ssp_space, pi_n_neurons, 0.05,
                                                      scaling_factor=vel_scaling_factor, stable=True,
                                                      solver_weights=False)

    nengo.Connection(vel_input, pathintegrator.velocity_input, synapse=None)
    nengo.Connection(init_state, pathintegrator.input, synapse=None)

    invassomemory = sspslam.networks.AssociativeMemory(mem_n_neurons, d, d, np.min([intercept,0.1]),
                                        voja_learning_rate=5e-4,
                                        pes_learning_rate=1e-2,
                                        voja=True, encoders=ssp_space.sample_grid_encoders(mem_n_neurons), radius=1.3)

    nengo.Connection(slam.landmark_ssp_ens.output, invassomemory.key_input, synapse=0.05)
    nengo.Connection(landmark_id, invassomemory.value_input, synapse=None)
    nengo.Connection(is_landmark, invassomemory.learning, synapse=None)


    ssp_p = nengo.Probe(slam.pathintegrator.output, synapse=0.05)
    ssp_pi_p = nengo.Probe(pathintegrator.output, synapse=0.05)
    #gate_p = nengo.Probe(slam.gate, synapse=None)
    newpos_p = nengo.Probe(slam.position_estimate.output, synapse=0.05)
    objssp_p = nengo.Probe(slam.landmark_ssp_ens.output, synapse=0.05)
    recall_p = nengo.Probe(slam.assomemory.recall, synapse=0.05)
    isitem_p = nengo.Probe(is_landmark, synapse=None)
    mem_weights = nengo.Probe(slam.assomemory.conn_out, "weights",sample_every=1000*dt)#, sample_every=T/10)
    meminv_weights = nengo.Probe(invassomemory.conn_out, "weights", sample_every=T)
    mem_encoders = nengo.Probe(slam.assomemory.conn_in.learning_rule, "scaled_encoders", sample_every=1000 * dt)
    meminv_encoders = nengo.Probe(invassomemory.conn_in.learning_rule, "scaled_encoders", sample_every=T)

nengo.rc['progress']['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'

sim = nengo_ocl.Simulator(model)
with sim:
    sim.run(T)


# np.savez('data/slam_results/slam_map_walls.npz', sim_ssp=sim.data[ssp_p],
#           ssp_space=ssp_space,ts = sim.trange(),path=path,
#           item_locations=item_locations,item_sps=landmark_sps,
#           sim_objssp = sim.data[objssp_p],
#           sim_recall = sim.data[recall_p],new_pos = sim.data[newpos_p],
#           mem_weights = sim.data[mem_weights],
#           meminv_weights = sim.data[meminv_weights],
#          mem_encoders =sim.data[mem_encoders],
#          meminv_encoders = sim.data[meminv_encoders],)

slam_sims = np.sum(real_ssp[::10,:] * sim.data[ssp_p][::10,:],axis=-1)/np.linalg.norm(sim.data[ssp_p][::10,:],axis=-1)
pi_sims = np.sum(real_ssp[::10,:] * sim.data[ssp_pi_p][::10,:],axis=-1)/np.linalg.norm(sim.data[ssp_pi_p][::10,:],axis=-1)
slam_path = ssp_space.decode(sim.data[ssp_p][::100,:])
pi_path = ssp_space.decode(sim.data[ssp_pi_p][::100,:])


fig,axs=plt.subplots(2,1,figsize=(4,3))
axs[0].plot(sim.trange()[::10], 1-slam_sims,label='ssp-slam')
axs[0].plot(sim.trange()[::10], 1-pi_sims,label='ssp-pi')
axs[0].set_ylabel("Cosine error")
axs[0].legend()
axs[1].plot(sim.trange()[::100], np.linalg.norm(path[::100,:]-slam_path,axis=-1),label='ssp-slam')
axs[1].plot(sim.trange()[::100], np.linalg.norm(path[::100,:]-pi_path,axis=-1),label='ssp-pi')
axs[1].set_ylabel("Distance error")
fig.show()
############################
### Object queries of memory at the end
#############################

def plot_env(ax):
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

    ax.plot(path[:, 0], path[:, 1], linewidth=2, color='k', zorder=5)
    ax.plot(path[0, 0], path[0, 1], markersize=10, color='k', marker='*', markeredgewidth=0, zorder=6)
    for idx in [5000, path.shape[0] - 5000]:
        ax.arrow(path[idx, 0], path[idx, 1],
                     path[idx + 1, 0] - path[idx, 0],
                     path[idx + 1, 1] - path[idx, 1], zorder=6,
                     shape='full', overhang=0.1, lw=0, length_includes_head=True, head_width=.1, color='k')

    for i in range(n_items):
        ax.plot(item_locations[i, 0], item_locations[i, 1], item_shapes[i], markersize=8,
                    markeredgewidth=0.5, markerfacecolor=item_cols[i], markeredgecolor='white')

    sspslam.utils.circles(item_locations[:, 0], item_locations[:, 1], view_rad, utils.grays[4], alpha=1,
            lw=5, edgecolor='none', linewidth=0, ax=ax)
    for i in np.arange(n_walls):
        sspslam.utils.circles(np.tile(wall_boundaries[i][0, :], 2),
                np.repeat(wall_boundaries[i][1, :], 2), view_rad,
                utils.grays[4], alpha=1, lw=5, edgecolor='none', linewidth=0, ax=ax)
        ax.add_patch(Rectangle((wall_boundaries[i, 0, 0] - view_rad, wall_boundaries[i, 1, 0]),
                                   wall_boundaries[i, 0, 1] - wall_boundaries[i, 0, 0] + 2 * view_rad,
                                   wall_boundaries[i, 1, 1] - wall_boundaries[i, 1, 0],
                                   linewidth=0, edgecolor=utils.grays[4], facecolor=utils.grays[4]))
        ax.add_patch(Rectangle((wall_boundaries[i, 0, 0], wall_boundaries[i, 1, 0] - view_rad),
                                   wall_boundaries[i, 0, 1] - wall_boundaries[i, 0, 0],
                                   wall_boundaries[i, 1, 1] - wall_boundaries[i, 1, 0] + 2 * view_rad,
                                   linewidth=0, edgecolor=utils.grays[4], facecolor=utils.grays[4]))
        ax.add_patch(Rectangle((wall_boundaries[i,0,0],wall_boundaries[i,1,0]),
                       wall_boundaries[i,0,1]-wall_boundaries[i,0,0],
                       wall_boundaries[i,1,1]-wall_boundaries[i,1,0],
                       linewidth=0,edgecolor='k',facecolor='k', zorder=2))

decoders = sim.data[mem_weights][-1].T
decodersinv = sim.data[meminv_weights][-1].T

def get_mem_out(x):
    x = np.dot(x, sim.data[mem_encoders][-1,:,:].T)
    with sim:
        activites=slam.assomemory.memory.neuron_type.rates(x,
                    sim.data[slam.assomemory.memory].gain, sim.data[slam.assomemory.memory].bias)
    return np.dot(activites, decoders)
def get_mem_out2(x):
    x = np.dot(x, sim.data[meminv_encoders][-1,:,:].T)
    with sim:
        activites=invassomemory.memory.neuron_type.rates(x,
                    sim.data[invassomemory.memory].gain, sim.data[invassomemory.memory].bias)
    return np.dot(activites, decodersinv)

item_ssp_hat = get_mem_out(landmark_sps)
query1 = ssp_space.normalize(ssp_space.bind(shape_sps[0,:], np.sum(col_sps[:,:],axis=0)))
query2 =ssp_space.normalize(ssp_space.bind(col_sps[0,:], np.sum(shape_sps[:,:],axis=0)))
query3 = ssp_space.normalize(np.sum(wall_sps,axis=0).reshape(1,-1))

vmin = 0.0
fig = plt.figure(figsize=(7.2, 3))
gs = fig.add_gridspec(2,3, width_ratios=[2,1,1],hspace = 0.5)
axs = [plt.subplot(gs[:,0]), plt.subplot(gs[0,1]), plt.subplot(gs[0,2]),
       plt.subplot(gs[1,1]), plt.subplot(gs[1,2])]
for i in range(len(axs)):
    axs[i].set_xlim(-1.2,1.2)
    axs[i].set_ylim(-1.2,1.2)
    axs[i].set_aspect('equal')
    axs[i].spines['right'].set_visible(True)
    axs[i].spines['top'].set_visible(True)

plot_env(axs[0])
ssp_space.similarity_plot(item_ssp_hat[0,:],cmap='Blues',plot_type='contourf',vmin=vmin,ax=axs[1]);
ssp_space.similarity_plot(get_mem_out(query1),cmap='Blues',plot_type='contourf',vmin=vmin,ax=axs[3]);
ssp_space.similarity_plot(get_mem_out(query2),cmap='Blues',plot_type='contourf',vmin=vmin,ax=axs[2]);
ssp_space.similarity_plot(get_mem_out(query3),cmap='Blues',plot_type='contourf',vmin=vmin,ax=axs[4]);
for i in np.arange(n_walls):
    axs[4].add_patch(Rectangle((wall_boundaries[i, 0, 0], wall_boundaries[i, 1, 0]),
                               wall_boundaries[i, 0, 1] - wall_boundaries[i, 0, 0],
                               wall_boundaries[i, 1, 1] - wall_boundaries[i, 1, 0],
                               linewidth=0.2, edgecolor='k', facecolor='none'))


axs[1].plot(item_locations[0,0],item_locations[0,1], 'X',markersize=6,
             markeredgewidth=0.5, markerfacecolor='k',markeredgecolor='white')
axs[2].plot(item_locations[[0,2],0],item_locations[[0,2],1], 'X',markersize=6,
             markeredgewidth=0.5, markerfacecolor='k',markeredgecolor='white')
axs[3].plot(item_locations[:2,0],item_locations[:2,1], 'X',markersize=6,
             markeredgewidth=0.5, markerfacecolor='k',markeredgecolor='white')

fig.text(0.2, 0.93, '\\textbf{A}', size=11, va="baseline", ha="left")
fig.text(0.52,0.93, '\\textbf{B}', size=11, va="baseline", ha="left")
fig.text(0.72,0.93, '\\textbf{C}', size=11, va="baseline", ha="left")
fig.text(0.52,0.45, '\\textbf{D}', size=11, va="baseline", ha="left")
fig.text(0.75,0.45, '\\textbf{E}', size=11, va="baseline", ha="left")

fig.text(0.3, 0.93, "Environment", va="baseline", ha="center",fontsize=11)
fig.text(0.6, 0.93, "Blue triangle", va="baseline", ha="center",fontsize=11)
fig.text(0.82, 0.93, "All blue objects", va="baseline", ha="center",fontsize=11)
fig.text(0.6, 0.45, "All triangles", va="baseline", ha="center",fontsize=11)
fig.text(0.82, 0.45, "Walls", va="baseline", ha="center",fontsize=11)
fig.show()
# utils.save(fig, "wall_env_obj_queries.pdf")
# fig.savefig("wall_env_obj_queries.pdf")
############################################################################


#####################################################
# Area query
########################################

def get_sims_for_area(query_area):
    qX,qY = np.meshgrid(np.linspace(query_area[0,0],query_area[0,1],20),
                        np.linspace(query_area[1,0],query_area[1,1],20))
    query_ssp = ssp_space.encode(np.vstack([qX.reshape(-1), qY.reshape(-1)]).T)
    query_ssp = ssp_space.normalize(np.sum(query_ssp, axis=0))
    # ssp_space.similarity_plot(query_ssp)
    # plt.show()
    item_sp_hat = get_mem_out2(query_ssp.reshape(1,-1))

    item_sims = item_sp_hat @ item_sps.T
    item_sims = np.append(item_sims,np.max(item_sp_hat @ wall_sps.T))
    return item_sims

#query_area = np.array([[-1.2,1.2],[0,1.2]])
# query_area = np.array([[-1.2,0.2],[-1.2,0.6]])
# item_sims = get_sims_for_area(query_area)

query_area = np.array([[-0.5,0.8],[-0.3,0.5]])
item_sims = get_sims_for_area(query_area)
# ssp_space.domain_bounds = bounds*1.4
sample_ssps,sample_points = ssp_space.get_sample_pts_and_ssps(100)
def clean_up_fun(x):
    sims =  sample_ssps @ x
    return sample_ssps[np.argmax(sims),:]
def clean_up_fun2(x):
    sims =  sample_ssps @ x.T
    return sample_ssps[np.argmax(sims,axis=0),:]
def decode(x):
    sims =  sample_ssps @ x.T
    return sample_points[np.argmax(sims,axis=0),:]

realvecs = (item_locations.reshape(-1,1,2) - path[::1000])
realvecssps = np.array([ssp_space.encode(realvecs[j]) for j in range(n_items)])
alldecoders = np.transpose(sim.data[mem_weights])
x = np.dot(landmark_sps, sim.data[mem_encoders][-1, :, :].T)
with sim:
    activites = slam.assomemory.memory.neuron_type.rates(x,
                                         sim.data[slam.assomemory.memory].gain,
                                         sim.data[slam.assomemory.memory].bias)
allitem_hat = np.einsum('ij,jkl->ikl', activites, alldecoders)
item_names = ['blue triangle','orange triangle', 'blue square', 'orange square', 'wall']
inv_ssps = ssp_space.invert(clean_up_fun2(sim.data[ssp_p][::1000]))
# simpest = decode(np.atleast_2d(sim.data[ssp_p][-1,:]))

####
fig = plt.figure(figsize=(7.2, 3.5))
Gs = fig.add_gridspec(2,1, height_ratios=[0.9,1], hspace = 0.8)
gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=Gs[0])
gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[1,2], subplot_spec=Gs[1])

axs = [plt.subplot(gs1[0]), plt.subplot(gs1[1]),
       plt.subplot(gs2[0]), plt.subplot(gs2[1])]

plot_env(axs[2])
axs[2].add_patch(Rectangle((query_area[0,0],query_area[1,0]),
                           query_area[0,1]-query_area[0,0], query_area[1,1]-query_area[1,0],
                   linewidth=0,edgecolor=utils.grays[1],facecolor=utils.grays[2],alpha=0.7,zorder=10))
axs[3].bar(np.arange(n_items+1), item_sims, color=utils.grays[2])
axs[3].set_xticks([])
axs[3].axhline(0, color='black')
axs[3].spines['bottom'].set_visible(False)
for i in range(n_items):
    axs[3].plot(i, -2, item_shapes[i],markersize=8,
         markeredgewidth=0.5, markerfacecolor=item_cols[i], clip_on=False, zorder=100,  markeredgecolor='white')
fig.text(0.84, 0.1, "Wall", va="baseline", ha="center",fontsize=11)

linestyles = ['-','--',':','-.']
for j in range(n_items):
    res1 = ssp_space.bind(inv_ssps, allitem_hat[j].T)
    res1 = res1 / np.maximum(1e-6, np.sqrt(np.sum(res1 ** 2, axis=1))).reshape(-1, 1)
    axs[0].plot(sim.trange()[::1000], 1 - np.sum(res1 * realvecssps[j], axis=1), label=item_names[j],linestyle=linestyles[j])

    res = decode(res1)
    axs[1].plot(sim.trange()[::1000], np.sqrt(np.sum((res - realvecs[j]) ** 2, axis=1)), label=item_names[j],linestyle=linestyles[j])
axs[0].set_ylabel('Cosine error')
axs[1].set_ylabel('Distance error')
axs[1].legend()
axs[1].set_xlabel('Time [s]')
axs[0].set_xlabel('Time [s]')
axs[0].set_xlim([0,T])
axs[1].set_xlim([0,T])

fig.text(0.08, 0.93, '\\textbf{A} $\quad$ Cosine error of landmark queries', size=11, va="baseline", ha="left")
fig.text(0.52, 0.93, '\\textbf{B} $\quad$ Distance error of landmark queries', size=11, va="baseline", ha="left")
# fig.text(0.55, 0.93, '\\textbf{C}', size=12, va="baseline", ha="left")

fig.text(0.08, 0.43, "\\textbf{C} $\quad$ Environment \& query area", va="baseline", ha="left",fontsize=11)
fig.text(0.65, 0.43, "\\textbf{D} $\quad$ Similarity between area query \& landmarks", va="baseline", ha="center",fontsize=11)
fig.tight_layout()
fig.show()
fig
# utils.save(fig, "wall_env_area_queries.pdf")
# fig.savefig("wall_env_area_queries.pdf")

####

