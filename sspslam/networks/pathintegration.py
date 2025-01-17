import nengo
import numpy as np
from sspslam.utils import sparsity_to_x_intercept

## Path integration network
# Continuously updates a SSP S (shape d) representing x(t) (shape n) given dx/dt. 
#
# Contains:
#	velocity_input: shape n. A node that receives dx/dt over time
#	input: shape d. A node that receives an SSP. Used to initialize or correct the network
#	oscillators: An EnsembleArray with d//2 + 1 populations each representing a 3 dim vector. Each population is a VCO that represents an component of the SSP in the Fourier domain -- in particular, the first dim is the component's freq (dot product of dx/dt and the VCO's preferred direction), the second is the real part of the FSSP, third dim is the imag part.
#	output: A node with the network's estimate of S(x(t))
#
# Input: 
#	ssp_space: A SSPSpace object
# 	recurrent_tau: synapse constant for recurrent connections
#	n_neurons: number of neurons per VCO population
#	scaling_factor: scaling factor on velocity signal
#	stable: if True use the attractor oscillator dynamics, if False just use a SHO, if callable use it for the recurrent dynamics

class PathIntegration(nengo.network.Network):
    r"""  Nengo network that performs path integration (PI)
    Uses velocity controlled oscillators (VCOs) with (optional) attractor dynamics
    
    Parameters
    ----------
        ssp_space : SSPSpace
            Specifies the SSP representation that is being used.
            
        n_neurons : int
            Number of neurons per VCO population.
            
        recurrent_tau : flaot
            Synaptic constant for recurrent connections. Default is 0.05.
            
        scaling_factor : float
            Scales the dynamics of the PI network. Used when velocity input has been normalized
            Default is 1.
            
        stable : bool
            Whether to use non-linear oscillator with attractor dynamics (True) 
            or linear oscillator (False). Default is True.
            
        max_radius : float
             Desired radius of oscillators. Only matters if stable=True. Default is 1.
             
        with_gcs : bool
            Whether to use a population of grid cells to represent output (True),
            or to use a node (False). Default is False.
            
        n_gcs : int
            number of neurons in grid cell population if with_gcs=True.
            
        solver_weights : bool
            If True solve for weights in recurrent connections, otherwise decoders

            
    Attributes
    ----------
        to_SSP, to_Fourier : np.ndarray
            Arrays used to convert SSPs to and from Fourier space.
            
        input : Node
            Input SSP. Can be used for initialization or corrections.
            
        oscillators : EnsembleArray
           The collection of recurrent VCO populations.
           
        output : Node
            Self-position estimate as an SSP. 

    Examples
    --------

    A basic example where data (exact or approximate) on agent's velocity is given

       from sspslam.networks import PathIntegration
       
       # Get data
       domain_dim = ... # dim of space agent moves in 
       initial_agent_position = ...
       velocity_data = ...
       vel_scaling_factor = ...
       velocity_func = ... # function that returns (possible scaled by vel_scaling_factor) agent's velocity at time t

       
       # Construct SSPSpace
       ssp_space = HexagonalSSPSpace(...)
       d = ssp_space.ssp_dim
       

       with nengo.Network():
           # If running agent online instead, these will be output from other networks
           velocty = nengo.Node(velocity_func)
           init_state = nengo.Node(lambda t: ssp_space.encode(initial_agent_position) if t<0.05 else np.zeros(d))
           #
           
           pathintegrator = PathIntegration(ssp_space, 500, scaling_factor=vel_scaling_factor)
           
           nengo.Connection(velocty, pathintegrator.velocity_input, synapse=0.01) 
           nengo.Connection(init_state, pathintegrator.input, synapse=None)
           
           pi_output_p = nengo.Probe(pathintegrator.output, synapse=0.05)


    """
    def __init__(self, ssp_space, n_neurons, recurrent_tau=0.05,
                 scaling_factor=1, stable=True, max_radius=1, 
                 with_gcs=False,n_gcs=1000,solver_weights=False,
                 label='pathint', **kwargs):
        super().__init__(label=label)
        #max_radius = max_radius*conn_scale
        if stable==True:
            def feedback(x):
                w = (x[2]/scaling_factor)/ssp_space.length_scale[0]
                r = np.maximum(np.sqrt(x[0]**2 + x[1]**2), 1e-9)
                dx0 = x[0]*(max_radius**2-r**2)/r - x[1]*w 
                dx1 = x[1]*(max_radius**2-r**2)/r + x[0]*w 
                out = np.array([recurrent_tau*dx0 + x[0], recurrent_tau*dx1 + x[1], [0]]).flatten()
                return out
        elif callable(stable):
            feedback = stable
        else:
            def feedback(x): 
                w = (x[2]/scaling_factor)/ssp_space.length_scale[0]
                dx0 = - x[1]*w
                dx1 =  x[0]*w
                out = np.array([recurrent_tau*dx0 + x[0], recurrent_tau*dx1 + x[1], [0]]).flatten()
                return out
        

        d = ssp_space.ssp_dim
        N = ssp_space.domain_dim
        n_oscs = (d+1)//2
        
        to_SSP = get_from_Fourier(d)
        to_Fourier = get_to_Fourier(d)
        
        self.to_SSP = to_SSP
        self.to_Fourier = to_Fourier
        with self:
            self.velocity_input = nengo.Node(label=label+"_vel_input", size_in=N)
            self.input = nengo.Node(label=label+"_input", size_in=d)
            if with_gcs:
                encoders = ssp_space.sample_grid_encoders(n_gcs)
                self.output = nengo.Ensemble(n_gcs,d, encoders = encoders,
                                             intercepts=nengo.dists.Choice([sparsity_to_x_intercept(d, 0.1)]),
                                             label=label+'_output')
            else:
                self.output = nengo.Node(size_in=d, label=label+'_output')
            
            # self.velocity = nengo.Ensemble(n_neurons, dimensions=N,label='velocity')
            # nengo.Connection(self.velocity_input, self.velocity)
            
            # An ensemble for each VCO
            self.oscillators = nengo.networks.EnsembleArray(n_neurons, n_oscs, 
                                                            ens_dimensions = 3,
                                                            radius = np.sqrt(2), 
                                                            label=label+"_vco",
                                                            **kwargs)
            self.oscillators.output.output = lambda t, x: x
            
            # Transform initialization (or some correction...) from phi(x) to FFT{phi(x)} with real and imag
            # components layed out
            nengo.Connection(self.input,self.oscillators.input, transform=to_Fourier)
            
            for i in np.arange(1,n_oscs):
                # nengo.Connection(self.input,self.oscillators.ea_ensembles[i], transform=to_Fourier[3*i:3*(i+1),:])
                # Pass the input freq
                nengo.Connection(self.velocity_input, self.oscillators.ea_ensembles[i], 
                                  transform = np.vstack([np.zeros((2,N)), ssp_space.phase_matrix[i,:].reshape(1,-1)]),
                                  synapse=recurrent_tau) # synapse for smoothing
                # Recurrent connection for dynamics
                nengo.Connection(self.oscillators.ea_ensembles[i], self.oscillators.ea_ensembles[i], 
                                  function=feedback, 
                                  synapse=recurrent_tau, solver=nengo.solvers.LstsqL2(weights=solver_weights))
                                  #nengo.solvers.LstsqDrop(weights=True, drop=0.25))
                                  #nengo.solvers.LstsqL2(weights=True))

            # The DC term is constant
            zerofreq = nengo.Node([1,0,0],label=label+'_zerofreq')
            nengo.Connection(zerofreq, self.oscillators.ea_ensembles[0], synapse=None)
            
            nengo.Connection(self.oscillators.output, self.output, transform=to_SSP)
            # for i in np.arange(0,n_oscs):
            #     nengo.Connection(self.oscillators.ea_ensembles[i], self.output, transform=to_SSP[:,3*i:3*(i+1)])
            
            
class PathIntegration_BCs_GCs(nengo.network.Network):
    def __init__(self, ssp_space, n_neurons, n_gc_neurons, recurrent_tau,
                 scaling_factor=1, stable=True, max_radius=1, conn_scale=1,
                 error_correction_factor = 0.1,
                 label='pathint', **kwargs):
        super().__init__(label=label)
        #max_radius = max_radius*conn_scale
        if stable==True:
            def feedback(x):
                w = (x[2]/scaling_factor)/ssp_space.length_scale[0]
                r = np.maximum(np.sqrt(x[0]**2 + x[1]**2), 1e-9)
                dx0 = x[0]*(max_radius**2-r**2)/r - x[1]*w 
                dx1 = x[1]*(max_radius**2-r**2)/r + x[0]*w 
                out = np.array([recurrent_tau*dx0 + x[0], recurrent_tau*dx1 + x[1], [0]]).flatten()
                return conn_scale*out
        elif callable(stable):
            feedback = stable
        else:
            def feedback(x): 
                w = (x[2]/scaling_factor)/ssp_space.length_scale[0]
                dx0 = - x[1]*w
                dx1 =  x[0]*w
                out = np.array([recurrent_tau*dx0 + x[0], recurrent_tau*dx1 + x[1], [0]]).flatten()
                return conn_scale*out
            
        def correction_feedback0(x):
             er_r0 = (x[0]*x[2]*x[4] - x[0]*x[3]*x[5] - x[1]*x[2]*x[5] - x[1]*x[3]*x[4])
             er_i0 =  (x[0]*x[2]*x[5] + x[0]*x[3]*x[4] + x[1]*x[2]*x[4] - x[1]*x[3]*x[5])
             res1 = (er_r0 + 1j*er_i0)**(1/3)
             er_r = res1.real
             er_i = res1.imag
             res = np.array([er_r*x[0]+er_i*x[1], er_r*x[1] - er_i*x[0]])
             return (error_correction_factor)*(res - x[:2]) + x[:2]
         
        def correction_feedback1(x):
             er_r0 = (x[0]*x[2]*x[4] - x[0]*x[3]*x[5] - x[1]*x[2]*x[5] - x[1]*x[3]*x[4])
             er_i0 =  (x[0]*x[2]*x[5] + x[0]*x[3]*x[4] + x[1]*x[2]*x[4] - x[1]*x[3]*x[5])
             res1 = (er_r0 + 1j*er_i0)**(1/3)
             er_r = res1.real
             er_i = res1.imag
             res = np.array([er_r*x[2]+er_i*x[3], er_r*x[3] - er_i*x[2]])
             return (error_correction_factor)*(res - x[2:4]) + x[2:4]
         
        def correction_feedback2(x):
             er_r0 = (x[0]*x[2]*x[4] - x[0]*x[3]*x[5] - x[1]*x[2]*x[5] - x[1]*x[3]*x[4])
             er_i0 =  (x[0]*x[2]*x[5] + x[0]*x[3]*x[4] + x[1]*x[2]*x[4] - x[1]*x[3]*x[5])
             res1 = (er_r0 + 1j*er_i0)**(1/3)
             er_r = res1.real
             er_i = res1.imag
             res = np.array([er_r*x[4]+er_i*x[5], er_r*x[5] - er_i*x[4]])
             return (error_correction_factor)*(res - x[4:]) + x[4:]
            
        self.ssp_space = ssp_space
        d = ssp_space.ssp_dim
        N = ssp_space.domain_dim
        n_oscs = (d+1)//2
         
        to_SSP = get_from_Fourier(d)
        to_Fourier = get_to_Fourier(d)
         
        self.to_SSP = to_SSP
        self.to_Fourier = to_Fourier
        with self:
             self.velocity_input = nengo.Node(label=label+"_vel_input", size_in=N)
             self.input = nengo.Node(label=label+"_input", size_in=d)
             self.output = nengo.Node(size_in=d, label=label+'_output')
             # self.output2 = nengo.Node(size_in=d, label=label+'_output2')
             
             # An ensemble for each VCO
             self.oscillators = nengo.networks.EnsembleArray(n_neurons, n_oscs, 
                                                             ens_dimensions = 3,
                                                             radius = np.sqrt(2), 
                                                             label=label+"_vco",
                                                             **kwargs)
             self.oscillators.output.output = lambda t, x: x
             self.gridcells = nengo.networks.EnsembleArray(n_gc_neurons, n_oscs//3 , 
                                                              ens_dimensions = 6,
                                                              radius=np.sqrt(2), label=label+"_gridcell")
             self.gridcells.output.output = lambda t, x: x
             
             # Transform initialization (or some correction...) from phi(x) to FFT{phi(x)} with real and imag
             # components layed out
             nengo.Connection(self.input,self.oscillators.input, transform=to_Fourier)
             
             for i in np.arange(1,n_oscs):
                 # Pass the input freq
                 nengo.Connection(self.velocity_input, self.oscillators.ea_ensembles[i], 
                                   transform = np.vstack([np.zeros((2,N)), ssp_space.phase_matrix[i,:].reshape(1,-1)]))
                 # Recurrent connection for dynamics
                 nengo.Connection(self.oscillators.ea_ensembles[i], self.oscillators.ea_ensembles[i], 
                                   function=feedback, 
                                   synapse=recurrent_tau, solver=nengo.solvers.LstsqL2(weights=True))
                 nengo.Connection(self.oscillators.ea_ensembles[i][:2], 
                                   self.gridcells.ea_ensembles[int((i-1)//3)][2*np.mod(i-1,3) + np.array([0,1])],
                                   synapse=recurrent_tau)

             # The DC term is constant
             zerofreq = nengo.Node([1,0,0],label=label+'_zerofreq')
             nengo.Connection(zerofreq, self.oscillators.ea_ensembles[0], synapse=None)
             
             for i in range(int(n_oscs//3)):
                nengo.Connection(self.gridcells.ea_ensembles[i], self.oscillators.ea_ensembles[3*i][:2], 
                                 function=correction_feedback0,synapse=recurrent_tau)
                nengo.Connection(self.gridcells.ea_ensembles[i], self.oscillators.ea_ensembles[3*i+1][:2], 
                                 function=correction_feedback1,synapse=recurrent_tau)
                nengo.Connection(self.gridcells.ea_ensembles[i], self.oscillators.ea_ensembles[3*i+2][:2], 
                                 function=correction_feedback2,synapse=recurrent_tau)
                 
             
             nengo.Connection(self.oscillators.output, self.output, transform=to_SSP)

    

# Helper functions

def get_to_Fourier(d):
    # Get matrix that transforms a d-dim SSP to the Fourier domain in the format needed for oscillators
    k = (d+1)//2
    M = np.zeros((3*k ,d))
    M[3:-1:3,:] = np.fft.fft(np.eye(d))[1:k,:].real
    M[4::3,:] = np.fft.fft(np.eye(d))[1:k:].imag
    return M

def get_from_Fourier(d):
    # Get matrix that transforms from the Fourier domain in oscillator format to an SSP
    k = (d+1) // 2 
    if d%2==0:
        shiftmat = np.zeros((4*k, 3*k))
        shiftmat[:k,0::3] = np.eye(k)
        shiftmat[k,0] = 1
        shiftmat[k+1:2*k,3::3] = np.flip(np.eye(k-1), axis=0)
        shiftmat[2*k:3*k,1::3] = np.eye(k)
        shiftmat[2*k,1] = 0
        shiftmat[3*k+1:,4::3] = -np.flip(np.eye(k-1), axis=0)
    else:
        shiftmat = np.zeros((4*k - 2, 3*k))
        shiftmat[:k,0::3] = np.eye(k)
        shiftmat[k:2*k-1,3::3] = np.flip(np.eye(k-1), axis=0)
        shiftmat[2*k-1:3*k-1,1::3] = np.eye(k)
        shiftmat[3*k-1:,4::3] = -np.flip(np.eye(k-1), axis=0)

    invW = np.fft.ifft(np.eye(d))
    M = np.hstack([invW.real, - invW.imag]) @ shiftmat 
    return M
