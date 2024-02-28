import nengo
import numpy as np

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

# Currecntly this version of PI is only set up for 2D paths
class PathHDIntegration(nengo.network.Network):
    def __init__(self, ssp_space, n_neurons, recurrent_tau,
                 vel_scaling_factor=1, rvel_scaling_factor=1, stable=True, **kwargs):
        super().__init__()
        
        if stable==True:
            def feedback(x):
                w = (x[2]/vel_scaling_factor)/ssp_space.length_scale[0]
                r = np.maximum(np.sqrt(x[0]**2 + x[1]**2), 1e-9)
                dx0 = x[0]*(1-r**2)/r - x[1]*w 
                dx1 = x[1]*(1-r**2)/r + x[0]*w 
                return recurrent_tau*dx0 + x[0], recurrent_tau*dx1 + x[1]
        elif callable(stable):
            feedback = stable
        else:
            def feedback(x): 
                w = (x[2]/vel_scaling_factor)/ssp_space.length_scale[0]
                dx0 = - x[1]*w
                dx1 =  x[0]*w
                return recurrent_tau*dx0 + x[0], recurrent_tau*dx1 + x[1]

        self.ssp_space = ssp_space
        d = ssp_space.ssp_dim
        N = ssp_space.domain_dim
        n_oscs = (d+1)//2
        
        to_SSP = get_from_Fourier(d)
        to_Fourier = get_to_Fourier(d)
        
        self.to_SSP = to_SSP
        self.to_Fourier = to_Fourier
        with self:
            self.speed_input = nengo.Node(lambda t,x: x*np.ones(N), label="speed_input", size_in=1)
            self.rvel_input = nengo.Node(label="rvel_input", size_in=N-1)
            self.input = nengo.Node(label="input", size_in=d)
            self.input_orientation = nengo.Node(label="input_orientation", size_in=N)
            
            def hdfeedback(x):
                w = (x[2]/rvel_scaling_factor) 
                r = np.maximum(np.sqrt(x[0]**2 + x[1]**2), 1e-9)
                dx0 = x[0]*(1-r**2)/r - x[1]*w 
                dx1 = x[1]*(1-r**2)/r + x[0]*w 
                return recurrent_tau*dx0 + x[0], recurrent_tau*dx1 + x[1]
           
            self.orientation = nengo.Ensemble(n_neurons, dimensions=3, radius=np.sqrt(2))
            nengo.Connection(self.input_orientation,  self.orientation[:-1], synapse=None)
            nengo.Connection(self.rvel_input, self.orientation[-1], synapse=None)
            nengo.Connection(self.orientation, self.orientation[:-1], 
                              function=hdfeedback, 
                              synapse=recurrent_tau)
            
            self.velocity = nengo.networks.Product(n_neurons, N,label='velocity')
            nengo.Connection(self.speed_input, self.velocity.input_a)
            nengo.Connection(self.orientation[:-1], self.velocity.input_b)
            
            # An ensemble for each VCO
            self.oscillators = nengo.networks.EnsembleArray(n_neurons, n_oscs, 
                                                            ens_dimensions = 3,
                                                            radius = np.sqrt(2), 
                                                            label="oscillators",
                                                            **kwargs)
            self.oscillators.output.output = lambda t, x: x
            
            # Transform initialization (or some correction...) from phi(x) to FFT{phi(x)} with real and imag
            # components layed out
            nengo.Connection(self.input,self.oscillators.input, transform=to_Fourier, synapse=None)

            for i in np.arange(1,n_oscs):
                # Pass the input freq
                nengo.Connection(self.velocity.output, self.oscillators.ea_ensembles[i][-1], 
                                  transform = ssp_space.phase_matrix[i,:].reshape(1,-1))
                # Recurrent connection for dynamics
                nengo.Connection(self.oscillators.ea_ensembles[i], self.oscillators.ea_ensembles[i][:-1], 
                                  function=feedback, 
                                  synapse=recurrent_tau)
            
            # The DC term is constant
            zerofreq = nengo.Node([1,0,0])
            nengo.Connection(zerofreq, self.oscillators.ea_ensembles[0], synapse=None)
            self.output = nengo.Node(size_in=d)
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
