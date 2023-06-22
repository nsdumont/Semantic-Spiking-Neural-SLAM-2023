import nengo
import numpy as np


def GridCellEncoders(n_G, ssp_space):
    def fractional_bind(basis, position):
        return np.fft.ifft(np.prod(np.fft.fft(basis, axis=0)**position, axis=1), axis=0)

    d = ssp_space.ssp_dim
    N = ssp_space.num_grids
    sub_dim=ssp_space.grid_basis_dim
    G_pos = np.random.uniform(low = ssp_space.domain_bounds[:,0], high = ssp_space.domain_bounds[:,1], size=(n_G,ssp_space.domain_dim))
    # G_sorts records which 'sub-grid' each encoder is 'picking out'
    # We'll do at least one of each sub-grid. This can be changed
    if N < n_G:
        G_sorts = np.hstack([np.arange(N), np.random.randint(0, N - 1, size = n_G - N)])
    else:
        G_sorts = np.arange(n_G)
    G_encoders = np.zeros((n_G,d))
    for i in range(n_G):
        sub_mat = _get_sub_SSP(G_sorts[i],N,sublen=sub_dim)
        proj_mat = _proj_sub_SSP(G_sorts[i],N,sublen=sub_dim)
        basis_i = sub_mat @ ssp_space.axis_matrix
        G_encoders[i,:] = N * proj_mat @ fractional_bind(basis_i, G_pos[i,:])
    return G_encoders, G_sorts

class SSPNetwork(nengo.network.Network):
    def __init__(self, ssp_space, n_neurons, **kwargs):
        super().__init__()
        d = ssp_space.ssp_dim
        G_encoders, _ = GridCellEncoders(n_neurons, ssp_space)
        with self:
            self.input = nengo.Node(size_in=d)
            self.ssp = nengo.Ensemble(n_neurons, d, encoders=G_encoders,**kwargs)
            nengo.Connection(self.input, self.ssp)
            
   
# Helper funstions 
def _get_sub_FourierSSP(n, N, sublen=3):
    # Return a matrix, \bar{A}_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \bar{A}_n F{S_{total}} = F{S_n}
    # i.e. pick out the sub vector in the Fourier domain
    tot_len = 2*sublen*N + 1
    FA = np.zeros((2*sublen + 1, tot_len))
    FA[0:sublen, sublen*n:sublen*(n+1)] = np.eye(sublen)
    FA[sublen, sublen*N] = 1
    FA[sublen+1:, tot_len - np.arange(sublen*(n+1),sublen*n,-1)] = np.eye(sublen)
    return FA

def _get_sub_SSP(n,N,sublen=3):
    # Return a matrix, A_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # A_n S_{total} = S_n
    # i.e. pick out the sub vector in the time domain
    tot_len = 2*sublen*N + 1
    FA = _get_sub_FourierSSP(n,N,sublen=sublen)
    W = np.fft.fft(np.eye(tot_len))
    invW = np.fft.ifft(np.eye(2*sublen + 1))
    A = invW @ np.fft.ifftshift(FA) @ W
    return A.real

def _proj_sub_FourierSSP(n,N,sublen=3):
    # Return a matrix, \bar{B}_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \sum_n \bar{B}_n F{S_{n}} = F{S_{total}}
    # i.e. project the sub vector in the Fourier domain such that summing all such projections gives the full vector in Fourier domain
    tot_len = 2*sublen*N + 1
    FB = np.zeros((2*sublen + 1, tot_len))
    FB[0:sublen, sublen*n:sublen*(n+1)] = np.eye(sublen)
    FB[sublen, sublen*N] = 1/N # all sub vectors have a "1" zero freq term so scale it so full vector will have 1 
    FB[sublen+1:, tot_len - np.arange(sublen*(n+1),sublen*n,-1)] = np.eye(sublen)
    return FB.T

def _proj_sub_SSP(n,N,sublen=3):
    # Return a matrix, B_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \sum_n B_n S_{n} = S_{total}
    # i.e. project the sub vector in the time domain such that summing all such projections gives the full vector
    tot_len = 2*sublen*N + 1
    FB = _proj_sub_FourierSSP(n,N,sublen=sublen)
    invW = np.fft.ifft(np.eye(tot_len))
    W = np.fft.fft(np.eye(2*sublen + 1))
    B = invW @ np.fft.ifftshift(FB) @ W
    return B.real