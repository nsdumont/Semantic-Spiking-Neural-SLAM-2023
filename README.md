# Spiking Semantic SLAM with Spatial Semantic Pointers
<img src="https://github.com/nsdumont/Semantic-Spiking-Neural-SLAM-2023/blob/main/slam.png?raw=true" alt="SLAM diagram" width="500px">

To navigate in new environments, an animal or robot must be able to keep track of it’s own position while simultaneously creating and updating an internal map of features in the environment, a problem known as simultaneous localization and mapping (SLAM). This requires integrating information from different domains, namely self-motion cues and sensory information. Recently, Spatial Semantic Pointers (SSPs) have been proposed as a vector representation of continuous space that can be encoded via neural activity. A key feature of this approach is that these spatial representations can be bound with other features, both continuous and discrete, to create compressed structures containing information from multiple domains (e.g. spatial, temporal, visual, conceptual). In this work, SSPs are used as the basis for a biological-plausible SLAM model called SSP-SLAM. It is shown that the self-motion driven dynamics of SSPs can be implemented with a hybrid oscillatory interference/ continuous attractor network of grid cells. The estimated self position represented by this network is used for online learning of an associative memory between landmarks and their positions – i.e. an environment map. This map in turn is used to provide corrections to the path integrator.


See [Exploiting semantic information in a spiking neural SLAM system](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1190515/full) for more detail. 


# Package Info
## Install
This requires numpy, scipy, nengo, nengo-ocl, and nengo-spa (tensorflow is optional if you'd like to use the neural network option for ssp decoding; nengo-loihi is optional if you'd like to use the loihi backend). To install the sspslam package:
 
```console
pip install -r requirements.txt
pip install .
```
Example usage is in the experiments folder. Experiments require matplotlib.

#### Plotting requirements
To run scripts in `make_plots` there additional requirements: you need GhostScript installed (the gs executable must be in your PATH), and an installation of TeXlive in your PATH that includes siunitx.sty, libertine.sty, libertinust1math.sty, mathrsfs.sty and amssymb.sty. If you are unable to install these, try commenting out lies 18-29 in `figure_utils.py`  and uncommenting lines 9-15. This will remove some of these requirements.

## Usage
An example of running the SSP-PI model on a randomly generated 2-D path:
```
python experiments/run_pathint.py --backend ocl --limit 0.1 --pi-n-neurons 1000 --save --plot
```
Running SSP-PI model on a path from a file and making a gif of the output:
```
python experiments/run_pathint_gif.py --path-data example_paths/oneRoom_path.npy --pi-n-neurons 500 --n-rotates 5 --n-scales 5
```
Running the SSP-SLAM model on a randomly generated 2-D path or making a gif to visualize output:
```
python experiments/run_slam.py --backend ocl --domain-dim 2 --seed 0  --save --plot --save-plot --ssp-dim 55 --pi-n-neurons 500
python experiments/run_slam_map_gif.py  --backend ocl --path-data example_paths/oneRoom_path2.npy
```
Other options are available, see `python run_slam.py --help` and `python run_pathint.py --help`. Note that, currently, the default in some of these example scripts are not consistent.  


## Files and Folders
* `sspslam`: The code for SSPs and the nengo SLAM and PI networks
    * `sspspace.py`: Defines classes for SSP representation mapping, including HexagonalSSPSpace, RandomSSPSpace, SSPSpace, SPSpace
    * `networks`: Nengo networks
        * `pathintegration.py`: The PathIntegration network. Continuously updates an SSP given a velocity signal via a set of VCOs with attractor dynamics.
        * `associativememory.py`: The AssociativeMemory network. Learns associations via PES and (optionally) Voja
        * `binding.py`: Contains CircularConvolution and Product networks. The same as the ones in nengo.networks but with additional labelling to help with debugging 
        * `workingmemory.py`, `pathHDintegration.py`: Not currently used.
        * `slam.py`: The SLAMNetwork network. Works with cpu and ocl backends.
        * `slam_loihi.py`: The SLAMLoihiNetwork network. Works with loihi backend.
        * `slam_view.py`: The SLAMViewNetwork network. Uses a mapping from local view cells to self-location for loop closure rather than mapping landmark SPs to landmarks SSPs. Works with cpu and ocl backends.
* `utils`: Various helper functions (only those used in current code are mentioned below)
	* `utils.py`: Includes `sparsity_to_x_intercept(d,p)`, a function for setting the intercept of d-dim ensemsble so that it has sparisty p (around p percent of neurons active at any time), assuming the population is representing unit length vectors; also includes `get_mean_and_ci` which will return the mean and 95% CIs of a dataset.
	* `figure_utils.py`: When imported, it will change the matplotlib defaults (see `matplotlibrc`), also includes `circles` for plotting a set of circles, and `save` for saving nice figures
* `experiments`: Scripts that use sspslam 
    * `run_pathint.py`: Runs the PathIntegration network on a random path or path from a data file. Used for benchmarking and as an example use of sspslam.networks.PathIntegration.
    * `run_slam.py`: Runs the SLAMNetwork on random path or path from a data file and with random landmarks. Used for benchmarking and as an example use of sspslam.networks.SLAMNetwork (or sspslam.networks.SLAMLoihiNetwork if a loihi backend is used).
    * * `run_slamview.py`: Runs the SLAMViewNetwork


## SSP-SLAM

The SSP-SLAM model, sspslam.networks.SLAMNetwork, is a [Nengo](https://www.nengo.ai) network. See the top of the readme for a diagram.

### Usage
A basic example of importing SLAMNetwork and using it in a nengo network:

```
import nengo
from sspslam.networks import SLAMNetwork, get_slam_input_functions

# Get data
domain_dim = ... # dim of space agent moves in 
initial_agent_position = ...
velocity_data = ...
vec_to_landmarks_data = ...
n_landmarks = ...
view_rad = ...

# Construct SSPSpace
ssp_space = HexagonalSSPSpace(...)
d = ssp_space.ssp_dim

# Construct SP space for discrete landmark representations
lm_space = SPSpace(n_landmarks, d)

# Convert data arrays to functions for nodes
velocity_func, vel_scaling_factor, is_landmark_in_view, landmark_id_func, _, landmark_vec_func, _ = get_slam_input_functions(ssp_space,lm_space, velocity_data, vec_to_landmarks_data, view_rad)

with nengo.Network():
   # If running agent online instead, these will be output from other networks
   velocty = nengo.Node(velocity_func)
   init_state = nengo.Node(lambda t: ssp_space.encode(initial_agent_position) if t<0.05 else np.zeros(d))
   landmark_vec = nengo.Node(landmark_vec_func, size_out = domain_dim)
   #
   
   slam = SLAMNetwork(ssp_space, lm_space, view_rad, n_landmarks,
			500, 500, 50, vel_scaling_factor=vel_scaling_factor)
   
   nengo.Connection(velocty, slam.velocity_input, synapse=0.01) 
   nengo.Connection(init_state, slam.pathintegrator.input, synapse=None)
   nengo.Connection(landmark_vec, slam.landmark_vec_input, synapse=None)
   
   slam_output_p = nengo.Probe(slam.pathintegrator.output, synapse=0.05)

```

### Spatial Semantic Pointers (SSPs)
A **Spatial Semantic Pointers (SSP)** represents a value $\mathbf{x}\in\mathbb{R}^n$ in the HRR VSA and is given by the output from a feature map $\phi: \mathbb{R}^n \rightarrow \mathbb{R}^d$ (with $d\gg n$),
```math
    \phi(\mathbf{x}) = \mathcal{F}^{-1}\left \{ e^{ i A  \frac{\mathbf{x}}{\ell} } \right \},
```
where $A \in \mathbb{R}^{d \times n}$ is the **phase matrix** of the representation $\ell \in \mathbb{R}^{n}$ is the **length scale** or **bandwidth** of the representation, and the exponential is applied element-wise. 
The SSP mapping is a projection of $\mathbf{x}$ to a higher-dimensional space parameterized by $A$ and $\ell$. The low-dimensional $\mathbf{x}$ is projected onto a set of $d$ vectors in $\mathbb{R}^{n}$ space, given by the rows of the phase matrix. These scalars are then stored in the phases of complex exponentials. 
This can be viewed as a type of combinatorial coding scheme, as $\mathbf{x}$ is encoded by the combination of many periodic elements, the phasors, $e^{i a_j \cdot \mathbf{x}}$. Or, this can be viewed as a continuous-valued version of a residue number system. 

SSPs are also known as fractional power encodings (Frady et al. 2022), and are closely related to the method of Random Fourier Features (Rahimi et al. 2007).

### Localization Module
The localization module consists of the SSP-PI model introduced in Dumont and Eliasmith (2021), together with a population of grid cells. The SSP-PI implements the dynamics of SSPs

A d-dimensional SSP representing an n-dimensional variable x is given by
```math
\phi(\mathbf{x}) = \mathcal{F}^{-1} \left \{ e^{ i A \mathbf{x} }\right \}
```
where A is the d by n encoding matrix of the representation. This matrix must have conjugate symmetry so that the SSP is real-valued. SSPs are also unit length.
In SSP-SLAM, SSPs are used to encode an animal's estimation of it's own location. This is continuously updated via self-motion cues using the dynamics of SSPs. The derivative of an SSP is
```math
\dot{\phi}(\mathbf{x}(t)) = \mathcal{F}^{-1} \{ e^{iA\mathbf{x}(t)} \odot iA\dot{\mathbf{x}}(t)  \}
```
A recurrently connected population of neurons is used to represent the components of the self-position SSP in the Fourier domain. The dynamics of the $j^{th}$ component are 
```math
\frac{d}{dt}
\begin{bmatrix}
 \text{Re}\mathcal{F}\{ \phi\}_j \\
 \text{Im}\mathcal{F}\{ \phi\}_j
\end{bmatrix} = 
\begin{bmatrix}
 0 & -\omega_j   \\
 \omega_j & 0
\end{bmatrix}\begin{bmatrix}
 \text{Re}\mathcal{F}\{ \phi\}_j \\
 \text{Im}\mathcal{F}\{ \phi\}_j
\end{bmatrix} 
```
where 

```math
\omega_j \equiv  A_{j,:} \cdot \dot{\mathbf{x}}(t) = -i\ln\mathcal{F} \{ \phi(\dot{\mathbf{x}(t) })\}_j
```
Each Fourier component of the SSP is an oscillator whose frequency is modulated by the animal's velocity -- e.g. a velocity controlled oscillator (VCO), the key module in oscillatory interference models of path integration. Instead of implementing the dynamics as given above, we substitute the dynamics of a nonlinear oscillator with a stable limit cycle. The resulting oscillator will be more stable, especially when implemented with spiking neurons. Considering the set of VCOs as a whole, the system has a toroidal attractor, making this a hybrid of oscillator-inference and continuous attractor models of path integration. 


The output of the SSP-PI network is an SSP estimate of self-location, $\hat{\phi}(\mathbf{ x}(t))$. However, this estimate will become noisy over time, eventually leaving the manifold of SSPs and hence requiring some sort of projection back to the SSP space. Consequently, a clean up operation is required to improve its performance.
We use the projection to grid cells to perform this clean up. SSPs can be used to construct probability distributions. Here, the SSP-PI network is initialized to encode the SSP $\phi(\mathbf{x}(0))$, from which a prior probability distribution can be computed. At every simulation timestep, this belief state is updated according to the dynamics given above. Then, the probability density of the agent being at a location $\hat{\mathbf{x}}$ is $\hat{f}(\hat{\mathbf{x}}) \approx (\phi(\hat{\mathbf{x}}) \cdot \hat{\phi}(\mathbf{x}(t)) - \xi)^+$, where $\xi$ is a normalization constant. The position estimate of the SSP-PI model is taken to be the $\hat{\mathbf{x}}$ that maximizes this probability, i.e., the MLE. The SSP representation of the MLE, $\phi(\hat{\mathbf{x}})$, is computed as a part of the clean-up process and represented by the grid cell population.  

### Landmark Perception Module
In the SSP-SLAM model, the agent not only receives a self-velocity signal as input, but additionally receives observations of its local environment.
We assume that information regarding distance to landmarks and landmark identity is provided directly as input. 
Specifically, we let $\{B_1, B_2, \dots \}$ be a set of Semantic Pointers (SPs) that represent features or landmarks in an environment, at locations $\{\mathbf{x}_1, \mathbf{x}_2, \dots\}$ (note that SSP-SLAM can also work with landmarks with spatial extent).  The SSP representation of the vector from the agent to each landmark within the agent's field of view, $\phi(\mathbf{x}_i - \mathbf{x}(t))$, is given as input to SSP-SLAM. The neurons in this population have activity patterns like those of object vector cells (OVCs) in the MEC, so we call the population the OVC population. The output of the path integrator and OVC populations are bound together to compute allocentric features locations, $\hat{\phi}(\mathbf{x}(t)) \circledast \phi(\mathbf{x}_i - \mathbf{x}(t)) = \hat{\phi}(\mathbf{x}_i) \approx \mathbf(\mathbf{x}_i)$. This is stored in the object location population. 
Like the SSP estimate of self-location, the allocentric SSP estimate of an landmark location, $\hat{\phi}(\mathbf{x}_i)$, can be converted to a probability distribution.

### Environment map
In SSP-SLAM, an environment map is stored in the weights of a heteroassociative memory network.  We use an existing SNN model of heteroassociative memory, presented in Stewart et al. (2011) and adapted for online learning by Voelker et al.(2014).

The memory network architecture consists of two neural populations. The first receives some input and projects to the second population -- the transformation is trained online to map input-output pairs. The PES learning rule is used to train the decoders (i.e., the outgoing synaptic weights) of the first population. Concurrently, the Voja learning rule is used to modify the encoders of the first population. This shifts neuron encoders to be more similar to the input they receive. The Voja learning rule sparsifies and separates activity in the population, which helps prevent interference.  

This heteroassociative memory network is used to map representations of features (e.g., objects, landmarks, barriers, colours, etc.) in the agent's field of view to the current estimate of those feature's locations as SSPs, $\hat{\phi}(\mathbf{x}_i)$, obtained from the landmark perception module. Notably, these environmental features can be structured representations. For example, vector representations of a colour, smell, and shape can be bound or bundled together to create a multi-sensory landmark. Using such representations, complex semantic environment maps can be learned. 

Other mappings can also be learned. For example, a network can be trained to map feature locations $\hat{\phi}(\mathbf{x}_i)$, to feature symbols. Or, alternatively, a mapping from feature locations to feature symbols bound with their location,  $\hat{\phi}(\mathbf{x}_i) \circledast B$, can be learned. Given an SSP input that represents the whole area of an environment, the network will approximately recall $\sum_i \phi(\mathbf{x}_i) \circledast B_i$, and so a single vector representation of a complete map can be recovered. 

## SLAM with local view cells
<img src="https://github.com/nsdumont/Semantic-Spiking-Neural-SLAM-2023/blob/main/slam_localview.png?raw=true" alt="SLAM diagram" width="300px">
We can also do SLAM with local view cells. Such a network is given in sspslam.networks.SLAMViewNetwork. 

