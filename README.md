# SSP SLAM

![SLAM diagram](https://github.com/ctn-waterloo/ssp_slam/blob/main/slam.png?raw=true)

To navigate in new environments, an animal or robot must be able to keep track of it’s own position while simultaneously creating and updating an internal map of features in the environment, a problem known as simultaneous localization and mapping (SLAM). This requires integrating information from different domains, namely self-motion cues and sensory information. Recently, Spatial Semantic Pointers (SSPs) have been proposed as a vector representation of continuous space that can be encoded via neural activity. A key feature of this approach is that these spatial representations can be bound with other features, both continuous and discrete, to create compressed structures containing information from multiple domains (e.g. spatial, temporal, visual, conceptual). In this work, SSPs are used as the basis for a biological-plausible SLAM model called SSP-SLAM. It is shown that the self-motion driven dynamics of SSPs can be implemented with a hybrid oscillatory interference/ continuous attractor network of grid cells. The estimated self position represented by this network is used for online learning of an associative memory between landmarks and their positions – i.e. an environment map. This map in turn is used to provide corrections to the path integrator.

### Path integrator
A d-dimensional SSP representing an n-dimensional variable x is given by
```math
S(\mathbf{x}) = \mathcal{F}^{-1} \left \{ e^{ i A \mathbf{x} }\right \}
```
where A is the d by n encoding matrix of the representation. This matrix must have conjugate symmetry so that the SSP is real-valued. SSPs are also unit length.

In SSP-SLAM, SSPs are used to encode an animal's estimation of it's own location. This is continuously updated via self-motion cues using the dynamics of SSPs. The derivative of an SSP is
```math
\dot{S}(\mathbf{x}(t)) = \mathcal{F}^{-1} \{ e^{iA\mathbf{x}(t)} \odot iA\dot{\mathbf{x}}(t)  \}
```
A recurrently connected population of neurons is used to represent the components of the self-position SSP in the Fourier domain. The dynamics of the $j^{th}$ component are 
```math
\frac{d}{dt}
\begin{bmatrix}
 \text{Re}\mathcal{F}\{ S\}_j \\
 \text{Im}\mathcal{F}\{ S\}_j
\end{bmatrix} = 
\begin{bmatrix}
 0 & -\omega_j   \\
 \omega_j & 0
\end{bmatrix}\begin{bmatrix}
 \text{Re}\mathcal{F}\{ S\}_j \\
 \text{Im}\mathcal{F}\{ S\}_j
\end{bmatrix} 
```
where 

```math
\omega_j \equiv  A_{j,:} \cdot \dot{\mathbf{x}}(t) = -i\ln\mathcal{F} \{ S(\dot{\mathbf{x}(t) })\}_j
```
Each Fourier component of the SSP is an oscillator whose frequency is modulated by the animal's velocity -- e.g. a velocity controlled oscillator (VCO), the key module in oscillatory interference models of path integration. Instead of implementing the dynamics as given above, we substitute the dynamics of a nonlinear oscillator with a stable limit cycle. The resulting oscillator will be more stable, especially when implemented with spiking neurons. Considering the set of VCOs as a whole, the system has a toroidal attractor, making this a hybrid of oscillator-inference and continuous attractor models of path integration. The neural population of oscillators is connected to a population of grid cells. The weights between the networks compute the inverse discrete Fourier transform to recover the SSP in the time domain. 

### Environment map
The animal's internal map of the environment is implemented as a heteroassociative memory network. It  consists of three layers of neurons: the first encodes a vector representation, $M$, of features (e.g. landmarks, barriers, colours, etc.) in the animal's field of view, and the last layer encodes the current memory of those feature's locations as an SSP, $S_M$. Biologically  plausible learning rules are used to learn the connection weights between these layers and the middle layer. Notable, the environment features can be multi-sensory structures. For example, the vector representations of a colour, smell, and shape can be bound together to create a multi-sensory landmark. Using such representations, complex environment maps can be learned.   

Sensory systems provide input in the form of an SSP representation of the vector between features in view and the animal, $\Delta S$ -- an egocentric representation of features currently seen. The neurons in the population representing this SSP will have activity patterns like those of boundary and object vector cells. The output of the path integrator and this population are used to compute the features' locations in space. The difference between this output and the third layer of the memory network is used as an error signal to drive learning of the map. The map is used to compute an estimate the animal's current position. This is fed to the path integrator as a correction to position estimation.

## Install
This requires numpy, scipy, nengo, and nengo-spa (pytorch is optional if you'd like to use the neural network option for ssp decoding). To install the sspslam package:
 
```console
pip install -r requirements.txt
pip install .
```
Example usage is in the experiments folder. These require matplotlib, pytry, nengo-ocl.

#### Plotting requirements
To run scripts in `make_plots` there additional requirements: you need GhostScript installed (the gs executable must be in your PATH), and an installation of TeXlive in your PATH that includes siunitx.sty, libertine.sty, libertinust1math.sty, mathrsfs.sty and amssymb.sty. If you are unable to install these, try commenting out lies 18-29 in `figure_utils.py`  and uncommenting lines 9-15. This will remove some of these requirements.

## Usage
An example of running the SLAM model on a randomly generated 2-D path:
```
cd experiments
python run_slam.py --domain-dim 2 --seed 0  --save True --plot True --save_plot_True --ssp_dim 55 --pi_n_neurons 500
```
Other options are available, see `python run_slam.py --help`. 


## Files and Folders


* `sspslam`: The code for SSPs and the nengo SLAM and PI networks
    * `sspspace.py`: Defines classes for SSP representation mapping, including HexagonalSSPSpace, RandomSSPSpace, SSPSpace, SPSpace
    * `solvers.py`: Includes a sparse solver, LstsqThres
    * `networks`: Nengo networks
        * `pathintegration.py`: The PathIntegration network. Continuously updates an SSP given a velocity signal via a set of VCOs with attractor dynamics.
        * `associativememory.py`: The AssociativeMemory network. Learns associations via PES and (optionally) Voja
        * `binding.py`: Contains CircularConvolution and Product networks. The same as the ones in nengo.networks but with additional labelling to help with debugging 
        * `workingmemory.py`, `pathHDintegration.py`: Not currently used.
        * `slam.py`: The SLAMNetwork network. Works with cpu and ocl backends.
* `utils`: Various helper functions (only those used in current code are mentioned below)
	* `utils.py`: Includes `sparsity_to_x_intercept(d,p)`, a function for setting the intercept of d-dim ensemsble so that it has sparisty p (~p% of neurons active at any time), assuming the population is representing unit length vectors; also includes `get_mean_and_ci` which will return the mean and 95% CIs of a dataset.
	* `figure_utils.py`: When imported it will change the matplotlib defaults (see `matplotlibrc`), also includes `circles` for plotting a set of circles, and `save` for saving nice figures
* `experiments`: Scripts that use sspslam 
    * `run_pathint.py`: Runs the PathIntegration network on a random path
    * `run_slam.py`: Runs the SLAMNetwork on random path
    * `slam_vs_pi_trials.py`: Runs PathIntegration and SLAMNetwork with different paths and different seeds and saves data


