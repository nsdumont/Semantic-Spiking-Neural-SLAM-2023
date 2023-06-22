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
pip install .
```
Example usage is in the experiments folder. These require matplotlib, pytry, nengo-ocl.

## Contents
- SSPSpace, RandomSSPSpace, HexagonalSSPSpace: Classes for constructing ssps. Includes encoding, decoding, similarity map plotting, etc.
- PathIntegration: Nengo network for path integration
- ObjectVectorCells: Nengo network for object vector cells
- AssociativeMemory: Nengo network for associative memory
- SLAMNetwork: Nengo network for SLAM, combines the above networks together. For current experiments, it requries object locations as input. The networks learn these online but the ground truth is needed to set up learning. Future packages will rework the network setup for more flexible use cases.
