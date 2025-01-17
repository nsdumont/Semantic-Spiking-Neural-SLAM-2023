import numpy as np
from nengo.connection import Connection
from nengo.exceptions import ObsoleteError
from nengo.network import Network
from nengo.networks.ensemblearray import EnsembleArray
from nengo.node import Node

# Adpated from nengo.networks.InputGatedMemory
# A working memory used for the cognitve mapping experiment

# 
class AdditiveInputGatedMemory(Network):
    def __init__(
        self,
        inputnet,
        inputnetneurons,
        n_neurons,
        dimensions,
        feedback=1.0,
        gain=1.0,
        recurrent_synapse=0.1,
        difference_synapse=None,
        **kwargs,
    ):
        super().__init__()

        if difference_synapse is None:
            difference_synapse = recurrent_synapse

        n_total_neurons = n_neurons * dimensions

        with self:
            # integrator to store value
            self.mem = EnsembleArray(n_neurons, dimensions, label="mem", **kwargs)
            Connection(
                self.mem.output,
                self.mem.input,
                transform=feedback,
                synapse=recurrent_synapse,
            )

            # calculate difference between stored value and input

            # feed difference into integrator
            Connection(
                inputnet,
                self.mem.input,
                transform=gain,
                synapse=difference_synapse,
            )

            # gate difference (if gate==0, update stored value,
            # otherwise retain stored value)
            self.gate = Node(size_in=1)
            if isinstance(inputnetneurons, list):
                for i, ns in enumerate(inputnetneurons):
                    ns.add_neuron_input()
                    Connection(
                        self.gate,
                        ns.neuron_input,
                        transform=np.ones((ns.n_neurons, 1)) * -10,
                        synapse=None,
                    )
            else:
                Connection(
                    self.gate,
                    inputnetneurons,
                    transform=np.ones((inputnetneurons.size_in, 1)) * -10,
                    synapse=None,
                )

            # reset input (if reset=1, remove all values, and set to 0)
            self.reset = Node(size_in=1)
            Connection(
                self.reset,
                self.mem.add_neuron_input(),
                transform=np.ones((n_total_neurons, 1)) * -3,
                synapse=None,
            )
        
        
        self.output = self.mem.output
