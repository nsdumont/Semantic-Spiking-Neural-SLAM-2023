import numpy as np

from nengo.connection import Connection
from nengo.network import Network
from nengo.networks.ensemblearray import EnsembleArray
from nengo.node import Node

# Adpated from nengo.networks.InputGatedMemory, but with external 'memory' pop
# Input:
#   net: a EnsembleArray that needs to be updated sometimes
#   n_neurons: to compute diff
#   dimensions: dim of variable represented by net
# When the node gate recives an input of 0, the difference between net.output and 
# node input is computed by pp diff and passed to net.input, updating its value to input
# When gate is given a value >0 (like 1), diff is inhibited and net is left alone
class InputGatedNetwork(Network):
    def __init__(
        self,
        net,
        n_neurons,
        dimensions,
        difference_gain=1.0,
        difference_synapse=0.05,
        **kwargs,
    ):

        
        super().__init__(**kwargs)

        n_total_neurons = n_neurons * dimensions

        with self:
            
            # calculate difference between stored value and input
            self.diff = EnsembleArray(n_neurons, dimensions, label="diff")
            Connection(net.output, self.diff.input, transform=-1)

            # feed difference into integrator
            Connection(
                self.diff.output,
                net.input,
                transform=difference_gain,
                synapse=difference_synapse,
            )

            # gate difference (if gate==0, update stored value,
            # otherwise leave net alone)
            self.gate = Node(size_in=1)
            self.diff.add_neuron_input()
            Connection(
                self.gate,
                self.diff.neuron_input,
                transform=np.ones((n_total_neurons, 1)) * -10,
                synapse=None,
            )

            

        self.input = self.diff.input

