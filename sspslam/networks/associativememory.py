import nengo
import numpy as np

# An asssociative memory network. Uses Voja + PES learning to learn a key-value mapping
# Contains:
# 	key_input: input to the network
#	value_input: the desired output
#	learning: if 0 learning is on, if positive learning is off
#	recall: the memeory network's output
# Code taken from https://www.nengo.ai/nengo/examples/learning/learn-associations.html
class AssociativeMemory(nengo.network.Network):
    def __init__(self, n_neurons, d_key, d_value,
                 intercept, voja_learning_rate=5e-2, pes_learning_rate=1e-3,
                 encoders=None,radius=1,voja=True,tau=0.05,**kwargs):
        super().__init__(**kwargs)
        with self:
            # Create the inputs/outputs
            self.key_input = nengo.Node(size_in=d_key, label='memory_input')
            self.value_input = nengo.Node(size_in=d_value)
            self.learning = nengo.Node(size_in=1)
            self.recall = nengo.Ensemble(n_neurons, d_value, label='memory_recall')
        
            # Create the memory
            if encoders is None:
                self.memory = nengo.Ensemble(n_neurons, d_key, intercepts=[intercept] * n_neurons,radius=radius, label='memory')
            else:
                self.memory = nengo.Ensemble(n_neurons, d_key, intercepts=[intercept] * n_neurons, encoders=encoders,radius=radius, label='memory')
        
            # Learn the encoders/keys
            if voja:
                voja = nengo.Voja(learning_rate=voja_learning_rate, post_synapse=None)
                self.conn_in = nengo.Connection(self.key_input, self.memory, synapse=None, learning_rule_type=voja, label='map_conn_in')
                nengo.Connection(self.learning, self.conn_in.learning_rule, synapse=None)
            else:
                self.conn_in = nengo.Connection(self.key_input, self.memory, synapse=None, label='map_conn_in')
        
            # Learn the decoders/values, initialized to a null function
            self.conn_out = nengo.Connection(
                self.memory,
                self.recall,
                learning_rule_type=nengo.PES(pes_learning_rate),
                function=lambda x: np.zeros(d_value), label='map_conn_pes'
            )
        
            # Create the error population
            self.error = nengo.Ensemble(n_neurons, d_value, label='memory_pes_error')
            nengo.Connection(
                self.learning, self.error.neurons, transform=[[-2.5]] * n_neurons, synapse=None
            )
        
            # Calculate the error and use it to drive the PES rule
            nengo.Connection(self.value_input, self.error, transform=-1, synapse=tau)
            nengo.Connection(self.recall, self.error, synapse=tau)
            nengo.Connection(self.error, self.conn_out.learning_rule, synapse=tau)
