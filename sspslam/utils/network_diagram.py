"""Build a visualization of the network diagram. The specific use case this was
designed for is post-compilation Loihi networks, to get a clearer vision of the network
layout after the passthrough remove function has run and DecodeNeurons are added in.

Ensembles - Octagons
- black: off-chip
- green: on-chip

Nodes - Brown Trapezoids

Probes - Blue mishaped ovals

Connections
- black: standard connection
- blue: on-chip to off-chip connection, involves Probes and HostReceiveNodes
- red: decode-neuron connections

Connection label: dim of pre_obj -> function output size x post_obj input size
"""

import numpy as np
import os

import nengo
import nengo_loihi
import logging

def add_network_obj_conns(net, depth=0, label="", sim_model=None):

    obj_conns = {"Name": label}
    # add objects and connections from any subnetworks
    for ii, subnet in enumerate(net.networks):
        subnet_label = f"{label}_{subnet.label}_{ii}"
        subnet_obj_conns = add_network_obj_conns(
            subnet,
            depth=depth + 1,
            label=subnet_label,
            sim_model=sim_model,
        )
        obj_conns[f"Sub-network: {subnet.label}_{ii}"] = subnet_obj_conns

    # add objects from this network
    obj_conns["Ensembles"] = net.ensembles
    obj_conns["Nodes"] = net.nodes
    obj_conns["Probes"] = net.probes
    obj_conns["Connections"] = net.connections

    logging.info("Building network graph for diagram")
    if sim_model is not None:
        logging.debug("\nChecking for objects to remove")

        for obj in sim_model.split.passthrough.to_remove:
            logging.debug(f"{id(obj)}: {obj}")
            if "Connection" in str(obj):
                if obj in obj_conns["Connections"]:
                    logging.debug(f"Removing {obj} from {obj_conns['Name']}")
                    obj_conns["Connections"].remove(obj)
            elif obj in obj_conns["Nodes"]:
                obj_conns["Nodes"].remove(obj)

        if depth == 0:
            logging.debug("\nChecking for objects to add")
            # only add things in at the main network level
            logging.debug(f"{sim_model.split.passthrough.to_add=}")
            for obj in sim_model.split.passthrough.to_add:
                if "Connection" in str(obj):
                    logging.debug(f"{id(obj.pre_obj)} -> {id(obj.post_obj)}")
                    if obj not in obj_conns["Connections"]:
                        obj_conns["Connections"].append(obj)
                elif obj:
                    logging.debug(f"{id(obj.pre_obj)}: {(obj)}")
                    obj_conns["Nodes"].append(obj)

    return obj_conns



def network_diagram(obj_conns, depth=0, sim_model=None):
    """Create a .dot file showing nodes, ensmbles, and connections.

    This can be useful for debugging and testing builders that manipulate
    the model graph before construction.

    Parameters
    ----------
    objs : list of Nodes and Ensembles
        All the nodes and ensembles in the model.
    connections : list of Connections
        All the connections in the model.

    Returns
    -------
    text : string
        Text content of the desired .dot file.
    """
    logging.debug(f"Processing {obj_conns['Name']} at depth {depth}")
    text = []
    if depth == 0:
        text.append("digraph G {")
    else:
        text.append(f"{' '*(depth*4)}subgraph cluster_{obj_conns['Name']} {{")
        text.append(f"{' '*(depth+1)*4}color=\"black\"")

    for key in obj_conns.keys():
        if "Sub-network" in key:
            logging.debug(key)
            text.append(
                network_diagram(obj_conns[key], depth=depth + 1, sim_model=sim_model)
            )

        elif key == "Ensembles":
            for ens in obj_conns[key]:
                fillcolor = 'gold'
                if isinstance(ens.neuron_type, nengo.Direct):
                    fillcolor = 'gray72'
                elif (sim_model is not None) and (sim_model.split.passthrough.hostchip is not None):
                    if sim_model.split.passthrough._on_chip(ens):
                        fillcolor = 'green'
                text.append(
                    f"{' '*((depth+1)*4)}{id(ens)} [style=filled label=\"{ens.label}\" fillcolor={fillcolor} shape=doubleoctagon];"
                )


        elif key == "Nodes":
            for node in obj_conns[key]:
                fillcolor='brown4'
                if (sim_model is not None) and sim_model.split.hostchip.on_chip(node):
                    fillcolor = 'green'
                text.append(
                    f"{' '*((depth+1)*4)}{id(node)} [style=filled label=\"{node.label}\" fillcolor={fillcolor} shape=trapezium];"
                )

        # elif key == "Probes":
        #     for probe in obj_conns[key]:
        #         text.append(
        #             f"{' '*((depth+1)*4)}{id(probe)} [style=filled label=\"{f'Probe of {probe.target}'}\" fillcolor=cornflowerblue shape=egg];"
        #         )

        elif key == "Connections":
            for conn in obj_conns[key]:
                color = '"black"'

                # if connecting to a neurons object, show that as connecting to the
                # ensemble the neurons belong to, instead of a separate object
                pre_obj = conn.pre_obj
                if isinstance(pre_obj, nengo.ensemble.Neurons):
                    pre_obj = pre_obj.ensemble
                post_obj = conn.post_obj
                if isinstance(post_obj, nengo.ensemble.Neurons):
                    post_obj = post_obj.ensemble

                # if the connection involves DecodeNeurons, show as red
                if (
                    sim_model is not None
                    and conn in sim_model.connection_decode_neurons
                ):
                    color = '"red"'
                # if the connection is from on-chip to off-chip from ensemble, show as blue
                # if the connection is from on-chip to off-chip from neurons, show as purple
                if sim_model is not None and (
                    sim_model.split.on_chip(pre_obj)
                    and not sim_model.split.on_chip(post_obj)
                ):
                    if isinstance(conn.pre_obj, nengo.ensemble.Neurons):
                        color = '"purple"'
                    else:
                        color = '"blue"'

                text.append(
                    f"{' '*((depth+1)*4)}{id(pre_obj)} -> {id(post_obj)} [label=\"{conn.size_in} -> {conn.size_mid} x {conn.size_out}\" fontcolor={color} color={color}];"
                )

    text.append(f"{' '*(depth*4 - 1)} }}")
    return "\n".join(text)


def process_and_save_diagram(net_dot, name, show=True):
    with open("temp.gv", "w") as f:
        f.write(net_dot)
    logging.info(os.system(f"fdp -Tsvg temp.gv > {name}.svg"))

    if show:
        os.system(f"eog {name}.svg")


if __name__ == "__main__":
    # test the network diagram generation

    with nengo.Network("main") as net:

        nengo_loihi.add_params(net)
        ens1 = nengo.Ensemble(n_neurons=4, dimensions=1, label="ens1")
        ens2 = nengo.Ensemble(n_neurons=100, dimensions=1, label="ens2")
        net.config[ens2].on_chip = True

        with nengo.Network("Sub1") as subnet1:
            subnet1.ens3 = nengo.Ensemble(n_neurons=100, dimensions=1, label="ens3")
        nengo.Connection(ens1, subnet1.ens3)

        ens4 = nengo.Ensemble(n_neurons=1, dimensions=1, label="ens4", neuron_type=nengo.Direct())

        with nengo.Network("Sub2") as subnet2:
            nengo_loihi.add_params(subnet2)
            subnet2.ens4 = nengo.Ensemble(n_neurons=100, dimensions=1, label="ens4")
            net.config[subnet2.ens4].on_chip = False
        nengo.Connection(ens1, subnet2.ens4)

        input = nengo.Node(np.sin, label="input")
        relay1 = nengo.Node(size_in=4)
        relay2 = nengo.Node(size_in=3)
        relay3 = nengo.Node(size_in=3)
        output = nengo.Node(lambda t, x: x * 2, size_in=1, label="output")

        nengo.Connection(input, ens1, label="conn1")
        nengo.Connection(ens1, relay1[0], label="relay conn 1", synapse=None)
        nengo.Connection(relay1[0], relay2[0], label="relay conn 2", synapse=0.005)
        nengo.Connection(relay2[1], relay3[0], label="relay conn 2", synapse=0.005)
        nengo.Connection(relay2[0], ens2, label="relay conn 3", synapse=None)
        nengo.Connection(relay3[0], output, label="conn3")
        nengo.Connection(subnet1.ens3, ens4, label="conn4")

        probe = nengo.Probe(output, label="probe output")

    # loihi_network_diagram(net)
    with nengo_loihi.Simulator(net) as sim:
        obj_conns = add_network_obj_conns(net, label="main", sim_model=sim.model)
        net_dot = network_diagram(obj_conns, sim_model=sim.model)
        process_and_save_diagram(net_dot, "net_output")
