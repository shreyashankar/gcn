import json
import argparse
import networkx as nx
import pickle as pkl


def get_adjacency_matrix():
    with open("synapses.json") as data_file:
        data = json.load(data_file)
        synapses = data["data"]

    adj_dict = {}

    for synapse in synapses:
        if len(synapse["partners"]) == 0:
            continue
        neuron = int(synapse["T-bar"]["body ID"])
        if neuron not in adj_dict:
            adj_dict[neuron] = set()
        for neighbor in synapse["partners"]:
            adj_dict[neuron].add(int(neighbor["body ID"]))

    G = nx.from_dict_of_lists(adj_dict)
    adj = nx.adjacency_matrix(G)
    pkl.dump( adj, open( "adj.p", "w" ) )
