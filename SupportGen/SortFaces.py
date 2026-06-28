import sys
import time
import heapq
import numpy as np
import trimesh
import networkx as nx

def path(mesh, start, end):
    edges = mesh.edges_unique
    lengths = mesh.edges_unique_length
    graph = nx.Graph()
    for edge, l in zip(edges,lengths):
        graph.add_edge(*edge, lengths=l)

    return nx.shortest_path(graph, source=start, target=end, weight="length")