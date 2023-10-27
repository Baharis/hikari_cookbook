from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from uncertainties import unumpy

from .atom_set import AtomSet


class AtomNetwork(nx.Graph):
    """A kind of graph with additional metadata associated with nodes"""

    @classmethod
    def from_atom_set(cls, atom_set: AtomSet):
        labels = atom_set.data.index
        xyz_map = {ll: xyz for ll, xyz in zip(labels, atom_set.cart_xyz.T)}
        dist_matrix = atom_set.cart_distances
        an = cls((0.1 < dist_matrix) & (dist_matrix < 1.7))
        an = nx.relabel_nodes(an, {i: k for i, k in enumerate(labels)})
        nx.set_node_attributes(an, xyz_map, name='cart_xyz')
        return an

    def angle(self, n1: str, n2: str, n3: str) -> float:
        vec21 = self.nodes[n2]['cart_xyz'] - self.nodes[n1]['cart_xyz']
        vec23 = self.nodes[n2]['cart_xyz'] - self.nodes[n3]['cart_xyz']
        norm21 = np.sum(vec21 ** 2) ** 0.5
        norm23 = np.sum(vec23 ** 2) ** 0.5
        cos = np.dot(vec21, vec23) / (norm21 * norm23)
        return unumpy.arccos(cos)  # noqa

    def distance(self, n1: str, n2: str) -> float:
        delta = self.nodes[n2]['cart_xyz'] - self.nodes[n1]['cart_xyz']
        return np.sum(delta ** 2) ** 0.5


    @property
    def distances(self) -> pd.DataFrame:
        records = []
        for n1, n2 in self.edges():
            vec21 = self.nodes[n2]['cart_xyz'] - self.nodes[n1]['cart_xyz']
            dist = np.sum(vec21 ** 2) ** 0.5
            records.append((n1, n2, dist))
        columns = 'atom1 atom2 dist'.split()
        return pd.DataFrame(records, columns=columns)

    @property
    def angles(self) -> pd.DataFrame:
        records = []
        for n2 in self.nodes():
            for n1, n3 in combinations(self.neighbors(n2), 2):
                angle = self.angle(n1, n2, n3) * 180 / np.pi
                records.append((n1, n2, n3, angle))
        columns = 'atom1 atom2 atom3 angle'.split()
        return pd.DataFrame(records, columns=columns)