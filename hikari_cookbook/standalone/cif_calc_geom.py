"""
A script to calculate distances and angles between atoms in supplied CIF file.
Please mind that since a CIF file does not carry information about correlation
coefficients between individual atoms, uncertainties of calculated metrics
are only an approximation of an actual value that can be calculated
during a full refinement.
"""
from copy import deepcopy
import glob
from itertools import combinations
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import uncertainties as uc
from uncertainties import unumpy

from hikari.dataframes import BaseFrame, CifFrame, UBaseFrame  # pip install hikari-toolkit
from hikari.symmetry import SymmOp


def ustr2float(s: str) -> float:
    return uc.ufloat_fromstr(s).nominal_value


class AtomSet:
    """Container class w/ atoms stored in pd.Dataframe & convenience methods"""

    def __init__(self, bf: BaseFrame, data: pd.DataFrame) -> None:
        self.base = bf
        self.data = data

    def __len__(self):
        return len(self.data.index)

    @classmethod
    def from_cif(cls, cif_path: str) -> 'AtomSet':
        bf = BaseFrame()
        cf = CifFrame()
        cf.read(cif_path)
        first_block_name = list(cf.keys())[0]
        cb = cf[first_block_name]
        bf.edit_cell(a=ustr2float(cb['_cell_length_a']),
                     b=ustr2float(cb['_cell_length_b']),
                     c=ustr2float(cb['_cell_length_c']),
                     al=ustr2float(cb['_cell_angle_alpha']),
                     be=ustr2float(cb['_cell_angle_beta']),
                     ga=ustr2float(cb['_cell_angle_gamma']))
        atoms_dict = {
            'label': cb['_atom_site_label'],
            'fract_x': [ustr2float(v) for v in cb['_atom_site_fract_x']],
            'fract_y': [ustr2float(v) for v in cb['_atom_site_fract_y']],
            'fract_z': [ustr2float(v) for v in cb['_atom_site_fract_z']],
        }
        atoms = pd.DataFrame.from_records(atoms_dict).set_index('label')
        return AtomSet(bf, atoms)

    @property
    def fract_xyz(self) -> np.ndarray:
        return np.vstack([self.data['fract_' + k].to_numpy() for k in 'xyz'])

    @property
    def cart_xyz(self) -> np.ndarray:
        return self.orthogonalise(self.fract_xyz)

    @property
    def cart_distances(self) -> np.ndarray:
        xyz = self.cart_xyz.T
        return np.sum((xyz[:, np.newaxis] - xyz) ** 2, axis=-1) ** 0.5

    def fractionalise(self, cart_xyz: np.ndarray) -> np.ndarray:
        """Multiply 3xN vector by crystallographic matrix to get fract coord"""
        return np.linalg.inv(self.base.A_d.T) @ cart_xyz

    def orthogonalise(self, fract_xyz: np.ndarray) -> np.ndarray:
        """Multiply 3xN vector by crystallographic matrix to get Cart. coord"""
        return self.base.A_d.T @ fract_xyz

    def select(self, label_regex: str) -> 'AtomSet':
        mask = self.data.index.str.match(label_regex)
        return self.__class__(self.base, deepcopy(self.data[mask]))

    def transform(self, symm_op: SymmOp) -> 'AtomSet':
        fract_xyz = symm_op.transform(self.fract_xyz.T)
        data = deepcopy(self.data)
        data['fract_x'] = fract_xyz[:, 0]
        data['fract_y'] = fract_xyz[:, 1]
        data['fract_z'] = fract_xyz[:, 2]
        return self.__class__(self.base, data)

    @property
    def centroid(self):
        """A 3-vector with average atom position."""
        return self.cart_xyz.T.mean(axis=0)

    @property
    def line(self):
        """A 3-vector describing line that best fits the cartesian
        coordinates of atoms. Based on https://stackoverflow.com/q/2298390/"""
        cart_xyz = self.cart_xyz.T
        uu, dd, vv = np.linalg.svd(cart_xyz - self.centroid)
        return vv[0]

    @property
    def plane(self):
        """A 3-vector normal to plane that best fits atoms' cartesian coords.
        Based on https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6"""
        cart_xyz = self.cart_xyz.T
        uu, dd, vv = np.linalg.svd((cart_xyz - self.centroid).T)
        return uu[:, -1]


class UAtomSet(AtomSet):
    @classmethod
    def from_cif(cls, cif_path: str) -> 'AtomSet':
        bf = UBaseFrame()
        cf = CifFrame()
        cf.read(cif_path)
        first_block_name = list(cf.keys())[0]
        cb = cf[first_block_name]
        bf.edit_cell(a=uc.ufloat_fromstr(cb['_cell_length_a']),
                     b=uc.ufloat_fromstr(cb['_cell_length_b']),
                     c=uc.ufloat_fromstr(cb['_cell_length_c']),
                     al=uc.ufloat_fromstr(cb['_cell_angle_alpha']),
                     be=uc.ufloat_fromstr(cb['_cell_angle_beta']),
                     ga=uc.ufloat_fromstr(cb['_cell_angle_gamma']))
        atoms_dict = {
            'label': cb['_atom_site_label'],
            'fract_x': [uc.ufloat_fromstr(v) for v in cb['_atom_site_fract_x']],
            'fract_y': [uc.ufloat_fromstr(v) for v in cb['_atom_site_fract_y']],
            'fract_z': [uc.ufloat_fromstr(v) for v in cb['_atom_site_fract_z']],
        }
        atoms = pd.DataFrame.from_records(atoms_dict).set_index('label')
        return AtomSet(bf, atoms)


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



def calculate_geometry(atom_set: AtomSet) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate and return lists of all distance and angles between atoms
    that lie closer to each other than a sum of their van der Waals radii."""
    an = AtomNetwork.from_atom_set(atom_set)
    return an.distances, an.angles


def degrees_between(v: np.ndarray, w: np.ndarray) -> float:
    """Calculate angle between two vectors in degrees"""
    assert v.shape == w.shape
    rad = np.arccos(sum(v * w) / (np.sqrt(sum(v * v)) * np.sqrt(sum(w * w))))
    return min([d := np.rad2deg(rad), 180. - d])


def main() -> None:
    cif_glob = './**/*.cif'
    cif_paths = sorted(glob.glob(cif_glob, recursive=True))
    if len(cif_paths) == 0:
        print('No cif files found, terminating without output.')
        exit(0)
    for cif_path in cif_paths:
        print(cif_path)
        atom_set = UAtomSet.from_cif(cif_path)
        distances, angles = calculate_geometry(atom_set)
        print(distances)
        print(angles)


if __name__ == '__main__':
    main()
