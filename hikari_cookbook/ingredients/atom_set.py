from copy import deepcopy

import numpy as np
import pandas as pd

from hikari.dataframes import BaseFrame, CifFrame, UBaseFrame  # pip install hikari-toolkit
from hikari.symmetry import SymmOp
import uncertainties as uc


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
