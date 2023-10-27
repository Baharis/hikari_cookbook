"""
A script to calculate distances and angles between atoms in supplied CIF file.
Please mind that since a CIF file does not carry information about correlation
coefficients between individual atoms, uncertainties of calculated metrics
are only an approximation of an actual value that can be calculated
during a full refinement.
"""

import glob
from typing import Tuple

import numpy as np
import pandas as pd

from hikari_cookbook.ingredients import AtomSet, UAtomSet, AtomNetwork


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
