import numpy as  np
from typing import List, Dict, Any

class OneHotEncoder:
    ###########################################
    # Class: One hot binary encoding.         #
    ###########################################
    def __init__(self, x: List[Any]):
        ###########################################
        # Input: List of objects to encode.       #
        #                                         #
        # Attributes:                             #
        #   - encoder = Dictionary formatted as   #
        #            {object: binary array}       #
        #   - decoder = Dictionary formatted as   #
        #            {int index: object}          #
        ###########################################

        x = list(sorted(set(x)))
        self.encoder, self.decoder = self.encode(x)

    def encode(self, x: List):
        ###########################################
        # Method: Construct dictionary of objects #
        # and one hot encodings.                  #
        #                                         #
        # Input: List of objects to encode.       #
        #                                         #
        # Output: Dictionary formatted as         #
        #         {object: binary array}.         #
        ###########################################

        codes = np.identity(len(x), dtype=int)
        x = {k: v for k, v in zip(x, codes)}
        y = {k: v for k, v in enumerate(x)}
        return (x, y)

    def transform(self, x: List[Any]) -> List[np.ndarray]:
        ###########################################
        # Method: Transform list of objects into  #
        # one hot encodings                       #
        #                                         #
        # Input: List of objects to encode.       #
        #                                         #
        # Output: List of encoded arrays.         #
        ###########################################

        return [self.encoder[i] for i in x]

    def reverse_transform(self, x: np.ndarray) -> List[Any]:
        ###########################################
        # Method: Transform list of one hot       #
        # encodings into objects.                 #
        #                                         #
        # Input: List of objects to encode.       #
        #                                         #
        # Output: List of encoded arrays.         #
        ###########################################
        return [self.decoder[np.where(i == 1)[0][0]] for i in x]

BASE_ATOM_TYPES: List[str] = [
    'C', 'N', 'O', 'S', 'Br', 'Cl',
    'F', 'I', 'P'
]

PDB_RECEPTOR_ATOM_TYPES: List[str] = [
    'C', 'CA', 'CB', 'N', 'O', 'OXT',
    'CD', 'CG', 'CZ', 'NE', 'NH1', 'NH2',
    'ND2', 'OD1', 'OD2', 'SG', 'NE2', 'OE1',
    'OE2', 'CD2', 'CE1', 'ND1', 'CD1', 'CG1',
    'CG2', 'CE', 'NZ', 'SD', 'CE2', 'OG',
    'OG1', 'CE3', 'CH2', 'CZ2', 'CZ3', 'NE1',
    'OH','P','O1P','O2P','O3P'
]

AMINO_ACID_CODES: Dict[str, str] = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N',
    'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'TPO':'TP'
}

BOND_TYPES: List[str] = [
    'SINGLE',
    'DOUBLE',
    'AROMATIC',
    'TRIPLE',
    'NON-COVALENT'
]

BASE_ATOM_ENCODER = OneHotEncoder(BASE_ATOM_TYPES)
BOND_TYPE_ENCODER = OneHotEncoder(BOND_TYPES)
