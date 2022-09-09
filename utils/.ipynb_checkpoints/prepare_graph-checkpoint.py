# Functions for calculating graph features
from typing import List, Dict, Any
from rdkit import Chem
from utils.ecifs import *
from utils.encoding import *
import numpy as np
import pandas as pd
from itertools import chain, product
import openbabel as ob
import re, os, sys
from tqdm import tqdm
from datetime import datetime
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RemoveIsolatedNodes
import pickle
import networkx as nx

ob_log = ob.OBMessageHandler()
ob_log.SetOutputLevel(0)

np.set_printoptions(linewidth=100)

class Mol(Data):
    ###########################################
    # Custom PyG Data class for a complex     #
    ###########################################
    def __init__(self, x=None, y=None, edge_index=None, edge_attr=None, covalent_index=None, covalent_attr=None, distance_index=None, distance_attr=None, ligand_index=None):
        super().__init__()
        self.x = x
        self.y = y
        self.covalent_index = covalent_index
        self.covalent_attr = covalent_attr
        self.distance_index = distance_index
        self.distance_attr = distance_attr
        self.edge_index = edge_index
        self.ligand_index = ligand_index
        self.edge_attr = edge_attr

def get_feature_min_max(array: np.ndarray) -> np.ndarray:
    ###########################################
    # Function: Return min and max values for #
    # each column in a 2D array.              #
    #                                         #
    # Inputs: 2D numpy array.                 #
    #                                         #
    # Output: 2D numpy array.                 #
    #         Row 0 = Min values              #
    #         Row 1 = Max values              #
    ###########################################
    res = np.zeros((2,array.shape[1]))
    res[0,:] = array.min(axis=0)
    res[1,:] = array.max(axis=0)
    return res

def get_bond_edges(mol: Chem.rdchem.Mol, adj: np.ndarray) -> (np.ndarray, np.ndarray):
    ###########################################
    # Function: Return bond indexes bond      #
    # types for rdkit molecule.               #
    #                                         #
    # Inputs: rdkit molecule and              #
    #         adjacency matrix.               #
    #                                         #
    # Output: Tuple of numpy arrays           #
    #         with bond indexes and type.     #
    ###########################################

    bonds = np.array(np.where(adj == 1.))
    bond_types = [str(mol.GetBondBetweenAtoms(int(bonds[0,b]),int(bonds[1,b])).GetBondType()) for b in range(bonds.shape[1])]

    return (bonds, bond_types)

def pdbqt_to_rdkit(pdbqt: str, type='smiles') -> Chem.rdchem.Mol:
    ###########################################
    # Function: Convert single pose .pdbqt    #
    # string block into rdkit molecule        #
    # object.                                 #
    #                                         #
    # Inputs: String block of one pose from   #
    #         in pdbqt format.                #
    #                                         #
    # Output: rdkit molecule object           #
    ###########################################
    if type == 'smiles':
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("pdbqt", "smiles")
        mol = ob.OBMol()
        obConversion.ReadString(mol, pdbqt)
        outMDL = obConversion.WriteString(mol)
        outMDL = Chem.MolFromSmiles(outMDL)
    if type == 'sdf':
        outMDL = SDF(pdbqt)
        suppl = Chem.SDMolSupplier()
        suppl.SetData(outMDL, sanitize=False)
        outMDL = next(suppl)
        outMDL.UpdatePropertyCache(strict=False)

    return outMDL

def ecif_to_array(fingerprint: str) -> np.ndarray:
    ###########################################
    # Function: Convert ECIF string into      #
    # array.                                  #
    #                                         #
    # Inputs: ECIF string in format           #
    # symbol; valence; no. non-hydrogen       #
    # neighbours; no. hydrogen neighbours;    #
    # aromatic; ring.                         #
    #                                         #
    # Output: numpy array with one hot        #
    # encoding for symbol.                    #
    ###########################################
    symbol, *fingerprint = fingerprint.split(';')
    fingerprint = np.array([int(i) for i in fingerprint], dtype=int)

    # One hot encode symbol
    if symbol in BASE_ATOM_TYPES:
        symbol = BASE_ATOM_ENCODER.transform([symbol])[0]
    else:
        symbol = np.zeros(len(BASE_ATOM_TYPES))

    return np.hstack((symbol, fingerprint))

def get_atom_fingerprint(atom: Chem.rdchem.Atom) -> np.ndarray:
    ###########################################
    # Function: Retrieve atom's ECIF          #
    # fingerprint.                            #
    #                                         #
    # Inputs: rdkit atom object.              #
    #                                         #
    # Output: ECIF fingerprint as an array.   #
    ###########################################

    fingerprint = GetAtomType(atom)
    fingerprint = ecif_to_array(fingerprint)
    return fingerprint

def get_pdbqt_atom(line: str) -> Dict[str, Any]:
    ###########################################
    # Function: Extract atom information from #
    # pdbqt str line.                         #
    #                                         #
    # Inputs: One line of a pdbqt file as a   #
    #         string.                         #
    #                                         #
    # Output: Dictionary containing atom      #
    #         information.                    #
    ###########################################

    pdbqt_cols = dict(
        amino_acid_serial_numeric = int(re.search(r'\d+', line[23:30].strip()).group()),
        amino_acid_serial = line[23:30].strip(),
        chain = line[21:23].strip(),
        atom_serial = int(line[7:11]),
        amino_acid = line[17:20].strip(),
        atom_PDB = line[11:16].strip(),
        coordinates = np.array([
            float(line[31:38]),
            float(line[38:46]),
            float(line[47:54]),
        ]),
        atomType = line[77:].strip(),
        partialChrg = float(line[67:76])
    )

    return pdbqt_cols

def replace_with_dict(ar:  np.ndarray, dic: Dict[str, Any]):
    ###########################################
    # Function: Replace values in array with  #
    # values from dictionary.                 #
    #                                         #
    # Inputs: One line of a pdbqt file as a   #
    #         string.                         #
    #                                         #
    # Output: Dictionary containing atom      #
    #         information.                    #
    ###########################################

    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    return vs[np.searchsorted(ks,ar)]

def get_distance_matrix(receptor, ligand):
    ###########################################
    # Function: Return pairwise distance      #
    # matrix for receptor-ligand complex      #
    #                                         #
    # Inputs:
    #                                         #
    # Output: Dictionary containing atom      #
    #         information.                    #
    ###########################################

    ligand = ligand.split('\n')

    ligand_atoms = [get_pdbqt_atom(line) for line in ligand if 'HETATM' in line]

    receptor = receptor.split('\n')
    receptor_atoms = [get_pdbqt_atom(line) for line in receptor if 'ATOM' in line]
    receptor_atoms = [a for a in receptor_atoms if a['atom_PDB'] in PDB_RECEPTOR_ATOM_TYPES]

    receptor_atoms = [a for a in receptor_atoms if a['atomType'] !='HD']

    atoms = receptor_atoms + ligand_atoms

    atoms = {k:v for k,v in enumerate(atoms)}
    coordinates = [a['coordinates'] for a in atoms.values()]

    x_matrix = np.array([np.repeat(i[0],len(coordinates)) for i in coordinates])
    y_matrix = np.array([np.repeat(i[1],len(coordinates)) for i in coordinates])
    z_matrix = np.array([np.repeat(i[2],len(coordinates)) for i in coordinates])
    distance_matrix = ((x_matrix - x_matrix.T)**2 + (y_matrix - y_matrix.T)**2 + (z_matrix - z_matrix.T)**2)**0.5

    return distance_matrix

def parse_ligand(pdbqt_block: str) -> Dict[str, Any]:
    ###########################################
    # Function: Prepare ligand pdbqt for      #
    # graph construction.                     #
    #                                         #
    # Inputs: String block of one pose from   #
    #         in pdbqt format.                #
    #                                         #
    # Output: Dictionary containing covlent   #
    #         bond adjacency matrix, feature  #
    #         matrix, and atom pdbqt details. #
    ###########################################

    # convert to rdkit molecule
    ligand = pdbqt_to_rdkit(pdbqt_block, type='sdf')

    # get hydrogen atom indexes
    Hs = []
    nonHs ={}
    count = 0
    for a in ligand.GetAtoms():
        if a.GetSymbol() == 'H':
            Hs.append(a.GetIdx())

        else:
            nonHs[a.GetIdx()] = count
            count += 1

    # get atom ecif fingerprints
    ligand_ecifs = np.array([get_atom_fingerprint(atom) for atom in ligand.GetAtoms() if atom.GetSymbol() != "H"])

    # get covalent adjacency matrix
    num_atoms = ligand_ecifs.shape[0]
    molecule_type = np.ones((num_atoms,1))
    adjacency_matrix = Chem.GetAdjacencyMatrix(ligand)

    # ignore hydrogen atoms in adjacency matrix
    adjacency_matrix[Hs,:] = 0
    adjacency_matrix[:,Hs] = 0

    # get atom bond pair indexes
    bonds, bond_types = get_bond_edges(ligand, adjacency_matrix)
    bonds = replace_with_dict(bonds,nonHs)

    # get atom coordinate and partial charges
    pdbqt_block = pdbqt_block.split('\n')
    atoms = [get_pdbqt_atom(line) for line in pdbqt_block if 'HETATM' in line]
    atoms = [a for a in atoms if 'H' not in a['atom_PDB']]
    atoms = {k:v for k,v in enumerate(atoms)}
    partial_charges = np.array([[v['partialChrg']] for v in atoms.values()])

#   # combine features
    node_features = np.hstack((molecule_type,ligand_ecifs,partial_charges))

    # remove hydrogens
    adjacency_matrix = np.delete(adjacency_matrix, Hs, axis=0)
    adjacency_matrix = np.delete(adjacency_matrix, Hs, axis=1)

    # return ligand dictionary
    return {'node_features':node_features, 'adjacency_matrix':adjacency_matrix, 'atoms':atoms, 'bonds':bonds, 'bond_types': bond_types}

def get_amino_acid_fingerprints() -> Dict[str, Any]:
    ###########################################
    # Function: Return dictionary of amino    #
    # acids' ECIF fingerprints, adjacency     #
    # matrices, atoms, and bonds              #
    ###########################################

    # Get ECIF dataframe
    PDB_ATOM_KEYS = pd.read_csv("utils/PDB_Atom_Keys.csv", sep=",")
    PDB_ATOM_KEYS['AMINO_ACID'] = PDB_ATOM_KEYS['PDB_ATOM'].apply(lambda x: x.split('-')[0])
    PDB_ATOM_KEYS['ATOM'] = PDB_ATOM_KEYS['PDB_ATOM'].apply(lambda x: x.split('-')[1])

    # group by amino acid three letter codes
    AMINO_ACID_GROUPS = PDB_ATOM_KEYS.groupby(['AMINO_ACID'])
    AMINO_ACIDS = {}

    # Loop over amino acids
    for k, v in AMINO_ACID_CODES.items():

        # Get rdkit molecule of amino acid
        if k == 'TPO':
            # Include phosphorylated threonine
            m = Chem.MolFromSmiles('C[C@@H](OP(O)(=O)O)[C@H](N)C(=O)O')
            adj = Chem.GetAdjacencyMatrix(m)
            num_atoms = m.GetNumAtoms()
            pdb_atoms = ['CG2','CB','OG1','P','O1P','O2P','O3P','CA','N','C','O','OXT']
            pdb_atoms = {idx: a for idx, a in enumerate(pdb_atoms)}
        else:
            # Mol from amino acid code
            m = Chem.MolFromSequence(v)
            adj = Chem.GetAdjacencyMatrix(m)
            num_atoms = m.GetNumAtoms()
            pdb_atoms = {idx: a.GetPDBResidueInfo().GetName().strip() for idx, a in enumerate(m.GetAtoms())}

        # get fingerprint of atoms from dataframe
        fingerprints = PDB_ATOM_KEYS.iloc[AMINO_ACID_GROUPS.groups[k],:]

        # convert to numpy arrays
        fingerprints = {i: ecif_to_array(j) for i,j in zip(fingerprints.ATOM.tolist(), fingerprints.ECIF_ATOM_TYPE.tolist())}

        # Add molecule type (0 for receptor)
        fingerprints = {i: np.hstack(([0],fingerprints[i])) for i in pdb_atoms.values()}

        # get bond indexes
        bonds, bond_types = get_bond_edges(m, adj)
        bonds = replace_with_dict(bonds, pdb_atoms)

        # add to amino acid dictionary
        AMINO_ACIDS[k] = {'fingerprints':fingerprints,'adjacency_matrix':adj,'atoms':pdb_atoms,'bonds':bonds,'bond_types':bond_types}

    return AMINO_ACIDS

AMINO_ACIDS = get_amino_acid_fingerprints()


def parse_receptor(pdbqt_block: str) -> Dict[str, Any]:
    ###########################################
    # Function: Prepare receptor pdbqt for    #
    # graph construction.                     #
    #                                         #
    # Inputs: String block of one pose from   #
    #         in pdbqt format.                #
    #                                         #
    # Output: Dictionary containing covlent   #
    #         bond adjacency matrix, feature  #
    #         matrix, and atom pdbqt details. #
    ###########################################

    num_features = 16

    # Read pdbqt into dataframe
    pdbqt_block = pdbqt_block.split('\n')
    atoms = [get_pdbqt_atom(line) for line in pdbqt_block if 'ATOM' in line]
    atoms = [a for a in atoms if a['atom_PDB'] in PDB_RECEPTOR_ATOM_TYPES]
    atoms = [a for a in atoms if a['amino_acid'] in AMINO_ACID_CODES.keys()]
    atoms = [atom for atom in atoms if atom['atomType'] !='HD']
    atoms_dict = {k:v for k,v in enumerate(atoms)}
    atoms = pd.DataFrame(atoms)

    # prepare empty arrays for receptor graph info
    receptor_adjacency_matrix = np.zeros((len(atoms),len(atoms)))
    receptor_node_features = np.zeros((len(atoms),num_features))
    receptor_bonds = np.array([[None],[None]])
    receptor_bond_types = []

    # group receptor atoms by amino acid
    amino_acids = atoms.groupby(['amino_acid_serial','chain'])
    amino_acid_groups = dict(sorted(amino_acids.groups.items(), key=lambda x: x[1][0]))

    # atom counter for array indexing
    count = 0

    # prepare peptide bond check
    previous_amino_acid = {'amino_acid_serial_numeric':-10, 'chain':None, 'C_index':None}

    # loop over receptor amino acids
    for group in amino_acid_groups:

        # get amino acid ecifs and bonds
        # ignore unresolved amino acid atoms

        # get amnino amid group rows
        group_indexes = amino_acids.groups[group]
        amino_acid_group = atoms.iloc[group_indexes,:]

        # get amino acid three letter code
        amino_acid_code = amino_acid_group['amino_acid'].iloc[0]

        # get pdb atoms as a list
        amino_acid_atoms = amino_acid_group.atom_PDB.tolist()

        # get total number of amino acid atoms
        num_atoms = len(amino_acid_atoms)

        # retrieve fingerprints and bond info for amino acid
        amino_acid_fingerprint = AMINO_ACIDS[amino_acid_code].copy()

        # get bonds and bond types
        bonds = amino_acid_fingerprint['bonds']
        bond_types = np.array(amino_acid_fingerprint['bond_types'])

        # remove bonds of unresolved atoms
        bond_types = list(bond_types[np.all(np.isin(bonds,amino_acid_atoms), axis=0)])
        bonds = bonds[:,np.all(np.isin(bonds,amino_acid_atoms), axis=0)]

        # reindex atoms in context of receptor
        amino_acid_atoms = {a: idx + count for idx, a in enumerate(amino_acid_atoms)}

        # reindex bonds
        bonds = replace_with_dict(bonds, amino_acid_atoms)

        # loop over atoms in amino acid
        for (a,idx), p in zip(amino_acid_atoms.items(), amino_acid_group['partialChrg'].tolist()):

            # add atom fingerprint to receptor feature array
            f = np.hstack((amino_acid_fingerprint['fingerprints'][a],p))
            receptor_node_features[idx,:] = f

            # get indexes of backbone C and N for peptide bonding
            if a == 'N':
                N_index = idx
            if a == 'C':
                C_index = idx



        # update receptor covalent adjacency matrix
        receptor_adjacency_matrix[bonds[0,:],bonds[1,:]] = 1

        # get amino acid serial number
        numeric_serial = amino_acid_group['amino_acid_serial_numeric'].iloc[0]

        # check for peptide bond
        if previous_amino_acid['chain'] == group[1] and previous_amino_acid['amino_acid_serial_numeric'] == numeric_serial or previous_amino_acid['amino_acid_serial_numeric'] == numeric_serial - 1:
            # add peptide bond
            receptor_adjacency_matrix[previous_amino_acid['C_index'], N_index] = 1
            receptor_adjacency_matrix[N_index, previous_amino_acid['C_index']] = 1
            peptide_bond_to_add = np.array([[N_index, previous_amino_acid['C_index']], [previous_amino_acid['C_index'], N_index]])
            bonds = np.hstack((bonds, peptide_bond_to_add))
            bond_types = bond_types + ['SINGLE','SINGLE']

        # add bonds to receptor arrays
        receptor_bond_types = receptor_bond_types + bond_types
        receptor_bonds = np.hstack((receptor_bonds,bonds))

        # update peptide bond check
        previous_amino_acid['C_index'] = C_index
        previous_amino_acid['chain'] = group[1]
        previous_amino_acid['amino_acid_serial_numeric'] = numeric_serial

        # update atom count
        count += num_atoms

    # return receptor dictionary
    return  {'adjacency_matrix':receptor_adjacency_matrix, 'node_features':receptor_node_features, 'atoms':atoms_dict, 'bonds':receptor_bonds[:,1:], 'bond_types':receptor_bond_types}

def min_max_scaling(array, min, max):
    # function for min-max scaling a feature array
    array = (array - min) / (max - min)
    array[np.where(array < 0)] = 0
    array[np.where(array > 1)] = 1

    return array

def scaling(array, min, max, axis=0):
    # run min-max scaling on specified axis
    ar_min = np.array([min]*array.shape[axis])
    ar_max = np.array([max]*array.shape[axis])
    array =  min_max_scaling(array, ar_min, ar_max)

    return array


def join_receptor_ligand(receptor: Dict[str, Any], ligand: Dict[str, Any], distance_cutoff: float, label=None):
    ###########################################
    # Function: Combine receptor and ligand   #
    # into one graph.                         #
    #                                         #
    # Inputs:                                 #
    # Dict for receptor and dict for          #
    # ligand with:                            #
    #           - atomised features           #
    #           - adjacency matrix            #
    #           - indexed atom dict           #
    #           - bond indexes                #
    # Edge distance threshold for             #
    # intermolecular contacts.                #
    # Optional graph label                    #
    #                                         #
    # Output: Molecular graph with covalent   #
    # and distance edges.                     #
    ###########################################

    # get number of atoms for each molecule
    num_receptor_atoms = receptor['adjacency_matrix'].shape[0]
    num_ligand_atoms = ligand['adjacency_matrix'].shape[0]

    # total number of atoms
    num_atoms = num_receptor_atoms + num_ligand_atoms

    # combine adjacency matrices
    covalent_adjacency_matrix = np.zeros((num_atoms,num_atoms))
    covalent_adjacency_matrix[:num_receptor_atoms, :num_receptor_atoms] = receptor['adjacency_matrix'].copy()
    covalent_adjacency_matrix[num_receptor_atoms:, num_receptor_atoms:] = ligand['adjacency_matrix']

    # reindex ligand atoms
    ligand['atoms'] = {k + num_receptor_atoms: ligand['atoms'][k] for k in ligand['atoms']}

    # combine atoms dictionaries
    atoms = {**receptor['atoms'],**ligand['atoms']}

    # get atom coordinates
    coordinates = [a['coordinates'] for a in atoms.values()]

    # calculate pairwise distances
    x_matrix = np.array([np.repeat(i[0],len(coordinates)) for i in coordinates])
    y_matrix = np.array([np.repeat(i[1],len(coordinates)) for i in coordinates])
    z_matrix = np.array([np.repeat(i[2],len(coordinates)) for i in coordinates])
    distance_matrix = ((x_matrix - x_matrix.T)**2 + (y_matrix - y_matrix.T)**2 + (z_matrix - z_matrix.T)**2)**0.5

    # remove receptor atoms more than 6A from ligand
    to_remove = list()
    for i in range(num_receptor_atoms):
        if np.any(distance_matrix[num_receptor_atoms:,i] < 6.):
            continue

        else:
            to_remove.append(i)
    [atoms.pop(i) for i in to_remove]
    distance_matrix = np.delete(distance_matrix, to_remove, axis=0)
    distance_matrix = np.delete(distance_matrix, to_remove, axis=1)
    covalent_adjacency_matrix = np.delete(covalent_adjacency_matrix, to_remove, axis=0)
    covalent_adjacency_matrix = np.delete(covalent_adjacency_matrix, to_remove, axis=1)
    num_receptor_atoms = num_receptor_atoms - len(to_remove)
    node_features = np.delete(receptor['node_features'].copy(), to_remove, axis=0)

    # copy distance matrix for identifying intermolecular edges
    non_covalent_distances = distance_matrix.copy()

    # set diagonal as greater than threshold to prevent nodes forming edges with themselves
    np.fill_diagonal(non_covalent_distances, distance_cutoff + 1)

    # set atom distances between molecules of the same type as greater than threshold
    # only intermolecular contacts considered to reduce noise
    non_covalent_distances[:num_receptor_atoms,:num_receptor_atoms] = distance_cutoff + 1
    non_covalent_distances[num_receptor_atoms:,num_receptor_atoms:] = distance_cutoff + 1

    # get indexes where distance is less than threshold
    non_covalent_edges = np.array(np.where(non_covalent_distances <= distance_cutoff))

    # get covalent bond indexes
    row_one, row_two = np.where(covalent_adjacency_matrix == 1.)
    bonds = np.array([row_one, row_two])

    # get covalent bond distances
    bonds_attr = np.array([[distance_matrix[bonds[0,e],bonds[1,e]]] for e in range(bonds.shape[1])])

    # get non-covalent distances
    non_covalent_attr = np.array([[distance_matrix[non_covalent_edges[0,e],non_covalent_edges[1,e]]] for e in range(non_covalent_edges.shape[1])])

    # stack edges
    edges = np.hstack((bonds, non_covalent_edges))

    # stack features
    node_features = np.vstack((node_features,ligand['node_features']))

    # remove atoms with no path to ligand
    G = nx.Graph()
    G.add_nodes_from([i for i in range(node_features.shape[0])])
    G.add_edges_from([i for i in zip(edges[0,:], edges[1,:])])
    to_remove = list()
    for i in range(node_features.shape[0]):
        if nx.has_path(G, num_receptor_atoms, i):
            pass
        else:
            to_remove.append(i)
            bonds = np.delete(bonds, np.where(np.any(bonds == i, axis=0)), axis=1)
            bonds_attr  =  np.delete(bonds_attr, np.where(np.any(bonds == i, axis=0)), axis=0)
    node_features = np.delete(node_features, to_remove, axis=0)

    # stack bonds
    edge_attr = np.vstack((bonds_attr, non_covalent_attr))

    # reindex bonds based on removed atoms
    changes_bonds = np.zeros((2, bonds.shape[1]))
    changes_distance = np.zeros((2, non_covalent_edges.shape[1]))
    for i in to_remove:
        changes_bonds[np.where(bonds > i)] -= 1
        changes_distance[np.where(non_covalent_edges > i)] -= 1
    bonds = bonds + changes_bonds
    non_covalent_edges = non_covalent_edges + changes_distance

    # stack edges
    edges = np.hstack((bonds, non_covalent_edges))

    # scale atom features
    node_features = scaling(
        node_features,
        min=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,-0.76],
        max=[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,6.,6.,4.,1.,1.,0.778]
        )

    # convert to tensors
    nodes = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges.astype('float32'), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    covalent_index = torch.tensor(bonds.astype('float32'), dtype=torch.long)
    distance_index = torch.tensor(non_covalent_edges.astype('float32'), dtype=torch.long)
    distance_attr = torch.tensor(non_covalent_attr, dtype=torch.float)

    # create graph
    graph = Mol(
        x=nodes, y=label,
        edge_index=edge_index,
        edge_attr=edge_attr,
        covalent_index=covalent_index,
        distance_index=distance_index, distance_attr=distance_attr,
        ligand_index=num_receptor_atoms
        )

    # return graph
    return graph
