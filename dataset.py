import torch
from torch.utils.data import Dataset
import pandas as pd
import os.path as osp
import os, sys
import typing
from typing import List, Dict, Any
import copy
from utils import poses, prepare_graph, multiprocess
from tqdm import tqdm
from itertools import chain
import numpy as np
import random


class MolDataset(Dataset):
        ###########################################
        # Class: Custom pyg dataset for molecular #
        # graphs.                                 #
        #                                         #
        # Inputs:                                 #
        #       - receptor pdbqt path (optional)  #
        #       - ligand path for pdbqt file or   #
        #         directory of pdbqt files        #
        #         (optional)                      #
        #       - intermolecular edge cutoff      #
        #       - threads for multiprocessing     #
        #       - dataset directory; either       #
        #         for saving after conversion or  #
        #         directory of already converted  #
        #         graphs.                         #
        #       - labels for complex pairs.       #
        ###########################################

    def __init__(self, receptor=None, ligands=None, distance_cutoff=5.,
                    threads=4, dataset_dir=None, labels=None, poses=None, idx=None):
        # initialise
        super().__init__()
        self.distance_cutoff = distance_cutoff
        self.threads = threads
        self.receptor = receptor
        self.ligands = ligands
        self.names = ligands
        self.dataset_dir=dataset_dir
        self.labels = labels
        self.poses = poses
        self.r = receptor
        self.idx=idx


    def run_graph_preparation(self, items: tuple):
        # run graph preparation on receptor ligand pair and save graph

        # get index and ligand dictionary
        idx, ligand = items
        
        if self.idx:
            idx = self.idx[idx]

        # retrieve labels if provided
        if type(self.labels) == np.ndarray:
            label = self.labels[idx]
        else:
            label = 0.

        # join receptor and ligand pose into graph
        graph = prepare_graph.join_receptor_ligand(self.receptor, ligand, self.distance_cutoff, label=label)

        # save data object
        torch.save(graph, osp.join(self.dataset_dir, f'sample_{idx}.pt'))

    def process(self):
        # method for preparing dataset

        # check if ligand is file or directory
        if osp.isdir(self.ligands):
            ligands = [osp.join(self.ligands, lig) for lig in os.listdir(self.ligands)]
        else:
            ligands = [self.ligands]

        # read ligand pdbqt files
        ligands = [open(lig, 'r').read() for lig in ligands]

        # get ligand poses
        if self.poses:
            ligands = [poses.get_poses(lig, poses=self.poses) for lig in ligands]
        else:    
            ligands = [poses.multiple_pose_check(lig) for lig in ligands]
        ligands = list(chain(*ligands))

        # prepare poses for graphs
        ligands = [prepare_graph.parse_ligand(lig) for lig in ligands]

        # prepare receptor for graphs
        self.receptor = open(self.receptor, 'r').read()
        self.receptor = prepare_graph.parse_receptor(self.receptor)

        # prepare sample indexing
        items = [(idx, lig) for idx, lig in enumerate(ligands)]

        # run graph construction
        multiprocess.multiprocess_wrapper(self.run_graph_preparation, items, self.threads)


    def get(self, idx: int):
        # Retrieve sample number idx

        if self.dataset_dir != None:
            graph = torch.load(osp.join(self.dataset_dir, f'sample_{idx}.pt'))

        else:
            ligand = self.ligands[idx]
            graph = prepare_graph.join_receptor_ligand(self.receptor, ligand, self.distance_cutoff)
        return graph


    def __getitem__(self, idx: int):

        return self.get(idx)

    def __len__(self):
        # get number of samples
        if self.dataset_dir != None:
            return len(os.listdir(self.dataset_dir))
        else:
            return len(self.ligands)

    def get_labels(self, verbose=True):
        # get labels from processed dataset
        labels = torch.zeros(self.__len__())
        if verbose:
            for i in tqdm(range(self.__len__())):

                labels[i] = self.get(i).y
        else:
            for i in range(self.__len__()):
                labels[i] = self.get(i).y
        return labels


    def rename(self):
        # rename sample files if sample missing
        count = 0
        for i in os.listdir(self.dataset_dir):
            v = int(i.split('_')[-1].split('.')[0])
            if v != count:
                new =  osp.join(self.dataset_dir, f'_sample_{count}.pt')
                os.rename(osp.join(self.dataset_dir, i), new)
            count += 1

        for i in range(len(os.listdir(self.dataset_dir))):
            r = osp.join(self.dataset_dir, f'_sample_{i}.pt')
            new =  osp.join(self.dataset_dir, f'sample_{i}.pt')
            if osp.exists(r):
                os.rename(r, new)

def parse_args(cmd):

    args = {i.replace('-',''): cmd[cmd.index(i) + 1] for i in cmd if '-' in i}

    if '-dir' not in cmd:
        if not osp.exists('temp'):
            os.mkdir('temp')
        args['dir'] = 'temp/'

    if 'receptor' not in args.keys():
        args['receptor'] = None

    if 'ligand' not in args.keys():
        args['ligand'] = None

    if 'process' not in args.keys():
        args['process'] = False
    return args

if __name__ == '__main__':

    args = parse_args(sys.argv)

    m = MolDataset(receptor=args['receptor'], ligands=args['ligand'], dataset_dir=args['dir'])

    if args['process']:
        m.process()
