import torch
from train import GNN
from dataset import MolDataset
from torch_geometric.loader import DataLoader
import sys, os
import os.path as osp
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["axes.labelsize"] = 15


def load_model(params, path):
    # load model from saved parameters
    model = GNN(16, 1, **params)
    model.load_state_dict(torch.load(path))
    return model


def predict(model, data):

    # predict on data
    num_samples = data.__len__()
    preds = np.zeros(num_samples)

    count=0
    model.eval()
    for d in tqdm(range(num_samples)):
        d = data.get(d)
        p = model(d).T[0].detach().numpy()
        preds[count:count+len(p)] = p
        count += len(p)

    return preds

def native_pose_rank(y, preds):
    # function for assessing csar pose ranking
    # returns the rank of the near native pose
    joined = list(zip(y, preds))
    joined = sorted(joined, key=lambda x: x[1], reverse=True)
    for i in range(len(joined)):
        if joined[i][0] == 1:
            rank = i
            break
    return rank

def enrichment(y, preds, perc=0.05):
    # function for calculating enrichment factor
    # perc - enrichment threshold
    n = len(y)
    top = round(n*perc)
    FA_total = np.count_nonzero(y==1) / n
    joined = list(zip(y, preds))
    joined = sorted(joined, key=lambda x: x[1], reverse=True)
    sorted_y = np.array([int(i[0]) for i in joined])
    sorted_y = sorted_y[:top]
    FA_x = np.count_nonzero(sorted_y==1) / (top)
    return FA_x / FA_total


def parse_args(args):
    args = {i.replace('-',''):args[args.index(i)+1] for i in args if '-' in i}

    return args


if __name__ == '__main__':

    args = parse_args(sys.argv)

    x = MolDataset(dataset_dir=args['x'])
    y = x.get_labels()

    # prepare model
    params = {
        'lr':0.001, 'weight_decay':0.01, 'lr_decay':1.,
        'early_stopping':3, 'batch_size':64,
        'out_channels':64, 'heads':2,
        'dropout':0.4, 'out_features':64,
        'n_gat':2,'n_ff':1
        }
    model = load_model(params, args['model'])

    preds = predict(model, x)

    ef = enrichment(y, preds, 0.01)

    print(f'Dataset enrichment factor at 1% threshold: {ef}')
