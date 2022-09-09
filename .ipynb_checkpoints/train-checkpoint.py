import torch
from torch.nn import Module, Linear, BCELoss
from torch.nn.functional import relu, dropout
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from dataset import MolDataset
import os
import os.path as osp
from sklearn.metrics import average_precision_score, roc_auc_score
import copy
from datetime import datetime
import numpy as np
from utils import *
from itertools import product
import sys
import pickle
from tqdm import tqdm
import random

class GNN(Module):
    # Class for GAT model

    def __init__(self, num_node_features, edge_dim,
                out_channels, out_features, heads, dropout,
                **kwargs):
        super().__init__()

        # initialise model layers
        self.conv = GATConv(in_channels=num_node_features, out_channels=out_channels, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.conv2 = GATConv(in_channels=out_channels*heads, out_channels=out_channels, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.conv3 = GATConv(in_channels=num_node_features, out_channels=out_channels, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.conv4 = GATConv(in_channels=out_channels*heads, out_channels=out_channels, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.ff = Linear(in_features=out_channels*heads, out_features=out_features)
        self.out = Linear(in_features=out_features, out_features=1)

    def forward(self, data):
        # unpack data object
        x, edge_index, covalent_index, distance_index, distance_attr, ligand_index, batch = data.x, data.edge_index, data.covalent_index, data.distance_index, data.distance_attr, data.ligand_index, data.batch

        # convert edges to long tensor
        distance_index = distance_index.to(torch.long)

        x_1 = self.conv(x, edge_index=covalent_index)
        x_1 = relu(x_1)
        x_1 = self.conv2(x_1, edge_index=covalent_index)
        x_2 = self.conv3(x, edge_index=distance_index, edge_attr=distance_attr)
        x_2 = relu(x_2)
        x_2 = self.conv4(x_2, edge_index=distance_index, edge_attr=distance_attr)

        x = (x_1 + x_2)/2

        x = global_mean_pool(x, batch=batch)

        x = self.ff(x)
        x = relu(x)
        x = dropout(x, p=0.3)
        x = torch.sigmoid(self.out(x))

        return x

class GridSearch():
    # grid search class if performing parameter tuning
    def __init__(self, train_data, val_data, params):
        self.params = params
        self.train_data=train_data
        self.val_data=val_data

    def train(self, params):
        model = GAT(16, 1, **params)
        optimizer=torch.optim.Adam(params=model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params['lr_decay'], last_epoch=-1)
        history = fit(model, self.train_data, self.val_data, epochs=30, optimizer=optimizer, lr_scheduler=lr_scheduler, early_stopping=params['early_stopping'], batch_size=params['batch_size'], verbose=False)
        return max(history['val_aucpr'])

    def construct_grid(self, params):
        grid = list(product(*self.params.values()))
        grid = [dict(zip(self.params.keys(), i)) for i in grid]
        return grid

    def grid_search(self, threads):
        self.params = self.construct_grid(self.params)
        return multiprocess.multiprocess_wrapper(self.train, self.params, threads)

def predict(model, data):
    # predict on data
    preds = np.zeros(data.dataset.__len__())
    count=0
    for d in data:
        p = model(d).T[0].detach().numpy()
        preds[count:count+len(p)] = p
        count += len(p)
    return preds

def train_step(model, loader, loss_fn, optimizer, verbose=True):

    loss = None
    pr = None

    if verbose:
        pbar = tqdm(bar_format="Loss: {postfix[0]} | PR: {postfix[1]} | Elapsed: {elapsed} | {rate_fmt}", postfix=[loss,pr], total=len(data))

    for data in train:
        optimizer.zero_grad()
        out = model(data)
        out = out.T[0]

        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        pr = average_precision_score(data.y, out.detach())
        t.postfix = [loss,pr]
        t.update()
        if verbose:
            pbar.update(1)

    return model

def fit(model, x, y, x_val, y_val, epochs, optimizer, lr_scheduler, early_stopping, batch_size, save_path, loss_fn=BCELoss(), verbose=True, **kwargs):

    if verbose:
        print('Preparing dataloaders...')

    # load data
    train = DataLoader(x, batch_size=batch_size)
    val = DataLoader(x_val, batch_size=32)

    # dictionary for storing training results
    history = dict(
        model=None,
        loss=np.zeros(epochs),
        val_loss=np.zeros(epochs),
        aucpr=np.zeros(epochs),
        val_aucpr=np.zeros(epochs)
    )

    checkpoint=0
    early_stopping_count=0
    if verbose:
        print('Fitting model...')

    # loop over epochs
    for e in range(epochs):

        start = datetime.now()

        # train step
        model.train()
        model = train_step(model, train, loss_fn, optimizer, verbose)
        if verbose:
            print('Train step complete.')

        # validation predictions
        model.eval()
        val_pred = predict(model, val)
        val_aucpr = average_precision_score(y_val, val_pred)
        v_roc = roc_auc_score(y_val, val_pred)
        loss = loss_fn(val_pred, y_val)

        # update history
        history['val_aucpr'][e] = val_aucpr
        history['val_loss'][e] = loss

        if verbose:
            print(f'Epoch: {e:03d}, Time: {datetime.now()-start}, Val PR: {val_aucpr:.4f}, Val ROC: {v_roc}')

        # check for validation set pr improvement
        if val_aucpr > checkpoint:
            if verbose:
                print('Saving checkpoint model.')
            torch.save(model.state_dict(), osp.join(save_path, 'model.pt'))
            history['model'] = copy.deepcopy(model)
            checkpoint = val_aucpr
            early_stopping_count=0

        else:
            early_stopping_count+=1

        if early_stopping_count==early_stopping:
            # end training if early stopping threshold met
            if verbose:
                print('Early stopping limit reached. Ending training.')
            break

        # update learning rate
        lr_scheduler.step()

    return history

def load_model(params, path):
    model = GNN(16, 1, **params)
    model.load_state_dict(torch.load(path))
    return model

def parse_args(args):
    args = {i.replace('-',''):args[args.index(i)+1] for i in args if '-' in i}

    return args


if __name__ == '__main__':

    args = parse_args(sys.argv)
    train_data = MolDataset(dataset_dir=args['train'])
    val_data = MolDataset(dataset_dir=args['val'])

    y = train_data.get_labels()
    val_y = val_data.get_labels()

    # prepare model
    params = {
        'lr':0.001, 'weight_decay':0.01, 'lr_decay':1.,
        'early_stopping':3, 'batch_size':64,
        'out_channels':64, 'heads':2,
        'dropout':0.4, 'out_features':64,
        'n_gat':2,'n_ff':1
        }

    model = GNN(16, 1, **params)

    print(f'Model: {model}')
    # run training
    optimizer=torch.optim.Adam(params=model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params['lr_decay'], last_epoch=-1)
    fit(model, train_data, y, val_data, val_y, epochs=250, optimizer=optimizer, lr_scheduler=lr_scheduler, early_stopping=100, batch_size=params['batch_size'], save_path=args['save'])
