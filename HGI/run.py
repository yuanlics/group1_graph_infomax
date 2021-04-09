# %matplotlib inline
import os.path as osp
import argparse
import json
import math
from math import ceil
import numpy as np
import sys
import logging

import nni
from nni.utils import merge_parameter

import torch
from torch import optim
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, PReLU, Conv1d
import torch.nn.functional as F
from torch.autograd import Variable

from layers import GCN, HGPSLPool
from utils import get_positive_expectation, get_negative_expectation

from torch_geometric.datasets import Planetoid, Reddit, TUDataset
from torch_geometric.data import NeighborSampler, DataLoader, DenseDataLoader
from torch_geometric.nn import GCNConv, SAGEConv, DeepGraphInfomax
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import GINConv, global_add_pool, global_sort_pool
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import dropout_adj
import torch_geometric.transforms as T

from livelossplot import PlotLosses
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC


device = "cuda:0"
logger = logging.getLogger('hgi')

group_patterns = []
unsup_groups = groups = {'Acccuracy': ['train_acc', 'val_acc', 'test_acc'], 'Loss': ['train_loss']}
unsup_kfold_groups = kfold_groups = {'Acccuracy': ['kfold_acc'], 'Loss': ['train_loss']}


def log(liveloss, loss, trnacc, valacc, tstacc):
    logs = {}
    if liveloss.logger.groups == unsup_groups:
        logs['train_loss'] = loss
        logs['train_acc'] = trnacc
        logs['val_acc'] = valacc
        logs['test_acc'] = tstacc
        nni.report_intermediate_result(tstacc)
    elif liveloss.logger.groups == unsup_kfold_groups:
        logs['train_loss'] = loss
        logs['kfold_acc'] = valacc
        nni.report_intermediate_result(valacc)
    else:
        raise Exception('invalid groups')
    liveloss.update(logs)
    liveloss.send()
    
    
def final_log(liveloss):
    if liveloss.logger.groups == unsup_groups:
        best_val_i = max(liveloss.logger.log_history['val_acc'], key=lambda i: i.value)
        step, best_val = best_val_i.step, best_val_i.value
        report_trn = liveloss.logger.log_history['train_acc'][step].value
        report_tst = liveloss.logger.log_history['test_acc'][step].value
        report_loss = liveloss.logger.log_history['train_loss'][step].value
        print('Best Epoch: {:04d}, Train_Loss: {:.4f}, Train_Acc: {:.4f}, Val_Acc: {:.4f}, Test_Acc: {:.4f}'
              .format(step, report_loss, report_trn, best_val, report_tst))
    elif liveloss.logger.groups == unsup_kfold_groups:
        best_val_i = max(liveloss.logger.log_history['kfold_acc'], key=lambda i: i.value)
        step, best_val = best_val_i.step, best_val_i.value
        report_loss = liveloss.logger.log_history['train_loss'][step].value
        print('Best Epoch: {:04d}, Train_Loss: {:.4f}, Kfold_Val_Acc: {:.4f}'
              .format(step, report_loss, best_val))
        nni.report_final_result(best_val)
        
    
def isnb():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


# LogReg Evaluation for unsupervised learning


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


def logistic_classify(x, y, args):
    nb_classes = np.unique(y).shape[0]
    xent = nn.CrossEntropyLoss()
    hid_units = x.shape[1]

    accs = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):
        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls= y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).to(device), torch.from_numpy(train_lbls).to(device)
        test_embs, test_lbls= torch.from_numpy(test_embs).to(device), torch.from_numpy(test_lbls).to(device)


        log = LogReg(hid_units, nb_classes)
        log.to(device)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_acc = 0
        test_acc = None
        for it in range(args.logreg_epochs):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            
            preds = torch.argmax(log(test_embs), dim=1)
            test_acc = (torch.sum(preds == test_lbls).float() / test_lbls.shape[0]).item()
            if test_acc > best_acc:
                best_acc = test_acc

            loss.backward()
            opt.step()

        accs.append(best_acc)
    return np.mean(accs)


def evaluate_embedding(embeddings, labels, args):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    logreg_accuracies = [logistic_classify(x, y, args) for _ in range(1)]
    return np.mean(logreg_accuracies)


# Infomax Loss


def local_global_loss_(l_enc, g_enc, edge_index, batch, measure):
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs)).to(device)
    neg_mask = torch.ones((num_nodes, num_graphs)).to(device)
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos


# Encoder


class Encoder(torch.nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.hdim
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)
        xs = []
        batches = []
        
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        xs.append(x)
        batches.append(batch)
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        xs.append(x)
        batches.append(batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        xs.append(x)
        batches.append(batch)
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        xs.append(x)
        batches.append(batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        xs.append(x)
        batches.append(batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        summary = F.relu(x1) + F.relu(x2) + F.relu(x3)
        xs = [xs[i] for i in self.args.takeout]
        batches = [batches[i] for i in self.args.takeout]
        return summary, xs, batches
    
    def get_graph_emb_and_label(self, loader):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _, _ = self.forward(data)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


class FF(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


# Infomax


class HGI(nn.Module):
    def __init__(self, args, alpha=0.5, beta=1., gamma=.1):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.hdim = args.hdim
        self.encoder = Encoder(args)

        self.local_d = FF(self.hdim, self.hdim)
        self.global_d = FF(self.hdim * 2, self.hdim)  # hdim*2 is the graph summary dimension, which is gmp||gap.

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):
        # data.batch would indicate which node belongs to which graph at initial layer.
        x, edge_index = data.x, data.edge_index

        summary, xs, batches = self.encoder(data)
        g_enc = self.global_d(summary)
        l_encs = [self.local_d(x) for x in xs]

        mode='fd'
        measure='JSD'
        local_global_losses = [local_global_loss_(l_enc, g_enc, edge_index, batch, measure) 
                               for l_enc, batch in zip(l_encs, batches)]
        local_global_loss = sum(local_global_losses)  # DEV can use weighted sum
        return local_global_loss


def train(model, optimizer, dataloader):
    loss_all = 0
    model.train()
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        
        loss = model(data)
        loss_all += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return loss_all / len(dataloader)
    
    
def test(model, dataloader, args):  # kfold logreg on whole dataset
    model.eval()
    emb, y = model.encoder.get_graph_emb_and_label(dataloader)
    acc = evaluate_embedding(emb, y, args)
    return acc
    
    
def get_params(nbargs=None):
    # Unsupervised args
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--dataset', type=str, default='PROTEINS', 
                        help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    
    # Pooling args
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--hdim', type=int, default=512, help='hidden size')
    parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
    parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
    parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
    parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    parser.add_argument('--epochs', type=int, default=30, help='maximum number of epochs')
    parser.add_argument('--logreg-epochs', type=int, default=150, help='maximum number of LogReg epochs')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    
    if nbargs is not None:
        args = parser.parse_args(nbargs)
    else:
        args = parser.parse_args()
    return args
    
    
def main(args):
    liveloss = PlotLosses(groups=kfold_groups, group_patterns=group_patterns)
    
    global device
    device = args.device
    epochs = args.epochs
    bs = args.batch_size
    lr = args.lr
    dataset = args.dataset
    args.takeout = [1, 3]  # Take out node representation layers for infomax.

    path = osp.join(osp.abspath(''), 'data', dataset)
    dataset = TUDataset(path, name=dataset).shuffle()
    dataloader = DataLoader(dataset, batch_size=bs)
    
    args.num_classes = dataset.num_classes
    args.num_features = max(dataset.num_features, 1)

    model = HGI(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        loss = train(model, optimizer, dataloader)
        kfoldacc = test(model, dataloader, args)
        log(liveloss, loss, None, kfoldacc, None)
    best_val = final_log(liveloss)
    return best_val


# Entry for single script run and nni experiment
if __name__ == '__main__' and not isnb():
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = (merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise

# main() is also the entry for ipynb cells, e.g.:
# main(get_params(['--device', 'cuda:0', '--dataset', 'PROTEINS', '--lr', '0.0005', '--hdim', '256', 
#                  '--pooling_ratio', '0.54', '--lamb', '1.26', '--weight_decay', '0.0001']))
