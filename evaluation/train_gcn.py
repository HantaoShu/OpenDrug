import argparse
import pickle

from tqdm import tqdm
import dgl
import os
import dgl.function as fn
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import pickle


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers=3, num_hidden_units=256, nonlin=F.relu):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.nonlin = nonlin
        for index in range(num_hidden_layers):
            if index == 0:
                self.layers.append(nn.Linear(input_dim, num_hidden_units))
            else:
                self.layers.append(nn.Linear(num_hidden_units, num_hidden_units))
        self.output_layer = nn.Linear(num_hidden_units, output_dim)

    def forward(self, X, **kwargs):
        for hidden_layer in self.layers:
            X = hidden_layer(X)
            X = self.nonlin(X)
        y = self.output_layer(X)
        return y, X


def pearson_corr(y, y_pred, **kwargs):
    if np.isnan(y).any() or np.isinf(y).any():
        return np.nan
    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        return np.nan
    y = y.reshape(-1)
    y_pred = y_pred.reshape(-1)
    corr, _ = pearsonr(y, y_pred)
    return corr


def my_loss_function(y_pred, y, weights, model, l2_regularization_coefficient):
    loss = F.mse_loss(y_pred, y, reduction='none')
    loss = loss.view(-1) * weights
    loss = loss.sum() / weights.sum()
    if model is None or l2_regularization_coefficient is None:
        return loss
    else:
        for W in model.parameters():
            loss += (l2_regularization_coefficient * W.norm(2).pow(2))
        return loss


def generate_random_graph(num_samples, num_neighbors):
    g_random_nx = nx.DiGraph()
    g_random_nx.add_nodes_from(range(num_samples))
    for dst in range(num_samples):
        g_random_nx.add_edge(dst, dst)
        src_arr = np.random.choice(np.delete(np.arange(num_samples), dst), num_neighbors, replace=False)
        for src in src_arr:
            g_random_nx.add_edge(src, dst)
    g_random_dgl = dgl.DGLGraph(g_random_nx)
    return g_random_dgl


def generate_graph(representer_value_matrix, num_samples, num_neighbors):
    g_nx = nx.DiGraph()
    g_nx.add_nodes_from(range(num_samples))
    for dst in range(num_samples):
        g_nx.add_edge(dst, dst)
        src_arr = np.argsort(np.absolute(representer_value_matrix[dst]))[::-1][:num_neighbors]
        for src in src_arr:
            g_nx.add_edge(src, dst)
    g_dgl = dgl.DGLGraph(g_nx)
    return g_dgl


gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.mean(msg='m', out='h')


class NodeApplyModule(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_features, out_features, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers, num_hidden_units, nonlin=F.relu):
        super(Net, self).__init__()
        self.gcn1 = GCN(input_dim, num_hidden_units, F.relu)
        self.nonlin = nonlin
        self.layers = nn.ModuleList()
        for index in range(1, num_hidden_layers):
            self.layers.append(nn.Linear(num_hidden_units, num_hidden_units))
        self.output_layer = nn.Linear(num_hidden_units, output_dim)

    def forward(self, g, features):
        g = g.to(torch.device('cuda'))
        x = self.gcn1(g, features)
        for hidden_layer in self.layers:
            x = hidden_layer(x)
            x = self.nonlin(x)
        y = self.output_layer(x)
        return y


def train_gcn(data_path,representer_value_matrix_dir_prefix):
    kf = KFold(n_splits=3, shuffle=True, random_state=1221)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    y = data['y']
    num_upsampling = int(X.shape[0] * 0.2)
    weights = np.ones(X.shape[0])
    weights[:num_upsampling] = weights[:num_upsampling] * 4.0
    num_samples = X.shape[0]
    num_features = X.shape[1]
    X = torch.FloatTensor(X).cuda()
    y = torch.FloatTensor(y).cuda()
    weights = torch.FloatTensor(weights).cuda()

    gcn_results = np.zeros((2, 100))
    gcn_results[0] = np.linspace(0.00, 0.99, 100)
    representer_value_matrix  = np.zeros([num_samples,num_samples])
    for i in range(num_samples):
        representer_value_matrix[i] = np.load(representer_value_matrix_dir_prefix + '_' + str(i) + '_example_importance.npy')
    for index, p in tqdm(enumerate(gcn_results[0])):

        num_neighbors = int(num_samples * p)
        g_dgl = generate_graph(representer_value_matrix, num_samples, num_neighbors)

        # train on graph
        score_list = []
        for train_index, test_index in kf.split(X):
            train_mask = np.zeros(num_samples, dtype=bool)
            test_mask = np.zeros(num_samples, dtype=bool)
            train_mask[train_index] = True
            test_mask[test_index] = True
            train_mask = torch.from_numpy(train_mask).cuda()
            test_mask = torch.from_numpy(test_mask).cuda()
            while True:
                model = Net(input_dim=num_features, output_dim=1, num_hidden_layers=3, num_hidden_units=50)
                model = model.cuda()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                for epoch in range(100):
                    optimizer.zero_grad()
                    y_pred = model(g_dgl, X)
                    loss = my_loss_function(y_pred[train_mask], y[train_mask], weights[train_mask], None, None)
                    loss.backward()
                    optimizer.step()
                y_pred = model(g_dgl, X)
                y_test_upsample = torch.cat((y[test_mask], y[:num_upsampling][test_mask[:num_upsampling]].repeat(3, 1)))
                y_pred_test_upsample = torch.cat(
                    (y_pred[test_mask], y_pred[:num_upsampling][test_mask[:num_upsampling]].repeat(3, 1)))
                correlation = pearson_corr(y_test_upsample.detach().cpu().numpy(), y_pred_test_upsample.detach().cpu().numpy())
                if not np.isnan(correlation):
                    break
            score_list.append(correlation)
        gcn_results[1, index] = np.mean(np.array(score_list))

    return gcn_results


def train_gcn_random(data_path):
    kf = KFold(n_splits=3, shuffle=True, random_state=1221)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']
    num_upsampling = int(X.shape[0] * 0.2)
    weights = np.ones(X.shape[0])
    weights[:num_upsampling] = weights[:num_upsampling] * 4.0
    num_samples = X.shape[0]
    num_features = X.shape[1]
    X = torch.FloatTensor(X).cuda()
    y = torch.FloatTensor(y).cuda()
    weights = torch.FloatTensor(weights).cuda()

    gcn_results = np.zeros((2, 100))
    gcn_results[0] = np.linspace(0.00, 0.99, 100)

    for index, p in enumerate(gcn_results[0]):

        num_neighbors = int(num_samples * p)
        g_random_dgl = generate_random_graph(num_samples, num_neighbors)

        # train on random graph
        score_list = []
        for train_index, test_index in kf.split(X):
            train_mask = np.zeros(num_samples, dtype=bool)
            test_mask = np.zeros(num_samples, dtype=bool)
            train_mask[train_index] = True
            test_mask[test_index] = True
            train_mask = torch.from_numpy(train_mask).cuda()
            test_mask = torch.from_numpy(test_mask).cuda()
            while True:
                model = Net(input_dim=num_features, output_dim=1, num_hidden_layers=3, num_hidden_units=50)
                model = model.cuda()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                for epoch in range(100):
                    optimizer.zero_grad()
                    y_pred = model(g_random_dgl, X)
                    loss = my_loss_function(y_pred[train_mask], y[train_mask], weights[train_mask], None, None)
                    loss.backward()
                    optimizer.step()
                y_pred = model(g_random_dgl, X)
                y_test_upsample = torch.cat((y[test_mask], y[:num_upsampling][test_mask[:num_upsampling]].repeat(3, 1)))
                y_pred_test_upsample = torch.cat(
                    (y_pred[test_mask], y_pred[:num_upsampling][test_mask[:num_upsampling]].repeat(3, 1)))
                correlation = pearson_corr(y_test_upsample.detach().cpu().numpy(), y_pred_test_upsample.detach().cpu().numpy())
                if not np.isnan(correlation):
                    break
            score_list.append(correlation)
        gcn_results[1, index] = np.mean(np.array(score_list))

    return gcn_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../example_data/11.pickle')
    parser.add_argument('--type', type=str, default='OpenDrug')
    parser.add_argument('--output_dir', type=str, default='../evaluation_output/gcn/')
    parser.add_argument('--representer_value_matrix_dir_prefix', type=str, default='../representer_matrix_output/11/11')
    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if args.type=='random':
        pickle.dump(train_gcn_random(args.data_path),open(args.output_dir+args.type+'.pkl','wb'))
    if args.type=='OpenDrug':
        pickle.dump(train_gcn(args.data_path,args.representer_value_matrix_dir_prefix),open(args.output_dir+args.type+'.pkl','wb'))
