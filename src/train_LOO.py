import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers, num_hidden_units):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for index in range(num_hidden_layers):
            if index == 0:
                self.layers.append(nn.Linear(input_dim, num_hidden_units))
            else:
                self.layers.append(nn.Linear(num_hidden_units, num_hidden_units))
        self.output_layer = nn.Linear(num_hidden_units, output_dim)

    def forward(self, X, **kwargs):
        for hidden_layer in self.layers:
            X = hidden_layer(X)
            X = F.leaky_relu(X, negative_slope=0.1)
        y = self.output_layer(X)
        return y, X


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


def train_nn(X, y, weights, train_index, test_index, num_layers, num_neurons, lr, l2_regularization_coefficient, gpu_id):
    device = torch.device('cuda:' + gpu_id)
    X_train = torch.FloatTensor(X[train_index]).cuda(device)
    y_train = torch.FloatTensor(y[train_index]).cuda(device)
    X_test = torch.FloatTensor(X[test_index]).cuda(device)
    y_test = torch.FloatTensor(y[test_index]).cuda(device)
    weights_train = torch.FloatTensor(weights[train_index]).cuda(device)

    model = MLP(
        input_dim=X_train.shape[1],
        output_dim=1,
        num_hidden_layers=num_layers,
        num_hidden_units=num_neurons
    )
    model = model.cuda(device)

    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0)
    lr_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=lr / 10)

    model.train()
    for epoch in range(1, 501):
        def closure():
            optimizer.zero_grad()
            y_train_pred, _ = model(X_train)
            loss = my_loss_function(y_train_pred, y_train, weights_train, model, l2_regularization_coefficient)
            loss.backward()
            return loss

        optimizer.step(closure)
        lr_schedular.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                train_loss = my_loss_function(model(X_train)[0], y_train, weights_train, model, l2_regularization_coefficient)
            if torch.isnan(train_loss):
                return {'success': False}
            model.train()

    return {'model': model.cpu(), 'success': True}


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default='0', type=str)
parser.add_argument('--data_dir', default='../example_data', type=str)
parser.add_argument('--drug_id', default='11', type=str)
parser.add_argument('--model_dir', default='../model', type=str)
args = parser.parse_args()

drug_id = args.drug_id
gpu_id = args.gpu_id

data_file_name = f'{args.data_dir}/{args.drug_id}.pickle'

with open( data_file_name,'rb') as f:
    data = pickle.load(f)
model_dir_name = args.model_dir+'/'+str(drug_id)
if not os.path.isdir(model_dir_name):
    os.makedirs(model_dir_name)

X = data['X']
y = data['y']

num_upsampling = int(X.shape[0] * 0.2)
weights = np.ones(X.shape[0])
weights[:num_upsampling] = weights[:num_upsampling] * 4.0

num_samples = X.shape[0]
num_features = X.shape[1]

lr = data['hyperparameter_search']['neural_net_config']['lr']
l2_regularization_coefficient = data['hyperparameter_search']['neural_net_config']['l2_regularization_coefficient']
num_layers = data['hyperparameter_search']['neural_net_config']['num_layers']
num_neurons = data['hyperparameter_search']['neural_net_config']['num_neurons']


# Train leave-one-out
for test_index in range(0, num_samples):
    print(test_index)
    model_file_name = f'{model_dir_name}/{args.drug_id}_{str(test_index)}.pickle'
    if os.path.exists(model_file_name):
        continue
    train_mask = np.ones(X.shape[0], dtype=bool)
    train_mask[test_index] = False

    tmp = train_nn(X, y, weights, train_mask, test_index, num_layers, num_neurons, lr, l2_regularization_coefficient, gpu_id)
    while tmp['success'] is False:
        tmp = train_nn(X, y, weights, train_mask, test_index, num_layers, num_neurons, lr, l2_regularization_coefficient, gpu_id)

    with open(model_file_name, 'wb') as f:
        pickle.dump(tmp['model'], f)


