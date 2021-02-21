import argparse
import pickle
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def my_loss_function(y_pred, y, weights, model=None, l2_regularization_coefficient=None):
    loss = F.mse_loss(y_pred, y, reduction='none')
    loss = loss.view(-1) * weights
    loss = loss.sum() / weights.sum()

    if model is None or l2_regularization_coefficient is None:
        return loss
    else:
        for W in model.parameters():
            loss += (l2_regularization_coefficient * W.norm(2).pow(2))
        return loss


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim,
                 num_hidden_layers=3, num_hidden_units=256, nonlin=F.leaky_relu):
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
            X = F.leaky_relu(X, negative_slope=0.1)  # self.nonlin(X)
        y = self.output_layer(X)
        return y, X


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--drug_id', type=str, help='drug ID',default='11')
    parser.add_argument('--data_dir', type=str, default='../example_data/')
    parser.add_argument('--model_dir', type=str,default='../model/' )
    parser.add_argument('--out_dir', type=str, default='../representer_matrix_output/')
    parser.add_argument('--index_list', type=str, help='test index range',default='../example_data/11_index_list')
    parser.add_argument('--gpu', type=int, help='GPU ID')
    args = parser.parse_args()

    drug_id = args.drug_id
    data_file_name = args.data_dir + '/' + args.drug_id + '.pickle'
    index_list_file = args.index_list
    gpu_id = args.gpu

    index_list = []
    file_handle = open(index_list_file)
    for index in file_handle:
        index_list.append(int(index.rstrip()))
    file_handle.close()

    with open(data_file_name, 'rb') as f:
        data = pickle.load(f)

    l2_regularization_coefficient = data['hyperparameter_search']['neural_net_config']['l2_regularization_coefficient']
    num_layers = data['hyperparameter_search']['neural_net_config']['num_layers']
    num_neurons = data['hyperparameter_search']['neural_net_config']['num_neurons']
    X = data['X']
    y = data['y']
    sort_index = np.argsort(y.reshape(-1))
    X = X[sort_index]
    y = y[sort_index]
    num_samples = X.shape[0]
    num_features = X.shape[1]
    num_upsampling = int(X.shape[0] * 0.2)
    weights = np.ones(X.shape[0])
    weights[:num_upsampling] = weights[:num_upsampling] * 4.0

    for test_index in index_list:

        tmp_data_file_name = f'{args.model_dir}/{args.drug_id}/{args.drug_id}_{str(test_index)}.pickle'
        with open(tmp_data_file_name, 'rb') as f:
            model = pickle.load(f)
        y_test_pred_list = []

        train_index = np.ones(num_samples, dtype=bool)
        train_index[test_index] = False

        X_t = X[test_index]
        X_i = X[train_index]
        y_t = y[test_index]
        y_i = y[train_index]
        weights_i = weights[train_index]

        X_t = torch.FloatTensor(X_t).requires_grad_()
        X_i = torch.FloatTensor(X_i)
        y_t = torch.FloatTensor(y_t)
        y_i = torch.FloatTensor(y_i)
        weights_i = torch.FloatTensor(weights_i)

        gradient = torch.zeros(num_features)
        gradient_matrix = torch.zeros(num_neurons, num_features)

        with torch.no_grad():
            y_i_pred, f_i = model(X_i)
        y_t_pred, f_t = model(X_t)

        if abs(torch.sum(f_t)) > 10e-05:
            y_t_pred.backward(retain_graph=True)
            gradient = X_t.grad.data.clone().detach()
            X_t.grad.zero_()

            for index, item in enumerate(f_t):
                item.backward(retain_graph=True)
                gradient_matrix[index] = X_t.grad.data.clone().detach()
                X_t.grad.zero_()

        else:
            for k in range(10):
                f_t.data = 0.001 * (2 * torch.rand(f_t.shape) - 1)

                y_t_pred.backward(retain_graph=True)
                gradient = X_t.grad.data.clone().detach()
                X_t.grad.zero_()

                for index, item in enumerate(f_t):
                    item.backward(retain_graph=True)
                    gradient_matrix[index] += X_t.grad.data.clone().detach()
                    X_t.grad.zero_()

            gradient_matrix[index] = gradient_matrix[index] / 10.0

        if abs(torch.sum(gradient_matrix)) < 10e-10:
            gradient_matrix = 0.001 * (2 * torch.rand(gradient_matrix.shape) - 1)

        decomposition_matrix = torch.matmul(f_i, gradient_matrix)
        decomposition_matrix = (y_i - y_i_pred) * decomposition_matrix
        decomposition_matrix = decomposition_matrix * weights_i.view(-1, 1)
        decomposition_matrix = decomposition_matrix / l2_regularization_coefficient
        decomposition_matrix = decomposition_matrix / weights_i.sum()
        abs_result = np.absolute(decomposition_matrix.numpy())
        model_dir_name = f'{args.out_dir}/{args.drug_id}/'
        if not os.path.isdir(model_dir_name):
            os.makedirs(model_dir_name)
        np.save(f'{args.out_dir}/{args.drug_id}/{args.drug_id}_{str(test_index)}.npy', abs_result)

        print(test_index, 'finished')
