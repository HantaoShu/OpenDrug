import argparse
import os
import pickle

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

    p_list = [1.00, 0.50, 0.40, 0.30, 0.20, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

    parser = argparse.ArgumentParser()
    parser.add_argument('--drug_id', type=str, help='drug ID', default='11')
    parser.add_argument('--data_dir', type=str, default='../example_data/')
    parser.add_argument('--index_list', type=str, help='test index range')
    parser.add_argument('--model_dir', type=str, default='../model/')
    parser.add_argument('--output_dir', type=str, default='../output/OpenDrug/')
    parser.add_argument('--gpu', type=int, help='GPU ID')
    args = parser.parse_args()

    drug_id = args.drug_id
    data_file_name = f'{args.data_dir}/{drug_id}.pickle'
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
        tmp_data_file_name = f'{args.data_dir}/{args.drug_id}/{args.drug_id}_{str(test_index)}.pickle'

        with open(tmp_data_file_name, 'rb') as f:
            tmp_data = pickle.load(f)
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
        model = tmp_data

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

        print('-=====', abs_result.shape)

        sort_abs_result = np.sort(abs_result.reshape(-1))


        for p_index, p in enumerate(p_list):

            index = int(sort_abs_result.size * (1 - p))
            key = sort_abs_result[index]

            mask = abs_result > key
            row_mask = mask.sum(axis=1) > 0
            col_mask = mask.sum(axis=0) > 0

            # print( 'col_mask', np.sum(col_mask))

            X_train = X[train_index]
            y_train = y[train_index]
            weights_train = weights[train_index]

            X_train = (X_train * mask)[row_mask, :][:, col_mask]
            y_train = y_train[row_mask, :]
            weights_train = weights_train[row_mask]

            X_test = X[test_index]
            X_test = X_test[col_mask]
            y_test = y[test_index]

            X_train = torch.FloatTensor(X_train).cuda(gpu_id)
            y_train = torch.FloatTensor(y_train).cuda(gpu_id)
            X_test = torch.FloatTensor(X_test).cuda(gpu_id)
            y_test = torch.FloatTensor(y_test).cuda(gpu_id)
            weights_train = torch.FloatTensor(weights_train).cuda(gpu_id)

            num_remain_features = X_train.size()[1]

            model = MLP(input_dim=num_remain_features, output_dim=1, num_hidden_layers=num_layers,
                        num_hidden_units=num_neurons).cuda(gpu_id)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=2 * l2_regularization_coefficient)

            for epoch in range(1, 501):
                optimizer.zero_grad()
                y_train_pred, _ = model(X_train)
                loss = my_loss_function(y_train_pred, y_train, weights_train)
                loss.backward()
                optimizer.step()

            y_test_pred, _ = model(X_test)
            y_test_pred_list.append(y_test_pred.item())

        new_data = {}

        new_data['representer_feature_selection_result'] = {'p_list': p_list, 'y_test_pred_list': y_test_pred_list}

        # print (y_test_pred_list)
        representor_file = f'{args.output_dir}/{drug_id}/r_{drug_id}_{test_index}.pickle'

        with open(representor_file, 'wb') as f:
            pickle.dump(new_data, f)

        print(test_index, 'finished')
