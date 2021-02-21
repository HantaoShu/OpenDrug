import argparse
import pickle

import lime.lime_tabular
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
    def __init__(self, input_dim, output_dim, num_hidden_layers=3, num_hidden_units=256, nonlin=F.leaky_relu):
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
            # X = self.nonlin(X)
            X = F.leaky_relu(X, negative_slope=0.1)
        y = self.output_layer(X)
        return y, X


if __name__ == '__main__':

    p_list = [1.00, 0.50, 0.40, 0.30, 0.20, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

    parser = argparse.ArgumentParser()
    parser.add_argument('--drug_id', type=str, help='drug ID', default='11')
    parser.add_argument('--data_dir', type=str, default='../example_data/')
    parser.add_argument('--index_list', type=str, help='test index range')
    parser.add_argument('--model_dir', type=str, default='../model/')
    parser.add_argument('--output_dir', type=str, default='../output/lime/')
    parser.add_argument('--gpu', type=int, help='GPU ID')
    args = parser.parse_args()

    drug_id = args.drug_id
    index_list_file = args.index_list
    gpu_id = args.gpu
    data_file_name = f'{args.data_dir}/{drug_id}.pickle'

    with open(data_file_name, 'rb') as f:
        data = pickle.load(f)

    index_list = []
    file_handle = open(index_list_file)
    for index in file_handle:
        index_list.append(int(index.rstrip()))
    file_handle.close()

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
    num_numerical_features = 1469
    num_categorical_features = num_features - num_numerical_features
    num_upsampling = int(X.shape[0] * 0.2)
    weights = np.ones(X.shape[0])
    weights[:num_upsampling] = weights[:num_upsampling] * 4.0

    for test_index in index_list:

        tmp_data_file_name = f'{args.model_dir}/{args.drug_id}/{args.drug_id}_{str(test_index)}.pickle'
        with open(tmp_data_file_name, 'rb') as f:
            tmp_data = pickle.load(f)

        y_test_pred_list = []

        train_index = np.ones(num_samples, dtype=bool)
        train_index[test_index] = False

        for p in p_list:

            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]


            def pred_fn(X_test):
                X_test = torch.FloatTensor(X_test)
                model = tmp_data

                with torch.no_grad():
                    y_test_pred, _ = model(X_test)
                y_test_pred = y_test_pred.clone().detach().numpy()
                return y_test_pred


            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train,
                categorical_features=range(num_numerical_features, num_features),
                mode='regression',
                discretize_continuous=False,
                verbose=False)

            num_selected_features = int(num_features * p)
            exp = explainer.explain_instance(X_test, pred_fn, num_features=num_selected_features)
            feature_index_list = [a for a, b in exp.as_list()]
            feature_index_list = [int(item.split('=')[0]) for item in feature_index_list]

            X_train = X[train_index]
            X_train = X_train[:, feature_index_list]
            X_test = X[test_index]
            X_test = X_test[feature_index_list]

            y_train = y[train_index]
            weights_train = weights[train_index]
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

            for epoch in range(1, 301):
                optimizer.zero_grad()
                y_train_pred, _ = model(X_train)
                loss = my_loss_function(y_train_pred, y_train, weights_train)
                loss.backward()
                optimizer.step()

            y_test_pred, _ = model(X_test)
            y_test_pred_list.append(y_test_pred.item())

        new_data = {}
        new_data['lime_feature_selection_result'] = {'p_list': p_list, 'y_test_pred_list': y_test_pred_list}
        lime_file = f'{args.output_dir}/{drug_id}/l_{drug_id}_{test_index}.pickle'
        with open(lime_file, 'wb') as f:
            pickle.dump(new_data, f)

        print(test_index, 'finished')
