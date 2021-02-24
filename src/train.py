import pickle
import argparse
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ax.service.ax_client import AxClient
from sklearn.model_selection import KFold
from scipy.stats import pearsonr


def pearson_corr(y, y_pred, **kwargs):
    if np.isnan(y).any() or np.isinf(y).any():
        return np.nan
    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        return np.nan
    y = y.reshape(-1)
    y_pred = y_pred.reshape(-1)
    corr, _ = pearsonr(y, y_pred)
    return corr


def upsampling(X, y):
    index = np.argsort(y.reshape(-1))
    index_subset = index[:int(y.size * 0.2)]
    X_subset = X[index_subset]
    y_subset = y[index_subset]
    X_upsample = np.tile(X_subset, (3, 1))
    y_upsample = np.tile(y_subset, (3, 1))
    X_upsample = np.concatenate([X, X_upsample], axis=0)
    y_upsample = np.concatenate([y, y_upsample], axis=0).reshape(-1, 1)
    return X_upsample, y_upsample


def cv_split(n_splits, X, y):
    kf = KFold(n_splits=n_splits, shuffle=True)
    dataset_list = [{'X_train': X[train_index].copy(),
                     'y_train': y[train_index].copy(),
                     'X_val'  : X[val_index].copy(),
                     'y_val'  : y[val_index].copy()} for (train_index, val_index) in kf.split(X)]
    upsample_dataset_list = []
    for dataset in dataset_list:
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_train_upsample, y_train_upsample = upsampling(X_train, y_train)
        X_val = dataset['X_val']
        y_val = dataset['y_val']
        X_val_upsample, y_val_upsample = upsampling(X_val, y_val)
        upsample_dataset_list.append({
            'X_train': X_train_upsample,
            'y_train': y_train_upsample,
            'X_val'  : X_val_upsample,
            'y_val'  : y_val_upsample
        })
    return upsample_dataset_list


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers, num_hidden_units, nonlin=F.relu):
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
        X = self.output_layer(X)
        return X


def my_loss_function(output, target, model, l2_regularization_coefficient):
    loss = F.mse_loss(output, target)
    for W in model.parameters():
        loss += (l2_regularization_coefficient * W.norm(2).pow(2))
    return loss


def train_and_evaluate_mlp(ax_parameters, dataset, num_epochs):
    num_layers = ax_parameters.get('num_layers')
    num_neurons = ax_parameters.get('num_neurons')
    lr = ax_parameters.get('lr')
    l2_regularization_coefficient = ax_parameters.get('l2_regularization_coefficient')
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    X_train = torch.FloatTensor(X_train).cuda()
    X_val = torch.FloatTensor(X_val).cuda()
    y_train = torch.FloatTensor(y_train).cuda()
    y_val = torch.FloatTensor(y_val).cuda()
    model = MLP(
        input_dim=X_train.shape[1],
        output_dim=1,
        num_hidden_layers=num_layers,
        num_hidden_units=num_neurons
    )
    model = model.cuda()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        def closure():
            optimizer.zero_grad()
            output = model(X_train)
            loss = my_loss_function(output, y_train, model, l2_regularization_coefficient)
            loss.backward()
            return loss

        optimizer.step(closure)
    return pearson_corr(y_val.detach().cpu().numpy(),
                        model(X_val).detach().cpu().numpy())


def evaluate(ax_parameters, data, num_splits, num_epochs):
    X = data['X']
    y = data['y']
    dataset_list = cv_split(num_splits, X, y)
    score_list = [train_and_evaluate_mlp(ax_parameters, dataset, num_epochs)
                  for dataset in dataset_list]
    score_arr = np.array(score_list)
    if np.isnan(score_arr).any():
        return {'score': (0.0, 0.0)}
    else:
        return {'score': (score_arr.mean(), score_arr.std())}


def automatic_hyperparameter_search(drug_id, num_trails=50, num_splits=3, num_epochs=50,
                                    data_folder='../example_data/'):
    data_file_name = f'{data_folder}/{drug_id}.pickle'

    with open(data_file_name, 'rb') as f:
        data = pickle.load(f)

    if 'hyperparameter_search' in data.keys():
        return

    ax_client = AxClient()
    ax_client.create_experiment(
        name=f"drug_id_{drug_id}",
        parameters=[
            {
                "name"          : "lr",
                "value_type": 'float',
                "type"          : "range",
                "bounds"        : [1e-5, 1e0],
                "log_scale"     : True
            },
            {
                "name"          : "l2_regularization_coefficient",
                "value_type": 'float',
                "type"          : "range",
                "bounds"        : [1e-5, 1e0],
                "log_scale"     : True
            },
            {
                "name"          : "num_layers",
                "value_type": 'int',
                "type"          : "range",
                "bounds"        : [1, 5]
            },
            {
                "name"          : "num_neurons",
                "value_type": 'int',
                "type"          : "range",
                "bounds"        : [10, 100]
            },
        ],
        objective_name="score",
        minimize=False
    )

    for i in range(num_trails):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index,
                                 raw_data=evaluate(parameters, data, num_splits, num_epochs))

    best_parameters, values = ax_client.get_best_parameters()
    trace = ax_client.get_optimization_trace()

    data['hyperparameter_search'] = {}
    data['hyperparameter_search']['score'] = values[0]['score']
    data['hyperparameter_search']['neural_net_config'] = best_parameters

    with open(data_file_name, 'wb') as f:
        pickle.dump(data, f)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../example_data', type=str)
    parser.add_argument('--drug_id', default='11', type=str)
    args = parser.parse_args()
    automatic_hyperparameter_search(drug_id= args.drug_id,data_folder=args.data_dir)