import argparse
import pickle

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV


def pearson_corr(y, y_pred, **kwargs):
    if np.isnan(y).any() or np.isinf(y).any():
        return np.nan
    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        return np.nan
    y = y.reshape(-1)
    y_pred = y_pred.reshape(-1)
    corr, _ = pearsonr(y, y_pred)
    return corr


def upsampling(index, y):
    index = index[np.argsort(y.reshape(-1)[index])]
    index_subset = index[:int(index.size * 0.2)]
    index_upsample = np.tile(index_subset, 3)
    index_upsample = np.concatenate([index, index_upsample])
    return index_upsample


def cv_split(n_splits, X, y):
    kf = KFold(n_splits=n_splits, shuffle=True)
    index_list = [(train_index, val_index) for (train_index, val_index) in kf.split(X)]
    upsample_index_list = []
    for train_index, val_index in index_list:
        upsample_train_index = upsampling(train_index, y)
        upsample_val_index = upsampling(val_index, y)
        upsample_index_list.append((upsample_train_index, upsample_val_index))
    return upsample_index_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--drug_id', type=str, help='drug ID',default='11')
    parser.add_argument('--data_dir', type=str,default='../example_data/')
    parser.add_argument('--model_dir', type=str, default='../model/')
    args = parser.parse_args()

    data_file_name = f'{args.data_dir}/{args.drug_id}.pickle'

    with open(data_file_name, 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    y = data['y']

    parameters = {'max_depth': [2, 4, 8, 16], 'n_estimators': [8, 16, 32, 64, 128]}
    rf = RandomForestRegressor()
    rf_cv = GridSearchCV(rf, parameters, cv=cv_split(5, X, y))
    rf_cv.fit(X, y)

    best_dict = rf_cv.best_params_
    rf_model = RandomForestRegressor(max_depth=best_dict['max_depth'], n_estimators=best_dict['n_estimators'])
    score = cross_val_score(rf_model, X, y, cv=cv_split(5, X, y), scoring=make_scorer(pearson_corr), n_jobs=1)

    data['RF_params'] = best_dict
    data['RF_score'] = np.nanmean(score)

    with open(f'{args.model_dir}/{args.drug_id}_rf.pkl', 'wb') as f:
        pickle.dump(data, f)
