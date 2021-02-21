import pickle
import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet, ElasticNetCV
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

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
    index_subset = index[:int(index.size*0.2)]
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

    drug_id = args.drug_id
    data_file_name = f'{args.data_dir}/{args.drug_id}.pickle'

    with open(data_file_name, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y = data['y']

    e_net_cv = ElasticNetCV(n_alphas=100, max_iter=10000, cv=cv_split(5,X,y), n_jobs=1)
    e_net_cv.fit(X,y)

    e_net = ElasticNet(l1_ratio=0.5, alpha=e_net_cv.alpha_)
    score = cross_val_score(e_net, X, y, cv=cv_split(5,X,y), scoring=make_scorer(pearson_corr), n_jobs=1)

    data['elastic_net_alpha'] = e_net_cv.alpha_
    data['elastic_net_score'] = np.nanmean(score)

    with open(f'{args.model_dir}/{args.drug_id}_elastic_net.pkl', 'wb') as f:
        pickle.dump(data, f) 
