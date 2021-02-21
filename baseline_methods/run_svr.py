
import pickle
import argparse
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet, ElasticNetCV
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

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
    parser.add_argument('--data_dir', type=str, default='../example_data/')
    parser.add_argument('--model_dir', type=str, default='../model/')
    args = parser.parse_args()

    data_file_name = f'{args.data_dir}/{args.drug_id}.pickle'

    with open(data_file_name, 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    y = data['y']
    
    
    parameters = {'C':range(1,101,1)}
    svr = SVR()
    svr_cv = GridSearchCV(svr, parameters, cv=cv_split(5,X,y))
    svr_cv.fit(X,y)

    best_dict = svr_cv.best_params_
    svr_model = SVR(C=best_dict['C'])
    score = cross_val_score(svr_model, X, y, cv=cv_split(5,X,y), scoring=make_scorer(pearson_corr), n_jobs=1)

    data['SVR_C'] = best_dict['C']
    data['SVR_score'] = np.nanmean(score)

    with open(f'{args.model_dir}/{args.drug_id}_svr.pkl', 'wb') as f:
        pickle.dump(data, f)
        

