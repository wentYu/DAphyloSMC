# DASMC - [module description]
# Copyright (C) 2026 Your Name / Your Institution
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import math

def find_x(a, alpha=0.95):
    if not a:
        raise ValueError("Null list!")
    if not 0 < alpha <= 1:
        raise ValueError("Invalid alpha out of [0,1]!")

    n = len(a)
    k = int(math.ceil(alpha * n))
    sorted_a = sorted(a)
    threshold = sorted_a[-k]
    return -threshold


def RandomForestValidation(X, y,  mark, dataset, random_seed=51):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=51
    )

    if dataset != 'M1809':
        param_grid = {
            'n_estimators': [50],
            'max_depth': [50],
            'min_samples_split': [2],
            'min_samples_leaf': [2],
            'max_features': [0.5],
            'bootstrap': [True],
            'random_state': [random_seed],
            'n_jobs': [-1]
        }
    else:
        param_grid = {
            'n_estimators': [100],
            'max_depth': [100],
            'min_samples_split': [3],
            'min_samples_leaf': [3],
            'max_features': [0.5],
            'bootstrap': [True],
            'random_state': [random_seed],
            'n_jobs': [-1]
        }

    #if need model selection

    # grid =GridSearchCV(
    #     RandomForestRegressor(random_state=random_seed, n_jobs=-1),
    #     param_grid,
    #     cv=5,
    #     scoring='neg_mean_squared_error',
    #     return_train_score=True,
    #     verbose=1,
    #     n_jobs=-1
    # )

    # grid.fit(X_train, y_train)
    # print(f"\n best params: {grid.best_params_}")
    # best_model = grid.best_estimator_
    # best_params = grid.best_params_

    params = {k: v[0] for k, v in param_grid.items()}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    best_train_mse = mean_squared_error(y_train, y_pred_train)
    best_test_mse = mean_squared_error(y_test, y_pred)

    print(f"training MSE: {best_train_mse:.6f}")
    print(f"testing MSE: {best_test_mse:.6f}")

    importance = model.feature_importances_
    print(f"feature importance range: [{importance.min():.4f}, {importance.max():.4f}]")

    model_filename = f'./RF_Output/RF_model_{dataset}_{mark}.joblib'
    dump(model, model_filename, compress=9)
    print(f"model save as: {model_filename}")

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred)

    return {
        'best_model': model,
        'best_train_mse': best_train_mse,
        'best_test_mse': best_test_mse,
        'importance': importance,
        'r2_train': r2_train,
        'r2_test': r2_test
    }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', default='primates')
    parser.add_argument('-m', '--mark', default='0')
    parser.add_argument('-a', '--advise_Delta', default='0')
    parser.add_argument('-K', '--K', default='-1000')
    parser.add_argument('-nP', '--nParticles', default='50')
    args = parser.parse_args()
    dataset = str(args.dataset)
    mark = str(args.mark)
    advise_Delta = int(args.advise_Delta)
    K = float(args.K)
    nParticles = int(args.nParticles)

    os.makedirs('./RF_Output',exist_ok=True)

    try:
        with open('./pilot_output/feature_' + dataset + '_' + mark+ '.txt',
                  'r') as f:
            lines = f.readlines()
    except Exception as e:
        with open('feature_' + dataset + '_' + mark + '.txt', 'r') as f:
            lines = f.readlines()


    all_Y = []
    all_Y_True = []

    for line in lines:
        all_Y_True.append(float(line.split(' ')[-5]))

    percent = np.percentile(all_Y_True, 2.5)
    for line in lines:

        if float(line.split(' ')[-5]) >= percent:
            for j in range(2):
                all_Y.append(float(line.split(' ')[-5]))
        all_Y.append(float(line.split(' ')[-5]))

    all_X = []
    all_X_R = []
    for line in lines:
        if float(line.split(' ')[-5]) >= percent:
            for j in range(2):
                all_X_R.append([float(x) for x in line.split(' ')[1:-6]])
        all_X.append([float(x) for x in line.split(' ')[1:-6]])
        all_X_R.append([float(x) for x in line.split(' ')[1:-6]])

    all_X = np.array(all_X)
    all_X_R = np.array(all_X_R)

    feature_names = ['Move mode', 'Total branch lengths', 'Longest branch', 'Variance of branch lengths',
                     'Prunning branch length', 'Regrafting branch length',
                     'Prunning and regrafting branch length ratio',
                     'Number of species influenced', 'Total branch lengths influenced', 'Longest branch influenced',
                     'Variance of branch lengths influenced',
                     'Topology distance from the pruned node', 'Branch length distance from the pruned node ',
                     'Longest branch from the pruned node',
                     'Variance of branch lengths from the pruned node ', 'Number of species in subtrees 1',
                     'Number of species in subtrees 2', 'Number of species in subtrees 3',
                     'Number of species in subtrees 4',
                     'Total branch lengths in subtrees 1', 'Total branch lengths in subtrees 2',
                     'Total branch lengths in subtrees 3', 'Total branch lengths in subtrees 4',
                     'Subtree 1 and subtree 3 branch length ratio', 'Longest branch of subtrees 1',
                     'Longest branch of subtrees 2', 'Longest branch of subtrees 3', 'Longest branch of subtrees 4',
                     'Variance of branch lengths of subtrees 1', 'Variance of branch lengths of subtrees 2',
                     'Variance of branch lengths of subtrees 3', 'Variance of branch lengths of subtrees 4',
                     'Long branch attraction risk',
                     'Pruned and regrafted branch length sum', 'Pruned and regrafted branch length product'
        , 'rAT_before', 'rAT_after'
        , 'rAC_before', 'rAC_after', 'rAG_before', 'rAG_after', 'rCT_before', 'rCT_after',
                     'rGT_before', 'rGT_after'
                     ]
    if 'GTR' in mark:
        feature_names = feature_names
    elif 'K2P' in mark or 'JC69' not in mark:
        feature_names = feature_names[:-8]
    else:
        feature_names = feature_names[:-10]

    corr_matrix = pd.DataFrame(all_X_R, columns=feature_names).corr(method='pearson')
    abs_corr = np.abs(corr_matrix.values)
    np.fill_diagonal(abs_corr, np.nan)  # 忽略对角线
    mac = np.nanmean(abs_corr)
    print(f"mean absolute correlation: {mac:.3f}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_X_R)
    U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    eigenvalues_svd = s ** 2 / (X_scaled.shape[0] - 1)

    cumulative_variance = np.cumsum(eigenvalues_svd) / np.sum(eigenvalues_svd)
    effective_dim = np.argmax(cumulative_variance >= 0.95) + 1

    print('effective dimensions=', effective_dim)

    random_seed=51
    result=RandomForestValidation(all_X_R, all_Y,  mark=mark, dataset=dataset,
                                              random_seed=random_seed)

    if advise_Delta == 1:
        best_model = result['best_model']
        rand_acc = []
        log_phi = []
        log_likelihood_ratio = []
        log_prior_ratio = []
        log_proposal_ratio = []
        log_likelihood_list = []
        ML_log_likelihood_list = []
        for line in lines:
            ones = line.split(' ')

            log_phi.append(float(ones[-6]))
            log_likelihood_ratio.append(float(ones[-5]))
            log_prior_ratio.append(float(ones[-4]))
            log_proposal_ratio.append(float(ones[-3]))
            rand_acc.append(float(ones[-2]))
            log_likelihood_list.append(float(ones[-1]))
        sur_log_likelihood_ratio_list = best_model.predict(all_X)
        for k in range(len(lines)):
            if k >= nParticles:
                ML_log_likelihood_list.append(sur_log_likelihood_ratio_list[k] + log_likelihood_list[k - nParticles])
            else:
                ML_log_likelihood_list.append(log_likelihood_list[k])
        our_log_likelihood_ratio_1 = np.subtract(np.add(sur_log_likelihood_ratio_list, np.array(log_likelihood_list)),
                                                 np.fmin(
                                                     np.array(log_likelihood_list), np.array(ML_log_likelihood_list)))

        check_list = np.fmin((np.array(log_prior_ratio) + np.array(log_proposal_ratio) - np.log(
            rand_acc)) / np.exp(log_phi) + our_log_likelihood_ratio_1,
                             np.array(sur_log_likelihood_ratio_list) - K)
        recall_ratio_list = [0.90, 0.95, 0.975]
        for recall_ratio in recall_ratio_list:
            delta = find_x(list(check_list), recall_ratio)
            print(f'advise delta under recall ratio {recall_ratio}: {delta}')

if __name__ == '__main__':
    main()