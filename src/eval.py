
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler

import numpy as np
import pandas as pd
from typing import Dict

import tqdm

def eval_models(models: Dict, X, y, n_trials: int) -> pd.DataFrame:
    """
    Eval a set of models
    """
    perf = pd.DataFrame()

    seeds = range(n_trials)    
    for name, model in tqdm.tqdm(models.items()):
        p = eval_performance(model, X, y, name, seeds=seeds)
        perf = perf.append(p)
    
    perf = perf.sort_values(['auc_test'], ascending=False)
    return perf

def eval_performance(model, X, y, model_name = "", test_size = .2, oversample = False, seeds = [42]):
    
    auc_test = []
    auc_train = []
    
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = test_size, random_state = seed, stratify = y
        )
        if oversample:
            ros = RandomOverSampler(random_state=0)
            X_train, y_train = ros.fit_resample(X_train, y_train)
            X_train = pd.DataFrame(X_train)
            
        model.fit(X_train, y_train)
        metric_test = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        auc_test.append(metric_test)
        metric_train = roc_auc_score(y_train, model.predict_proba(X_train)[:,1])
        auc_train.append(metric_train)
        
    auc_train = round(np.mean(auc_train),4)
    auc_test = round(np.mean(auc_test),4)
    
    data = [model_name, auc_train, auc_test, auc_train - auc_test, len(seeds)]
    return pd.DataFrame([data], columns = ['model','auc_train', 'auc_test', 'delta','n_models'])
