
"""
Simulations with balanced random forest for imbalanced data.

To run: 

```bash
seq 0 10 | parallel -j 4 python run_simulations.py
```
"""

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import BalancedRandomForestClassifier

from datetime import datetime
import numpy as np
import pandas as pd

import random
import sys
import random

from src.RandomForest import CustomRandomForestClassifier, StratifiedRandomForest, BalancedRandomForest, OverUnderRandomForest
from src.eval import eval_models


def simulate_performance(imbalance = .9, n_tree = 100, seeds = [42]):
            
    print(f"{datetime.now().time()}: Start imbalance={imbalance}, trees={n_tree}")
    run = pd.DataFrame()
    
    X, y = make_classification(
                n_samples = 10000, 
                n_features = 20, 
                n_informative=15, 
                n_redundant=2, 
                n_classes=2, 
                flip_y=0.05,
                weights = [imbalance], 
                random_state = 42)
    X = pd.DataFrame(X)

    rf_params = {"max_depth": 7, "min_samples_leaf": 200, "n_estimators": n_tree, "max_features": 'sqrt', "random_state": 42}

    run = eval_models(models={
        'sklearn RF' : RandomForestClassifier(**rf_params),
        'custom RF' : CustomRandomForestClassifier(**rf_params),
        'class_weight balanced RF' : RandomForestClassifier(**rf_params, class_weight="balanced"),
        'Stratified RF' : StratifiedRandomForest(**rf_params),
        'Balanced RF' : BalancedRandomForest(**rf_params),
        'OverUnder RF' : OverUnderRandomForest(**rf_params),
        'imblearn balancedRF' : BalancedRandomForestClassifier(**rf_params),
        }, X=X, y=y, n_trials=100
    )

    # results
    run['imbalance'] = imbalance
    run['n_trees'] = n_tree
    return run


if __name__ == "__main__":

    random_seeds = random.sample(range(10000), 100)
    imbalances = [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99]
    
    # From bash `seq`
    i = int(sys.argv[1])
    
    print(f"Running simulation {i}")
    simulations = simulate_performance(
        imbalance=imbalances[i],
        n_tree=100,
        seeds = random_seeds
    )
    
    filename = "output/simulations-" + format(datetime.now(),"%Y%m") + ".csv"
    with open(filename, 'a') as f:
        simulations.to_csv(f, header=f.tell()==0, index = False)
    
