from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

import numpy as np
import pandas as pd

from typing import Tuple


class CustomRandomForestClassifier():
    """
    Custom implementation of RF where we can change the bootstrapping method.
    """
    def __init__(self, 
               n_estimators: int = 100, 
               max_features: str = "sqrt",
               max_depth: int = None, 
               min_samples_leaf: int = 1, 
               random_state: int = None) -> None:
        
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf 
        self.random_state = random_state
      
    def fit(self, X: pd.DataFrame, y: np.array):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        assert isinstance(y, np.ndarray)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        trees = list()
        for i in range(self.n_estimators):

            # Get our bootstrapped data
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
        
            # Fit a decision tree
            tree = DecisionTreeClassifier(
              max_depth = self.max_depth,
              min_samples_leaf = self.min_samples_leaf,
              max_features = self.max_features,
              random_state = self.random_state+i
            )
            tree.fit(X_bootstrap, y_bootstrap)
            trees.append(tree)

        self.trees = trees
        return self

    def _bootstrap_sample(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Returns bootstrapped indices of X
        (same number of rows, sampled with replacement)

        Args:
          X: pandas dataframes

        Return:
          nd.array with indices
        """
        n_samples = X.shape[0]
        indices = np.random.randint(low=0, high=n_samples, size=n_samples)
        X_bootstrap = X.iloc[indices]
        y_bootstrap = y[indices]
        return X_bootstrap, y_bootstrap

    def predict_proba(self, X: pd.DataFrame):
        """
        Here we use a 'soft' voting ensemble
        average all probabilities

        See https://github.com/scikit-learn/scikit-learn/blob/1495f69242646d239d89a5713982946b8ffcf9d9/sklearn/ensemble/voting.py#L320
        """
        probas = np.asarray([clf.predict_proba(X) for clf in self.trees])
        avg = np.average(probas, axis=0)
        return avg


class StratifiedRandomForest(CustomRandomForestClassifier):    
    def _bootstrap_sample(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Stratified bootstrap sample.

        This means the class ratio should be the same in the bootstrap.
        """
        X_bootstrap, y_bootstrap = resample(X, y, stratify = y)
        return X_bootstrap, y_bootstrap
    

class BalancedRandomForest(CustomRandomForestClassifier):
    def _bootstrap_sample(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Balanced bootstrap. Implementation of Breiman's BalancedRandomForest.

        We create a dataset with an articial equal class distribution:
        first bootstraps the minority class and then samples with replacement the same number of cases from the majority class.
        """
        # Find majority class, 0 or 1
        counts = np.bincount(y)
        minority_class = np.argmin(counts)
        n_minority = counts[minority_class]

        # bootstrap minority class
        indices_minority = np.random.choice(
            np.where(y == minority_class)[0],
            size = n_minority,
            replace = True)

        # bootstrap majority class with minority size
        indices_majority = np.random.choice(
            np.where(y != minority_class)[0],
            size = n_minority,
            replace = True)

        indices = np.hstack([indices_majority, indices_minority])
        np.random.shuffle(indices) # in-place
        
        X_bootstrap = X.iloc[indices]
        y_bootstrap = y[indices]
        return X_bootstrap, y_bootstrap


class OverUnderRandomForest(CustomRandomForestClassifier):
    def _bootstrap_sample(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Oversample minority and undersample bootstrap. A variation on the BalancedRandomForest.

        We create a dataset with an artificial equal class distribution.
        Basically sample with replacement, but equal amounts for each class. So an undersample of majority class, oversample of minority.
        """
        # Determine number of samples from each class (50%-50%)
        n_majority = n_minority = int(len(y) / 2)
        if len(y) % 2 == 1:
            n_majority += 1
        assert n_minority + n_majority == len(y)

        # Find majority class, 0 or 1
        counts = np.bincount(y)
        majority_class = np.argmax(counts)

        # oversample minority class
        indices_minority = np.random.choice(
            np.where(y != majority_class)[0],
            size = n_minority,
            replace = True)

        # undersample majority class
        indices_majority = np.random.choice(
            np.where(y == majority_class)[0],
            size = n_majority,
            replace = True)

        indices = np.hstack([indices_majority, indices_minority])
        np.random.shuffle(indices) # in-place
        
        X_bootstrap = X.iloc[indices]
        y_bootstrap = y[indices]
        return X_bootstrap, y_bootstrap
        

