from sklearn.base import BaseEstimator, TransformerMixin
from sklearn._typing import ArrayLike, MatrixLike
from typing import Optional
import numpy as np 
import pandas as pd
    
class RemoveCorrelated(BaseEstimator, TransformerMixin):
    
    def __init__(self, max_correlation: int=0.95):
        """
        A transformer which removes highly correlated features.


        Parameters:
        -----------
        max_correlation : float,
            The maximum correlation between features before they are removed. 

        Attributes:
        -----------
        self.c_mat_ : np.array,
            The correlation matrix of X.
        """
        self.max_correlation = max_correlation

    def fit(self, X:MatrixLike, y:Optional[ArrayLike]=None):
        """
        Fit the transformer according to the given training data.

        Calculates the correlation matrix (c_mat_) of X and calculates
        indexes of columns which display more than 'max_correlation'
        to another column.

        Parameters:
        -----------
        X : {array-like, sparse matrix}
            Training vector of shape (n_samples, n_features), where 'n_samples'
            is the number of samples and 'n_features' is the number of features.

        y : array-like of shape (n_samples,),
            Target vector relative to X. Not used, included for API
            compatability.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        # TO DO: check all entries are float
        self.c_mat_ = np.corrcoef(X)
        correlated_mask = (self.c_mat_>self.max_correlation) & (self.c_mat_!=1)
        corr_cols = np.argwhere(correlated_mask)
        self.cols_to_drop = np.unique((np.sort(corr_cols, axis=1)), axis=0)[:,0]
        return self
    
    def transform(self, X, y=None):
        """
        """
        X = np.delete(X, self.cols_to_drop, axis=1)
        return X