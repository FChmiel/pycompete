from sklearn.base import BaseEstimator, TransformerMixin
from numpy.typing import ArrayLike
from typing import Optional
import numpy as np
import pandas as pd


class RemoveCorrelated(BaseEstimator, TransformerMixin):
    def __init__(self, max_correlation: int = 0.95):
        """
        A transformer which removes highly correlated features.

        This is a naive implementation which removes correlated features, it is
        not stable for cases with groups of correlated features. This
        implementation keeps the features with the lowest column index.

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

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None):
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
        self.c_mat_ = np.absolute(np.corrcoef(X, rowvar=False))
        upper = np.triu(self.c_mat_, k=1)
        corr_cols = np.argwhere(upper > self.max_correlation)
        self.cols_to_drop = np.unique(corr_cols[:, 1])
        return self

    def transform(self, X: ArrayLike, y: Optional[ArrayLike] = None):
        """
        Transform X by removing the columns which are highly correlated.

        Parameters:
        -----------
        X : {array-like, sparse matrix}
            Training vector of shape (n_samples, n_features), where 'n_samples'
            is the number of samples and 'n_features' is the number of features.

        y : array-like of shape (n_samples,),
            Target vector relative to X. Not used, included for API
            compatability.

        Returns:
        --------
        X : {array-like, sparse matrix}
            Training vector of shape (n_samples, n_features), where 'n_samples'
        """
        # X is edited in-place, perhaps make a copy?
        X = np.delete(X, self.cols_to_drop, axis=1)
        return X
