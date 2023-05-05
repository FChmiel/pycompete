from typing import Iterator, Optional
import numpy as np
from numpy import ndarray
from sklearn._typing import ArrayLike, MatrixLike
from sklearn.model_selection._split import _BaseKFold, indexable


class TimeSeriesSplitGroupsPurged(_BaseKFold):
    def __init__(
        self, n_splits: int = 5, purge_size: int = 0, test_size: Optional[int] = None
    ):
        """
        Walk forward cross validation with purging between train and
        test.

        Splits are done at the group/moon level, so a instances from a given
        moon will appear exclusively in either the train or test set.

        Parameters:
        -----------
        n_splits : int, default=5
            Number of splits. Must be at least 2.

        purge_size : int, default=0,
            Number of groups/moons to purge between train and test sets. Must be at least 0.

        test_size : int, default=None
            Number of groups/moons to include in the test set. If None, test_size is set to  n_groups // n_folds.
        """
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.purge_size = purge_size
        self.test_size = test_size

    def split(
        self,
        X: MatrixLike,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.


        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set. This should be the 'Moon' column, and must be
            chronologically ordered, for example: [1,1,1,1,2,2,2] and not
            [2,2,2,1,1,1,1].

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        unique_groups = np.sort(np.unique(groups))
        n_groups = len(unique_groups)
        test_size = (
            self.test_size if self.test_size is not None else n_groups // n_folds
        )
        purge_size = self.purge_size

        if n_splits > n_groups:
            raise ValueError(
                f"Number of folds cannot exceed the number of groups ({n_groups})."
            )

        if purge_size > n_groups - n_splits * test_size:
            raise ValueError(
                f"Purge size ({purge_size}) too large for number of groups and test_size."
            )

        test_starts = range(n_groups - n_splits * test_size, n_groups, test_size)
        train_ends = [test_start - purge_size for test_start in test_starts]

        for test_start, train_end in zip(test_starts, train_ends):
            test_groups = unique_groups[test_start : test_start + test_size]
            train_groups = unique_groups[:train_end]
            yield (
                np.argwhere(np.isin(groups, train_groups))[:, 0],
                np.argwhere(np.isin(groups, test_groups))[:, 0],
            )
