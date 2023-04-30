import numpy as np


def sharpe_ratio(preds: np.array, targets: np.array, era: np.array) -> float:
    """
    Compute the Sharpe ratio for the predictions.

    The sharpe ratio is defined

    Parameters
    ----------
    preds : np.array
        Predictions to be evaluated.

    targets : np.array
        Targets used for evaluation.

    era : np.array
        Era used for evaluation, this should be the 'moon' of each respective
        instance.

    Returns
    -------
    float
        Sharpe ratio of the predictions.
    """
    moons = np.unique(era)
    correlations = [
        np.corrcoef(preds[era == moon], targets[era == moon])[0, 1] for moon in moons
    ]
    mean_correlation = np.mean(correlations)
    std_correlation = np.std(correlations)
    return mean_correlation / std_correlation
