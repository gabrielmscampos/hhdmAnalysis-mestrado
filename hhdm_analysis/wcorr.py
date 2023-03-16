import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch


def wmean(x: np.array, w: np.array):
    """
    Weighted Mean
    """
    return np.sum(x * w) / np.sum(w)


def wcov(x: np.array, y: np.array, w: np.array):
    """
    Weighted Covariance
    """
    return np.sum(w * (x - wmean(x, w)) * (y - wmean(y, w))) / np.sum(w)


def wcorr(x: np.array, y: np.array, w: np.array):
    """
    Weighted Correlation
    """
    return wcov(x, y, w) / np.sqrt(wcov(x, x, w) * wcov(y, y, w))


def pd_wcorr(df: pd.DataFrame, weights: np.array, rmv_neg_weights: bool = False):
    """
    Weighted Correlation of Pandas DataFrame
    """
    if rmv_neg_weights is True:
        df = df.iloc[np.where(weights > 0)[0]]

    mtx = [
        [wcorr(df[col_x], df[col_y], weights) for col_y in df.columns]
        for col_x in df.columns
    ]

    return pd.DataFrame(mtx, index=df.columns, columns=df.columns)


def cluster_corr(corr_array, inplace=False):
    """
    https://wil.yegelwel.com/cluster-correlation-matrix/

    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method="complete")
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(
        linkage, cluster_distance_threshold, criterion="distance"
    )
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]
