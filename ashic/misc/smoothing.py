import numpy as np


def mean_filter(mat, mask=None, h=1):
    x = np.array(mat)
    n = x.shape[0]
    if mask is None:
        mask = np.invert(np.isnan(mat))
        # make sure to exclude main diagonal and lower traingle
        mask = np.triu(mask, k=1)
    # !!! treat zeros as missing
    x[x == 0] = np.nan
    x[~mask] = np.nan
    y = np.full((n, n), np.nan, dtype=float)
    # only use upper traingle indices
    # !!! do not include lower traingle in window
    indices = np.argwhere(np.triu(mask))
    for i, j in indices:
        low = max(0, i-h)
        upper = min(i+h, n-1)
        left = max(0, j-h)
        right = min(j+h, n-1)
        y[i, j] = np.nanmean(x[low:upper+1, left:right+1])
        y[j, i] = y[i, j]
    return y
