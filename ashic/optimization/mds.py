"""
Adapted from PASTIS: https://github.com/hiclib/pastis
"""

import numpy as np
from scipy import optimize
from scipy import sparse
from sklearn.utils import check_random_state
from sklearn.metrics import euclidean_distances


def compute_wish_distances(counts, alpha=-3., beta=1., bias=None):
    if beta == 0:
        raise ValueError("beta cannot be equal to 0.")
    counts = counts.copy()
    if sparse.issparse(counts):
        if not sparse.isspmatrix_coo(counts):
            counts = counts.tocoo()
        if bias is not None:
            bias = bias.flatten()
            counts.data /= bias[counts.row] * bias[counts.col]
        wish_distances = counts / beta
        wish_distances.data[wish_distances.data != 0] **= 1. / alpha
        return wish_distances
    else:
        wish_distances = counts.copy() / beta
        wish_distances[wish_distances != 0] **= 1. / alpha

        return wish_distances


def smooth_intra(distances, h, diag=0):
    if sparse.issparse(distances):
        d = distances.toarray()
    else:
        d = np.array(distances)
    n = d.shape[0]
    d[d == 0] = np.nan
    d[np.isinf(d)] = np.nan
    # set the lower triangle to np.nan
    d[np.tril_indices(n, k=diag)] = np.nan
    # find valid loci
    notna = ~np.isnan(d)
    loci = (notna.sum(axis=0) + notna.sum(axis=1)) > 0
    smooth_dis = np.full(d.shape, np.nan, dtype=float)
    np.fill_diagonal(smooth_dis, 0)
    for gdis in range(diag+1, n):
        for i in np.where(loci)[0]:
            j = i + gdis
            if j < n and loci[j] and np.isnan(d[i, j]):
                # mean filter
                low = max(0, i-h)
                upper = min(i+h, n-1)
                left = max(0, j-h)
                right = min(j+h, n-1)
                m = np.nanmean(d[low:upper+1, left:right+1])
                # shortest distance
                dpair = smooth_dis[i, i:j+1] + smooth_dis[i:j+1, j]
                shortestd = np.nanmin(dpair)
                smooth_dis[i, j] = np.nanmin(np.array([m, shortestd]))
                smooth_dis[j, i] = smooth_dis[i, j]
            if j < n and loci[j] and not np.isnan(d[i, j]):
                smooth_dis[i, j] = smooth_dis[j, i] = d[i, j]
    smooth_dis[np.isnan(smooth_dis)] = 0
    return smooth_dis


def MDS_obj(X, distances):
    X = X.reshape(-1, 3)
    dis = euclidean_distances(X)
    X = X.flatten()
    return ((dis - distances)**2).sum()


def MDS_obj_sparse(X, distances):
    X = X.reshape(-1, 3)
    dis = np.sqrt(((X[distances.row] - X[distances.col])**2).sum(axis=1))
    return ((dis - distances.data)**2 / distances.data**2).sum()


def MDS_gradient(X, distances):
    X = X.reshape(-1, 3)
    m, n = X.shape
    tmp = X.repeat(m, axis=0).reshape((m, m, n))
    dif = tmp - tmp.transpose(1, 0, 2)
    dis = euclidean_distances(X).repeat(3, axis=1).flatten()
    distances = distances.repeat(3, axis=1).flatten()
    grad = dif.flatten() * (dis - distances) / dis / distances.data**2
    grad[(distances == 0) | np.isnan(grad)] = 0
    X = X.flatten()
    return grad.reshape((m, m, n)).sum(axis=1).flatten()


def MDS_gradient_sparse(X, distances):
    X = X.reshape(-1, 3)
    dis = np.sqrt(((X[distances.row] - X[distances.col])**2).sum(axis=1))

    grad = ((dis - distances.data) /
            dis / distances.data**2)[:, np.newaxis] * (
        X[distances.row] - X[distances.col])
    grad_ = np.zeros(X.shape)

    for i in range(X.shape[0]):
        grad_[i] += grad[distances.row == i].sum(axis=0)
        grad_[i] -= grad[distances.col == i].sum(axis=0)

    X = X.flatten()
    return grad_.flatten()


def estimate_X(counts, alpha=-3., beta=1., ini=None,
               verbose=0,
               bias=None,
               factr=1e12,
               precompute_distances="auto",
               random_state=None,
               maxiter=10000,
               smooth=False,
               h=0,
               diag=0,
               numchr=1):
    n = counts.shape[0]

    random_state = check_random_state(random_state)
    if ini is None or ini == "random":
        ini = 1 - 2 * random_state.rand(n * 3)

    if precompute_distances == "auto":
        distances = compute_wish_distances(counts, alpha=alpha, beta=beta,
                                           bias=bias)
        if smooth:
            if numchr == 1:
                distances = smooth_intra(distances, h=h, diag=diag)
                distances = sparse.coo_matrix(distances)
            elif numchr == 2:
                disarray = distances.toarray()
                m = int(n/2)
                # smooth intra-chromosomal distance
                disarray[:m, :m] = smooth_intra(disarray[:m, :m], h=h, diag=diag)
                disarray[m:, m:] = smooth_intra(disarray[m:, m:], h=h, diag=diag)
                # TODO smooth inter-chromosomal distance
                distances = sparse.coo_matrix(disarray)
            else:
                raise ValueError("The number of chromosomes should be 1 or 2.")
    elif precompute_distances == "precomputed":
        distances = counts

    results = optimize.fmin_l_bfgs_b(
        MDS_obj_sparse, ini.flatten(),
        MDS_gradient_sparse,
        (distances, ),
        iprint=verbose,
        factr=factr,
        maxiter=maxiter)
    return results[0].reshape(-1, 3)


class MDS(object):
    def __init__(self, alpha=-3., beta=1.,
                 max_iter=5000, random_state=None,
                 precompute_distances="auto", bias=None,
                 init=None, verbose=False, factr=1e12,
                 smooth=False, h=0, diag=0, numchr=1):
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.random_state = check_random_state(random_state)
        self.precompute_distances = precompute_distances
        self.init = init
        self.verbose = verbose
        self.bias = bias
        self.factr = factr
        self.smooth = smooth
        self.h = h
        self.diag = diag
        self.numchr = numchr

    def fit(self, counts):
        if not sparse.isspmatrix_coo(counts):
            counts = sparse.coo_matrix(counts)

        X_ = estimate_X(counts,
                        alpha=self.alpha,
                        beta=self.beta,
                        ini=self.init,
                        verbose=self.verbose,
                        precompute_distances=self.precompute_distances,
                        random_state=self.random_state,
                        bias=self.bias,
                        factr=self.factr,
                        maxiter=self.max_iter,
                        smooth=self.smooth,
                        h=self.h,
                        diag=self.diag,
                        numchr=self.numchr)
        return X_
