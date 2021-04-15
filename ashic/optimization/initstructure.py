import numpy as np
from ashic.optimization import rmds
from ashic.optimization.poissonmodel import estimate_x
from ashic.optimization.poissonmodelinter import estimate_rotation
from operator import itemgetter


def apply_mask(t, mask):
    t[~np.tile(mask, (2,2))] = np.nan
    return t

def sub_aa(mat):
    n = mat.shape[0] / 2
    return mat[:n,:n]

def sub_bb(mat):
    n = mat.shape[0] / 2
    return mat[n:,n:]

def sub_ab(mat):
    n = mat.shape[0] / 2
    return mat[:n,n:]

def unpack_params(params):
    f = itemgetter('n', 'alpha_mat', 'alpha_pat', 'alpha_inter', 'beta', 'bias')
    return f(params)

def mds_structure(t, params, mask, loci, seed, factr):
    t = apply_mask(t, mask)
    n, am, ap, ai, beta, bias = unpack_params(params)
    xm = rmds.haploid(sub_aa(t), alpha=am, beta=beta, verbose=1, seed=seed, factr=factr)
    xp = rmds.haploid(sub_bb(t), alpha=ap, beta=beta, verbose=1, seed=seed, factr=factr)
    return rmds.combine(t, xm, xp, alpha=ai, beta=beta, loci=loci, verbose=1, seed=seed)

def mds(t, params, mask, loci, seed, ensemble, n_structure, factr=1e5):
    if not ensemble:
        return mds_structure(t, params, mask, loci, seed, factr)
    x = []
    for sd in range(seed, seed + n_structure):
        x.append(mds_structure(
            t, params, mask, loci, sd, factr
        ))
    return x

def poisson_structure(t, params, mask, loci, seed, maxiter):
    t = apply_mask(t, mask)
    n, am, ap, ai, beta, bias = unpack_params(params)
    triu_mask = np.triu(mask)
    xm = estimate_x(sub_aa(t), alpha=np.full((n,n), am), beta=beta,
                    bias=bias[:n], mask=triu_mask, symask=mask, 
                    random_state=np.random.RandomState(seed=seed),
                    maxiter=maxiter)
    xp = estimate_x(sub_bb(t), alpha=np.full((n,n), ap), beta=beta, 
                    bias=bias[n:], mask=triu_mask, symask=mask, 
                    random_state=np.random.RandomState(seed=seed),
                    maxiter=maxiter)
    cdist = np.power(
        np.nansum(sub_ab(t)) / 
        (np.nansum(np.outer(bias[:n], bias[n:])[mask]) * beta), 
        1. / ai)
    xm -= np.nanmean(xm, axis=0)
    xp -= np.nanmean(xp, axis=0)
    xp += np.tile([cdist, 0, 0], (n, 1))
    x = np.concatenate((xm, xp), axis=0)
    x = estimate_rotation(x=x, tab=sub_ab(t), alpha=ai, beta=beta, 
                            mask=mask, loci=loci, bias=bias)
    x[~np.tile(loci, 2), :] = np.nan
    return x

def poisson(t, params, mask, loci, seed, ensemble, n_structure, maxiter=2000):
    if not ensemble:
        return poisson_structure(t, params, mask, loci, seed, maxiter)
    x = []
    for sd in range(seed, seed + n_structure):
        x.append(poisson_structure(
            t, params, mask, loci, sd, maxiter
        ))
    return x