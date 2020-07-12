"""
Estimate initial gamma per diagonal.
step 1: initial amb counts assignment by allele-certain counts 
step 2: estimate intra-gamma per diagonal, binning, spline fitting and istonic regression
step 3: estimate inter-gamma for whole matrix
ref: https://austinrochford.com/posts/2015-03-03-mle-python-statsmodels.html
"""
import os
import click
import numpy as np
from scipy import stats
import cPickle as pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.isotonic import IsotonicRegression
from statsmodels.base.model import GenericLikelihoodModel
from allelichicem.utils import init_counts, join_matrix


def mask_data(mat, mask, loci):
    """
    set filtered bins to NaN when estimate gamma
    """
    mat[~mask] = np.nan
    mat[~loci, :] = np.nan
    mat[:, ~loci] = np.nan
    return mat


def zip_pmf(x, pi, lambda_):
    if pi < 0 or pi > 1 or lambda_ <= 0:
        return np.zeros_like(x)
    else:
        return (x == 0) * pi + (1 - pi) * stats.poisson.pmf(x, lambda_)


class ZeroInflatedPoisson(GenericLikelihoodModel):
    def __init__(self, endog, exog=None, **kwds):
        if exog is None:
            exog = np.zeros_like(endog)
        super(ZeroInflatedPoisson, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        pi = params[0]
        lambda_ = params[1]
        return -np.log(zip_pmf(self.endog, pi=pi, lambda_=lambda_))

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            p0 = (self.endog == 0).mean()
            excess_zeros = p0 * 0.5
            lambda_start = self.endog.sum() * 1. / (self.endog.size * (1 - excess_zeros))
            start_params = np.array([excess_zeros, lambda_start])
        return super(ZeroInflatedPoisson, self).fit(start_params=start_params,
                                                maxiter=maxiter, maxfun=maxfun, **kwds)


def fit_zip(mat, seq):
    """
    estimate ZIP params mu and gamma per diagonal
    """
    mus = []
    gammas = []
    for i in seq:
        x = np.diag(mat, i)
        if (~np.isnan(x)).sum() > 0:
            model = ZeroInflatedPoisson(x[~np.isnan(x)])
            results = model.fit()
            gamma = 1. - results.params[0]
            mu = results.params[1]
            gammas.append(gamma)
            mus.append(mu)
        else:
            gammas.append(np.nan)
            mus.append(np.nan)
    return np.array(mus), np.array(gammas)


def estimate_intra_gamma(conmat, conpat, start, nbins=200, plot=False, outdir=None):
    n = conmat.shape[0]
    x = np.arange(start, n)
    _, gammas_mat = fit_zip(conmat, x)
    _, gammas_pat = fit_zip(conpat, x)
    bins = np.unique(np.logspace(np.log10(x.min()), np.log10(x.max()),
                                 num=nbins, dtype=int))
    binning = []
    for i in range(len(bins) - 1):
        binning.append((bins[i], bins[i + 1] - 1))
    meangammas = map(lambda t: np.nanmean(
                     np.concatenate((gammas_mat[t[0]-start:t[1]-start+1],
                                     gammas_pat[t[0]-start:t[1]-start+1]))),
                     binning)
    meanbinning = map(lambda t: np.nanmean(np.arange(t[0], t[1] + 1)), binning)
    meanbinning = np.append(meanbinning, x.max())
    meangammas = np.append(meangammas, meangammas[-1])
    uspl = UnivariateSpline(x=meanbinning, y=meangammas,
                            s=min(np.nanmin(gammas_mat), np.nanmin(gammas_pat)) ** 2)
    ir = IsotonicRegression(increasing=False)
    newgammas = ir.fit_transform(x, uspl(x))
    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x, gammas_mat, 'r.', alpha=0.5, label='maternal gamma')
        ax.plot(x, gammas_pat, 'c.', alpha=0.3, label='paternal gamma')
        ax.plot(meanbinning, meangammas, 'yo', label='binning gamma')
        ax.plot(x, uspl(x), 'g-', lw=2, label='spline fitting')
        ax.plot(x, newgammas, 'b-', lw=2, label='isotonic regression')
        ax.legend()
        plt.savefig(os.path.join(outdir, 'init_gamma.png'))
    gammas = np.full(n-1, np.nan)
    gammas[start-1:] = newgammas
    return gammas


def estimate_inter_gamma(conab):
    # estimate inter-gamma using whole inter-contact matrix
    model = ZeroInflatedPoisson(conab[~np.isnan(conab)])
    results = model.fit()
    gamma = 1. - results.params[0]
    return gamma


def estimate_gamma(data, mask, loci, diag, out, nbins):
    # use allele-certain as init Poisson lambda
    initcon = init_counts(
        join_matrix(data['aa'], data['ab'], data['ba'], data['bb']),
        data['ax'], data['bx'], data['xx']
    )
    n = data['aa'].shape[0]
    # set filtered bins as NaN to exclude from gamma estimation
    taa = mask_data(initcon[:n, :n], mask, loci)
    tbb = mask_data(initcon[n:, n:], mask, loci)
    tab = mask_data(initcon[:n, n:], mask, loci)
    intra_gamma = estimate_intra_gamma(taa, tbb, 
                                       start=diag+1, 
                                       nbins=nbins,
                                       plot=True, 
                                       outdir=out)
    inter_gamma = estimate_inter_gamma(tab)
    gamma = np.append(intra_gamma, inter_gamma)
    return gamma


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("out")
@click.option("--nbins", default=200, type=int)
def cli(filename, out, nbins):
    if not os.path.exists(out):
        os.makedirs(out)
    with open(filename, 'rb') as fh:
        pk = pickle.load(fh)
        data = pk["obs"]
        params = pk["params"]
        gamma = estimate_gamma(data, params["mask"], 
                               params["loci"], params["diag"], out, nbins)
        np.savetxt(os.path.join(out, "init_gamma.txt"), gamma)


if __name__ == "__main__":
    cli()

