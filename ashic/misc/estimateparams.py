import os
import numpy as np
from scipy import stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from sklearn.isotonic import IsotonicRegression
from statsmodels.base.model import GenericLikelihoodModel


def logspace_diagmean(mat, start=1, stop=None, num=100):
    n = mat.shape[0]
    if stop is None:
        stop = n-1
    seq = np.unique(np.logspace(np.log10(start), np.log10(stop),
                                num=num, dtype=int))
    m = seq.shape[0]
    dm = np.zeros(m)
    for i in np.arange(m):
        dm[i] = np.nanmean(np.diag(mat, seq[i]))
    return dm, seq


def diagmean(mat, start=1, stop=None):
    n = mat.shape[0]
    if stop is None:
        stop = n
    seq = np.arange(start, stop, dtype=int)
    m = seq.shape[0]
    dm = np.zeros(m)
    for i in np.arange(m):
        dm[i] = np.nanmean(np.diag(mat, seq[i]))
    return dm, seq


def estimate_alpha(c, d, start, plot=False, savefile=None):
    dmean, seq = diagmean(d, start=start)
    idx = ~np.isnan(dmean)
    # fit relation between d = A1*s^B1
    p1, _ = curve_fit(lambda s, A, B: A*np.power(s, B),
              seq[idx], dmean[idx])
    A1, B1 = p1
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(24, 8))
        axes[0].loglog(seq, dmean, label='distance')
        axes[0].loglog(seq, A1 * np.power(seq, B1), label='curve fit')
        axes[0].set_xlabel('diagonals')
        axes[0].set_ylabel('average distance')
        axes[0].legend()
    cmean, seq = diagmean(c, start=start)
    idx = ~np.isnan(cmean)
    # fit relation between c = A2*s^B2
    p2, _ = curve_fit(lambda s, A, B: A*np.power(s, B),
              seq[idx], cmean[idx])
    A2, B2 = p2
    if plot:
        axes[1].loglog(seq, cmean, label='contact')
        axes[1].loglog(seq, A2 * np.power(seq, B2), label='curve fit')
        axes[1].set_xlabel('diagonals')
        axes[1].set_ylabel('average contact')
        axes[1].legend()
        plt.savefig(savefile)
    # d=A1*s^B1 => s=(d/A1)^(1/B1)
    # c=A2*s^B2
    # c=[A2*A1^(-B2/B1)]*d^(B2/B1)
    # c=beta*d^alpha
    alpha = B2/B1
    return alpha


def estimate_beta(conmat, conpat, dmat, dpat, alpha_mat, alpha_pat, mask):
    beta = (np.sum(conmat[mask]) + np.sum(conpat[mask])) / \
           (np.sum(np.power(dmat[mask], alpha_mat)) + np.sum(np.power(dpat[mask], alpha_pat)))
    return beta


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
    mus = []
    gammas = []
    for idx, i in enumerate(seq):
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


def estimate_gamma(conmat, conpat, start, plot=False, outdir=None):
    # TODO remove outliers
    n = conmat.shape[0]
    x = np.arange(start, n)
    _, gammas_mat = fit_zip(conmat, x)
    _, gammas_pat = fit_zip(conpat, x)
    bins = np.unique(np.logspace(np.log10(x.min()), np.log10(x.max()),
                                 num=200, dtype=int))
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
        plt.savefig(os.path.join(outdir, 'simulated_gamma.png'))
    gammas = np.full(n-1, np.nan)
    gammas[start-1:] = newgammas
    return gammas


def sampling(beta, d, alpha):
    mu = beta * (d ** alpha)
    mu[np.isinf(mu)] = 0
    t = np.random.poisson(mu)
    np.fill_diagonal(t, 0)
    t[np.tri(t.shape[0], k=-1).astype(bool)] = 0
    t = (t + t.T).astype(float)
    return t


def plot_simulated(simmat, simpat, conmat, conpat, start, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))
    con_sim, seq_sim = logspace_diagmean(simmat, start=start, num=500)
    con_pop, seq_pop = logspace_diagmean(conmat, start=start, num=500)
    axes[0].loglog(seq_sim, con_sim, label='maternal simulated')
    axes[0].loglog(seq_pop, con_pop, label='maternal population')
    axes[0].legend()
    con_sim, seq_sim = logspace_diagmean(simpat, start=start, num=500)
    con_pop, seq_pop = logspace_diagmean(conpat, start=start, num=500)
    axes[1].loglog(seq_sim, con_sim, label='paternal simulated')
    axes[1].loglog(seq_pop, con_pop, label='paternal population')
    axes[1].legend()
    plt.savefig(os.path.join(outdir, 'simulated_population.png'))