from scipy.stats import poisson
from sklearn.metrics import euclidean_distances
import itertools
import numpy as np
import iced
from allelichicem.model.basemodel import BaseModel
from allelichicem.optimization.zipmodel import estimate_x
from allelichicem.utils import encodejson
from allelichicem.utils import centroid_distance
import json


def poisson_lambda(x, cdis, beta, alpha, bias=None):
    d = euclidean_distances(x)
    n = int(x.shape[0] / 2)
    # set inter distance to centroid distance
    d[:n, n:] = cdis
    d[n:, :n] = cdis
    if bias is None:
        bias = np.ones(d.shape[0], dtype=float)
    lambda_mat = (beta * d ** alpha) * np.outer(bias, bias)
    return lambda_mat.astype(float)


def disjoin_matrix(mat, n, mask=None):
    if mask is None:
        return mat[:n, :n], mat[:n, n:], mat[n:, :n], mat[n:, n:]
    else:
        return mat[:n, :n][mask], mat[:n, n:][mask], mat[n:, :n][mask], mat[n:, n:][mask]


def join_matrix(m1, m2, m3, m4):
    r1 = np.concatenate((m1, m2), axis=1)
    r2 = np.concatenate((m3, m4), axis=1)
    return np.concatenate((r1, r2), axis=0)


def fill_diagonal(mat, k, val):
    """
    Fill the k-th diagonal of the given 2-d square array.
    :param mat: array.
    :param k: int.
        if positive, above the main diagonal,
        else, below the main diagonal.
    :param val: scalar.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("mat should be a 2-d square array.")
    n = mat.shape[1]
    if abs(k) > n - 1:
        raise ValueError("k should not larger than n-1.")
    if k >= 0:
        start = k
        end = n * n - k * n
    else:
        start = (-k) * n
        end = n * n + k
    step = n + 1
    mat.flat[start:end:step] = val


def estimate_p(data, mask):
    # copy observed data
    oaa, oab, oba, obb = np.array(data['aa']), np.array(data['ab']), np.array(data['ba']), np.array(data['bb'])
    oax, oxa, obx, oxb, oxx = np.array(data['ax']), np.array(data['xa']), \
        np.array(data['bx']), np.array(data['xb']), np.array(data['xx'])
    for obs in (oaa, oab, oba, obb, oax, oxa, obx, oxb, oxx):
        obs[~mask] = 0
    s1 = (oaa.sum(axis=1) + oaa.sum(axis=0)) + (oab.sum(axis=1) + oba.sum(axis=0)) + \
         (oab.sum(axis=0) + oba.sum(axis=1)) + (obb.sum(axis=1) + obb.sum(axis=0)) + \
         (oax.sum(axis=1) + oxa.sum(axis=0)) + (obx.sum(axis=1) + oxb.sum(axis=0))
    s2 = (oax.sum(axis=0) + oxa.sum(axis=1)) + (obx.sum(axis=0) + oxb.sum(axis=1)) + \
         (oxx.sum(axis=1) + oxx.sum(axis=0))

    p = s1 * 1.0 / (s1 + s2)
    return p


def multinomial_p(p, mask=None):
    p1 = np.outer(p, p)
    p2 = np.outer(p, 1 - p)
    p3 = np.outer(1 - p, p)
    p4 = np.outer(1 - p, 1 - p)
    if mask is None:
        return p1, p2, p3, p4
    else:
        return p1[mask], p2[mask], p3[mask], p4[mask]


def gamma_matrix(gamma, n, mask):
    gamma_mat = np.zeros((n * 2, n * 2), dtype=float)
    gaa, gab, gba, gbb = disjoin_matrix(gamma_mat, n)
    gab[:] = gamma[-1]  # inter-gamma
    gba[:] = gamma[-1]  # inter-gamma
    # intra-gamma: only fill upper traingle
    for offset in range(1, n):
        fill_diagonal(gaa, offset, gamma[offset - 1])
        fill_diagonal(gbb, offset, gamma[offset - 1])
    log_gaa, log_gab, log_gba, log_gbb = np.log(gaa[mask]), np.log(gab[mask]), \
        np.log(gba[mask]), np.log(gbb[mask])
    log_ngaa, log_ngab, log_ngba, log_ngbb = np.log(1 - gaa[mask]), np.log(1 - gab[mask]), \
        np.log(1 - gba[mask]), np.log(1 - gbb[mask])
    return (log_gaa, log_gab, log_gba, log_gbb), (log_ngaa, log_ngab, log_ngba, log_ngbb)


class ZeroInflatedPoissonFI(BaseModel):
    def __init__(self, params, merge=None, normalize=False, loci=None, diag=0, random_state=None):
        super(ZeroInflatedPoissonFI, self).__init__("Zero-Inflated Poisson Model with Flat-inter")
        assert params.get('n', None) is not None, \
            "Chromosome bin-size (N) must be provided!"
        self.n = params['n']
        self.alpha = params.get('alpha', -3.0)
        self.beta = params.get('beta', 1.0)
        self.normalize = normalize
        if random_state is None:
            random_state = np.random.RandomState()
        if merge is not None:
            assert (merge >= 1) and (merge <= self.n - 1), "Merge should between 1 and N - 1 !"
            self.merge = merge
        else:
            self.merge = self.n - 1
        # check if structure is provided, otherwise initial with random
        if params.get('x', None) is not None:
            try:
                self.x = np.array(params['x'], dtype=float).reshape((self.n * 2, 3))
            except ValueError:
                print "Chromosome structure size does not match bin-size! Should be (N * 2, 3)."
                raise
        else:
            self.x = 1 - 2 * random_state.rand(self.n * 2, 3)
            print "No initial chromosome structure provided! Initialize with random instead."
        # check if centroid distance is provided, otherwise initial from x
        if params.get('cdis', None) is not None:
            self.cdis = float(params['cdis'])
        else:
            self.cdis = centroid_distance(self.x[:self.n, :], self.x[self.n:, :])
            print "No centroid distance provided! Initialize from structure."
        # check if p is provided, otherwise calculate in later step (only calculate once)
        if params.get('p', None) is not None:
            self.p = np.array(params['p'], dtype=float).flatten()
            assert self.p.shape[0] == self.n, \
                "Assignable probabilities size does not match bin-size! Should be (N)."
        else:
            self.p = None
            print "No assignable probabilities provided! Will calculated using observed data."
        # check if gamma is provided, otherwise initial with random, size = N (N-1 intra and 1 inter)
        if params.get('gamma', None) is not None:
            gamma = np.array(params['gamma'], dtype=float).flatten()
            assert (gamma.shape[0] >= self.merge + 1) and (gamma.shape[0] <= self.n), \
                "Gamma size does not match bin-size! Should be at least (Merge + 1) or (N) if no merge."
            self.gamma = np.zeros(self.n, dtype=float)
            self.gamma[:self.merge-1] = gamma[:self.merge-1]  # one gamma per diagonal
            self.gamma[self.merge-1:-1] = gamma[self.merge-1]  # merged gamma
            self.gamma[-1] = gamma[-1]  # inter-chr gamma
        else:
            self.gamma = np.zeros(self.n, dtype=float)
            self.gamma[:self.merge-1] = random_state.uniform(0, 1, self.merge-1)
            self.gamma[self.merge-1:-1] = random_state.uniform(0, 1, 1)
            self.gamma[-1] = random_state.uniform(0, 0.1, 1)
            print "No initial gamma values provided! Initialize with random instead."
        # check if bias is provided, otherwise initialize with ones
        if not self.normalize:
            self.bias = np.ones(self.n * 2, dtype=float)
        elif params.get('bias', None) is None:
            self.bias = np.ones(self.n * 2, dtype=float)
            print "No bias provided! Initialize with ones instead."
        else:
            self.bias = np.array(params['bias'], dtype=float).flatten()
            assert self.bias.shape[0] == self.n * 2, \
                "Bias size does not match bin-size! Should be (N * 2)."
        if loci is not None:
            self.loci = np.array(loci, dtype=bool).flatten()
            assert self.loci.shape[0] == self.n, \
                "Valid loci size does not match bin-size! Should be (N)."
        else:
            self.loci = np.ones(self.n, dtype=bool)
            print "No valid loci provided! Will use all loci."
        # use only the upper traingle and exclude unmappable loci
        assert (diag >= 0) and (diag < self.n - 1), "Exclude diags should be [0, N - 1) !"
        self.diag = diag
        self.mask = ~np.tri(self.n, k=diag, dtype=bool)
        self.mask[~self.loci, :] = False
        self.mask[:, ~self.loci] = False

    def getparams(self):
        _params = {
            'n': self.n,
            'alpha': self.alpha,
            'beta': self.beta,
            'x': np.array(self.x),
            'cdis': self.cdis,
            'p': np.array(self.p),
            'gamma': np.array(self.gamma),
            'bias': np.array(self.bias)
        }
        return _params

    def getstate(self):
        _state = {
            'params': self.getparams(),
            'merge': self.merge,
            'normalize': self.normalize,
            'loci': np.array(self.loci),
            'diag': self.diag
        }
        return _state

    def dumpjson(self, filename, **kwargs):
        _state = self.getstate()
        encodejson(_state)
        with open(filename, 'w') as fh:
            json.dump(_state, fh, **kwargs)

    @classmethod
    def fromjson(cls, filename, **kwargs):
        with open(filename, 'r') as fh:
            _state = json.load(fh, **kwargs)
            return cls(**_state)

    def log_likelihood(self, data):
        # if p is not provided, estiamte with observed data
        if self.p is None:
            self.p = estimate_p(data, mask=self.mask)
            print "Assignable probabilities are calculated using observed data."
        p1, p2, p3, p4 = multinomial_p(self.p, mask=self.mask)
        # masked observed data
        oaa, oab, oba, obb = data['aa'][self.mask], data['ab'][self.mask], data['ba'][self.mask], data['bb'][self.mask]
        oax, oxa, obx, oxb, oxx = data['ax'][self.mask], data['xa'][self.mask], data['bx'][self.mask], \
            data['xb'][self.mask], data['xx'][self.mask]
        # masked poisson lambda
        lambda_mat = poisson_lambda(self.x, self.cdis, self.beta, self.alpha, self.bias)
        laa, lab, lba, lbb = disjoin_matrix(lambda_mat, self.n, mask=self.mask)
        # masked gamma
        log_gs, log_ngs = gamma_matrix(self.gamma, self.n, self.mask)
        # sumout log-likelihood
        ll = float('-inf')  # start with log(0)
        for ztuple in itertools.product((0, 1), repeat=4):
            # ztuple = zaa(0), zab(1), zba(2), zbb(3)
            f = 0  # log f(obs|z) * f(z) for a given z assignment
            # log f(z) = z * log(g) + (1-z) * log(1-g)
            for z, log_g, log_ng in itertools.izip(ztuple, log_gs, log_ngs):
                f += log_g if z == 1 else log_ng
            # log f(obs) = z * log_poisson(p1 * lambda) + (1-z) * log_1(obs==0)
            # poisson.logpmf(k,0) = 0 if k == 0 else = -inf
            for z, lmd, obs in itertools.izip(ztuple,
                                              (laa, lab, lba, lbb),
                                              (oaa, oab, oba, obb)):
                f += poisson.logpmf(obs, z * p1 * lmd)
            # ax = aa + ab
            lax = ztuple[0] * laa + ztuple[1] * lab
            f += poisson.logpmf(oax, p2 * lax)
            # xa = aa + ba
            lxa = ztuple[0] * laa + ztuple[2] * lba
            f += poisson.logpmf(oxa, p3 * lxa)
            # bx = ba + bb
            lbx = ztuple[2] * lba + ztuple[3] * lbb
            f += poisson.logpmf(obx, p2 * lbx)
            # xb = ab + bb
            lxb = ztuple[1] * lab + ztuple[3] * lbb
            f += poisson.logpmf(oxb, p3 * lxb)
            # xx = aa + ab + ba + bb
            lxx = ztuple[0] * laa + ztuple[1] * lab + ztuple[2] * lba + ztuple[3] * lbb
            f += poisson.logpmf(oxx, p4 * lxx)
            # log(exp(ll) + exp(f))
            ll = np.logaddexp(ll, f)

        return ll.sum()

    def maximization(self, data, expected, iced_iter=300, max_func=10000):
        # if p is not provided, estiamte with observed data
        if self.p is None:
            self.p = estimate_p(data, mask=self.mask)
            print "Assignable probabilities are calculated using observed data."
        z, zt = expected
        # form zaa,zbb matrix so we can extract diagonal
        # NOTE could use slicing to save memory, but leave it for now
        zaa, zab, zba, zbb = np.zeros((self.n, self.n), dtype=float), np.zeros((self.n, self.n), dtype=float), \
            np.zeros((self.n, self.n), dtype=float), np.zeros((self.n, self.n), dtype=float)
        zaa[self.mask] = z[0]
        zaa[~self.mask] = np.nan
        zab[self.mask] = z[1]
        zab[~self.mask] = np.nan
        zba[self.mask] = z[2]
        zba[~self.mask] = np.nan
        zbb[self.mask] = z[3]
        zbb[~self.mask] = np.nan
        # update intra-gamma
        for offset in range(1, self.merge):
            diag_aa = np.diagonal(zaa, offset=offset)
            diag_bb = np.diagonal(zbb, offset=offset)
            diag = np.concatenate((diag_aa, diag_bb))
            self.gamma[offset-1] = np.nanmean(diag)
        # update merge-gamma
        diag_merge = []
        for offset in range(self.merge, self.n):
            diag_merge = np.concatenate((diag_merge,
                                        np.diagonal(zaa, offset=offset),
                                        np.diagonal(zbb, offset=offset)))
        gamma_merge = np.nanmean(diag_merge)
        self.gamma[self.merge-1:-1] = gamma_merge
        # update inter-gamma
        self.gamma[-1] = np.concatenate((z[1], z[2])).mean()
        # form ZaaTaa = Zaa * Oaa + ZaaCaa* + ZaaCa*a + ZaaCa*a*
        ztaa, ztab, ztba, ztbb = np.zeros((self.n, self.n), dtype=float), np.zeros((self.n, self.n), dtype=float), \
            np.zeros((self.n, self.n), dtype=float), np.zeros((self.n, self.n), dtype=float)
        ztaa[self.mask] = zt[0]
        ztab[self.mask] = zt[1]
        ztba[self.mask] = zt[2]
        ztbb[self.mask] = zt[3]
        # make zt symmetric so we can call iced on zt
        ztaa = ztaa + ztaa.T
        ztab = ztab + ztba.T
        ztbb = ztbb + ztbb.T
        ztmat = join_matrix(ztaa, ztab, ztab.T, ztbb)
        if self.normalize:
            _, bias = iced.normalization.ICE_normalization(np.array(ztmat), max_iter=iced_iter, output_bias=True)
            # TODO check if bias make ll decrease or count iteration
            self.bias = bias
        # estimate structure of a and b separately
        xa = estimate_x(ztaa, zaa, self.alpha, self.beta, bias=self.bias[:self.n], ini=self.x[:self.n, :],
                        mask=self.mask, maxiter=max_func)
        xb = estimate_x(ztbb, zbb, self.alpha, self.beta, bias=self.bias[self.n:], ini=self.x[self.n:, :],
                        mask=self.mask, maxiter=max_func)
        self.x = np.concatenate((xa, xb))
        # estimate centroid distance
        zbias = join_matrix(zaa, zab, zba, zbb) * np.outer(self.bias, self.bias)
        _, zbias_ab, zbias_ba, _ = disjoin_matrix(zbias, self.n, mask=self.mask)
        self.cdis = pow(
            (zt[1].sum() + zt[2].sum()) / (self.beta * (zbias_ab.sum() + zbias_ba.sum())),
            1.0 / self.alpha)

    def expectation(self, data):
        # if p is not provided, estiamte with observed data
        if self.p is None:
            self.p = estimate_p(data, mask=self.mask)
            print "Assignable probabilities are calculated using observed data."
        p1, p2, p3, p4 = multinomial_p(self.p, mask=self.mask)
        # masked observed data
        oaa, oab, oba, obb = data['aa'][self.mask], data['ab'][self.mask], data['ba'][self.mask], data['bb'][self.mask]
        oax, oxa, obx, oxb, oxx = data['ax'][self.mask], data['xa'][self.mask], data['bx'][self.mask], \
            data['xb'][self.mask], data['xx'][self.mask]
        # masked poisson lambda
        lambda_mat = poisson_lambda(self.x, self.cdis, self.beta, self.alpha, self.bias)
        laa, lab, lba, lbb = disjoin_matrix(lambda_mat, self.n, mask=self.mask)
        # masked gamma
        log_gs, log_ngs = gamma_matrix(self.gamma, self.n, self.mask)
        # expectation of Z
        zaa, zab, zba, zbb = float('-inf'), float('-inf'), float('-inf'), float('-inf')
        # expectation of Z * C
        zcaa_, zca_a, zca_a_ = float('-inf'), float('-inf'), float('-inf')
        zcab_, zca_b, zca_b_ = float('-inf'), float('-inf'), float('-inf')
        zcba_, zcb_a, zcb_a_ = float('-inf'), float('-inf'), float('-inf')
        zcbb_, zcb_b, zcb_b_ = float('-inf'), float('-inf'), float('-inf')
        # observed loglikelihood for each i,j
        ll = float('-inf')
        for ztuple in itertools.product((0, 1), repeat=4):
            # ztuple = zaa(0), zab(1), zba(2), zbb(3)
            f = 0  # log f(obs|z) * f(z) for a given z assignment
            # log f(z) = z * log(g) + (1-z) * log(1-g)
            for z, log_g, log_ng in itertools.izip(ztuple, log_gs, log_ngs):
                f += log_g if z == 1 else log_ng
            # log f(obs) = z * log_poisson(p1 * lambda) + (1-z) * log_1(obs==0)
            # poisson.logpmf(k,0) = 0 if k == 0 else = -inf
            for z, lmd, obs in itertools.izip(ztuple,
                                              (laa, lab, lba, lbb),
                                              (oaa, oab, oba, obb)):
                f += poisson.logpmf(obs, z * p1 * lmd)
            # ax = aa + ab
            lax = ztuple[0] * laa + ztuple[1] * lab
            f += poisson.logpmf(oax, p2 * lax)
            # xa = aa + ba
            lxa = ztuple[0] * laa + ztuple[2] * lba
            f += poisson.logpmf(oxa, p3 * lxa)
            # bx = ba + bb
            lbx = ztuple[2] * lba + ztuple[3] * lbb
            f += poisson.logpmf(obx, p2 * lbx)
            # xb = ab + bb
            lxb = ztuple[1] * lab + ztuple[3] * lbb
            f += poisson.logpmf(oxb, p3 * lxb)
            # xx = aa + ab + ba + bb
            lxx = ztuple[0] * laa + ztuple[1] * lab + ztuple[2] * lba + ztuple[3] * lbb
            f += poisson.logpmf(oxx, p4 * lxx)
            # log(exp(ll) + exp(f))
            ll = np.logaddexp(ll, f)
            if ztuple[0] == 1:
                zaa = np.logaddexp(zaa, f)
                zcaa_ = np.logaddexp(zcaa_, f + np.log(laa) - np.log(lax))
                zca_a = np.logaddexp(zca_a, f + np.log(laa) - np.log(lxa))
                zca_a_ = np.logaddexp(zca_a_, f + np.log(laa) - np.log(lxx))
            if ztuple[1] == 1:
                zab = np.logaddexp(zab, f)
                zcab_ = np.logaddexp(zcab_, f + np.log(lab) - np.log(lax))
                zca_b = np.logaddexp(zca_b, f + np.log(lab) - np.log(lxb))
                zca_b_ = np.logaddexp(zca_b_, f + np.log(lab) - np.log(lxx))
            if ztuple[2] == 1:
                zba = np.logaddexp(zba, f)
                zcba_ = np.logaddexp(zcba_, f + np.log(lba) - np.log(lbx))
                zcb_a = np.logaddexp(zcb_a, f + np.log(lba) - np.log(lxa))
                zcb_a_ = np.logaddexp(zcb_a_, f + np.log(lba) - np.log(lxx))
            if ztuple[3] == 1:
                zbb = np.logaddexp(zbb, f)
                zcbb_ = np.logaddexp(zcbb_, f + np.log(lbb) - np.log(lbx))
                zcb_b = np.logaddexp(zcb_b, f + np.log(lbb) - np.log(lxb))
                zcb_b_ = np.logaddexp(zcb_b_, f + np.log(lbb) - np.log(lxx))
        # log E[Z] = log sum f(obs|z) * f(z) - log f(obs)
        zaa, zab, zba, zbb = np.exp(zaa - ll), np.exp(zab - ll), np.exp(zba - ll), np.exp(zbb - ll)
        # log E[ZaaCaa*] = log cax - log f(obs) + log sum laa/(laa + zab * lab) * f(obs | z) * f(z)
        zcaa_, zca_a, zca_a_ = oax * np.exp(zcaa_ - ll), oxa * np.exp(zca_a - ll), oxx * np.exp(zca_a_ - ll)
        zcab_, zca_b, zca_b_ = oax * np.exp(zcab_ - ll), oxb * np.exp(zca_b - ll), oxx * np.exp(zca_b_ - ll)
        zcba_, zcb_a, zcb_a_ = obx * np.exp(zcba_ - ll), oxa * np.exp(zcb_a - ll), oxx * np.exp(zcb_a_ - ll)
        zcbb_, zcb_b, zcb_b_ = obx * np.exp(zcbb_ - ll), oxb * np.exp(zcb_b - ll), oxx * np.exp(zcb_b_ - ll)

        ztaa = zaa * oaa + zcaa_ + zca_a + zca_a_
        ztab = zab * oab + zcab_ + zca_b + zca_b_
        ztba = zba * oba + zcba_ + zcb_a + zcb_a_
        ztbb = zbb * obb + zcbb_ + zcb_b + zcb_b_

        return (zaa, zab, zba, zbb), (ztaa, ztab, ztba, ztbb)
