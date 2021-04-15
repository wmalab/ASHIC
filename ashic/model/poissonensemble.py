import os
import json
import numpy as np
from scipy.stats import poisson
from ashic.model.basemodel import BaseModel
from ashic.model.zipoissonhuman import estimate_p, multinomial_p, \
    disjoin_matrix, join_matrix
from ashic.utils import naneuclidean_distances, form_alphamatrix
from ashic.optimization.poissonmodel import estimate_x
from ashic.optimization.poissonmodelinter import estimate_rotation
from ashic.utils import encodejson
from sklearn.utils import check_random_state


def poisson_lambda_ensemble(x, beta, alpha_mat, alpha_pat, alpha_inter, bias):
    m = x[0].shape[0]
    d = np.zeros((m, m))
    for _x in x:
        d += naneuclidean_distances(_x)
    d /= len(x)
    alpha = form_alphamatrix(alpha_mat, alpha_pat, alpha_inter, int(m / 2))
    return (beta * np.power(d, alpha)) * np.outer(bias, bias).astype(float)


class PoissonEnsemble(BaseModel):
    def __init__(self, params, n_structure, loci=None, diag=0, mask=None, random_state=None):
        super(PoissonEnsemble, self).__init__("Poisson-Multinomial Ensemble Model")
        assert params.get('n', None) is not None, \
            "Chromosome bin-size (N) must be provided!"
        self.n = params['n']
        self.alpha_mat = params['alpha_mat']
        self.alpha_pat = params['alpha_pat']
        self.alpha_inter = params['alpha_inter']
        self.beta = params.get('beta', 1.0)
        random_state = check_random_state(random_state)
        # check if structure is provided, otherwise initial with random
        if params.get('x', None) is not None:
            self.x = params['x']
        else:
            self.x = [1 - 2 * random_state.rand(self.n * 2, 3) for _ in range(n_structure)]
            print "No initial chromosome structure provided! Initialize with random instead."
        # check if p is provided, otherwise calculate in later step (only calculate once)
        if params.get('p', None) is not None:
            self.p = np.array(params['p'], dtype=float).flatten()
            assert self.p.shape[0] == self.n, \
                "Assignable probabilities size does not match bin-size! Should be (N)."
        else:
            self.p = None
            print "No assignable probabilities provided! Will calculated using observed data."
        # check if bias is provided, otherwise initialize with ones
        if params.get('bias', None) is None:
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
        if mask is not None:
            self.mask = self.mask & mask
        self.symask = np.logical_or(self.mask, self.mask.T)

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
        lambda_mat = poisson_lambda_ensemble(self.x, self.beta,
                                            self.alpha_mat, self.alpha_pat, self.alpha_inter,
                                            self.bias)
        laa, lab, lba, lbb = disjoin_matrix(lambda_mat, self.n, mask=self.mask)
        # sumout log-likelihood
        ll = 0
        ll += poisson.logpmf(oaa, p1 * laa).sum()
        ll += poisson.logpmf(oab, p1 * lab).sum()
        ll += poisson.logpmf(oba, p1 * lba).sum()
        ll += poisson.logpmf(obb, p1 * lbb).sum()
        ll += poisson.logpmf(oax, p2 * (laa + lab)).sum()
        ll += poisson.logpmf(oxa, p3 * (laa + lba)).sum()
        ll += poisson.logpmf(obx, p2 * (lba + lbb)).sum()
        ll += poisson.logpmf(oxb, p3 * (lab + lbb)).sum()
        ll += poisson.logpmf(oxx, p4 * (laa + lab + lba + lbb)).sum()
        return ll

    def maximization(self, data, expected, max_func=200, separate=True):
        # if p is not provided, estiamte with observed data
        if self.p is None:
            self.p = estimate_p(data, mask=self.mask)
            print "Assignable probabilities are calculated using observed data."
        taa, tab, tba, tbb = expected
        tmataa, tmatab, tmatba, tmatbb = np.zeros((self.n, self.n), dtype=float), \
            np.zeros((self.n, self.n), dtype=float), \
            np.zeros((self.n, self.n), dtype=float), \
            np.zeros((self.n, self.n), dtype=float)
        tmataa[self.mask] = taa
        tmatab[self.mask] = tab
        tmatba[self.mask] = tba
        tmatbb[self.mask] = tbb
        # make t symmetric
        tmataa = tmataa + tmataa.T
        tmatab = tmatab + tmatba.T
        tmatbb = tmatbb + tmatbb.T
        # !!! replace NaN value with 0 in structure optimization to avoid problem
        if separate:
            for i in range(len(self.x)):
                _x = self.x[i]
                _x[~np.tile(self.loci, 2), :] = 0
                x1 = estimate_x(tmataa, np.full_like(tmataa, self.alpha_mat),
                                self.beta, bias=self.bias[:self.n], ini=_x[:self.n, :],
                                mask=self.mask, symask=self.symask, maxiter=max_func)
                x2 = estimate_x(tmatbb, np.full_like(tmatbb, self.alpha_pat),
                                self.beta, bias=self.bias[self.n:], ini=_x[self.n:, :],
                                mask=self.mask, symask=self.symask, maxiter=max_func)
                _x = np.concatenate((x1, x2))
                self.x[i] = estimate_rotation(x=_x, tab=tmatab, alpha=self.alpha_inter,
                            beta=self.beta, mask=self.symask, loci=self.loci, bias=self.bias)
                self.x[i][~np.tile(self.loci, 2), :] = np.nan
        else:
            tmat = join_matrix(tmataa, tmatab, tmatab.T, tmatbb)
            alpha = form_alphamatrix(self.alpha_mat, self.alpha_pat, self.alpha_inter, self.n)
            mask_full = np.tile(self.mask, (2, 2))
            symask_full = np.tile(self.symask, (2, 2))
            for i in range(len(self.x)):
                _x = self.x[i]
                _x[~np.tile(self.loci, 2), :] = 0
                self.x[i] = estimate_x(tmat, alpha, self.beta, bias=self.bias, ini=_x,
                                    mask=mask_full, symask=symask_full, maxiter=max_func)
                self.x[i][~np.tile(self.loci, 2), :] = np.nan

    def expectation(self, data):
        # masked observed data
        oaa, oab, oba, obb = data['aa'][self.mask], data['ab'][self.mask], data['ba'][self.mask], data['bb'][self.mask]
        oax, oxa, obx, oxb, oxx = data['ax'][self.mask], data['xa'][self.mask], data['bx'][self.mask], \
            data['xb'][self.mask], data['xx'][self.mask]
        # masked poisson lambda
        lambda_mat = poisson_lambda_ensemble(self.x, self.beta,
                                            self.alpha_mat, self.alpha_pat, self.alpha_inter,
                                            self.bias)
        laa, lab, lba, lbb = disjoin_matrix(lambda_mat, self.n, mask=self.mask)
        lax = laa + lab
        lxa = laa + lba
        lbx = lba + lbb
        lxb = lab + lbb
        lxx = laa + lab + lba + lbb
        caa_ = oax * np.true_divide(laa, lax)
        ca_a = oxa * np.true_divide(laa, lxa)
        ca_a_ = oxx * np.true_divide(laa, lxx)
        cab_ = oax * np.true_divide(lab, lax)
        ca_b = oxb * np.true_divide(lab, lxb)
        ca_b_ = oxx * np.true_divide(lab, lxx)
        cba_ = obx * np.true_divide(lba, lbx)
        cb_a = oxa * np.true_divide(lba, lxa)
        cb_a_ = oxx * np.true_divide(lba, lxx)
        cbb_ = obx * np.true_divide(lbb, lbx)
        cb_b = oxb * np.true_divide(lbb, lxb)
        cb_b_ = oxx * np.true_divide(lbb, lxx)

        taa = oaa + caa_ + ca_a + ca_a_
        tab = oab + cab_ + ca_b + ca_b_
        tba = oba + cba_ + cb_a + cb_a_
        tbb = obb + cbb_ + cb_b + cb_b_

        return (taa, tab, tba, tbb)

    def tomatrix(self, values):
        """
        unpack values into two intra-(aa, bb) and one inter-(ab) matrix
        """
        n, mask, symask = self.n, self.mask, self.symask
        aa, ab, ba, bb = np.zeros((n, n), dtype=float), np.zeros((n, n), dtype=float), \
            np.zeros((n, n), dtype=float), np.zeros((n, n), dtype=float)
        aa[mask] = values[0]
        ab[mask] = values[1]
        ba[mask] = values[2]
        bb[mask] = values[3]
        # make it symmetric
        aa = aa + aa.T
        ab = ab + ba.T
        bb = bb + bb.T
        aa[~symask] = np.nan
        ab[~symask] = np.nan
        bb[~symask] = np.nan
        return aa, ab, bb

    def savematrix(self, data, mtxdir):
        expected = self.expectation(data)
        taa, tab, tbb = self.tomatrix(expected)
        np.savetxt(os.path.join(mtxdir, 't_aa.txt'), taa)
        np.savetxt(os.path.join(mtxdir, 't_ab.txt'), tab)
        np.savetxt(os.path.join(mtxdir, 't_bb.txt'), tbb)
