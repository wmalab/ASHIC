import numpy as np
from sklearn.metrics import euclidean_distances
from allelichicem import structure
from sklearn.utils import check_random_state
from utils import naneuclidean_distances, form_alphamatrix, mask_diagonals


def sample_same_structure(x, y):
    """
    copy a new x which has same rotation and center as y
    """
    xa, xb = structure.copy_structure(x, y)
    return np.concatenate((xa, xb), axis=0)


def sample_localdiff_structure(x, y, local_inds, diff_d, randstate=None):
    xa, xb = structure.copy_structure(x, y)
    randstate = check_random_state(randstate)
    for ind in local_inds:
        xb[ind, :] += structure.random_translate(diff_d, randstate)
    return np.concatenate((xa, xb), axis=0)


def sample_diff_structure(x, y):
    n = min(x.shape[0], y.shape[0])
    return np.concatenate((x[:n, :], y[:n, :]), axis=0)


def sample_p(a, b, n, randstate=None):
    randstate = check_random_state(randstate)
    return randstate.beta(a, b, size=n)


def sample_gamma(a, b, inter, n):
    gd = np.arange(1, n)
    gamma = a * (gd ** b)
    return np.append(gamma, inter)


class Simulation(object):
    def __init__(self, params, seed=0):
        self.params = {
            'beta': params.get('beta', 1.0),
            'alpha': params.get('alpha', -3.0)}
        self.params['n'] = params['n']
        try:
            self.params['x'] = np.array(params['x'], dtype=float).reshape((self.params['n'] * 2, 3))
        except ValueError:
            print "Chromosome structure size does not match bin-size! Should be (N * 2, 3)."
            raise
        self.params['gamma'] = np.array(params['gamma']).flatten()
        assert self.params['gamma'].shape[0] == self.params['n'], "Gamma size should be N."
        self.params['p'] = np.array(params['p']).flatten()
        assert self.params['p'].shape[0] == self.params['n'], "p size sholud be N."
        self.randstate = np.random.RandomState(seed)
        self.hidden = None
        self.obs = None

    def simulate_obs_counts(self, t):
        p = self.params['p']
        n = self.params['n']  # len of one chr
        ax = np.zeros((n, n))  # matrix for ax and xa
        bx = np.zeros((n, n))  # matrix for bx and xb
        xx = np.zeros((n, n))
        certain = np.zeros(t.shape)  # matrix for certain counts

        for i in range(t.shape[0]):
            for j in range(i + 1, t.shape[0]):
                bi, bj = i % n, j % n
                if t[i, j] > 0:
                    c_ij, c_ix, c_xj, c_xx = self.randstate.multinomial(t[i, j],
                                                                        (p[bi] * p[bj],
                                                                         p[bi] * (1 - p[bj]),
                                                                         (1 - p[bi]) * p[bj],
                                                                         (1 - p[bi]) * (1 - p[bj])))
                else:
                    c_ij, c_ix, c_xj, c_xx = 0, 0, 0, 0
                certain[i, j] += c_ij
                certain[j, i] += c_ij
                if i < n:  # i belong to chr a
                    ax[bi, bj] += c_ix
                else:  # i belong to chr b
                    bx[bi, bj] += c_ix
                if j < n:  # j belong to chr a
                    ax[bj, bi] += c_xj
                else:  # j belong to chr b
                    bx[bj, bi] += c_xj
                xx[bi, bj] += c_xx
                xx[bj, bi] += c_xx
        return certain, ax, bx, xx

    def simulate_data(self):
        n = self.params['n']
        x = self.params['x']
        beta = self.params['beta']
        alpha = self.params['alpha']
        gamma = self.params['gamma']
        # distance matrix
        d = euclidean_distances(x)
        # poisson lambda matrix
        mu = beta * d ** alpha
        mu[np.isinf(mu)] = 0
        # simulate Z matrix
        gamma_inter = np.full((n, n), gamma[-1])
        gamma_intra = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                gamma_intra[i, j] = gamma[abs(i-j) - 1]
                gamma_intra[j, i] = gamma_intra[i, j]
        mat_1 = np.concatenate((gamma_intra, gamma_inter), axis=1)
        mat_2 = np.concatenate((gamma_inter, gamma_intra), axis=1)
        gamma_mat = np.concatenate((mat_1, mat_2), axis=0)
        z = self.randstate.binomial(1, gamma_mat)
        
        # set diagonal of Z to 0
        np.fill_diagonal(z, 0)
        np.fill_diagonal(z[:n, n:], 0)
        np.fill_diagonal(z[n:, :n], 0)
        # set Z to symmertric
        z[np.tri(z.shape[0], k=-1).astype(bool)] = 0
        z = z + z.T
        # simulate complete matrix T
        t = self.randstate.poisson(mu)
        # set diagonal of T to 0
        np.fill_diagonal(t, 0)
        np.fill_diagonal(t[:n, n:], 0)
        np.fill_diagonal(t[n:, :n], 0)
        # set T to symmetric
        t[np.tri(t.shape[0], k=-1).astype(bool)] = 0
        t = (t + t.T).astype(float)
        # mask missing entries
        zt = np.multiply(t, z).astype(float)
        # simulate hidden counts
        certain, ax, bx, xx = self.simulate_obs_counts(zt)
        self.hidden = {
            'z': z,
            't': t,
            'zt': zt
        }
        self.obs = {
            'aa': certain[:n, :n],
            'ab': certain[:n, n:],
            'ba': certain[n:, :n],
            'bb': certain[n:, n:],
            'ax': ax,
            'xa': ax.T,
            'bx': bx,
            'xb': bx.T,
            'xx': xx
        }


class SimulationHuman(object):
    def __init__(self, params, seed=0):
        self.params = {
            'alpha_mat': params['alpha_mat'],
            'alpha_pat': params['alpha_pat'],
            'alpha_inter': params['alpha_inter'],
            'beta': params['beta'],
            'n': params['n']
        }
        self.diag = params['diag']
        self.filter_high = params['filter_high']
        try:
            self.params['x'] = np.array(params['x'], dtype=float).reshape((self.params['n'] * 2, 3))
        except ValueError:
            print "Chromosome structure size does not match bin-size! Should be (N * 2, 3)."
            raise
        self.params['gamma'] = np.array(params['gamma']).flatten()
        assert self.params['gamma'].shape[0] == self.params['n'], "Gamma size should be N."
        self.params['p'] = np.array(params['p']).flatten()
        assert self.params['p'].shape[0] == self.params['n'], "p size sholud be N."
        self.loci = np.array(params['loci'], dtype=bool).flatten()
        assert self.loci.shape[0] == self.params['n'], 'Loci size should be N.'
        self.mask = np.ones((params['n'], params['n']), dtype=bool)
        self.mask[~self.loci, :] = False
        self.mask[:, ~self.loci] = False
        self.randstate = np.random.RandomState(seed)
        self.hidden = None
        self.obs = None

    def simulate_obs_counts(self, t):
        p = self.params['p']
        n = self.params['n']  # len of one chr
        ax = np.zeros((n, n))  # matrix for ax and xa
        bx = np.zeros((n, n))  # matrix for bx and xb
        xx = np.zeros((n, n))
        certain = np.zeros(t.shape)  # matrix for certain counts
        # asym matrix for hidden counts of one-end ambiguous
        # row: a, b; col: a*, b*
        hx = np.zeros(t.shape)
        # sym matrix for hidden counts of both-end ambiguous
        # row: a*, b*; col: a*, b*
        hxx = np.zeros(t.shape)

        for i in range(t.shape[0]):
            for j in range(i + 1, t.shape[0]):
                bi, bj = i % n, j % n
                if np.isnan(t[i, j]):
                    c_ij, c_ix, c_xj, c_xx = np.nan, np.nan, np.nan, np.nan
                elif t[i, j] > 0:
                    c_ij, c_ix, c_xj, c_xx = self.randstate.multinomial(t[i, j],
                                                                        (p[bi] * p[bj],
                                                                         p[bi] * (1 - p[bj]),
                                                                         (1 - p[bi]) * p[bj],
                                                                         (1 - p[bi]) * (1 - p[bj])))
                else:
                    c_ij, c_ix, c_xj, c_xx = 0, 0, 0, 0
                certain[i, j] += c_ij
                certain[j, i] += c_ij
                hx[i, j] += c_ix
                hx[j, i] += c_xj
                hxx[i, j] += c_xx
                hxx[j, i] += c_xx
                if i < n:  # i belong to chr a
                    ax[bi, bj] += c_ix
                else:  # i belong to chr b
                    bx[bi, bj] += c_ix
                if j < n:  # j belong to chr a
                    ax[bj, bi] += c_xj
                else:  # j belong to chr b
                    bx[bj, bi] += c_xj
                xx[bi, bj] += c_xx
                xx[bj, bi] += c_xx
        return certain, ax, bx, xx, hx, hxx

    def simulate_data(self):
        n = self.params['n']
        x = self.params['x']
        beta = self.params['beta']
        alpha_mat = self.params['alpha_mat']
        alpha_pat = self.params['alpha_pat']
        alpha_inter = self.params['alpha_inter']
        gamma = self.params['gamma']
        # distance matrix
        d = naneuclidean_distances(x)
        # TODO mask
        self.mask = (d[:n, :n] > np.nanpercentile(d[:n, :n], q=100-self.filter_high)) & \
                    (d[n:, n:] > np.nanpercentile(d[n:, n:], q=100-self.filter_high))
        self.mask = self.mask & mask_diagonals(n, k=self.diag)
        self.mask[~self.loci, :] = False
        self.mask[:, ~self.loci] = False
        mask_full = np.tile(self.mask, (2, 2))
        d[~mask_full] = np.nan
        alpha = form_alphamatrix(alpha_mat, alpha_pat, alpha_inter, n)
        # poisson lambda matrix
        mu = beta * np.power(d, alpha)
        mu[np.isinf(mu)] = 0
        # simulate Z matrix
        gamma_inter = np.full((n, n), gamma[-1])
        gamma_intra = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                gamma_intra[i, j] = gamma[abs(i - j) - 1]
                gamma_intra[j, i] = gamma_intra[i, j]
        mat_1 = np.concatenate((gamma_intra, gamma_inter), axis=1)
        mat_2 = np.concatenate((gamma_inter, gamma_intra), axis=1)
        gamma_mat = np.concatenate((mat_1, mat_2), axis=0)
        # TODO random sampling error with np.nan
        z = np.full_like(gamma_mat, np.nan)
        z[~np.isnan(gamma_mat)] = self.randstate.binomial(1, gamma_mat[~np.isnan(gamma_mat)])
        # set diagonal of Z to 0
        np.fill_diagonal(z, 0)
        np.fill_diagonal(z[:n, n:], 0)
        np.fill_diagonal(z[n:, :n], 0)
        # set Z to symmertric
        z[np.tri(z.shape[0], k=-1).astype(bool)] = 0
        z = z + z.T
        # simulate complete matrix T
        # TODO random sampling error with np.nan
        t = np.full_like(mu, np.nan)
        t[~np.isnan(mu)] = self.randstate.poisson(mu[~np.isnan(mu)])
        # set diagonal of T to 0
        np.fill_diagonal(t, 0)
        np.fill_diagonal(t[:n, n:], 0)
        np.fill_diagonal(t[n:, :n], 0)
        # set T to symmetric
        t[np.tri(t.shape[0], k=-1).astype(bool)] = 0
        t = (t + t.T).astype(float)
        # mask missing entries
        zt = np.multiply(t, z).astype(float)
        # simulate hidden counts
        certain, ax, bx, xx, hx, hxx = self.simulate_obs_counts(zt)
        self.hidden = {
            'z': z,
            't': t,
            'zt': zt,
            'hx': hx,
            'hxx': hxx,
        }
        self.obs = {
            'aa': certain[:n, :n],
            'ab': certain[:n, n:],
            'ba': certain[n:, :n],
            'bb': certain[n:, n:],
            'ax': ax,
            'xa': ax.T,
            'bx': bx,
            'xb': bx.T,
            'xx': xx
        }
