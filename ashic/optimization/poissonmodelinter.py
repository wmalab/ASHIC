import numpy as np
from scipy import optimize
from sklearn.metrics import euclidean_distances
from allelichicem.utils import rotation, fill_array3d


def prepare_data(x, tab, alpha, beta, mask, loci):
    n = int(x.shape[0]/2)
    x1 = x[:n, :][loci, :]
    x2 = x[n:, :][loci, :]
    # find the centroids of x1, x2
    c1 = x1.mean(axis=0)
    c2 = x2.mean(axis=0)
    d = c2 - c1
    # centering x1, x2
    x1 = x1 - c1
    x2 = x2 - c2
    return x1, x2, d, tab[loci, :][:, loci], \
        alpha, beta, mask[loci, :][:, loci]


def poisson_complete_ll(x1, x2, t, alpha, beta, mask):
    d = euclidean_distances(x1, x2)
    mu = beta * np.power(d, alpha)[mask]
    ll = t[mask] * np.log(mu) - mu
    return - ll.sum()


def eval_f(angles, data=None):
    """
    function to minimize
    """
    x1, x2, d, t, alpha, beta, mask = data
    thetaxm, thetaym, thetazm, thetaxp, thetayp, thetazp = angles
    rm = rotation(thetaxm, thetaym, thetazm)
    rp = rotation(thetaxp, thetayp, thetazp)
    x1r = rm.dot(x1.T).T
    x2r = rp.dot(x2.T).T + d
    obj = poisson_complete_ll(x1r, x2r, t, alpha, beta, mask)
    return obj


def estimate_rotation(x, tab, alpha, beta, mask, loci, maxiter=1000):
    data = prepare_data(x, tab, alpha, beta, mask, loci)
    ini = np.repeat(0., 6).astype(float)
    results = optimize.fmin_l_bfgs_b(
        eval_f,  # function to minimize
        x0=ini.flatten(),  # initial guess
        approx_grad=True,
        # fprime=eval_grad_f,  # gradient of function
        args=(data, ),  # args to pass to function
        iprint=1,
        maxiter=maxiter)
    thetaxm, thetaym, thetazm, thetaxp, thetayp, thetazp = results[0]
    rm = rotation(thetaxm, thetaym, thetazm)
    rp = rotation(thetaxp, thetayp, thetazp)
    x1 = fill_array3d(data[0], loci, 0.)
    x2 = fill_array3d(data[1], loci, 0.)
    x1 = rm.dot(x1.T).T
    x2 = rp.dot(x2.T).T + data[2]
    return np.concatenate((x1, x2))
