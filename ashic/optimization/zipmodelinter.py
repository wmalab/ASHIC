import numpy as np
from scipy import optimize
from sklearn.metrics import euclidean_distances
from allelichicem.utils import rotation, fill_array3d


def prepare_data(x, ztab, zab, alpha, beta, mask, loci):
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
    return x1, x2, d, ztab[loci, :][:, loci], zab[loci, :][:, loci], \
        alpha, beta, mask[loci, :][:, loci]


def poisson_complete_ll(x1, x2, zt, z, alpha, beta, mask):
    d = euclidean_distances(x1, x2)
    mu = beta * np.power(d, alpha)[mask]
    ll = zt[mask] * np.log(mu) - z[mask] * mu
    return - ll.sum()


def _poisson_gradient_x(x, zt, z, alpha, beta, bias=None, mask=None):
    if bias is None:
        bias = np.ones(x.shape[0], dtype=float)
    row, col = np.where(mask)
    d = np.sqrt(((x[row] - x[col]) ** 2).sum(axis=1))
    bij = (bias[row] * bias[col]).reshape((-1,))
    # TODO alpha is a matrix
    mu = (beta * np.power(d, alpha[mask])) * bij
    diff = x[row] - x[col]
    # TODO alpha is a matrix
    grad = - ((zt[mask] / mu - z[mask]) * mu * alpha[mask] /
              (d ** 2))[:, np.newaxis] * diff
    grad_ = np.zeros(x.shape)
    # TODO try vectorize see if faster
    for i in range(x.shape[0]):
        grad_[i] += grad[row == i].sum(axis=0)
        grad_[i] -= grad[col == i].sum(axis=0)
    return grad_


def poisson_gradient_x(x, zt, z, alpha, beta, bias=None, mask=None):
    # TODO check gradients calculation *mask*
    m = x.shape[0]
    tmp = x.repeat(m, axis=0).reshape((m, m, 3))
    diff = (tmp - tmp.transpose(1, 0, 2)).flatten()
    d = euclidean_distances(x).repeat(3)
    a = alpha.repeat(3)
    grad = (zt.repeat(3) - z.repeat(3) * beta * np.power(d, a)) * a * diff / (d ** 2)
    grad[np.invert(mask.repeat(3))] = 0
    return - grad.reshape((m, m, 3)).sum(1).flatten()


def eval_f(angles, data=None):
    """
    function to minimize
    """
    x1, x2, d, zt, z, alpha, beta, mask = data
    thetaxm, thetaym, thetazm, thetaxp, thetayp, thetazp = angles
    rm = rotation(thetaxm, thetaym, thetazm)
    rp = rotation(thetaxp, thetayp, thetazp)
    x1r = rm.dot(x1.T).T
    x2r = rp.dot(x2.T).T + d
    obj = poisson_complete_ll(x1r, x2r, zt, z, alpha, beta, mask)
    return obj


def eval_grad_f(x, data=None):
    """
    gradient of function to minimize
    """
    m, zt, z, alpha, beta, bias, _, symask = data
    x = x.reshape((m, 3))
    grad = poisson_gradient_x(x, zt, z, alpha, beta, bias=bias, mask=symask)
    x = x.flatten()
    return grad.flatten()


def estimate_rotation(x, ztab, zab, alpha, beta, mask, loci, maxiter=1000):
    data = prepare_data(x, ztab, zab, alpha, beta, mask, loci)
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
