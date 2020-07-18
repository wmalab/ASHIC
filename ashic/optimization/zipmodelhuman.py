import numpy as np
from scipy import optimize
from sklearn.metrics import euclidean_distances
from ashic.utils import naneuclidean_distances


# TODO multiple alpha
def poisson_complete_ll(x, zt, z, alpha, beta, bias=None, mask=None):
    d = euclidean_distances(x)  # NaN values replaced by 0
    if bias is None:
        bias = np.ones(d.shape[0], dtype=float)
    mu = (beta * np.power(d, alpha)[mask]) * (np.outer(bias, bias)[mask])
    ll = zt[mask] * np.log(mu) - z[mask] * mu
    return - ll.sum()


# TODO multiple alpha
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
    b = np.outer(bias, bias).repeat(3)
    grad = (zt.repeat(3) - z.repeat(3) * beta * b * np.power(d, a)) * a * diff / (d ** 2)
    grad[np.invert(mask.repeat(3))] = 0
    return - grad.reshape((m, m, 3)).sum(1).flatten()


def eval_f(x, data=None):
    """
    function to minimize
    """
    m, zt, z, alpha, beta, bias, mask, _ = data
    x = x.reshape((m, 3))
    obj = poisson_complete_ll(x, zt, z, alpha, beta, bias=bias, mask=mask)
    x = x.flatten()
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


def estimate_x_human(zt, z, alpha, beta, bias=None, ini=None,
                     mask=None, symask=None, random_state=None, maxiter=1000):
    m = zt.shape[0]
    if random_state is None:
        random_state = np.random.RandomState()
    if ini is None:
        ini = 1 - 2 * random_state.rand(m * 3)
    if bias is None:
        bias = np.ones(m, dtype=float)
    data = (m, zt, z, alpha, beta, bias, mask, symask)
    results = optimize.fmin_l_bfgs_b(
        eval_f,  # function to minimize
        x0=ini.flatten(),  # initial guess
        fprime=eval_grad_f,  # gradient of function
        args=(data, ),  # args to pass to function
        iprint=1,
        maxiter=maxiter)
    results = results[0].reshape((-1, 3))
    return results
