import numpy as np
from scipy import optimize
from sklearn.metrics import euclidean_distances


def poisson_complete_ll(x, zt, z, alpha, beta, bias=None, mask=None):
    d = euclidean_distances(x)
    if bias is None:
        bias = np.ones(d.shape[0], dtype=float)
    mu = (beta * d[mask] ** alpha) * (np.outer(bias, bias)[mask])
    ll = zt[mask] * np.log(mu) - z[mask] * mu
    return - ll.sum()


def poisson_gradient_x(x, zt, z, alpha, beta, bias=None, mask=None):
    if bias is None:
        bias = np.ones(x.shape[0], dtype=float)
    row, col = np.where(mask)
    d = np.sqrt(((x[row] - x[col]) ** 2).sum(axis=1))
    bij = (bias[row] * bias[col]).reshape((-1,))
    mu = (beta * d ** alpha) * bij
    diff = x[row] - x[col]
    grad = - ((zt[mask] / mu - z[mask]) * mu * alpha /
              (d ** 2))[:, np.newaxis] * diff
    grad_ = np.zeros(x.shape)
    # TODO try vectorize see if faster
    for i in range(x.shape[0]):
        grad_[i] += grad[row == i].sum(axis=0)
        grad_[i] -= grad[col == i].sum(axis=0)
    return grad_


def eval_f(x, data=None):
    """
    function to minimize
    """
    m, zt, z, alpha, beta, bias, mask = data
    x = x.reshape((m, 3))
    obj = poisson_complete_ll(x, zt, z, alpha, beta, bias=bias, mask=mask)
    x = x.flatten()
    return obj


def eval_grad_f(x, data=None):
    """
    gradient of function to minimize
    """
    m, zt, z, alpha, beta, bias, mask = data
    x = x.reshape((m, 3))
    grad = poisson_gradient_x(x, zt, z, alpha, beta, bias=bias, mask=mask)
    x = x.flatten()
    return grad.flatten()


def estimate_x(zt, z, alpha, beta, bias=None, ini=None, mask=None, random_state=None, maxiter=10000):
    m = zt.shape[0]
    if random_state is None:
        random_state = np.random.RandomState()
    if ini is None:
        ini = 1 - 2 * random_state.rand(m * 3)
    if bias is None:
        bias = np.ones(m, dtype=float)
    data = (m, zt, z, alpha, beta, bias, mask)
    results = optimize.fmin_l_bfgs_b(
        eval_f,  # function to minimize
        ini.flatten(),  # initial guess
        eval_grad_f,  # gradient of function
        (data, ),  # args to pass to function
        iprint=1,
        maxiter=maxiter)
    results = results[0].reshape(-1, 3)
    return results
