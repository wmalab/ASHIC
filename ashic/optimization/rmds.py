"""
Compute structure for maternal and paternal separately,
then rotate to combine.
"""

import numpy as np
from scipy import optimize
from sklearn.utils import check_random_state
from sklearn.metrics import euclidean_distances
from ashic.misc import smoothing


def compute_wish_distances(counts, alpha=-3., beta=1.):
    # c = beta * d^alpha
    wish_distances = counts / beta
    wish_distances[wish_distances != 0] **= 1. / alpha
    return wish_distances


def mds_objective(X, distances):
    # stress(X) = sum_ij { (d_ij - ||xi-xj||)^2 / d_ij^2 }
    X = X.reshape((-1, 3))
    dis = euclidean_distances(X)
    X = X.flatten()
    obj = 1. / (distances ** 2) * ((dis - distances) ** 2)
    return obj[np.invert(np.isnan(obj) | np.isinf(obj))].sum()


def mds_gradient(X, distances):
    X = X.reshape((-1, 3))
    m, n = X.shape
    tmp = X.repeat(m, axis=0).reshape((m, m, n))
    dif = tmp - tmp.transpose(1, 0, 2)
    dis = euclidean_distances(X).repeat(3, axis=1).flatten()
    wish_distances = distances.repeat(3, axis=1).flatten()
    grad = 2 * dif.flatten() * (dis - wish_distances) / dis / wish_distances**2
    grad[(wish_distances == 0) | np.isnan(grad)] = 0
    X = X.flatten()
    return grad.reshape((m, m, n)).sum(axis=1).flatten()


def estimate_X(counts, alpha=-3., beta=1., ini=None,
               verbose=0,
               random_state=None,
               factr=1e12,
               maxiter=10000):
    n = counts.shape[0]
    random_state = check_random_state(random_state)
    if ini is None:
        ini = 1 - 2 * random_state.rand(n * 3)
    distances = compute_wish_distances(counts, alpha=alpha, beta=beta)
    results = optimize.fmin_l_bfgs_b(
        mds_objective, ini.flatten(),
        mds_gradient,
        (distances, ),
        iprint=int(verbose),
        factr=factr,
        maxiter=maxiter)
    return results[0].reshape((-1, 3))

####################################
## combine two haploid structures ##
####################################


def rotation(thetax, thetay, thetaz):
    """
    Generate rotation matrix around X, Y, Z axes.
    thetax, thetaz: [-pi, pi]
    thetay: [-pi/2, pi/2]
    Return:
        R: 3 * 3 rotation matrix
    """
    sx, cx = np.sin(thetax), np.cos(thetax)
    sy, cy = np.sin(thetay), np.cos(thetay)
    sz, cz = np.sin(thetaz), np.cos(thetaz)
    X = np.array([[1, 0, 0],
                  [0, cx, -sx],
                  [0, sx, cx]])
    Y = np.array([[cy, 0, sy],
                  [0, 1, 0],
                  [-sy, 0, cy]])
    Z = np.array([[cz, -sz, 0],
                  [sz, cz, 0],
                  [0, 0, 1]])
    R = Z.dot(Y).dot(X)
    return R


def naneuclidean_distances(x):
    loci = np.isnan(x).sum(axis=1) == 0
    x_ = np.array(x, copy=True)
    x_[~loci, :] = 0
    d = euclidean_distances(x_)
    d[~loci, :] = np.nan
    d[:, ~loci] = np.nan
    return d


def fill_array3d(x, loci):
    x_ = np.full((loci.shape[0], 3), np.nan, dtype=float)
    x_[loci, :] = x
    return x_


def centering(X):
    center = np.nanmean(X, axis=0).reshape((1, 3))
    return X - np.tile(center, (X.shape[0], 1))


def f(R, *params):
    thetaxm, thetaym, thetazm, thetaxp, thetayp, thetazp = R
    d, X, Y, distances = params
    Rm = rotation(thetaxm, thetaym, thetazm)
    Rp = rotation(thetaxp, thetayp, thetazp)
    Xr = Rm.dot(X.T).T
    Yr = Rp.dot(Y.T).T + np.tile([d, 0, 0], (Y.shape[0], 1))
    dis = euclidean_distances(Xr, Yr)
    obj = 1. / (distances ** 2) * ((dis - distances) ** 2)
    return obj[np.invert(np.isnan(obj) | np.isinf(obj))].sum()


def rotation_matrices(thetax, thetay, thetaz):
    sx, cx = np.sin(thetax), np.cos(thetax)
    sy, cy = np.sin(thetay), np.cos(thetay)
    sz, cz = np.sin(thetaz), np.cos(thetaz)
    X = np.array([[1, 0, 0],
                  [0, cx, -sx],
                  [0, sx, cx]])
    Y = np.array([[cy, 0, sy],
                  [0, 1, 0],
                  [-sy, 0, cy]])
    Z = np.array([[cz, -sz, 0],
                  [sz, cz, 0],
                  [0, 0, 1]])
    return X, Y, Z


def grad_rotation(thetax, thetay, thetaz):
    sx, cx = np.sin(thetax), np.cos(thetax)
    sy, cy = np.sin(thetay), np.cos(thetay)
    sz, cz = np.sin(thetaz), np.cos(thetaz)
    gradX = np.array([[0, 0, 0],
                  [0, -sx, -cx],
                  [0, cx, -sx]])
    gradY = np.array([[-sy, 0, cy],
                  [0, 0, 0],
                  [-cy, 0, -sy]])
    gradZ = np.array([[-sz, -cz, 0],
                  [cz, -sz, 0],
                  [0, 0, 0]])
    return gradX, gradY, gradZ


def Rdot(X, Y, Z):
    return Z.dot(Y).dot(X)


def f_grad(R, *params):
    thetaxm, thetaym, thetazm, thetaxp, thetayp, thetazp = R
    d, X, Y, distances = params
    # rotation matrix
    RXm, RYm, RZm = rotation_matrices(thetaxm, thetaym, thetazm)
    RXp, RYp, RZp = rotation_matrices(thetaxp, thetayp, thetazp)
    # derivation of rotation matrix
    gRXm, gRYm, gRZm = grad_rotation(thetaxm, thetaym, thetazm)
    gRXp, gRYp, gRZp = grad_rotation(thetaxp, thetayp, thetazp)
    # X'=RX
    Xr = Rdot(RXm, RYm, RZm).dot(X.T).T
    # Y'=RY+d
    Yr = Rdot(RXp, RYp, RZp).dot(Y.T).T + np.tile([d, 0, 0], (Y.shape[0], 1))
    dis = euclidean_distances(Xr, Yr)
    n = Xr.shape[0]
    dif = Xr.repeat(n, axis=0).reshape((n, n, 3)) - Yr.repeat(n, axis=0).reshape((n, n, 3)).transpose(1, 0, 2)
    tmp = 2. * (dis - distances) / (dis * distances ** 2)
    # Theta_x gradients
    grad_xm = tmp * (dif * Rdot(gRXm, RYm, RZm).dot(X.T).T.repeat(n, axis=0).reshape((n, n, 3))).sum(axis=2)
    grad_ym = tmp * (dif * Rdot(RXm, gRYm, RZm).dot(X.T).T.repeat(n, axis=0).reshape((n, n, 3))).sum(axis=2)
    grad_zm = tmp * (dif * Rdot(RXm, RYm, gRZm).dot(X.T).T.repeat(n, axis=0).reshape((n, n, 3))).sum(axis=2)
    # Theta_y gradients
    grad_xp = -tmp * (dif * Rdot(gRXp, RYp, RZp).dot(Y.T).T.repeat(n, axis=0).reshape((n, n, 3)).transpose(1, 0, 2)).sum(axis=2)
    grad_yp = -tmp * (dif * Rdot(RXp, gRYp, RZp).dot(Y.T).T.repeat(n, axis=0).reshape((n, n, 3)).transpose(1, 0, 2)).sum(axis=2)
    grad_zp = -tmp * (dif * Rdot(RXp, RYp, gRZp).dot(Y.T).T.repeat(n, axis=0).reshape((n, n, 3)).transpose(1, 0, 2)).sum(axis=2)
    return np.array(map(lambda x: x[np.invert(np.isnan(distances) | (distances == 0))].sum(),
                        (grad_xm, grad_ym, grad_zm, grad_xp, grad_yp, grad_zp)))


def combine_X(counts, iniX, iniY, alpha=-3., beta=1., loci=None, verbose=0, random_state=None):
    random_state = check_random_state(random_state)
    n = int(counts.shape[0]/2)
    if loci is None:
        loci = np.ones(n, dtype=bool)
    mat_pat = counts[:n, n:][loci, :][:, loci]
    X = iniX[loci, :]
    Y = iniY[loci, :]
    # centroid distance = (mean(counts_inter) / beta)^(1/alpha)
    d = np.power(np.nanmean(mat_pat) / beta, 1./alpha)
    # actually not the two centroids distance,
    # but the two interacting *surface* distance
    # radiusm = np.nanpercentile(euclidean_distances(X), 99) / 2.
    # radiusp = np.nanpercentile(euclidean_distances(Y), 99) / 2.
    # d = d + (radiusm + radiusp)
    # center X and Y
    X = centering(X)
    Y = centering(Y)
    distances = compute_wish_distances(mat_pat, alpha=alpha, beta=beta)
    params = (d, X, Y, distances)
    rranges = ((-np.pi, np.pi), (-0.5*np.pi, 0.5*np.pi), (-np.pi, np.pi),
               (-np.pi, np.pi), (-0.5*np.pi, 0.5*np.pi), (-np.pi, np.pi))
    thetaxm, thetaxp, thetazm, thetazp = random_state.uniform(-1, 1, 4) * np.pi
    thetaym, thetayp = random_state.uniform(-0.5, 0.5, 2) * np.pi
    ini = (thetaxm, thetaym, thetazm,
           thetaxp, thetayp, thetazp)
    results = optimize.fmin_l_bfgs_b(
                        f,
                        x0=np.array(ini),
                        fprime=f_grad,
                        # approx_grad=True,
                        bounds=rranges,
                        args=params,
                        iprint=int(verbose))
    thetaxm, thetaym, thetazm, thetaxp, thetayp, thetazp = results[0]
    # value of the function at minimum
    fvalue = results[1]
    Rm = rotation(thetaxm, thetaym, thetazm)
    Rp = rotation(thetaxp, thetayp, thetazp)
    X = fill_array3d(X, loci)
    Y = fill_array3d(Y, loci)
    X = Rm.dot(X.T).T
    Y = Rp.dot(Y.T).T + np.tile([d, 0, 0], (n, 1))
    return np.concatenate((X, Y)), fvalue


def haploid(counts, alpha=-3., beta=1., ini=None, verbose=1, seed=None,
            maxiter=5000, factr=1e12, smooth=False, h=1, mask=None):
    c = np.array(counts)
    if smooth:
        c = smoothing.mean_filter(c, mask=mask, h=h)
    X_ = estimate_X(c, alpha=alpha, beta=beta, ini=ini,
                    verbose=verbose, random_state=seed, maxiter=maxiter, factr=factr)
    return X_


def combine(counts, iniX, iniY, alpha=-3., beta=1., loci=None, verbose=1, seed=None, nruns=1):
    # TODO multiple random starts
    random_state = check_random_state(seed)
    # !!! allow mirror symmertry
    mirror = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, -1]], dtype=float)
    fvalue = np.inf
    S = None
    for _ in range(nruns):
        S_, fvalue_ = combine_X(counts, iniX, iniY, alpha=alpha, beta=beta, loci=loci,
                                verbose=verbose, random_state=random_state)
        if fvalue_ < fvalue:
            fvalue = fvalue_
            S = S_
    for _ in range(nruns):
        S_, fvalue_ = combine_X(counts, iniX.dot(mirror), iniY, alpha=alpha, beta=beta, loci=loci,
                                verbose=verbose, random_state=random_state)
        if fvalue_ < fvalue:
            fvalue = fvalue_
            S = S_
    return S
