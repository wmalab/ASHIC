import numpy as np
from sklearn.metrics import euclidean_distances


def _random_rotation(randstate):
    """
    Generate random rotation matrix around X, Y, Z axes.
    Return:
        R: 3 * 3 rotation matrix
    """
    thetax, thetaz = randstate.uniform(-1, 1, 2) * np.pi
    thetay = (randstate.uniform(-0.5, 0.5, 1) * np.pi)[0]
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


def random_translate(length, randstate):
    """
    Generate random translation vector with given length.
    Return:
        v: 3 * 1 vector
    """
    v = randstate.uniform(size=(3,))
    v *= length / np.linalg.norm(v)
    return v


def random_transform(X, length, randstate):
    """
    Generate randomly transformed strcuture from X,
    the distance between two centroids is given by length.
    X: N * 3 matrix
    Return:
        Xc: N * 3 matrix, centered X
        Yt: transformed X
    """
    n = X.shape[0]
    # Calculate centroids of X
    centerX = (X.sum(axis=0) / n).reshape((1, 3))
    # # Centering X
    Xc = X - np.tile(centerX, (n, 1))
    R = _random_rotation(randstate)
    v = random_translate(length, randstate)
    Y = R.dot(Xc.T).T
    Yt = Y + np.tile(v, (n, 1))
    return Xc, Yt


def rmsd(X, Y):
    """
    Calculate the RMSD between X and Y
    X, Y are two N * 3 matrix
    Return:
        RMSD: float
    """
    n, _ = X.shape
    RMSD = (((X - Y) ** 2).sum() / n) ** 0.5
    return RMSD


def center_distance(X, Y):
    """
    Calculate the center distance between X and Y
    truncate the longer one so X and Y have same length
    X: N * 3 matrix
    Y: M * 3 matrix
    """
    n = min(X.shape[0], Y.shape[0])
    centerX = X[:n, :].sum(axis=0) / n
    centerY = Y[:n, :].sum(axis=0) / n
    return np.sqrt(np.sum((centerX - centerY) ** 2))


def compare_two_structure(Xo, Yo):
    """
    Xo, Yo are two N * 3 matrix
    Transform X (tranlation and rotation) to align Y.
    Return:
        rmsd, X, Y
    """
    X = Xo.T
    Y = Yo.T
    _, n = X.shape  # n is the number of beads
    if n != Y.shape[1]:
        raise ValueError("Two structures have different number of beads")
    # Calculate centroids of X and Y
    centerX = (X.sum(axis=1) / n).reshape((3, 1))
    centerY = (Y.sum(axis=1) / n).reshape((3, 1))
    # Centering X and Y
    Xc = X - np.tile(centerX, (1, n))
    Yc = Y - np.tile(centerY, (1, n))
    # Scaling Xc and Yc
    sx = ((Xc ** 2).sum() / n) ** 0.5
    sy = ((Yc ** 2).sum() / n) ** 0.5
    Xc = Xc / sx
    Yc = Yc / sy
    # Rotate Xc
    C = np.dot(Xc, Yc.transpose())
    V, S, Wt = np.linalg.svd(C)
    if np.linalg.det(C) > 0:
        d = 1
    else:
        d = -1
    I = np.identity(3)
    I[-1, -1] = d
    U = Wt.transpose().dot(I).dot(V.transpose())
    # U(X-cx) - (Y-cy) = UX-Ucx-Y+cy = (UX+(cy-Ucx) - Y)
    # Calculate translation vector
    # t = centerY - U.dot(centerX)  # shape (3,1)
    Xs = U.dot(Xc)
    return rmsd(Xs.T, Yc.T), Xs.T, Yc.T


def compare_two_structure_reflection(Xo, Yo):
    """
    Xo, Yo are two N * 3 matrix
    Transform X (scaling, tranlation and rotation and/or reflection) to align Y.
    Return:
        rmsd, X, Y
    """
    X = Xo.T
    Y = Yo.T
    _, n = X.shape  # n is the number of beads
    # Calculate centroids of X and Y
    centerX = (X.sum(axis=1) / n).reshape((3, 1))
    centerY = (Y.sum(axis=1) / n).reshape((3, 1))
    # Centering X and Y
    Xc = X - np.tile(centerX, (1, n))
    Yc = Y - np.tile(centerY, (1, n))
    # Scaling Xc and Yc
    sx = ((Xc ** 2).sum() / n) ** 0.5
    sy = ((Yc ** 2).sum() / n) ** 0.5
    Xc = Xc / sx
    Yc = Yc / sy
    # Rotate Xc
    C = np.dot(Xc, Yc.transpose())
    V, S, Wt = np.linalg.svd(C)
    U = Wt.transpose().dot(V.transpose())
    Xs = U.dot(Xc)

    dx = euclidean_distances(Xs.T)
    dy = euclidean_distances(Yc.T)
    derror = np.absolute(dx-dy).sum() / dx.sum()
    return rmsd(Xs.T, Yc.T), derror, Xs.T, Yc.T


def uniformscaling_distance(x):
    n = x.shape[0]
    center = x.sum(axis=0) / n
    x_ = x - np.tile(center, (n, 1))
    s = ((x_ ** 2).sum() / n) ** 0.5
    x_ = x_ / s
    d = euclidean_distances(x_)
    return d


def superposition(Xo, Yo):
    """
    Xo, Yo are two N * 3 matrix
    Transform X (tranlation and rotation) to align Y.
    Return:
        X, Y
    """
    X = Xo.T
    Y = Yo.T
    _, n = X.shape  # n is the number of beads
    if n != Y.shape[1]:
        raise ValueError("Two structures have different number of beads")
    # Calculate centroids of X and Y
    centerX = (X.sum(axis=1) / n).reshape((3, 1))
    centerY = (Y.sum(axis=1) / n).reshape((3, 1))
    # Centering X and Y
    Xc = X - np.tile(centerX, (1, n))
    Yc = Y - np.tile(centerY, (1, n))
    # Rotate Xc
    C = np.dot(Xc, Yc.transpose())
    V, S, Wt = np.linalg.svd(C)
    if np.linalg.det(C) > 0:
        d = 1
    else:
        d = -1
    I = np.identity(3)
    I[-1, -1] = d
    U = Wt.transpose().dot(I).dot(V.transpose())
    # U(X-cx) - (Y-cy) = UX-Ucx-Y+cy = (UX+(cy-Ucx) - Y)
    # Calculate translation vector
    # t = centerY - U.dot(centerX)  # shape (3,1)
    Xs = U.dot(Xc)
    return Xs.T, Yc.T, U


def copy_structure(x, y):
    """
    copy a new x which has same rotation and center as y
    """
    n = min(x.shape[0], y.shape[0])
    x1 = x[:n, :]
    x2 = y[:n, :]
    center1 = x1.mean(axis=0)
    center2 = x2.mean(axis=0)
    _, _, u = superposition(x2, x1)
    xa = x1
    # centering x
    xb = x1 - np.tile(center1, (n, 1))
    # rotate x
    xb = np.linalg.inv(u).dot(xb.T)
    # translate to y center
    xb = xb.T + np.tile(center2, (n, 1))
    return xa, xb


def duplicate_structure(x, y, loci):
    """
    duplicate a new x, calculate rotation angles
    and translation vector from y
    """
    n = min(x.shape[0], y.shape[0])
    x[~loci, :] = np.nan
    y[~loci, :] = np.nan
    x1 = x[loci, :]
    x2 = y[loci, :]
    center1 = x1.mean(axis=0)
    center2 = x2.mean(axis=0)
    # x1-center1 ~ u.dot(x2-center2)
    _, _, u = superposition(x2, x1)
    xa = x
    # centering x
    xb = x - np.tile(center1, (n, 1))
    # rotate x: inv(u).dot(x1-center1) ~ x2-center2
    xb = np.linalg.inv(u).dot(xb.T)
    # translate to y center
    xb = xb.T + np.tile(center2, (n, 1))
    return xa, xb
