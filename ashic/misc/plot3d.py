import os
import numpy as np
import plotly
import plotly.graph_objs as go
from sklearn.metrics import euclidean_distances


def plot_haploid(X):
    trace = go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        marker=dict(
            size=4,
            color=np.arange(X.shape[0]),
            colorscale='Reds',
        ),
        line=dict(
            width=4,
            color=np.arange(X.shape[0]),
            colorscale='Reds',
        )
    )
    data = [trace]
    return data


def plot_diploid(X, Y, name=None):
    tracem = go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        name=name + " maternal" if name else "maternal",
        marker=dict(
            size=4,
            color=np.arange(X.shape[0]),
            colorscale='Reds',
        ),
        line=dict(
            width=4,
            color=np.arange(X.shape[0]),
            colorscale='Reds',
        )
    )

    tracep = go.Scatter3d(
        x=Y[:, 0], y=Y[:, 1], z=Y[:, 2],
        name=name + " paternal" if name else "paternal",
        marker=dict(
            size=4,
            color=np.arange(Y.shape[0]),
            colorscale='Blues',
        ),
        line=dict(
            width=4,
            color=np.arange(Y.shape[0]),
            colorscale='Blues',
        )
    )

    data=[tracem, tracep]
    return data


def alignment(Xo, Yo, m):
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
    # sx = ((Xc ** 2).sum() / n) ** 0.5
    # sy = ((Yc ** 2).sum() / n) ** 0.5
    # !!! scale use the maternal one as reference
    # !!! need substract the center of maternal
    # !!! s = \sqrt{sum_i(x_i - mean)^2 / m}
    sx = (((Xc[:, :m] - np.tile(Xc[:, :m].mean(axis=1).reshape((3, 1)), (1, m))) ** 2).sum() / m) ** 0.5
    sy = (((Yc[:, :m] - np.tile(Yc[:, :m].mean(axis=1).reshape((3, 1)), (1, m))) ** 2).sum() / m) ** 0.5
    Xc = Xc / sx
    Yc = Yc / sy
    # Rotate Xc
    C = np.dot(Xc, Yc.transpose())
    V, _, Wt = np.linalg.svd(C)
    U = Wt.transpose().dot(V.transpose())
    Xs = U.dot(Xc)
    return Xs.T, Yc.T


def compare(combinex, truex, name=None, prefix=None, out=None, scale=True):
    # scale the combined and true structures
    n = int(combinex.shape[0]/2)
    loci = (np.isnan(combinex).sum(axis=1) == 0) & (np.isnan(truex).sum(axis=1) == 0)
    m = loci[:n].sum()
    if scale:
        tcombinex, ttruex = alignment(combinex[loci, :], truex[loci, :], m)
    else:
        tcombinex, ttruex = combinex[loci, :], truex[loci, :]
    # translate the true structure to 2*diameter
    diameter = np.nanmax([
        np.nanmax(euclidean_distances(tcombinex[:m, :])),
        np.nanmax(euclidean_distances(tcombinex[m:, :])),
        np.nanmax(euclidean_distances(ttruex[:m, :])),
        np.nanmax(euclidean_distances(ttruex[m:, :]))
    ])
    ttruex = ttruex + np.tile([2*diameter, 0, 0], (ttruex.shape[0], 1))
    data1 = plot_diploid(tcombinex[:m, :], tcombinex[m:, :], name=name)
    data2 = plot_diploid(ttruex[:m, :], ttruex[m:, :], name='True')
    data = data1 + data2
    savef = 'compare3d.html'
    if prefix:
        savef = prefix + savef
    if out:
        if not os.path.exists(out):
            os.makedirs(out)
        savef = os.path.join(out, savef)
    plotly.offline.plot(data, filename=savef, auto_open=False)


def plot(X, Y=None, diploid=False, prefix=None, out=None):
    if diploid:
        if Y:
            data = plot_diploid(X, Y)
        else:
            n = int(X.shape[0]/2)
            data = plot_diploid(X[:n, :], X[n:, :])
    else:
        data = plot_haploid(X)
    savef = '3d.html'
    if prefix:
        savef = prefix + savef
    if out:
        if not os.path.exists(out):
            os.makedirs(out)
        savef = os.path.join(out, savef)
    plotly.offline.plot(data, filename=savef, auto_open=False)
