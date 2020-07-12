import numpy as np
from sklearn.metrics import euclidean_distances


def encodejson(obj):
    for key in obj:
        if isinstance(obj[key], np.ndarray):
            obj[key] = obj[key].flatten().tolist()
        elif isinstance(obj[key], dict):
            encodejson(obj[key])


def join_matrix(m1, m2, m3, m4, n=None, mask=None):
    if mask is None:
        r1 = np.concatenate((m1, m2), axis=1)
        r2 = np.concatenate((m3, m4), axis=1)
        return np.concatenate((r1, r2), axis=0)
    else:
        mat = np.full((n*2, n*2), np.nan, dtype=m1.dtype)
        mat[:n, :n][mask] = m1
        mat[:n, n:][mask] = m2
        mat[n:, :n][mask] = m3
        mat[n:, n:][mask] = m4
        return mat


def disjoin_matrix(mat, n, mask=None):
    if mask is None:
        return mat[:n, :n], mat[:n, n:], mat[n:, :n], mat[n:, n:]
    else:
        return mat[:n, :n][mask], mat[:n, n:][mask], mat[n:, :n][mask], mat[n:, n:][mask]


def centroid_distance(x, y):
    cx = x.sum(axis=0) / x.shape[0]
    cy = y.sum(axis=0) / y.shape[0]
    return np.sqrt(np.sum((cx - cy) ** 2))


def find_closestlength_chrom(chroms, chrom):
    """
    find the chromosome with length closest to chrom
    """
    mindiff = np.inf
    minchrom = None
    for ch in chroms:
        if ch != chrom:
            diff = abs(chroms[chrom].shape[0]-chroms[ch].shape[0])
            if diff < mindiff:
                mindiff = diff
                minchrom = ch
    return minchrom


def get_localinds(n, percentile=0.2, fragment_size=5):
    length = int(n * percentile)  # num of bins to change
    # num of fragments, each with length fragment_size
    if length % fragment_size == 0:
        nfragments = int(length / fragment_size)
    else:
        nfragments = int(length / fragment_size) + 1
    localinds = []
    # the distance between center of two fragments
    centerdis = int(n / (nfragments + 1))
    center = centerdis
    for i in range(nfragments):
        if i < nfragments - 1:
            s = max(0, center-int(fragment_size/2))
            e = min(s+fragment_size, n)
            localinds.extend(range(s, e))
            center += centerdis
        else:
            if length % fragment_size == 0:
                end = fragment_size
            else:
                end = length % fragment_size
            s = max(0, center-int(fragment_size/2))
            e = min(s+end, n)
            localinds.extend(range(s, e))
    return localinds


def parse_localinds(indstr):
    """
    local indices string format:
        start1-end1,start2-end2...
    """
    localinds = []
    for rg in indstr.split(','):
        start, end = rg.split('-')
        start, end = int(start), int(end)
        localinds.extend(range(start, end+1))
    return localinds


def get_rdis(x):
    d = euclidean_distances(x)
    rdis = np.nanmean(np.diag(d, k=1)) / 2
    return rdis


def init_counts(certain, ax, bx, xx):
    """
    Assign allelic-uncertain counts proportion to certain counts.
    certain: ndarray
        Allele certain counts matrix.
    ax: ndarray
        chr_a allele certain counts.
    bx: ndarray
        chr_b allele certain counts.
    xx: ndarray
        Both alleles uncertain counts.
    """
    n = ax.shape[0]
    aa = certain[:n, :n] + 1e-6  # add pesudocount to avoid divide by 0
    ab = certain[:n, n:] # + 1e-6  # TODO make inter pesudocount to 0
    bb = certain[n:, n:] + 1e-6
    diploid = aa + ab + ab.T + bb
    # TODO change intra inter ratio
    # TODO add pesudocount only at the denominater
    # so if the numerator is 0, then the ratio will also be 0

    raa_ax = np.true_divide(aa, (aa + ab))
    rab_ax = np.true_divide(ab, (aa + ab))
    rbb_bx = np.true_divide(bb, (bb + ab.T))
    rba_bx = np.true_divide(ab.T, (bb + ab.T))
    raa_xx = np.true_divide(aa, diploid)
    rbb_xx = np.true_divide(bb, diploid)
    rab_xx = np.true_divide(ab, diploid)
    # assign each uncertain counts to different sources
    aa_ax = np.multiply(raa_ax, ax)
    ab_ax = np.multiply(rab_ax, ax)
    bb_bx = np.multiply(rbb_bx, bx)
    ba_bx = np.multiply(rba_bx, bx)  # !!! FIX: change from ab to ba
    aa_xx = np.multiply(raa_xx, xx)
    bb_xx = np.multiply(rbb_xx, xx)
    ab_xx = np.multiply(rab_xx, xx)
    # combine reassign counts
    add_aa = aa_ax + aa_ax.T + aa_xx  # aa = aa* + a*a + a*a*
    add_bb = bb_bx + bb_bx.T + bb_xx  # bb = bb* + b*b + b*b*
    add_ab = ab_ax + ba_bx.T + ab_xx  # ab = ab* + a*b + a*b*
    add_mat = np.concatenate((np.concatenate((add_aa, add_ab), axis=1),
                              np.concatenate((add_ab.T, add_bb), axis=1)),
                             axis=0)
    return add_mat + certain


def init_gamma(data):
    pass


def naneuclidean_distances(x):
    loci = np.isnan(x).sum(axis=1) == 0
    x_ = np.array(x, copy=True)
    x_[~loci, :] = 0
    d = euclidean_distances(x_)
    d[~loci, :] = np.nan
    d[:, ~loci] = np.nan
    return d


def form_alphamatrix(alpha_mat, alpha_pat, alpha_inter, n):
    alpha = np.full((n*2, n*2), np.nan)
    alpha[:n, :n] = alpha_mat
    alpha[n:, n:] = alpha_pat
    alpha[:n, n:] = alpha_inter
    alpha[n:, :n] = alpha_inter
    return alpha


def mask_diagonals(n, k):
    """
    return bool matrix with 0 to +k and 0 to -k diagonals masked as false
    elsewhere as true
    """
    return np.tri(n, k=-(abs(k)+1), dtype=bool) | ~np.tri(n, k=abs(k), dtype=bool)


def nansampling():
    pass


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


def fill_array3d(x, loci, v=np.nan):
    x_ = np.full((loci.shape[0], 3), v, dtype=float)
    x_[loci, :] = x
    return x_