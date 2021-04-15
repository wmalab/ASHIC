import os
import json
import iced
import itertools
import numpy as np
import cPickle as pickle
from ashic.em import emfit
from ashic.utils import join_matrix
from ashic.utils import init_counts_allele_certain
from ashic.utils import init_counts_mate_rescue
from ashic.utils import init_gamma_complete
from ashic.misc.smoothing import mean_filter
from ashic.optimization import rmds
from ashic.model.zipoissonhuman import ZeroInflatedPoissonHuman
from ashic.model.poisson import Poisson
from ashic.model.poissonensemble import PoissonEnsemble
import ashic.optimization.initstructure as initstructure
from ashic.optimization.initstructure import apply_mask, sub_aa, sub_bb, sub_ab


def create_model(init_params, model_type, 
                ensemble, n_structure, 
                normalize, seed, merge, 
                diag, loci, mask):
    if model_type == 'ASHIC-ZIPM':
        if not ensemble:
            return ZeroInflatedPoissonHuman(init_params, merge=merge, normalize=normalize, 
                                            loci=loci, diag=diag, mask=mask,
                                            random_state=np.random.RandomState(seed=seed))
        raise NotImplementedError("ASHIC-ZIPM does not support ensemble mode yet.")
    if model_type == 'ASHIC-PM':
        if not ensemble:
            return Poisson(init_params, normalize=normalize, loci=loci, diag=diag, mask=mask,
                            random_state=np.random.RandomState(seed=seed))
        return PoissonEnsemble(init_params, n_structure=n_structure, 
                                loci=loci, diag=diag, mask=mask, random_state=seed)
    raise ValueError("Model should be ASHIC-ZIPM or ASHIC-PM only, not {}.".format(model_type))

def load_data(f):
    """
    Load input data from file `f`
    """
    with open(f, 'rb') as fh:
        pkl = pickle.load(fh)
    data = pkl['obs']
    params = pkl['params']
    return data, params

def create_mask(params, diag):
    n, loci = params['n'], params['loci']
    diag = max(diag, params.get('diag', 0))
    mask = ~np.tri(n, k=diag, dtype=bool)
    mask[~loci, :] = False
    mask[:, ~loci] = False
    if 'mask' in params:
        mask = mask & params['mask']
    # make the mask symmetric
    return np.logical_or(mask, mask.T)

def assign_ambiguous(pois_param, ax, bx, xx, add_pseudo=False):
    """
    Assign ambiguous counts ax, bx, and xx by pois_param
    """
    aa = sub_aa(pois_param)
    ab = sub_ab(pois_param)
    bb = sub_bb(pois_param)
    if add_pseudo:
        aa += 1e-6
        bb += 1e-6
    agg = aa + ab + ab.T + bb

    raa_ax = np.true_divide(aa, (aa + ab))
    rab_ax = np.true_divide(ab, (aa + ab))
    rbb_bx = np.true_divide(bb, (bb + ab.T))
    rba_bx = np.true_divide(ab.T, (bb + ab.T))
    raa_xx = np.true_divide(aa, agg)
    rbb_xx = np.true_divide(bb, agg)
    rab_xx = np.true_divide(ab, agg)
    # assign each uncertain counts to different sources
    aa_ax = np.multiply(raa_ax, ax)
    ab_ax = np.multiply(rab_ax, ax)
    bb_bx = np.multiply(rbb_bx, bx)
    ba_bx = np.multiply(rba_bx, bx)
    aa_xx = np.multiply(raa_xx, xx)
    bb_xx = np.multiply(rbb_xx, xx)
    ab_xx = np.multiply(rab_xx, xx)
    # combine reassign counts
    add_aa = aa_ax + aa_ax.T + aa_xx  # aa = aa* + a*a + a*a*
    add_bb = bb_bx + bb_bx.T + bb_xx  # bb = bb* + b*b + b*b*
    add_ab = ab_ax + ba_bx.T + ab_xx  # ab = ab* + a*b + a*b*
    return join_matrix(add_aa, add_ab, add_ab.T, add_bb)

def assign_complete(data, t, bias, add_pseudo=False):
    """
    Assign ambiguous counts by t
    """
    t = t * np.outer(bias, bias)
    certain = join_matrix(data['aa'], data['ab'], data['ba'], data['bb'])
    return certain + assign_ambiguous(t, data['ax'], data['bx'], data['xx'], add_pseudo=True)

def smooth_matrix(t, mask, h):
    smooth_aa = mean_filter(sub_aa(t), mask=mask, h=h)
    smooth_bb = mean_filter(sub_bb(t), mask=mask, h=h)
    # TODO check if need to smooth inter-matrix as well
    smooth_ab = mean_filter(sub_ab(t), mask=mask, h=h)
    return join_matrix(smooth_aa, smooth_ab, smooth_ab.T, smooth_bb)

def estimate_bias(t, normalize=True, output_matrix=True):
    if not normalize:
        if output_matrix:
            return t, np.ones(t.shape[0])
        return np.ones(t.shape[0])
    # if normalize, estimate allelic bias with ICE
    t_norm, bias = iced.normalization.ICE_normalization(np.array(t), output_bias=True)
    bias = bias.flatten()
    if output_matrix:
        return t_norm, bias
    return bias


# --model=ASHIC-PM --max-iter=1 --seed=0 --init-x=PM --init-c=mate-rescue --ensemble --n-structure=10 --smooth --h=1 --normalize
def run_ashic(inputfile, outputdir, model_type,
              diag, max_iter, tol, seed,
              gamma_share, init_gamma, init_x,
              init_c, ensemble, n_structure,
              normalize, smooth, h, save_iter,
              **kwargs):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    data, params = load_data(inputfile)
    mask = create_mask(params, diag)
    # initialize allele-specific counts by 
    # treat allele-certain or mate-rescue counts 
    # as the Poisson parameters
    init_counts = {
        "allele-certain": init_counts_allele_certain,
        "mate-rescue": init_counts_mate_rescue
    }
    init_t = init_counts[init_c](
        join_matrix(data['aa'], data['ab'], data['ba'], data['bb']),
        data['ax'], data['bx'], data['xx']
    )
    init_t = apply_mask(init_t, mask)
    init_t_norm, init_bias = estimate_bias(init_t, normalize=normalize)
    # if smooth, apply mean_filter to matrix then reassign
    if smooth:
        # if normalize is True, init_t_norm is the normalized complete matrix
        # else init_t_norm equals the unnormalized complete matrix init_t
        init_t = assign_complete(data, smooth_matrix(init_t_norm, mask, h), init_bias)
        # TODO need to reapply mask since we reassign
        init_t = apply_mask(init_t, mask)
        # TODO maybe re-estimate bias here
        init_t_norm, init_bias = estimate_bias(init_t, normalize=normalize)
    # init_t should always be the unnormalized complete matrix
    # initialize model parameters
    init = {
        'n': params['n'],
        'alpha_mat': params['alpha_mat'],
        'alpha_pat': params['alpha_pat'],
        'alpha_inter': params['alpha_inter'],
        'beta': params.get('beta', 1.0),
        'bias': init_bias,
    }
    # initialize gamma from given file, initial complete matrix or a single value
    if init_gamma is not None:
        if os.path.isfile(init_gamma):
            init['gamma'] = np.loadtxt(init_gamma)
        elif init_gamma == 'complete':
            init['gamma'] = init_gamma_complete(init_t, params['loci'], diag, params['mask'])
        else:
            try:
                init_gamma = float(init_gamma)
                init['gamma'] = np.full(init['n'], init_gamma, dtype=float)
            except ValueError:
                print "init_gamma should either be a file name, 'complete', or a single value."
    # initialize x as random, MDS, PoissonModel, or from a given file
    if init_x == 'random':
        init['x'] = None # leave it as None so that model will init it randomly
    elif init_x == 'MDS':
        # need normalized complete matrix here
        # c_ij = b_i * b_j * beta * d^alpha
        # c_norm_ij = c_ij / (b_i * b_j)
        # c_norm_ij = beta * d^alpha
        init['x'] = initstructure.mds(init_t_norm, params=init, mask=mask, loci=params['loci'],
                                        seed=seed, ensemble=ensemble, n_structure=n_structure)
    elif init_x == 'PM':
        init['x'] = initstructure.poisson(init_t, params=init, mask=mask, loci=params['loci'],
                                            seed=seed, ensemble=ensemble, n_structure=n_structure)
    elif os.path.isfile(init_x):
        if init_x.endswith('.pkl') or init_x.endswith('.pickle'):
            with open(init_x, 'rb') as xfh:
                init['x'] = pickle.load(xfh)
        else:
            raise ValueError("Structures init file should be in pickle format.")
    else:
        raise ValueError("Structures can only be initialized by: " + 
                         "'random', 'MDS' or a precomputed file, not {}.".format(init_x))
    # merge is the diagonal where gamma sharing starts from
    if gamma_share is None:
        merge = None
    elif 1 <= gamma_share < init['n']:
        merge = init['n'] - gamma_share
    else:
        raise ValueError("gamma_share should between 1 and {}.".format(init['n'] - 1))
    # TODO previously for normalize, we start with bias=1 and run a few EM iterations
    # then estimate bias, and run EM again
    # it seems use bias estimated from initial complete matrix is ok
    # so we use the init_bias directly and do not update bias later
    model = create_model(init_params=init, model_type=model_type, 
                        ensemble=ensemble, n_structure=n_structure,
                        normalize=normalize, seed=seed, merge=merge, 
                        diag=diag, loci=params['loci'], mask=mask)
    model, converge, loglikelihood, expected, message = emfit(model, data, maxiter=max_iter, tol=tol, **kwargs)

    # TODO save expected Z and T matrices
    mtxpath = os.path.join(outputdir, 'matrices')
    if not os.path.exists(mtxpath):
        os.makedirs(mtxpath)
    if model_type == 'ASHIC-ZIPM':
        zaa, zab, zbb = model.tomatrix(expected[0])
        ztaa, ztab, ztbb = model.tomatrix(expected[1])
        np.savetxt(os.path.join(mtxpath, 'z_mm.txt'), zaa)
        np.savetxt(os.path.join(mtxpath, 'z_mp.txt'), zab)
        np.savetxt(os.path.join(mtxpath, 'z_pp.txt'), zbb)
    else:
        ztaa, ztab, ztbb = model.tomatrix(expected)
    np.savetxt(os.path.join(mtxpath, 't_mm.txt'), ztaa)
    np.savetxt(os.path.join(mtxpath, 't_mp.txt'), ztab)
    np.savetxt(os.path.join(mtxpath, 't_pp.txt'), ztbb)
    #  save result as JSON
    with open(os.path.join(outputdir, 'log.json'), 'w') as fh:
        retdict = {
            'loglikelihood': loglikelihood,
            'converge': converge,
            'message': message,
            'seed': seed,
        }
        json.dump(retdict, fh, indent=4, sort_keys=True)
    # save the final model
    # model.dumpjson(os.path.join(outputdir, 'model.json'), indent=4, sort_keys=True)
    # save structure file
    with open(os.path.join(mtxpath, 'structure.pkl'), 'wb') as xfh:
        pickle.dump(model.x, xfh)
