import os
import json
import numpy as np
# from ashic import simulation
# from ashic.utils import naneuclidean_distances, encodejson
# from ashic.commands.fit import create_model_human
from ashic.em import emfit
from ashic.progresscb import BasicCallback
# from ashic.model.zipoisson import ZeroInflatedPoisson
# from ashic.model.poisson import Poisson
# from ashic.misc import estimateparams
from ashic.utils import init_counts, join_matrix
from ashic.optimization import rmds
from ashic.misc import plot3d
import cPickle as pickle
import iced
# from time import time
# from datetime import timedelta
# from ashic.structure import duplicate_structure
from ashic.model.zipoissonhuman import ZeroInflatedPoissonHuman
from ashic.model.poisson import Poisson


def create_model(init_params=None, model_type='ASHIC-ZIPM', 
                 seed=0, merge=None, loci=None, diag=0, 
                 normalize=False, mask=None):
    if model_type == 'ASHIC-ZIPM':
        model = ZeroInflatedPoissonHuman(init_params, merge=merge, normalize=normalize, 
                                         loci=loci, diag=diag, mask=mask,
                                         random_state=np.random.RandomState(seed=seed))
    elif model_type == 'ASHIC-PM':
        model = Poisson(init_params, normalize=normalize, loci=loci, diag=diag, mask=mask,
                        random_state=np.random.RandomState(seed=seed))
    else:
        raise ValueError("Model should be ASHIC-ZIPM or ASHIC-PM only, not {}.".format(model_type))
    return model


def run_ashic(inputfile, outputdir, model_type,
              diag, max_iter, tol, seed,
              gamma_share, init_gamma, init_x, 
              normalize, save_iter, smooth=False, h=1, 
              **kwargs):
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    with open(inputfile, 'rb') as fh:
        pk = pickle.load(fh)
        data = pk['obs']
        params = pk['params']
    # initialize allele-specific contact by treat allele-certain as the Poisson lambda
    init_con = init_counts(
        join_matrix(data['aa'], data['ab'], data['ba'], data['bb']),
        data['ax'], data['bx'], data['xx']
    )
    # initialize model parameters
    init = {
        'n': params['n'],
        'alpha_mat': params['alpha_mat'],
        'alpha_pat': params['alpha_pat'],
        'alpha_inter': params['alpha_inter'],
        'beta': params.get('beta', 1.0),
    }
    # initialize gamma from given file
    if init_gamma is not None:
        if os.path.isfile(init_gamma):
            init['gamma'] = np.loadtxt(init_gamma)
        else:
            try:
                init_gamma = float(init_gamma)
                init['gamma'] = np.full(init['n'], init_gamma, dtype=float)
            except ValueError:
                print "init_gamma should either be a file or a single value."
    # initialize x as ranom, MDS or from given file
    if init_x == 'random':
        init['x'] = None
    elif init_x == 'MDS':
        inimat = rmds.haploid(init_con[:init['n'], :init['n']], alpha=init['alpha_mat'],
                              beta=init['beta'], verbose=1, seed=seed, factr=1e5, smooth=smooth, h=h)
        inipat = rmds.haploid(init_con[init['n']:, init['n']:], alpha=init['alpha_pat'],
                              beta=init['beta'], verbose=1, seed=seed, factr=1e5, smooth=smooth, h=h)
        init['x'] = rmds.combine(init_con, inimat, inipat, alpha=init['alpha_inter'],
                                 beta=init['beta'], loci=params['loci'], verbose=1, seed=seed)
    elif os.path.isfile(init_x):
        if init_x.endswith('.txt'):
            init['x'] = np.loadtxt(init_x)
        elif init_x.endswith('.json'):
            with open(init_x) as xfh:
                init['x'] = np.array(
                    json.load(xfh)['params']['x'],
                    dtype=float).reshape((-1, 3)) 
        else:
            raise ValueError("Structures init file should be in text or json format.")
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
    # add normalize
    if not normalize:
        model = create_model(init_params=init, model_type=model_type, seed=seed, merge=merge,
                            loci=params['loci'], diag=diag, normalize=normalize, mask=params['mask'])
        progress = BasicCallback(model, outdir=outputdir, simobj=None, 
                                save=save_iter, seed=seed,
                                maxiter=max_iter, tol=tol)
        model, converge, loglikelihood, expected, message = emfit(model, data, maxiter=max_iter, tol=tol,
                                                                callback=progress.callback, **kwargs)
    else:
        # start with draft model which bias=1 and normalize=False
        model = create_model(init_params=init, model_type=model_type, seed=seed, merge=merge,
                            loci=params['loci'], diag=diag, normalize=False, mask=params['mask'])
        # remove draft_progress later
        # draft_progress = BasicCallback(model, outdir=os.path.join(outputdir, 'draft'), simobj=None, 
        #                         save=save_iter, seed=seed,
        #                         maxiter=max_iter, tol=tol)
        # TODO set maxiter for draft model as 30 now, may need to change later
        model, _, _, expected, _ = emfit(model, data, maxiter=30, tol=tol,
                                         callback=None, **kwargs)
        if model_type == 'ASHIC-ZIPM':
            expected = expected[1]
        taa, tab, tbb = model.tomatrix(expected)
        t = join_matrix(taa, tab, tab.T, tbb)
        # estimate bias using ICE on draft contact matrix t
        _, bias = iced.normalization.ICE_normalization(np.array(t), max_iter=300, output_bias=True)
        # set bias and normalize=True in the draft model
        model.bias = bias
        model.normalize = True
        progress = BasicCallback(model, outdir=outputdir, simobj=None, 
                                save=save_iter, seed=seed,
                                maxiter=max_iter, tol=tol)
        # continue optimization on the draft model with bias
        model, converge, loglikelihood, expected, message = emfit(model, data, maxiter=max_iter, tol=tol,
                                                                callback=progress.callback, **kwargs)
    # save expected Z and T matrices
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
    model.dumpjson(os.path.join(outputdir, 'model.json'), indent=4, sort_keys=True)
    # save 3D plot
    plot3d.plot(np.array(model.x), diploid=True, prefix="structure_", out=outputdir)
    # save structure TEXT file
    np.savetxt(os.path.join(mtxpath, 'structure.txt'), model.x)
