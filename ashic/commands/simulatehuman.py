import os
import json
import numpy as np
from allelichicem import simulation
from allelichicem.utils import naneuclidean_distances, encodejson
from allelichicem.commands.fit import create_model_human
from allelichicem.em import emfit
from allelichicem.progresscb import basic_callback, SimulationProgress, BasicCallback
from allelichicem.model.zipoisson import ZeroInflatedPoisson
from allelichicem.model.poisson import Poisson
from allelichicem.misc import estimateparams
from allelichicem.utils import init_counts, join_matrix, mask_diagonals
from allelichicem.optimization import rmds
from allelichicem.misc import plot3d
import cPickle as pickle
import iced
from time import time
from datetime import timedelta
from allelichicem.structure import duplicate_structure


# TODO mask unmappable loci
# TODO filter and mask high values
# TODO mask first diagonal
# TODO three different alpha for maternal, paternal and interchromosomal
# TODO try different chromosome
# TODO use normalize contacts
# TODO merge gamma tail
# FIX sampling nan is 0


def load_humanstructures(chrom, res=100000):
    datadir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           '../examples/human/Tan et al.'))
    xmat = np.loadtxt(os.path.join(datadir, 'chr{}_{}.mat.txt'.format(chrom.upper(), res)))
    xpat = np.loadtxt(os.path.join(datadir, 'chr{}_{}.pat.txt'.format(chrom.upper(), res)))
    return xmat, xpat


def load_humancontacts(chrom, res=100000):
    datadir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           '../examples/human/Rao et al.'))
    conmat = np.load(os.path.join(datadir, 'chr{}_{}.mat.con.npy'.format(chrom.upper(), res)))
    conpat = np.load(os.path.join(datadir, 'chr{}_{}.pat.con.npy'.format(chrom.upper(), res)))
    return conmat, conpat


def cmd_estimate_params(chrom, outdir, alpha_inter=-3.0, gamma_inter=0.05,
                        p_a=2.0, p_b=2.0, seed=0, diag=1, filter_high=99.9,
                        plot=True):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # TODO merge settings into params.json
    # settings = {
    #     'chrom': chrom,
    #     'p_a': p_a,
    #     'p_b': p_b,
    #     'seed': seed,
    # }
    xmat, xpat = load_humanstructures(chrom)
    conmat, conpat = load_humancontacts(chrom)
    assert xmat.shape[0] == xpat.shape[0], \
        "maternal and paternal structures have different length"
    n = xmat.shape[0]
    dmat = naneuclidean_distances(xmat)
    dpat = naneuclidean_distances(xpat)
    # filter too small distance/high contacts
    # mask unmappable loci
    # mask first diagonal
    mask = (dmat > np.nanpercentile(dmat, q=100-filter_high)) & \
           (dpat > np.nanpercentile(dpat, q=100-filter_high)) & \
           (~np.tri(n, k=diag, dtype=bool))
    loci = (conmat.sum(axis=0) > 0) & (conpat.sum(axis=0) > 0) & \
           (np.isnan(xmat).sum(axis=1) == 0) & (np.isnan(xpat).sum(axis=1) == 0)
    mask[~loci, :] = False
    mask[:, ~loci] = False
    dmat[~mask] = np.nan
    dpat[~mask] = np.nan
    conmat[~mask] = np.nan
    conpat[~mask] = np.nan
    # estimate alpha using curve fitting
    alpha_mat = estimateparams.estimate_alpha(conmat, dmat, diag+1,
                                              plot=plot,
                                              savefile=os.path.join(outdir, 'maternal_alpha.png'))
    alpha_pat = estimateparams.estimate_alpha(conpat, dpat, diag+1,
                                              plot=plot,
                                              savefile=os.path.join(outdir, 'paternal_alpha.png'))
    # estimate beta using MLE
    beta = estimateparams.estimate_beta(conmat, conpat, dmat, dpat,
                                        alpha_mat, alpha_pat, mask)
    if plot:
        estimateparams.plot_simulated(estimateparams.sampling(beta, dmat, alpha_mat),
                                      estimateparams.sampling(beta, dpat, alpha_pat),
                                      conmat, conpat, diag+1, outdir)
    gamma_intra = estimateparams.estimate_gamma(conmat, conpat, diag+1, plot=plot, outdir=outdir)
    p = simulation.sample_p(a=p_a, b=p_b, n=n, randstate=seed)
    params = {
        'chrom': chrom,
        'alpha_mat': alpha_mat,
        'alpha_pat': alpha_pat,
        'alpha_inter': alpha_inter,
        'beta': beta,
        'gamma': np.append(gamma_intra, gamma_inter),
        'p': p,
        'p_a': p_a,
        'p_b': p_b,
        'p_seed': seed,
        'x': np.concatenate((xmat, xpat), axis=0),
        'n': n,
        'loci': loci,
        'diag': diag,
        'filter_high': filter_high
    }
    encodejson(params)
    # encodejson(settings)
    with open(os.path.join(outdir, 'params.json'), 'w') as fh:
        json.dump(params, fh, indent=4, sort_keys=True)
    # with open(os.path.join(outdir, 'settings.json'), 'w') as fh:
    #     json.dump(settings, fh, indent=4, sort_keys=True)


def cmd_duplicate_structure(paramsfile, outdir, parent):
    """
    duplicate 'parent' chromatin structure
    use superposition to find the rotation angles and translation vector
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(paramsfile, 'r') as fh:
        params = json.load(fh)
    n = params['n']
    loci = np.array(params['loci'], dtype=bool)
    x = np.array(params['x']).reshape((-1, 3))
    xmat = x[:n, :]
    xpat = x[n:, :]
    if parent == 'mat':  # duplicate maternal chromosome
        xmat, xpat = duplicate_structure(xmat, xpat, loci)
        # change alpha_pat to alpha_mat
        params['alpha_pat'] = params['alpha_mat']
    elif parent == 'pat':  # duplicate paternal chromosome
        xpat, xmat = duplicate_structure(xpat, xmat, loci)
        params['alpha_mat'] = params['alpha_pat']
    else:
        raise ValueError('No such type of parental choice: {}'.format(parent))
    params['x'] = np.concatenate((xmat, xpat), axis=0)
    plot3d.compare(params['x'], x, name='duplicate',
                   prefix='duplicate_', out=outdir, scale=False)
    encodejson(params)
    with open(os.path.join(outdir, 'params.json'), 'w') as fh:
        json.dump(params, fh, indent=4, sort_keys=True)


# TODO downsampling beta or p
# TODO change downsampling p to use positively skewed Beta distribution
def cmd_downsample_params(paramsfile, outdir, frac_beta=None, frac_p=None, seed=0):
    """
    downsampling coverage (beta) to frac_beta * beta
    downsampling SNP density (p) to mean=frac_p * 0.5 with positively skewed Beta
    2/(2+beta) = frac_p*0.5
    """
    with open(paramsfile, 'r') as fh:
        params = json.load(fh)
    if frac_beta is not None:
        params['beta'] *= frac_beta
    if frac_p is not None:
        p_a = 2.
        p_b = (p_a - 0.5*frac_p*p_a) / (0.5 * frac_p)
        params['p'] = simulation.sample_p(a=p_a, b=p_b, n=params['n'], randstate=seed)
        params['p'] = params['p'].flatten().tolist()
        # merge settings to params
        params['p_a'] = p_a
        params['p_b'] = p_b
        params['p_seed'] = seed
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir, 'params.json'), 'w') as fw:
        json.dump(params, fw, indent=4, sort_keys=True)


def cmd_simulate_data(paramsfile, outdir, seed=0):
    with open(paramsfile, 'r') as fh:
        params = json.load(fh)
    sim = simulation.SimulationHuman(params, seed=seed)
    sim.simulate_data()
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    picklefile = os.path.join(outdir, "simulate_data_{}.pickle".format(seed))
    with open(picklefile, 'wb') as fh:
        pickle.dump(sim, fh, protocol=pickle.HIGHEST_PROTOCOL)


# TODO allow simulated data and real data
def cmd_run(inputs, outdir, paramsfile=None, modeltype='ziphuman', is_simulation=False, savemat=True, saveprog=True,
            beta=1.0, diag=1, maxiter=30, tol=1e-4, seed=0, tail=None, initgamma=None, initx=None, smooth=False, h=1, **kwargs):
    start_time = time()
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if is_simulation:
        with open(inputs[0], 'rb') as fh:
            sim = pickle.load(fh)
            data = sim.obs
        with open(paramsfile, 'r') as fh:
            params = json.load(fh)
            params['mask'] = sim.mask
            params['loci'] = sim.loci
            params['diag'] = sim.diag
    else:
        sim = None
        with open(inputs[0], 'rb') as fh:
            pk = pickle.load(fh)
            data = pk['obs']
            params = pk['params']
        # # TODO readin real data, filter low bins
        # certain = np.load(inputs[0])
        # ax = np.load(inputs[1])
        # bx = np.load(inputs[2])
        # xx = np.load(inputs[3])
        # n = int(certain.shape[0] / 2)
        # data = {
        #     'aa': certain[:n, :n],
        #     'ab': certain[:n, n:],
        #     'ba': certain[n:, :n],
        #     'bb': certain[n:, n:],
        #     'ax': ax,
        #     'xa': ax.T,
        #     'bx': bx,
        #     'xb': bx.T,
        #     'xx': xx
        # }
        # mask = mask_diagonals(n, k=diag)
        # for i in data:
        #     data[i][~mask] = 0
        # filtered = iced.filter.filter_low_counts(np.array(data['aa']), sparsity=False,
        #                                          percentage=0.02)
        # loci = np.nansum(filtered, axis=0) > 0
        # filtered = iced.filter.filter_low_counts(np.array(data['bb']), sparsity=False,
        #                                          percentage=0.02)
        # loci = loci & (np.nansum(filtered, axis=0) > 0)
        # # loci[:600] = False
        # # for i in data:
        # #     data[i][:600, :] = 0
        # #     data[i][:, :600] = 0
        # # TODO estimate alpha_mat and alpha_pat
        # params = {
        #     'n': n,
        #     'alpha_mat': -3.,
        #     'alpha_pat': -3.,
        #     'alpha_inter': -3.,
        #     'diag': diag,
        #     'loci': loci,
        #     'mask': mask
        # }

    initcon = init_counts(
        join_matrix(data['aa'], data['ab'], data['ba'], data['bb']),
        data['ax'], data['bx'], data['xx']
    )

    init = {
        'n': params['n'],
        'alpha_mat': params['alpha_mat'],
        'alpha_pat': params['alpha_pat'],
        'alpha_inter': params['alpha_inter'],
        'beta': beta,
    }

    if initgamma is not None and os.path.isfile(initgamma):
        init['gamma'] = np.loadtxt(initgamma)

    if initx is None or initx == 'random':
        init['x'] = (1 - 2 * np.random.rand(params['n'] * 2 * 3)).reshape((params['n'] * 2, 3))
    elif initx == 'true':
        init['x'] = params['x']
        # the scale beta=1.0 here is different from true structure
        init['beta'] = params['beta']
        init['gamma'] = params['gamma']
        init['p'] = params['p']
    elif initx == 'MDS':
        inimat = rmds.haploid(initcon[:params['n'], :params['n']], alpha=params['alpha_mat'],
                              beta=beta, verbose=1, seed=seed, factr=1e5, smooth=smooth, h=h)
        inipat = rmds.haploid(initcon[params['n']:, params['n']:], alpha=params['alpha_pat'],
                              beta=beta, verbose=1, seed=seed, factr=1e5, smooth=smooth, h=h)
        init['x'] = rmds.combine(initcon, inimat, inipat, alpha=params['alpha_inter'],
                                 beta=beta, loci=params['loci'], verbose=1, seed=seed)
    elif os.path.isfile(initx):
        init['x'] = np.loadtxt(initx)
    else:
        raise NotImplementedError("Structure initialization method ({}) not implemented.".format(initx))

    if tail is None:
        merge = None
    elif 1 <= tail < params['n']:
        merge = params['n'] - tail
    else:
        raise ValueError("tail should between 1 and {}.".format(params['n'] - 1))
    model = create_model_human(init, modeltype=modeltype, seed=seed, merge=merge,
                               loci=params['loci'], diag=params['diag'], mask=params['mask'])
    simprogress = BasicCallback(model, outdir=outdir, simobj=sim, save=saveprog, seed=seed,
                                maxiter=maxiter, tol=tol)
    model, converge, loglikelihood, expected, message = emfit(model, data, maxiter=maxiter, tol=tol,
                                                     callback=simprogress.callback, **kwargs)
    # DONE add save for Poisson
    # DONE add option for save
    if savemat:
        # save expected Z and T matrices
        mtxpath = os.path.join(outdir, 'expected_matrices')
        if not os.path.exists(mtxpath):
            os.makedirs(mtxpath)
        if modeltype == 'ziphuman':
            zaa, zab, zbb = model.tomatrix(expected[0])
            ztaa, ztab, ztbb = model.tomatrix(expected[1])
            np.savetxt(os.path.join(mtxpath, 'z_aa.txt'), zaa)
            np.savetxt(os.path.join(mtxpath, 'z_ab.txt'), zab)
            np.savetxt(os.path.join(mtxpath, 'z_bb.txt'), zbb)
        else:
            ztaa, ztab, ztbb = model.tomatrix(expected)
        np.savetxt(os.path.join(mtxpath, 't_aa.txt'), ztaa)
        np.savetxt(os.path.join(mtxpath, 't_ab.txt'), ztab)
        np.savetxt(os.path.join(mtxpath, 't_bb.txt'), ztbb)
    #  save result as JSON
    with open(os.path.join(outdir, 'result.json'), 'w') as fh:
        retdict = {
            'loglikelihood': loglikelihood,
            'converge': converge,
            'message': message,
            'seed': seed,
            'elapsed_time': str(timedelta(seconds=time()-start_time))
        }
        if is_simulation:
            retdict['params_filepath'] = os.path.abspath(paramsfile)
            retdict['pickle_filepath'] = os.path.abspath(inputs[0])
            retdict['errors'] = simprogress.errors
        json.dump(retdict, fh, indent=4, sort_keys=True)
    model.dumpjson(os.path.join(outdir, 'model.json'), indent=4, sort_keys=True)
    # save 3D plot
    if is_simulation:
        plot3d.compare(np.array(model.x), np.array(sim.params['x']),
                       name=modeltype, prefix="final_", out=outdir)
    else:
        plot3d.plot(np.array(model.x), diploid=True, prefix="final_", out=outdir)


def cmd_simulate_fromparams(paramsfile, outdir, modeltype,
                            maxiter=20, tol=1e-4, beta=1.0,
                            seed=0, rseed=0, tail=None,
                            initx=None, smooth=False, h=1,
                            **kwargs):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(paramsfile, 'r') as fh:
        params = json.load(fh)
    sim = simulation.SimulationHuman(params, seed=seed)
    sim.simulate_data()

    # np.savetxt(os.path.join(outdir, 'oaa.txt'), sim.obs['aa'])
    # np.savetxt(os.path.join(outdir, 'oab.txt'), sim.obs['ab'])
    # np.savetxt(os.path.join(outdir, 'obb.txt'), sim.obs['bb'])
    # mate_aa = sim.obs['aa'] + sim.obs['ax'] + sim.obs['ax'].T
    # mate_bb = sim.obs['bb'] + sim.obs['bx'] + sim.obs['bx'].T
    # np.savetxt(os.path.join(outdir, 'maa.txt'), mate_aa)
    # np.savetxt(os.path.join(outdir, 'mbb.txt'), mate_bb)

    initcon = init_counts(
        join_matrix(sim.obs['aa'], sim.obs['ab'], sim.obs['ba'], sim.obs['bb']),
        sim.obs['ax'], sim.obs['bx'], sim.obs['xx']
    )

    init = {
        'n': params['n'],
        'alpha_mat': params['alpha_mat'],
        'alpha_pat': params['alpha_pat'],
        'alpha_inter': params['alpha_inter'],
        'beta': beta,
    }

    if initx is None or initx == 'random':
        init['x'] = (1 - 2 * np.random.rand(params['n'] * 2 * 3)).reshape((params['n'] * 2, 3))
    elif initx == 'true':
        init['x'] = params['x']
        # the scale beta=1.0 here is different from true structure
        init['beta'] = params['beta']
        init['gamma'] = params['gamma']
        init['p'] = params['p']
    elif initx == 'MDS':
        inimat = rmds.haploid(initcon[:params['n'], :params['n']], alpha=params['alpha_mat'],
                              beta=beta, verbose=1, seed=rseed, factr=1e5, smooth=smooth, h=h)
        inipat = rmds.haploid(initcon[params['n']:, params['n']:], alpha=params['alpha_pat'],
                              beta=beta, verbose=1, seed=rseed, factr=1e5, smooth=smooth, h=h)
        init['x'] = rmds.combine(initcon, inimat, inipat, alpha=params['alpha_inter'],
                                 beta=beta, loci=sim.loci, verbose=1, seed=rseed)
    elif os.path.isfile(initx):
        init['x'] = np.loadtxt(initx)
    else:
        raise NotImplementedError("Structure initialization method ({}) not implemented.".format(initx))

    if tail is None:
        merge = None
    elif 1 <= tail < params['n']:
        merge = params['n'] - tail
    else:
        raise ValueError("tail should between 1 and {}.".format(params['n'] - 1))
    model = create_model_human(init, modeltype=modeltype, seed=rseed, merge=merge,
                               loci=sim.loci, diag=sim.diag, mask=sim.mask)
    simprogress = BasicCallback(model, outdir=outdir, simobj=sim,
                                seed=seed, rseed=rseed,
                                maxiter=maxiter, tol=tol)
    model, converge, loglikelihood, expected = emfit(model, sim.obs, maxiter=maxiter, tol=tol,
                                                     callback=simprogress.callback, **kwargs)
    # save expected Z and T matrices
    zaa, zab, zbb = model.tomatrix(expected[0])
    ztaa, ztab, ztbb = model.tomatrix(expected[1])
    mtxpath = os.path.join(outdir, 'expected_matrices')
    if not os.path.exists(mtxpath):
        os.makedirs(mtxpath)
    np.savetxt(os.path.join(mtxpath, 'z_aa.txt'), zaa)
    np.savetxt(os.path.join(mtxpath, 'z_ab.txt'), zab)
    np.savetxt(os.path.join(mtxpath, 'z_bb.txt'), zbb)
    np.savetxt(os.path.join(mtxpath, 't_aa.txt'), ztaa)
    np.savetxt(os.path.join(mtxpath, 't_ab.txt'), ztab)
    np.savetxt(os.path.join(mtxpath, 't_bb.txt'), ztbb)
    #  save result as JSON
    with open(os.path.join(outdir, 'result.json'), 'w') as fh:
        retdict = {
            'loglikelihood': loglikelihood,
            'converge': converge,
            'seed_ini': rseed,
            'seed_sim': seed,
            'params_filepath': os.path.relpath(paramsfile, outdir),
            'errors': simprogress.errors
        }
        json.dump(retdict, fh, indent=4, sort_keys=True)
    model.dumpjson(os.path.join(outdir, 'model.json'), indent=4, sort_keys=True)
    # save 3D plot
    plot3d.compare(np.array(model.x), np.array(sim.params['x']),
                   name="ZIP", prefix="final_", out=outdir)

    # n = model.n
    # mask = model.mask
    # zt = expected[1]
    # ztaa, ztab, ztba, ztbb = np.zeros((n, n), dtype=float), np.zeros((n, n), dtype=float), \
    #     np.zeros((n, n), dtype=float), np.zeros((n, n), dtype=float)
    # ztaa[mask] = zt[0]
    # ztab[mask] = zt[1]
    # ztba[mask] = zt[2]
    # ztbb[mask] = zt[3]
    # # make zt symmetric so we can call iced on zt
    # ztaa = ztaa + ztaa.T
    # ztab = ztab + ztba.T
    # ztbb = ztbb + ztbb.T
    # ztaa[~model.symask] = np.nan
    # ztab[~model.symask] = np.nan
    # ztbb[~model.symask] = np.nan
    # np.savetxt(os.path.join(outdir, 'ztaa.txt'), sim.hidden['zt'][:n, :n])
    # np.savetxt(os.path.join(outdir, 'ztab.txt'), sim.hidden['zt'][:n, n:])
    # np.savetxt(os.path.join(outdir, 'ztbb.txt'), sim.hidden['zt'][n:, n:])
