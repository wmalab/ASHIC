import os
import json
import numpy as np
from allelichicem import simulation
from allelichicem import structure
from allelichicem.utils import find_closestlength_chrom, encodejson
from allelichicem.utils import get_localinds, get_rdis, parse_localinds
from allelichicem.utils import centroid_distance
from allelichicem.commands.fit import initialx, create_model
from allelichicem.em import emfit
from allelichicem.progresscb import SimulationProgress
from allelichicem.model.zipoisson import ZeroInflatedPoisson
from allelichicem.model.poisson import Poisson


def load_yeaststructures():
    datadir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           '../examples/yeast'))
    #  load yeast structures
    structures = np.loadtxt(os.path.join(datadir, 'yeast.pdb.txt'))
    structures = structures.reshape((len(structures) / 3, 3))
    lengths = np.loadtxt(os.path.join(datadir, "yeast_lengths.txt"), dtype='int32')

    start = 0
    chroms = {}
    for i, length in enumerate(lengths, start=1):
        end = start + length
        chroms['chr'+str(i)] = structures[start:end, :]
        start = end
    return chroms


def simulate_yeastdata(params, seed=0):
    sim = simulation.Simulation(params, seed=seed)
    sim.simulate_data()
    return sim


def sample_simulation_params(chroms, chrom1=None, simtype=None,
                             alpha=-3.0, beta=1.0,
                             p_a=2.0, p_b=2.0,
                             gamma_a=0.9, gamma_b=-0.2, gamma_inter=0.05,
                             seed=0, **kwargs):
    settings = {
        'chrom1': chrom1,
        'simtype': simtype,
        'alpha': alpha,
        'beta': beta,
        'p_a': p_a,
        'p_b': p_b,
        'gamma_a': gamma_a,
        'gamma_b': gamma_b,
        'gamma_inter': gamma_inter,
        'seed': seed
    }
    if simtype == 'diff':
        assert kwargs.get('chrom2') is not None, "chrom2 must be provided."
        settings['chrom2'] = kwargs['chrom2']
        x = simulation.sample_diff_structure(chroms[chrom1], chroms[kwargs['chrom2']])
        n = int(x.shape[0] / 2)
    elif simtype == 'same' or simtype == 'local':
        # center distance is estiamted from chrom1 and the chrom with closest length
        # cut at the same length
        if kwargs.get('chrom2') is not None:
            minchrom = kwargs['chrom2']
        else:
            minchrom = find_closestlength_chrom(chroms, chrom1)
        settings['chrom2'] = minchrom
        n = min(chroms[chrom1].shape[0], chroms[minchrom].shape[0])
        # TODO change center
        # if kwargs.get('centerdis') is None:
        #     cd = structure.center_distance(chroms[chrom1], chroms[minchrom])
        # else:
        #     cd = kwargs['centerdis']
        # settings['centerdis'] = cd

        if simtype == 'same':
            x = simulation.sample_same_structure(chroms[chrom1], chroms[minchrom])
            cd = centroid_distance(x[:n, :], x[n:, :])
            settings['centerdis'] = cd
        else:
            if kwargs.get('localinds') is None:
                localinds = get_localinds(n=n, percentile=kwargs.get('percentile', 0.2),
                                          fragment_size=kwargs.get('fragmentsize', 5))
            else:
                localinds = parse_localinds(kwargs['localinds'])
            settings['localinds'] = localinds

            if kwargs.get('diffd') is None:
                # TODO multiply 2?
                radius = kwargs.get('radius', 2)
                diffd = get_rdis(chroms[chrom1]) * radius
                settings['radius'] = radius
            else:
                diffd = kwargs['diffd']
            settings['diffd'] = diffd

            x = simulation.sample_localdiff_structure(chroms[chrom1], chroms[minchrom],
                                                      localinds, diffd, randstate=seed)
            cd = centroid_distance(x[:n, :], x[n:, :])
            settings['centerdis'] = cd
    else:
        raise ValueError("Simulation type should be diff, same or local.")

    p = simulation.sample_p(a=p_a, b=p_b, n=n, randstate=seed)
    gamma = simulation.sample_gamma(a=gamma_a, b=gamma_b, inter=gamma_inter, n=n)
    params = {
        'alpha': alpha,
        'beta': beta,
        'p': p,
        'gamma': gamma,
        'x': x,
        'n': n
    }
    return params, settings


def cmd_sample_params(chrom1, simtype, outdir, **kwargs):
    yeastchroms = load_yeaststructures()
    simparams, simsettings = sample_simulation_params(yeastchroms,
                                                      chrom1=chrom1, simtype=simtype, **kwargs)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    encodejson(simparams)
    encodejson(simsettings)
    with open(os.path.join(outdir, 'params.json'), 'w') as fh:
        json.dump(simparams, fh, indent=4, sort_keys=True)
    with open(os.path.join(outdir, 'settings.json'), 'w') as fh:
        json.dump(simsettings, fh, indent=4, sort_keys=True)


def cmd_simulate_fromparams(paramsfile, outdir, modeltype,
                            numruns=5, maxiter=20, tol=1e-2,
                            alpha=-3.0, beta=1.0, seed=0, tail=None, **kwargs):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(paramsfile, 'r') as fh:
        params = json.load(fh)
    sim = simulate_yeastdata(params, seed=seed)
    # np.savetxt(os.path.join(outdir, 'z.txt'), sim.hidden['z'])
    # np.savetxt(os.path.join(outdir, 't.txt'), sim.hidden['t'])
    # for var in ('aa', 'ab', 'bb', 'ax', 'bx', 'xx'):
    #     np.savetxt(os.path.join(outdir, var+'.txt'), sim.obs[var])
    # TODO change as a function
    best_loglikelihood = -np.inf
    best_model = None
    best_converge = False
    best_expected = None
    best_sim = None
    best_rseed = None
    for rseed in range(numruns):
        init = {
            'n': params['n'],
            'x': initialx(sim.obs, alpha=alpha, beta=beta, seed=rseed, **kwargs),
            'alpha': alpha,
            'beta': beta
        }
        if tail is None:
            merge = None
        elif 1 <= tail < params['n']:
            merge = params['n'] - tail
        else:
            raise ValueError("tail should between 1 and {}.".format(params['n'] - 1))
        model = create_model(init, modeltype=modeltype, seed=rseed, merge=merge)
        simprogress = SimulationProgress(model, outdir=os.path.join(outdir, 'em_seed_'+str(rseed)),
                                         simobj=sim, seed=rseed, maxiter=maxiter, tol=tol)
        model, converge, loglikelihood, expected = emfit(model, sim.obs, maxiter=maxiter, tol=tol,
                                                         callback=simprogress.callback)
        with open(simprogress.logfile, 'a') as fh:
            fh.write("# converge={}\n".format(converge))
        # choose the model with maximum loglikelihood in all runs
        if loglikelihood > best_loglikelihood:
            best_loglikelihood = loglikelihood
            best_model = model
            best_converge = converge
            best_expected = expected
            best_sim = simprogress
            best_rseed = rseed
    #  save best result
    with open(os.path.join(outdir, 'result.json'), 'w') as fh:
        retdict = {
            'loglikelihood': best_loglikelihood,
            'converge': best_converge,
            'em_seed': best_rseed,
            'simulation_seed': seed,
            'params_filepath': os.path.relpath(paramsfile, outdir),
            'relative_error': best_sim.errors
        }
        json.dump(retdict, fh, indent=4, sort_keys=True)
    best_model.dumpjson(os.path.join(outdir, 'result_model.json'),
                        indent=4, sort_keys=True)
    with open(os.path.join(outdir, 'result_expected.json'), 'w') as fh:
        row, col = np.where(best_model.mask)
        values = {}
        if isinstance(best_model, Poisson):
            values['T'] = {
                'aa': best_expected[0],
                'ab': best_expected[1],
                'ba': best_expected[2],
                'bb': best_expected[3]
            }
        elif isinstance(best_model, ZeroInflatedPoisson):
            values['Z'] = {
                'aa': best_expected[0][0],
                'ab': best_expected[0][1],
                'ba': best_expected[0][2],
                'bb': best_expected[0][3],
            }
            values['T'] = {
                'aa': best_expected[1][0],
                'ab': best_expected[1][1],
                'ba': best_expected[1][2],
                'bb': best_expected[1][3],
            }
        else:
            raise ValueError("model should be zip or poisson.")
        encodejson(values)
        expectdict = {
            'n': params['n'],
            'row': row.flatten().tolist(),
            'col': col.flatten().tolist(),
            'values': values
        }
        json.dump(expectdict, fh, indent=4, sort_keys=True)


def cmd_simulate_fromsettings():
    pass
