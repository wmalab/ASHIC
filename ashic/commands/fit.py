import numpy as np
from allelichicem.model.zipoisson import ZeroInflatedPoisson
from allelichicem.model.zipoissonhuman import ZeroInflatedPoissonHuman
from allelichicem.model.poisson import Poisson
from allelichicem.utils import init_counts, join_matrix
from allelichicem.optimization.mds import MDS


def initialx(data, alpha=-3.0, beta=1.0, seed=0, max_iter=5000, smooth=False, h=0, diag=0):
    mds = MDS(alpha=alpha, beta=beta, random_state=seed, max_iter=max_iter,
              smooth=smooth, h=h, diag=diag, numchr=2)
    # init full_counts by using certain counts as poisson lambda
    certain = join_matrix(data['aa'], data['ab'], data['ba'], data['bb'])
    full_counts = init_counts(certain, data['ax'], data['bx'], data['xx'])
    x = mds.fit(full_counts)
    return x


def create_model(params, modeltype='zip', seed=0, merge=None):
    if modeltype == 'zip':
        model = ZeroInflatedPoisson(params, merge=merge, normalize=False,
                                    random_state=np.random.RandomState(seed=seed))
    elif modeltype == 'poisson':
        model = Poisson(params, normalize=False, random_state=np.random.RandomState(seed=seed))
    else:
        raise ValueError("Model type should be zip or poisson.")
    return model


def create_model_human(params, modeltype='ziphuman', seed=0, merge=None, loci=None, diag=0, mask=None):
    if modeltype == 'ziphuman':
        model = ZeroInflatedPoissonHuman(params, merge=merge, normalize=False, loci=loci, diag=diag, mask=mask,
                                         random_state=np.random.RandomState(seed=seed))
    elif modeltype == 'poisson':
        model = Poisson(params, normalize=False, loci=loci, diag=diag, mask=mask,
                        random_state=np.random.RandomState(seed=seed))
    else:
        raise ValueError("Model type not implemented.")
    return model


def fit(data, outdir, modeltype, n=None, numruns=5, maxiter=20, tol=1e-2, alpha=-3.0, beta=1.0, tail=None, **kwargs):
    best_loglikelihood = -np.inf
    best_model = None
    best_converge = False
    best_expected = None
    best_sim = None
    best_rseed = None
    if n is None:
        n = data['aa'].shape[0]
    for rseed in range(numruns):
        init = {
            'n': n,
            'x': initialx(data, alpha=alpha, beta=beta, seed=rseed, **kwargs),
            'alpha': alpha,
            'beta': beta
        }
        if tail is None:
            merge = None
        elif 1 <= tail < n:
            merge = n - tail
        else:
            raise ValueError("tail should between 1 and {}.".format(n - 1))
        model = create_model(init, modeltype=modeltype, seed=rseed, merge=merge)
        simprogress = SimulationProgress(model, outdir=os.path.join(outdir, 'seed'+str(rseed)),
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
            'seed': best_rseed,
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