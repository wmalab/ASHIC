import os
import numpy as np
from sklearn.metrics import euclidean_distances
from allelichicem.structure import uniformscaling_distance
from allelichicem.model.zipoisson import ZeroInflatedPoisson
from allelichicem.model.zipoissonhuman import ZeroInflatedPoissonHuman
from allelichicem.model.poisson import Poisson
from allelichicem.utils import join_matrix
from allelichicem.utils import disjoin_matrix
from allelichicem.misc import plot3d


def basic_callback(i, model, loglikehood, expected):
    print "{name}: iteration {i} (observed log-likelihood={ll})".format(
        name=model.name,
        i=i,
        ll=loglikehood)


class BasicCallback(object):
    def __init__(self, model, outdir=None, simobj=None, save=False, **kwargs):
        if isinstance(model, ZeroInflatedPoissonHuman):
            self.modeltype = 'ZIP'
        elif isinstance(model, Poisson):
            self.modeltype = 'Poisson'
        else:
            raise NotImplementedError("Model not implemented.")
        self.outdir = outdir
        self.modeldir = os.path.join(outdir, 'saved_models')
        self.plotdir = os.path.join(outdir, 'saved_plots')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if save and (not os.path.exists(self.modeldir)):
            os.makedirs(self.modeldir)
        if save and (not os.path.exists(self.plotdir)):
            os.makedirs(self.plotdir)
        self.simobj = simobj
        self.save = save
        self.errors = {}  # save the last computed errors
        self.logfile = os.path.join(outdir, 'progress.txt')
        with open(self.logfile, 'w') as fh:
            fh.write("# modelname={}\n".format(model.name))
            for arg in kwargs:
                fh.write("# {}={}\n".format(arg, kwargs[arg]))
            if simobj:
                if self.modeltype == 'ZIP':
                    fh.write("# iteration\tloglikelihood\tRE(p)\tRE(gamma)\t" +
                             "RE(Dm)\tRE(Dp)\tRE(D)\tRE(T)\n")
                elif self.modeltype == 'Poisson':
                    fh.write("# iteration\tloglikelihood\tRE(p)\tRE(Dm)\tRE(Dp)\tRE(D)\tRE(T)\n")
            else:
                fh.write("# iteration\tloglikelihood\n")

    def callback(self, i, model, loglikehood, expected):
        logstr = "{}\t{}".format(i, loglikehood)
        if self.simobj:
            if self.save:
                plot3d.compare(np.array(model.x), np.array(self.simobj.params['x']),
                               name=self.modeltype, prefix=str(i)+"_", out=self.plotdir)
            n = model.n
            loci = model.loci
            x = model.x
            y = self.simobj.params['x']
            # relative error for p
            re_p = np.nansum(np.absolute(model.p[loci] - self.simobj.params['p'][loci])) \
                / np.nansum(self.simobj.params['p'][loci])
            # relative error for intra-distance
            x1 = x[:n, :][loci, :]
            x2 = x[n:, :][loci, :]
            y1 = y[:n, :][loci, :]
            y2 = y[n:, :][loci, :]
            m = x1.shape[0]
            sx1 = (((x1 - np.tile(x1.mean(axis=0), (m, 1))) ** 2).sum() / m) ** 0.5
            sx2 = (((x2 - np.tile(x2.mean(axis=0), (m, 1))) ** 2).sum() / m) ** 0.5
            sy1 = (((y1 - np.tile(y1.mean(axis=0), (m, 1))) ** 2).sum() / m) ** 0.5
            sy2 = (((y2 - np.tile(y2.mean(axis=0), (m, 1))) ** 2).sum() / m) ** 0.5
            x1 = x1 / sx1
            x2 = x2 / sx2
            y1 = y1 / sy1
            y2 = y2 / sy2
            mask = model.mask[loci, :][:, loci]
            dx1, dx2 = euclidean_distances(x1)[mask], euclidean_distances(x2)[mask]
            dy1, dy2 = euclidean_distances(y1)[mask], euclidean_distances(y2)[mask]
            re_dm = np.absolute(dx1 - dy1).sum() / dy1.sum()
            re_dp = np.absolute(dx2 - dy2).sum() / dy2.sum()
            re_d = (np.absolute(dx1 - dy1).sum() + np.absolute(dx2 - dy2).sum()) \
                / (dy1.sum() + dy2.sum())
            # relative error for intra-contact
            ztaa, _, _, ztbb = disjoin_matrix(self.simobj.hidden['zt'], n, model.mask)
            if self.modeltype == 'ZIP':
                # relative error for gamma
                re_gamma = np.nansum(np.absolute(model.gamma - self.simobj.params['gamma'])) \
                    / np.nansum(self.simobj.params['gamma'])
                re_t = (np.absolute(expected[1][0] - ztaa).sum() + np.absolute(expected[1][3] - ztbb).sum()) \
                    / (ztaa.sum() + ztbb.sum())
                logstr += "\t{}\t{}\t{}\t{}\t{}\t{}".format(re_p, re_gamma, re_dm, re_dp, re_d, re_t)
                self.errors['gamma'] = re_gamma
            elif self.modeltype == 'Poisson':
                re_t = (np.absolute(expected[0] - ztaa).sum() + np.absolute(expected[3] - ztbb).sum()) \
                    / (ztaa.sum() + ztbb.sum())
                logstr += "\t{}\t{}\t{}\t{}\t{}".format(re_p, re_dm, re_dp, re_d, re_t)
            self.errors['p'] = re_p
            self.errors['dm'] = re_dm
            self.errors['dp'] = re_dp
            self.errors['d'] = re_d
            self.errors['t'] = re_t
        else:
            if self.save:
                plot3d.plot(np.array(model.x), diploid=True,
                            prefix=str(i)+"_", out=self.plotdir)
        with open(self.logfile, 'a') as fh:
            fh.write(logstr + "\n")
        if self.save:
            model.dumpjson(os.path.join(self.modeldir, '{}_model.json'.format(i)),
                           indent=4, sort_keys=True)


class SimulationProgress(object):
    def __init__(self, model, outdir=None, simobj=None, **kwargs):
        self.outdir = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.simobj = simobj
        self.errors = {}
        # create log file to record progress
        self.logfile = os.path.join(outdir, 'progress.txt')
        with open(self.logfile, 'w') as fh:
            fh.write("# modelname={}\n".format(model.name))
            for arg in kwargs:
                fh.write("# {}={}\n".format(arg, kwargs[arg]))
            if isinstance(model, ZeroInflatedPoisson):
                if simobj:
                    fh.write("# iteration\tloglikelihood\tRE(p)\tRE(gamma)\t" +
                             "RE(D)\tRE(intra-D)\tRE(T)\tRE(intra-T)\n")
                else:
                    fh.write("# iteration\tloglikelihood\n")
            elif isinstance(model, Poisson):
                if simobj:
                    fh.write("# iteration\tloglikelihood\tRE(p)\t" +
                             "RE(D)\tRE(intra-D)\tRE(T)\tRE(intra-T)\n")
                else:
                    fh.write("# iteration\tloglikelihood\n")

    def callback(self, i, model, loglikehood, expected):
        logstr = "{}\t{}".format(i, loglikehood)
        if self.simobj:
            model_params = model.getparams()
            re_p = np.absolute(
                model_params['p'][model.loci] -
                self.simobj.params['p'][model.loci]).sum() \
                / self.simobj.params['p'][model.loci].sum()
            self.errors['p'] = re_p
            d1 = uniformscaling_distance(model_params['x'])
            d2 = uniformscaling_distance(self.simobj.params['x'])
            mask = np.tile(model.mask, (2, 2))
            re_d = np.absolute(d1[mask] - d2[mask]).sum() / d2[mask].sum()
            self.errors['D'] = re_d
            daa1, _, _, dbb1 = disjoin_matrix(d1, model_params['n'], model.mask)
            daa2, _, _, dbb2 = disjoin_matrix(d2, model_params['n'], model.mask)
            re_d_intra = (np.absolute(daa1 - daa2).sum() + np.absolute(dbb1 - dbb2).sum())\
                / (daa2.sum() + dbb2.sum())
            self.errors['intraD'] = re_d_intra
            logstr += "\t{}".format(re_p)
            ztaa2, _, _, ztbb2 = disjoin_matrix(self.simobj.hidden['zt'], model_params['n'], model.mask)
            if isinstance(model, ZeroInflatedPoisson):
                zt = join_matrix(expected[1][0], expected[1][1], expected[1][2], expected[1][3],
                                 n=model_params['n'], mask=model.mask)
                re_t = np.absolute(zt[mask] - self.simobj.hidden['zt'][mask]).sum() \
                    / self.simobj.hidden['zt'][mask].sum()
                self.errors['T'] = re_t
                re_t_intra = (np.absolute(expected[1][0] - ztaa2).sum() + np.absolute(expected[1][3] - ztbb2).sum())\
                    / (ztaa2.sum() + ztbb2.sum())
                self.errors['intraT'] = re_t_intra
                re_gamma = np.absolute(model_params['gamma'] - self.simobj.params['gamma']).sum() \
                    / self.simobj.params['gamma'].sum()
                self.errors['gamma'] = re_gamma
                logstr += "\t{}\t{}\t{}\t{}\t{}".format(re_gamma, re_d, re_d_intra, re_t, re_t_intra)
            if isinstance(model, Poisson):
                t = join_matrix(expected[0], expected[1], expected[2], expected[3],
                                n=model_params['n'], mask=model.mask)
                re_t = np.absolute(t[mask] - self.simobj.hidden['zt'][mask]).sum() \
                    / self.simobj.hidden['zt'][mask].sum()
                self.errors['T'] = re_t
                re_t_intra = (np.absolute(expected[0] - ztaa2).sum() + np.absolute(expected[3] - ztbb2).sum())\
                    / (ztaa2.sum() + ztbb2.sum())
                self.errors['intraT'] = re_t_intra
                logstr += "\t{}\t{}\t{}\t{}".format(re_d, re_d_intra, re_t, re_t_intra)
        with open(self.logfile, 'a') as fh:
            fh.write(logstr + "\n")
        model.dumpjson(os.path.join(self.outdir, 'model_iteration{}.json'.format(i)),
                       indent=4, sort_keys=True)
