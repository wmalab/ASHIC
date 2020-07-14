"""
Collections of utils to post-processing outputs
"""
import os
import json
import cPickle as pickle
import numpy as np
from ashic.model.zipoissonhuman import ZeroInflatedPoissonHuman
from ashic.model.poisson import Poisson


def symlink_force(source, link_name):
    """
    if symbolic link already exists, remove it first then make symlink
    """
    if os.path.exists(link_name) and os.path.islink(link_name):
        os.remove(link_name)
    os.symlink(source, link_name)

def model_rank(path, model_type, method, data):
    """
    return rank statistic for given model
    higher value means higher rank
    method: "ll", "der", "intra_ll"
    """
    result_file = os.path.join(path, 'result.json')
    model_path = os.path.join(path, 'model.json')
    if not os.path.isfile(result_file):
        return float('-inf')
    with open(result_file, 'r') as fh:
        result = json.load(fh)
        if method == "ll":
            return float(result['loglikelihood'])
        elif method == "der":
            return -float(result['errors']['d'])
        elif method == "intra_ll":
            if model_type == "ziphuman":
                model = ZeroInflatedPoissonHuman.fromjson(model_path, mask=data.mask)
            elif model_type == "poisson":
                model = Poisson.fromjson(model_path, mask=data.mask)
            else:
                raise ValueError("modeltype should be one of ziphuman or poisson.")
            return model.intra_log_likelihood(data.obs)
        else:
            raise ValueError("method value should be one of ll, der, and intra_ll.")


# TODO find best run and make symbolink
# TODO if no expected matrices generate
def find_opt_run(output_dir, rs=0, nrun=10, modeltype='ziphuman', method='ll'):
    """
    :param output_dir: output directory for a given data e.g. simulate_data_0
    :param modeltype: ziphuman or poisson
    :return:
    """
    opt_dir = os.path.join(output_dir, 'opt_run')
    dir_list = [os.path.join(output_dir, 'run_{}'.format(i)) for i in xrange(rs, rs+nrun)]
    with open(os.path.join(dir_list[0], 'result.json'), 'r') as fh:
        with open(json.load(fh)['pickle_filepath'], 'rb') as fr:
            data = pickle.load(fr)
    opt_run = max(dir_list, key=lambda x: model_rank(x, modeltype, method, data))
    symlink_force(os.path.abspath(opt_run), os.path.abspath(opt_dir))
    # generate expected matrices if not found
    mtxdir = os.path.join(opt_dir, 'expected_matrices')
    modelpath = os.path.join(opt_dir, 'model.json')
    if not os.path.exists(mtxdir):
        os.makedirs(mtxdir)
        # load the pickle data
        with open(os.path.join(opt_dir, 'result.json'), 'r') as fh:
            res = json.load(fh)
        with open(res['pickle_filepath'], 'rb') as fr:
            data = pickle.load(fr)
        if modeltype == 'ziphuman':
            model = ZeroInflatedPoissonHuman.fromjson(modelpath, mask=data.mask)
        elif modeltype == 'poisson':
            model = Poisson.fromjson(modelpath, mask=data.mask)
        else:
            raise NotImplementedError('Model not implemented.')
        model.savematrix(data.obs, mtxdir)


def suppmat(data_file, output_dir):
    with open(data_file, 'rb') as fh:
        data = pickle.load(fh)
    n = data.params['n']
    mask = data.mask
    oaa, oab, obb = data.obs['aa'], data.obs['ab'], data.obs['bb']
    mate_aa = data.obs['aa'] + data.obs['ax'] + data.obs['ax'].T
    mate_bb = data.obs['bb'] + data.obs['bx'] + data.obs['bx'].T
    taa, tab, tbb = data.hidden['zt'][:n, :n], data.hidden['zt'][:n, n:], data.hidden['zt'][n:, n:]
    for mat in (oaa, oab, obb, mate_aa, mate_bb, taa, tab, tbb):
        mat[~mask] = np.nan
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(os.path.join(output_dir, 'oaa.txt'), oaa)
    np.savetxt(os.path.join(output_dir, 'oab.txt'), oab)
    np.savetxt(os.path.join(output_dir, 'obb.txt'), obb)
    np.savetxt(os.path.join(output_dir, 'maa.txt'), mate_aa)
    np.savetxt(os.path.join(output_dir, 'mbb.txt'), mate_bb)
    np.savetxt(os.path.join(output_dir, 'taa.txt'), taa)
    np.savetxt(os.path.join(output_dir, 'tab.txt'), tab)
    np.savetxt(os.path.join(output_dir, 'tbb.txt'), tbb)


# output
#   - errors
#   - simulate_data_*
#       - opt_run
#           - model.json
#           - result.json
# def func(output_dir):
#     # get the gamma from model.json and true gamma from sim
#     gamma_errors = []
#     true_gamma = None
#     for subdir in os.listdir(output_dir):
#         opt_dir = os.path.join(output_dir, subdir, 'opt_run')
#         if os.path.exists(opt_dir):
#             model_file = os.path.join(opt_dir, 'model.json')
#             res_file = os.path.join(opt_dir, 'result.json')
#             with open(model_file, 'r') as fh:
#                 model = json.load(fh)
#                 gamma = np.array(model['params']['gamma'])
#             if true_gamma is None:
#                 with open(res_file, 'r') as fh:
#                     res = json.load(fh)
#                 with open(res['pickle_filepath'], 'rb') as fh:
#                     data = pickle.load(fh)
#                     true_gamma = np.array(data.params['gamma'])
#             gamma_errors.append(np.abs(gamma-true_gamma))
#     np.savetxt('test.txt', gamma_errors)
