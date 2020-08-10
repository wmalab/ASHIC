"""
Find the random run with highest loglikelihood.
"""

import os
import json
import click
import numpy as np
import cPickle as pickle
from ashic.model.poisson import Poisson
from ashic.model.zipoissonhuman import ZeroInflatedPoissonHuman


def symlink_force(source, link_name):
    """
    if symbolic link already exists, remove it first then make symlink
    """
    if os.path.exists(link_name) and os.path.islink(link_name):
        os.remove(link_name)
    os.symlink(source, link_name)


def model_rank(path):
    """
    return rank statistic for given model
    higher value means higher rank
    """
    result_file = os.path.join(path, 'log.json')
    if not os.path.isfile(result_file):
        return float('-inf')
    with open(result_file, 'r') as fh:
        result = json.load(fh)
        return float(result['loglikelihood'])


def find_bestrun(output_dir, data_path, model_type):
    """
    :param output_dir: output directory for a given data e.g. simulate_data_0
    :param model_type: ASHIC-ZIPM or ASHIC-PM
    :return:
    """
    opt_dir = os.path.join(output_dir, 'opt_run')
    dir_list = [os.path.join(output_dir, subdir) for subdir in os.listdir(output_dir)]
    opt_run = max(dir_list, key=lambda x: model_rank(x))
    symlink_force(os.path.abspath(opt_run), os.path.abspath(opt_dir))
    # generate expected matrices if not found
    mtxdir = os.path.join(opt_dir, 'matrices')
    modelpath = os.path.join(opt_dir, 'model.json')
    if not os.path.exists(mtxdir):
        os.makedirs(mtxdir)
        # load the pickle data
        with open(data_path, 'rb') as fr:
            data = pickle.load(fr)
        if model_type == 'ASHIC-ZIPM':
            model = ZeroInflatedPoissonHuman.fromjson(modelpath, mask=data['params']['mask'])
        elif model_type == 'ASHIC-PM':
            model = Poisson.fromjson(modelpath, mask=data['params']['mask'])
        else:
            raise NotImplementedError('Model not implemented.')
        # TODO change model.savematrix files to t_mm.txt, t_pp.txt, etc.
        model.savematrix(data['obs'], mtxdir)


@click.command()
@click.option('--model', default='ASHIC-ZIPM', show_default=True,
              type=click.Choice(['ASHIC-ZIPM', 'ASHIC-PM']))
@click.argument('outdir')
@click.argument('datapath')
def cli(outdir, datapath, model):
    find_bestrun(output_dir=outdir, data_path=datapath, model_type=model)


if __name__ == '__main__':
    cli()