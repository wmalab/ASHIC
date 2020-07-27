import os
import click
import cPickle as pickle
import numpy as np
from ashic.misc.zipgamma import estimate_gamma


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """Collection of helper commands for ASHIC."""

@cli.command(name='fitgamma')
@click.argument("datafile", type=click.Path(exists=True))
@click.argument("outputdir")
@click.option("--nbins", default=200, type=int, show_default=True,
              help='Number of genomic intervals used for fitting.')
def fitgamma(datafile, outputdir, nbins):
    """Fit initial gamma values from data."""
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    with open(datafile, 'rb') as fh:
        pk = pickle.load(fh)
        data = pk["obs"]
        params = pk["params"]
        gamma = estimate_gamma(data, params["mask"], 
                               params["loci"], params["diag"], outputdir, nbins)
        np.savetxt(os.path.join(outputdir, "init_gamma.txt"), gamma)