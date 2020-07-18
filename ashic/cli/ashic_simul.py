import click
from ashic.commands import simulateyeast
from ashic.commands import simulatehuman
from ashic.misc import postprocessing


@click.group()
def cli():
    """
    ASHIC: EM algorithm to assign uncertain contacts in allelic Hi-C
    """


@cli.command()
@click.option('--alpha', default=-3.0, type=float)
@click.option('--beta', default=1.0, type=float)
@click.option('--p_a', default=2.0, type=float)
@click.option('--p_b', default=2.0, type=float)
@click.option('--gamma_a', default=0.9, type=float)
@click.option('--gamma_b', default=-0.2, type=float)
@click.option('--gamma_inter', default=0.05, type=float)
@click.option('--seed', default=0, type=int)
@click.option('--chrom2')
@click.option('--centerdis', type=float)
@click.option('--localinds')
@click.option('--radius', default=2, type=int)
@click.option('--diffd', type=float)
@click.argument('chrom1')
@click.argument('simtype')
@click.argument('outdir')
def params(chrom1, simtype, outdir, alpha, beta,
           p_a, p_b, gamma_a, gamma_b, gamma_inter,
           seed, chrom2, centerdis, localinds, radius, diffd):
    """
    Sample simulation parameters.
    """
    simulateyeast.cmd_sample_params(chrom1, simtype, outdir,
                                    alpha=alpha, beta=beta,
                                    p_a=p_a, p_b=p_b,
                                    gamma_a=gamma_a, gamma_b=gamma_b, gamma_inter=gamma_inter,
                                    seed=seed, chrom2=chrom2, centerdis=centerdis,
                                    localinds=localinds, radius=radius, diffd=diffd)


@cli.command()
@click.option('--seed', default=0, type=int)
@click.option('--numruns', default=5, type=int)
@click.option('--maxiter', default=20, type=int)
@click.option('--tol', default=1e-2, type=float)
@click.option('--alpha', default=-3.0, type=float)
@click.option('--beta', default=1.0, type=float)
@click.option('--tail', type=int)
@click.option('--smooth/--no-smooth', default=False)
@click.option('--h', default=0, type=int)
@click.argument('paramsfile', type=click.Path(exists=True))
@click.argument('outdir')
@click.argument('modeltype')
def simulation(paramsfile, outdir, modeltype, numruns, maxiter, tol,
               alpha, beta, tail, smooth, h, seed):
    """
    Simulation using yeast structure.
    """
    simulateyeast.cmd_simulate_fromparams(paramsfile, outdir, modeltype,
                                          numruns=numruns, maxiter=maxiter, tol=tol,
                                          alpha=alpha, beta=beta, tail=tail, seed=seed,
                                          smooth=smooth, h=h)


# TODO run EM
@cli.command()
@click.option('--alpha-inter', default=-3.0, type=float)
@click.option('--gamma-inter', default=0.05, type=float)
@click.option('--p-a', default=2.0, type=float)
@click.option('--p-b', default=2.0, type=float)
@click.option('--seed', default=0, type=int)
@click.option('--diag', default=1, type=int)
@click.option('--filter-high', default=99.9, type=float)
@click.argument('chrom')
@click.argument('outdir')
def paramshuman(chrom, outdir, alpha_inter, gamma_inter, p_a, p_b, seed, diag, filter_high):
    """
    estimate and generate simulation parameters for human diploid structures
    """
    simulatehuman.cmd_estimate_params(chrom, outdir, alpha_inter=alpha_inter, gamma_inter=gamma_inter,
                                      p_a=p_a, p_b=p_b, seed=seed, diag=diag, filter_high=filter_high,
                                      plot=True)


@cli.command()
@click.option('--parent', required=True, type=click.Choice(['mat', 'pat']))
@click.argument('paramsfile', type=click.Path(exists=True))
@click.argument('outdir')
def duplicate(paramsfile, outdir, parent):
    simulatehuman.cmd_duplicate_structure(paramsfile, outdir, parent)


@cli.command()
@click.option('--frac-beta', type=float)
@click.option('--frac-p', type=float)
@click.option('--seed', default=0, type=int)
@click.argument('paramsfile', type=click.Path(exists=True))
@click.argument('outdir')
def downparams(paramsfile, outdir, frac_beta, frac_p, seed):
    """
    downsampling beta or p
    """
    simulatehuman.cmd_downsample_params(paramsfile, outdir, frac_beta=frac_beta, frac_p=frac_p, seed=seed)


@cli.command()
@click.option('--seed', default=0, type=int)
@click.argument('paramsfile', type=click.Path(exists=True))
@click.argument('outdir')
def simulatedata(paramsfile, outdir, seed):
    """
    generate simulated data from parameters file
    """
    simulatehuman.cmd_simulate_data(paramsfile, outdir, seed=seed)


@cli.command()
@click.option('-o', '--outdir', required=True)
@click.option('--paramsfile', type=click.Path(exists=True))
@click.option('--model', default='ziphuman', show_default=True,
              type=click.Choice(['ziphuman', 'poisson']))
@click.option('--simulated', is_flag=True)
@click.option('--beta', default=1.0, type=float)
@click.option('--diag', default=1, type=int)
@click.option('--maxiter', default=30, type=int)
@click.option('--tol', default=1e-4, type=float)
@click.option('--seed', default=0, type=int)
@click.option('--tail', type=int)
@click.option('--initgamma', type=click.Path(exists=True))
@click.option('--initx', default='MDS', show_default=True)
@click.option('--max-func', default=200, type=int, show_default=True)
@click.option('--separate', is_flag=True)
@click.option('--save/--no-save', default=True)
@click.option('--prog/--no-prog', default=True)
@click.argument('inputs', nargs=-1)
def run(inputs, outdir, paramsfile, model, simulated, beta, diag,
        maxiter, tol, seed, tail, initgamma, initx, max_func, separate, save, prog):
    """
    run EM algorithm on simulated and real data
    """
    simulatehuman.cmd_run(inputs, outdir, paramsfile=paramsfile, modeltype=model,
                          is_simulation=simulated, savemat=save, saveprog=prog, beta=beta, diag=diag,
                          maxiter=maxiter, tol=tol, seed=seed, tail=tail, initgamma=initgamma, initx=initx, smooth=False, h=1,
                          max_func=max_func, separate=separate)


@cli.command()
@click.option('--numruns', default=1, type=int)
@click.option('--maxiter', default=20, type=int)
@click.option('--tol', default=1e-2, type=float)
@click.option('--beta', default=1.0, type=float)
@click.option('--tail', type=int)
@click.option('--seed', default=0, type=int)
@click.option('--initx', type=click.Path(exists=True))
@click.argument('paramsfile', type=click.Path(exists=True))
@click.argument('outdir')
@click.argument('modeltype')
def simulationhuman(paramsfile, outdir, modeltype, numruns, maxiter, tol,
                    beta, tail, seed, initx):
    """
    Simulation using human structure.
    """
    simulatehuman.cmd_simulate_fromparams(paramsfile, outdir, modeltype,
                                          numruns=numruns, maxiter=maxiter, tol=tol,
                                          beta=beta, seed=seed, tail=tail, initx=initx)


@cli.command()
@click.option('--rs', default=0, show_default=True)
@click.option('--nrun', default=10, show_default=True)
@click.option('--method', default='ll', show_default=True,
              type=click.Choice(['ll', 'der', 'intra_ll']))
@click.option('--model', default='ziphuman', show_default=True,
              type=click.Choice(['ziphuman', 'poisson']))
@click.argument('outdir')
def getopt(outdir, rs, nrun, model, method):
    postprocessing.find_opt_run(outdir, rs=rs, nrun=nrun, 
                                modeltype=model, method=method)


@cli.command()
@click.argument('data', type=click.Path(exists=True))
@click.argument('outdir')
def suppmat(data, outdir):
    postprocessing.suppmat(data, outdir)

if __name__ == '__main__':
    cli()
