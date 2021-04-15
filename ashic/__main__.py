import click
from ashic.commands import fitrealdata
from ashic.commands import fitraodata

# change --help to both --help and -h to display help message
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--inputfile', required=True, 
              type=click.Path(exists=True),
              help='Input file name.')
@click.option('-o', '--outputdir', required=True,
              type=click.Path(),
              help='Directory to save output files.')
# @click.option('--paramsfile', type=click.Path(exists=True))
@click.option('--model', default='ASHIC-ZIPM', 
              show_default=True,
              type=click.Choice(['ASHIC-ZIPM', 'ASHIC-PM']),
              help='Choose between the Poisson-multinomial (ASHIC-PM) or ' +
              'the zero-inflated Poisson-multinomial (ASHIC-ZIPM) model.')
# @click.option('--simulated', is_flag=True)
# @click.option('--beta', default=1.0, type=float)
@click.option('--diag', default=1, 
              show_default=True,
              type=int,
              help='Number of diagonals ignored in the contact matrix.')
@click.option('--max-iter', default=30,  
              show_default=True,
              type=int,
              help='Maximum iterations allowed for EM algorithm.')
@click.option('--tol', default=1e-4, 
              show_default=True, 
              type=float,
              help='Minimum relative difference between the last two iterations ' +
              'when EM algorithm converges.')
@click.option('--seed', default=0, 
              show_default=True,
              type=int,
              help='Seed of the random number generator ' +
              'initializing structures and gamma values.')
@click.option('--gamma-share', 
              type=int,
              help='Number of diagonals from the last share the same gamma value. ' +
              'If not provided, each diagonal will use different gamma value.')
@click.option('--init-gamma', help='TEXT file containing precomputed gamma values, ' +
              'or a single value.')
# @click.option('--init-gamma', type=click.Path(exists=True),
#               help='TEXT file containing precomputed gamma values. ' +
#               'Initial gamma values can be computed with command `ashic-utils fitgamma`. ' +
#               'If not provided, gamma values will be initialized randomly.')
@click.option('--init-x', default='MDS', 
              show_default=True,
              help="Method to initialize structure: 'random', 'MDS' or " +
              "a TEXT file containing precomputed structure.")
@click.option('--init-c', default='allele-certain',
              show_default=True,
              type=click.Choice(['allele-certain', 'mate-rescue']),
              help='Method to initialize complete matrix.')
@click.option('--ensemble', is_flag=True,
              show_default=True,
              help='Use ensemble mode with multiple structures.')
@click.option('--n-structure', type=int,
              help='Number of structures in ensemble mode.')
@click.option('--smooth', is_flag=True,
              show_default=True,
              help='Use 2D mean filter to smooth the initial complete matrix.')
@click.option('--h', type=int,
              help='Top/bottom/left/right padding of the mean filter window.')
# @click.option('--init-model', type=click.Path(exists=True),
#               help='Precomputed model file to initialize parameters.')
@click.option('--max-func', default=200, 
              show_default=True, 
              type=int,
              help='Maximum iterations allowed for each structure optimization by L-BFGS-B.')
@click.option('--separate/--no-separate', default=True, 
              show_default=True,
              help='Whether or not optimize each structure separately then combine.')
@click.option('--normalize', is_flag=True, 
              show_default=True,
              help='Incorporate ICE bias into model.')
# @click.option('--save/--no-save', default=True)
@click.option('--save-iter/--no-save-iter', default=False,
              show_default=True,
              help='Whether or not save model file from each iteration.')
# @click.argument('inputs', nargs=-1)
def cli(inputfile, outputdir, model, diag,
        max_iter, tol, seed, gamma_share, 
        init_gamma, init_x, 
        init_c, ensemble, n_structure,
        smooth, h,
        max_func, separate, normalize, save_iter):
    """ASHIC: Hierarchical Bayesian modeling of diploid chromatin contacts and structures.\n
    Example:
    ashic -i <INPUT> -o <OUTPUT>

    Refer to README.md for <INPUT> format detail and how to generate with command `ashic-data`."""
    if ensemble:
        return fitraodata.run_ashic(
            inputfile, outputdir, model_type=model,
            diag=diag, max_iter=max_iter, tol=tol, seed=seed,
            gamma_share=gamma_share, init_gamma=init_gamma, init_x=init_x,
            init_c=init_c, ensemble=ensemble, n_structure=n_structure,
            normalize=normalize, smooth=smooth, h=h, save_iter=save_iter,
            max_func=max_func, separate=separate
        )
    fitrealdata.run_ashic(inputfile, outputdir, model_type=model,
                          diag=diag, max_iter=max_iter, tol=tol, seed=seed, 
                          gamma_share=gamma_share, init_gamma=init_gamma, init_x=init_x,
                          normalize=normalize, save_iter=save_iter, 
                          max_func=max_func, separate=separate)


if __name__ == '__main__':
    cli()
