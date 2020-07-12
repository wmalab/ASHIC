import os
import click
import numpy as np

def convert(mat, ch, r, fh):
    n = mat.shape[0]
    start = 0
    for i in range(n):
        end = start + r
        fh.write("{}\t{}\t{}\t".format(ch, start, end) + '\t'.join(map(str, mat[i])) + "\n")
        start = end

@click.command()
@click.argument('matfile')
@click.argument('outdir')
@click.option('-r', '--resolution', type=int)
@click.option('-p', '--prefix', default='')
@click.option('-c', '--ch', default='chrX')
def cli(matfile, outdir, resolution, prefix, ch):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    contactfile = os.path.join(outdir, prefix + "_hicrep.txt")
    cfw = open(contactfile, 'w')
    if matfile.endswith('.npy'):
        mat = np.load(matfile)
    else:
        mat = np.loadtxt(matfile)
    mat[np.isnan(mat)] = 0
    convert(mat, ch, resolution, cfw)
    cfw.close()

if __name__ == '__main__':
    cli()