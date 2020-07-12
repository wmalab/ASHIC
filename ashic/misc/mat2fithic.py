"""
mat2fithic.py - convert numpy ndarray to fithic inputs
"""
import os
import gzip
import click
import numpy as np


def get_fragments(size, r):
    fragments = []
    for i in range(size):
        fragments.append(i * r + int(r / 2))
    return fragments


def marginalize(mat, fragments, ch, fw):
    """
    chrom 0 midpoint marinalcount mappable
    """
    np.fill_diagonal(mat, 0)
    n = mat.shape[0]
    margin = np.nansum(mat, axis=0)
    for i in range(n):
        mappable = 1 if margin[i] > 0 else 0
        fw.write("{ch}\t0\t{mid}\t{ct}\t{map}\n".format(
            ch=ch, mid=fragments[i], ct=int(margin[i]), map=mappable
        ))


def contact(mat, fragments, ch, fw):
    """
    chr1 mid1 chr2 mid2 contact
    """
    n = mat.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if mat[i, j] > 0:
                fw.write("{ch}\t{mid1}\t{ch}\t{mid2}\t{ct}\n".format(
                    ch=ch, mid1=fragments[i], mid2=fragments[j], ct=mat[i, j]
                ))


@click.command()
@click.argument('matfiles', nargs=-1)
@click.option('-r', '--resolution', type=int)
@click.option('-o', '--outdir', required=True)
@click.option('-p', '--prefix', default='')
def cli(matfiles, resolution, outdir, prefix):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fragmentfile = os.path.join(outdir, prefix + "_fragments.gz")
    contactfile = os.path.join(outdir, prefix + "_contacts.gz")
    ffw = gzip.open(fragmentfile, 'w')
    cfw = gzip.open(contactfile, 'w')
    ch = 1
    for matfile in matfiles:
        if matfile.endswith('.npy'):
            mat = np.load(matfile)
        else:
            mat = np.loadtxt(matfile)
        fragments = get_fragments(mat.shape[0], resolution)
        marginalize(mat, fragments, ch, ffw)
        contact(mat, fragments, ch, cfw)
        ch += 1
    ffw.close()
    cfw.close()


if __name__ == '__main__':
    cli()
