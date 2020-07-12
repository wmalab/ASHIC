"""
Convert .validpairs file into input data format.
step1:
- split the input file into chunks
- split into chunk00_chrX_ref-ref, chrX.ref-alt, chrX.alt-alt, chrX.ref-x, chrX.alt-x, chrX.x-x
step2:
- merge all chunks
step3:
- binning matrices
total ~360000000 valid pairs
"""

import os
import sys
import iced
import click
import shutil
import subprocess
import numpy as np
from glob import glob
import cPickle as pickle
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from itertools import combinations
from itertools import combinations_with_replacement
from allelichicem.utils import mask_diagonals


@click.group()
def cli():
	pass


def write_slurm_script(out, njobs, file_prefix):
	slurm_script = """\
#!/bin/bash -x
#SBATCH --job-name=split_chrom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --partition=wmalab
#SBATCH --array=0-{}
#SBATCH --output=array_%A_%a.out

python {} split2chrs {}$SLURM_ARRAY_TASK_ID {}
	""".format(njobs-1, os.path.abspath(__file__), 
			   file_prefix, os.path.join(out, "chroms"))

	with open(os.path.join(out, 'slurm_step1.sh'), 'w') as fh:
		fh.write(slurm_script)


@cli.command(name="split2chunks")
@click.option("--prefix")
@click.option("--nlines", default=20000000, type=int)
@click.argument("filename", type=click.Path(exists=True))
@click.argument("out")
def split_chunks(filename, out, prefix, nlines):
	if prefix is not None:
		prefix = os.path.join(out, prefix + "_chunk_")
	else:
		prefix = os.path.join(out, os.path.basename(filename) + "_chunk_")
	if not os.path.exists(out):
		os.makedirs(out)
	bash_cmd = "zcat {} | split -l {} -d - {}".format(filename, nlines, prefix)
	subprocess.check_call(bash_cmd, shell=True)
	# rename *_chunk_00 to *_chunk_0
	files = glob(prefix + "*")
	njobs = len(files)
	for file in files:
		dst, chunk = file.split("_chunk_")
		dst = dst + "_chunk_" + str(int(chunk))
		shutil.move(file, dst)
	write_slurm_script(out, njobs, prefix)


def get_chroms(genome):
	if genome == "mm10":
		return ['chr' + str(x) for x in range(1, 20)] + ['chrX']
	else:
		raise NotImplementedError("Not implemented for genome: " + genome)


def get_chrom_size(chrom, genome):
	if genome == "mm10":
		chrom_sizes = {
		"chr1":	195471971,
		"chr2":	182113224,
		"chrX":	171031299,
		"chr3":	160039680,
		"chr4":	156508116,
		"chr5":	151834684,
		"chr6":	149736546,
		"chr7":	145441459,
		"chr10":	130694993,
		"chr8":	129401213,
		"chr14":	124902244,
		"chr9":	124595110,
		"chr11":	122082543,
		"chr13":	120421639,
		"chr12":	120129022,
		"chr15":	104043685,
		"chr16":	98207768,
		"chr17":	94987271,
		"chr18":	90702639,
		"chr19":	61431566,
		}
		return chrom_sizes[chrom]
	else:
		raise NotImplementedError("Not implemented for genome: " + genome)


@cli.command(name="split2chrs")
@click.option("--mat", default="ref", 
			  help="string for maternal allele")
@click.option("--pat", default="alt", 
			  help="string for paternal allele")
@click.option("--amb", default="both-ref", 
			  help="string for ambiguous allele")
@click.option("--chr1", default=3, type=int, 
			  help="column index (start from 1) for first chr")
@click.option("--allele1", default=5, type=int,
			  help="column index for first allele")
@click.option("--chr2", default=8, type=int,
			  help="column index for second chr")
@click.option("--allele2", default=10, type=int,
			  help="column index for second allele")
@click.argument("filename", type=click.Path(exists=True))
@click.argument("out")
def split_chroms(filename, out, mat, pat, amb,
				 chr1, allele1, chr2, allele2):
	def get_suffix(a1, a2):
		order = {mat: 0, pat: 1, amb: 2}
		a1, a2 = sorted([a1, a2], key=lambda x: order[x])
		return a1 + "_" + a2
	if not os.path.exists(out):
		os.makedirs(out)
	try:
		files = {}
		with open(filename, 'r') as fr:
			for line in fr:
				entries = line.split('\t')
				chr1_, alle1 = entries[chr1-1], entries[allele1-1]
				chr2_, alle2 = entries[chr2-1], entries[allele2-1]
				if chr1_ != chr2_: continue
				suffix = get_suffix(alle1, alle2)
				if (chr1_, suffix) not in files:
					files[(chr1_, suffix)] = open(os.path.join(
						out, "{}_{}_{}".format(os.path.basename(filename), chr1_, suffix)
					), 'w')
				files[(chr1_, suffix)].write(line)
	finally:
		for fh in files.values():
			fh.close()
	

@cli.command(name="merge")
@click.option("--genome", default="mm10")
@click.option("--mat", default="ref", 
			  help="string for maternal allele")
@click.option("--pat", default="alt", 
			  help="string for paternal allele")
@click.option("--amb", default="both-ref", 
			  help="string for ambiguous allele")
@click.argument("src", type=click.Path(exists=True))
def merge(src, genome, mat, pat, amb):
	chroms = get_chroms(genome)
	for a1, a2 in combinations_with_replacement([mat, pat, amb], 2):
		for chrom in chroms:
			files = glob(os.path.join(src, "*_chunk_*_{}_{}_{}".format(chrom, a1, a2)))
			if len(files) < 1: continue
			merge_file = "{}_{}_{}_{}".format(files[0].split("_chunk_")[0], chrom, a1, a2)
			bash_cmd = ["cat"] + files + ['>', merge_file]
			bash_cmd = " ".join(bash_cmd)
			subprocess.check_call(bash_cmd, shell=True)
			# remove chunk files
			for file in files:
				os.remove(file)


@cli.command(name="binning")
@click.option("--res", default=100000, type=int)
@click.option("--chrom", required=True)
@click.option("--genome", default="mm10")
@click.option("--start", type=int)
@click.option("--end", type=int)
@click.option("--mat", default="ref")
@click.option("--pat", default="alt")
@click.option("--amb", default="both-ref")
@click.option("--c1", default=3, type=int)
@click.option("--p1", default=4, type=int)
@click.option("--a1", default=5, type=int)
@click.option("--c2", default=8, type=int)
@click.option("--p2", default=9, type=int)
@click.option("--a2", default=10, type=int)
@click.argument("filename", type=click.Path(exists=True))
@click.argument("out")
def binning(filename, out, res, chrom, genome,
			start, end,
			mat, pat, amb,
			c1, p1, a1, c2, p2, a2):
	def sort_pos(loc1, loc2):
		order = {mat: 0, pat: 1, amb: 2}
		loc1, loc2 = sorted([loc1, loc2], key=lambda x: order[x[1]])
		suffix = filename.split('_' + chrom + '_')[-1]
		assert "{}_{}".format(loc1[1], loc2[1]) == suffix, \
			"{}_{} are not equal to {}".format(loc1[1], loc2[1], suffix)
		return loc1[0], loc2[0]
	nbases = get_chrom_size(chrom, genome)
	if (start is None) or (end is None):
		nbins = int(int(nbases) / res) + 1
		s, e = 0, np.inf
	else:
		s, e = int(int(start) / res), int(int(end) / res)  # inclusive [s, e] 
		nbins = e - s + 1
	matrix = np.zeros((nbins, nbins))
	with open(filename, 'r') as fh:
		for line in fh:
			entries = line.split('\t')
			chr1, pos1, alle1 = entries[c1-1], entries[p1-1], entries[a1-1]
			chr2, pos2, alle2 = entries[c2-1], entries[p2-1], entries[a2-1]
			assert chr1 == chr2 == chrom, \
				"{} and {} are not equal to {}".format(chr1, chr2, chrom)
			pos1 = int(int(pos1) / res)
			pos2 = int(int(pos2) / res)
			if (not (s <= pos1 <= e)) or (not (s <= pos2 <= e)):  # skip contact not in [s, e]
				continue 
			pos1, pos2 = sort_pos((pos1, alle1), (pos2, alle2))
			matrix[pos1-s, pos2-s] += 1
			if (pos1 != pos2) and (alle1 == alle2):
				matrix[pos2-s, pos1-s] += 1
		if not os.path.exists(out): os.makedirs(out)
		if (start is None) or (end is None):
			np.save(os.path.join(out, "{}_{}.npy".format(
					os.path.basename(filename), res)), matrix)
		else:
			np.save(os.path.join(out, "{}_{}_{}_{}.npy".format(
					os.path.basename(filename), res, s, e)), matrix)


@cli.command("prepare")
@click.option("--diag", default=1, type=int)
@click.option("--perc", default=2, type=float)
@click.option("--mat", default="ref")
@click.option("--pat", default="alt")
@click.option("--amb", default="both-ref")
@click.argument("datadir", type=click.Path(exists=True))
@click.argument("out")
def prepare_data(datadir, out, diag, perc, mat, pat, amb):
	# read in matrix data
	f = {mat: 'a', pat: 'b', amb: 'x'}
	obs = {}
	prefix = None
	if not os.path.exists(out):
		os.makedirs(out)
	for a1, a2 in combinations_with_replacement([mat, pat, amb], 2):
		file = glob(os.path.join(datadir, "*_{}_{}_*.npy".format(a1, a2)))
		assert len(file) == 1, \
			"number of {}_{}.npy file is not 1: {}".format(a1, a2, len(file))
		if prefix is None:
			prefix = os.path.basename(file[0]).replace("_{}_{}".format(a1, a2), '')
			prefix = os.path.splitext(prefix)[0]
		obs[f[a1]+f[a2]] = np.load(file[0])
		if a1 != a2: 
			obs[f[a2]+f[a1]] = obs[f[a1]+f[a2]].T
	n = obs['aa'].shape[0]
	# mask: bool matrix with -k~+k diagonals as False
	mask = mask_diagonals(n, k=diag)
	for i in obs:
		obs[i][~mask] = 0
	# plot percentiles of marginal sum so we can choose a cutoff
	perc_aa, sum_aa = [], np.nansum(obs['aa'], axis=0)
	perc_bb, sum_bb = [], np.nansum(obs['bb'], axis=0)
	for q in range(1, 101):
		perc_aa.append(np.percentile(sum_aa, q))
		perc_bb.append(np.percentile(sum_bb, q))
	plt.plot(np.arange(1, 101), perc_aa, 
			 label="{}-{}: {}".format(mat, mat, perc_aa[int(perc-1)]))
	plt.plot(np.arange(1, 101), perc_bb, 
			 label="{}-{}: {}".format(pat, pat, perc_bb[int(perc-1)]))
	plt.axvline(x=perc, ls=":", color="red")
	plt.legend()
	plt.savefig(os.path.join(out, "percentiles.png"), bbox_inches="tight")
	# filter low mappable (< q-th percentile marginal sum) bins
	filtered = iced.filter.filter_low_counts(np.array(obs['aa']), 
											 sparsity=False,
											 percentage=np.true_divide(perc, 100.0))
	loci = np.nansum(filtered, axis=0) > 0
	filtered = iced.filter.filter_low_counts(np.array(obs['bb']), 
											 sparsity=False,
											 percentage=np.true_divide(perc, 100.0))
	loci = loci & (np.nansum(filtered, axis=0) > 0)
	# change low mappable bins to 0
	for i in obs:
		obs[i][~loci, :] = 0
		obs[i][:, ~loci] = 0
	# plot matrix heatmap
	for a1, a2 in combinations_with_replacement([mat, pat, amb], 2):
		plt.matshow(obs[f[a1]+f[a2]], norm=LogNorm(), cmap="OrRd")
		plt.savefig(os.path.join(out, "{}_{}.png".format(a1, a2)), bbox_inches="tight")
	params = {
		'n': n,
		'alpha_mat': -3.,
		'alpha_pat': -3.,
		'alpha_inter': -3.,
		'diag': diag,
		'loci': loci,
		'mask': mask
	}
	data = {
		'obs': obs,
		'params': params,
	}
	with open(os.path.join(out, "{}.pickle".format(prefix)), 'wb') as fh:
		pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)


def convert_to_ndarray(filename, chr_size, resolution, chrom):
	"""
	Convert interaction file to numpy matrix.
	chr_size: int. Total bin size of chromosome
	"""
	# init ref_alt, ref, alt, xx counts matrix
	ref_alt_counts = np.zeros((2*chr_size, 2*chr_size))
	#  |---ref---|---alt---|			
	#  ref       |         |
	#  |---------|---------|
	#  alt       |         |
	#  |---------|---------|
	ref_counts = np.zeros((chr_size, chr_size))
	#  |---X---|
	#  ref     |
	#  |-------|
	alt_counts = np.zeros((chr_size, chr_size))
	#  |---X---|
	#  alt     |
	#  |-------|
	xx_counts = np.zeros((chr_size, chr_size))
	#  |---X---|
	#  X       |
	#  |-------|
	# for ref_alt matrix, row/col index ordered in: chr1_ref, chr2_ref, ..., chrx_ref, chr1_alt, chr2_alt, ..., chrx_alt
	# the start index of ref genome is 0, the start index of alt genome is chr_size
	start_pos = {'ref': 0, 'alt': chr_size}
	with open(filename, 'r') as fh:
		## file format ##
		## chromosome1 midpoint1	chromosome2	midpoint2	source1	source2	count
		# chr5	92500000	chr10	51500000	ref	both-ref	4
		# ...
		for line in fh:
			l = line.strip().split('\t')
			if l[0] != chrom or l[2] != chrom:
				continue
			allele1 = l[4] # allele source of first fragment ref/alt/both-ref
			allele2 = l[5] # allele source of second fragment
			# compute the bin index of fragment1 (on haploid genome)
			row = int(int(l[1]) / resolution)
			# compute the bin index of fragment2 (on haploid genome)
			col = int(int(l[3]) / resolution)
			
			# for both allele certain fragments
			if allele1 != 'both-ref' and allele2 != 'both-ref':
				# if it's on alt genome, add chr_size to the index
				row += start_pos[allele1] 
				col += start_pos[allele2]
				# set contact frequency symmetrically
				ref_alt_counts[row, col] = ref_alt_counts[col, row] = int(l[6])
			elif allele1 != 'both-ref': # if fragment1 source is known, 2 is unknown
				if allele1 == 'ref':
					ref_counts[row, col] = int(l[6])
				elif allele1 == 'alt':
					alt_counts[row, col] = int(l[6])
			elif allele2 != 'both-ref': # if fragment2 source is known, 1 is unknown
				if allele2 == 'ref':
					ref_counts[col, row] = int(l[6])
				elif allele2 == 'alt':
					alt_counts[col, row] = int(l[6])
			else:
				xx_counts[row, col] = int(l[6])
				xx_counts[col, row] = int(l[6])
		return ref_alt_counts, ref_counts, alt_counts, xx_counts


def get_chr_size(filename, resolution, chrom):
	with open(filename, 'r') as fh:
		## file format ##
		## chromosome	bp
		# chr1	197195432
		# chr2	181748087
		# ...
		for line in fh:
			l = line.strip().split('\t')
			if l[0] == chrom:
				size = int(int(l[1]) / resolution) + 1 # bin size of chromosome
				return size


if __name__ == "__main__":
	cli()
# chrom = sys.argv[1]
# res = int(sys.argv[2])
# chr_size = get_chr_size('mm9.chrom', res, chrom)
# sp_chr, ref_chr, alt_chr, xx_chr = convert_to_ndarray(str(res), chr_size, res, chrom)
# outdir = "each_chr_matrix_{}/{}".format(res, chrom)
# if not os.path.exists(outdir):
# 	os.makedirs(outdir)
# np.save(outdir + '/' + 'sp_' + chrom + '.npy',sp_chr)
# np.save(outdir + '/' + 'ref_' + chrom + '.npy',ref_chr)
# np.save(outdir + '/' + 'alt_' + chrom + '.npy',alt_chr)
# np.save(outdir + '/' + 'xx_' + chrom + '.npy',xx_chr)
