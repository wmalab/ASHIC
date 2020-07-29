# ASHIC

Allele-specific modeling of diploid Hi-C data

## Installation

Create a conda env with `ASHIC` installed:

```bash
conda env create -f environment.yml
```

## Usage

```
> ashic -h

Usage: ashic [OPTIONS]

  ASHIC: Hierarchical Bayesian modeling of diploid chromatin contacts and structures.

  Example: ashic -i <INPUT> -o <OUTPUT>

  Refer to README.md for <INPUT> format detail and how to generate with command `ashic-data`.

Options:
  -i, --inputfile PATH           Input file name.  [required]
  -o, --outputdir PATH           Directory to save output files.  [required]
  --model [ASHIC-ZIPM|ASHIC-PM]  Choose between the Poisson-multinomial
                                 (ASHIC-PM) or the zero-inflated Poisson-
                                 multinomial (ASHIC-ZIPM) model.  [default:
                                 ASHIC-ZIPM]
  --diag INTEGER                 Number of diagonals ignored in the contact
                                 matrix.  [default: 1]
  --max-iter INTEGER             Maximum iterations allowed for EM algorithm.
                                 [default: 30]
  --tol FLOAT                    Minimum relative difference between the last
                                 two iterations when EM algorithm converges.
                                 [default: 0.0001]
  --seed INTEGER                 Seed of the random number generator
                                 initializing structures and gamma values.
                                 [default: 0]
  --gamma-share INTEGER          Number of diagonals from the last share the
                                 same gamma value. If not provided, each
                                 diagonal will use different gamma value.
  --init-x TEXT                  Method to initialize structure: 'random',
                                 'MDS' or a TEXT file containing precomputed
                                 structure.  [default: MDS]
  --max-func INTEGER             Maximum iterations allowed for each structure
                                 optimization by L-BFGS-B.  [default: 200]
  --separate / --no-separate     Whether or not optimize each structure
                                 separately then combine.  [default: True]
  --normalize                    Incorporate ICE bias into model.  [default:
                                 False]
  --save-iter / --no-save-iter   Whether or not save model file from each
                                 iteration.  [default: False]
  -h, --help                     Show this message and exit.
```

### Example

Sample data can be found at `examples/sample_data/GSM2863686_chrX_1000000.pickle`. The data file can be generated with `ashic-data` command.

Run `ASHIC-ZIPM` model on the example data:
```bash
ashic -i examples/sample_data/GSM2863686_chrX_1000000.pickle -o output --model ASHIC-ZIPM
```

Run `ASHIC-PM` model on the example data:
```bash
ashic -i examples/sample_data/GSM2863686_chrX_1000000.pickle -o output --model ASHIC-PM
```

The `output` directory includes:

- model.json: ASHIC-ZIPM/ASHIC-PM model saved in JSON format
- structure_3d.html: interactive view of predicted 3D structure
- progress.txt: saved log-likelihood for each iteration
- log.json: log information about convergence
- matrices:
  - t_mm.txt (maternal), t_mp.txt (inter-homologous), t_pp.txt (paternal): diploid contact matrices in text format (shape: n * n)
  - structure.txt: predicted 3D structure in text format (shape: n * 3)

## Generate input data from mapped read pairs

We use the mapped read pairs file from Bonora et al. (2018) ([GSM2863686](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2863686)) as an example to show the steps to generate ASHIC input data.

The mapped read pairs file should be tab-delimited text format and contains at least the following columns: 
- chromosome of 1st read end
- genomic coordinate of 1st read end
- allele of 1st read end
- chromosome of 2nd read end
- genomic coordinate of 2nd read end
- allele of 2nd read end

The *allele* of each read end should be some tag representing maternal-specific, paternal-specific, and allele-ambiguous reads.
In the case of the example, we use 'ref' for maternal-specific reads, 'alt' for paternal-specific reads, and 'both-ref' for allele-ambiguous reads. 

The preprocessing step would be performed using `ashic-data` command.

First, we split the read pairs into allele-certain and allele-ambiguous read pairs per chromosome using `ashic-data split2chrs` command.

- allele-certain
  - both-end maternal-specific
  - both-end paternal-specific
  - one-end maternal-specific, one-end paternal-specific
- allele-ambiguous
  - one-end maternal-specific, one-end allele-ambiguous
  - one-end paternal-specific, one-end allele-ambiguous
  - both-end allele-ambiguous

```
ashic-data split2chrs --prefix=<PREFIX> <FILENAME> <OUTPUT>
```

Replace `<FILENAME>` with the read pairs file name, and `<OUTPUT>` with the destination directory to store split allele-certain and allele-ambiguous read pairs per chromosome.
Use `<PREFIX>` to specify the prefix of the output files.

If you use input file format different from Bonora et al. (2018), you need to specify the *allele* tags used, and the index (1-based) of each column as listed before. Please refer to the following detailed explanation of options to do so. The default option values are for Bonora et al. (2018) data. 

```
> ashic-data split2chrs -h

Usage: ashic-data split2chrs [OPTIONS] FILENAME OUTPUTDIR

  Split contacts into intra-chromosomal allele-certain and allele-ambiguous contacts.

  The input file will be split into the followings:
  Allele-certain contacts:
          <FILENAME>_<CHR>_<MAT>_<MAT>
          <FILENAME>_<CHR>_<PAT>_<PAT>
          <FILENAME>_<CHR>_<MAT>_<PAT>
  Allele-ambiguous contacts:
          <FILENAME>_<CHR>_<MAT>_<AMB>
          <FILENAME>_<CHR>_<PAT>_<AMB>
          <FILENAME>_<CHR>_<AMB>_<AMB>

Options:
  --prefix TEXT      Prefix of the output files. If not provided, the input
                     filename will be used.
  --mat TEXT         Allele flag of maternal-specific reads.  [default: ref]
  --pat TEXT         Allele flag of paternal-specific reads.  [default: alt]
  --amb TEXT         Allele flag of allele-ambiguous reads.  [default: both-ref]
  --chr1 INTEGER     Column index (1-based) of chromosome of the 1st end.
                     [default: 3]
  --allele1 INTEGER  Column index (1-based) of allele of the 1st end.
                     [default: 5]
  --chr2 INTEGER     Column index (1-based) of chromosome of the 2nd end.
                     [default: 8]
  --allele2 INTEGER  Column index (1-based) of allele of the 2nd end.
                     [default: 10]
  -h, --help         Show this message and exit.
```

Then, bin the split read pairs files of a chromosome into contact matrices at chosen resolution.

```
ashic-data binning --res=<RES> --chrom=<CHROM> <PREFIX> <OUTPUT>
```

Replace `<RES>` with the resolution in base pair, `<CHROM>` with chosen chromosome, `<OUTPUT>` with the destination directory to store binned '.npy' matrix files. `<PREFIX>` is the common prefix of the split read pairs files with *directory* included, e.g. `<PREFIX>_chrX_ref_ref` is the file containing both-end maternal-specific read pairs on chrX.

If you use input file format different from Bonora et al. (2018), you need to specify the *allele* tags used, and the index (1-based) of each column as listed before. If the reads were mapped use reference genome other than 'mm10', a two-column tab-delimited text file ('.chrom.sizes') containing each chromosome name and size in base pair should be provided.
Please refer to the following detailed explanation of options to do so. The default option values are for Bonora et al. (2018) data.

```
> ashic-data binning -h

Usage: ashic-data binning [OPTIONS] PREFIX OUTPUTDIR

  Bin the mapped read pairs of chromosome <CHROM> into contact matrices.

  The input files containing allele-certain and allele-ambiguous read pairs
  should have name in format of <PREFIX>_<CHROM>_<MAT>_<MAT> and etc. PREFIX
  need to include the PATH to the files as well.

Options:
  --res INTEGER    Resolution in base pair of the binned contact matrices.
                   [default: 100000]
  --chrom TEXT     Chromosome to generate the binned contact matrices.
                   [required]
  --genome TEXT    Genome reference. Built-in supports 'mm10'. Other genome
                   can be supported by providing a '**.chrom.sizes' file.
                   [default: mm10]
  --start INTEGER  If provided, instead of binning the whole chromosome, only
                   region after <start> (inclusive) will be binned.
  --end INTEGER    If provided, instead of binning the whole chromosome, only
                   region before <end> (inclusive) will be binned.
  --mat TEXT       Allele flag of maternal-specific reads.  [default: ref]
  --pat TEXT       Allele flag of paternal-specific reads.  [default: alt]
  --amb TEXT       Allele flag of allele-ambiguous reads.  [default: both-ref]
  --c1 INTEGER     Column index (1-based) of chromosome of the 1st end.
                   [default: 3]
  --p1 INTEGER     Column index (1-based) of coordinate of the 1st end.
                   [default: 4]
  --a1 INTEGER     Column index (1-based) of allele of the 1st end.  
                   [default: 5]
  --c2 INTEGER     Column index (1-based) of chromosome of the 2nd end.
                   [default: 8]
  --p2 INTEGER     Column index (1-based) of coordinate of the 2nd end.
                   [default: 9]
  --a2 INTEGER     Column index (1-based) of allele of the 2nd end.  
                   [default: 10]
  -h, --help       Show this message and exit.
```

Finally, we pack the binned matrices into pickle data with some essential parameters (length of chromosome, distance decay exponent, etc.) and filter low mappability loci.
The output pickle data can be used as input for ASHIC.

```
ashic-data pack --perc=<PERC> <INPUT> <OUTPUT>
```
Replace `<INPUT>` with the *directory* containing the binned '.npy' matrix files of a chromosome, and `<OUTPUT>` with the destination directory to store the pickle data. 
Use `<PERC>` to specify the percentage of rows or columns have the lowest contacts to filter out.

Specify the *allele* tags if different from the example.

```
> ashic-data pack -h

Usage: ashic-data pack [OPTIONS] DATADIR OUTPUTDIR

  Pack allele-certain and allele-ambiguous binned matrices in DATADIR into
  ASHIC input format. Filter and mask loci with low mappability in allele-
  certain intra-chromosomal contact matrices.

  DATADIR should contain .npy format matrices files with names in format of

  *_<MAT>_<MAT>_*.npy,
  *_<MAT>_<PAT>_*.npy,
  *_<PAT>_<PAT>_*.npy,
  *_<MAT>_<AMB>_*.npy,
  *_<PAT>_<AMB>_*.npy,
  *_<AMB>_<AMB>_*.npy

Options:
  --diag INTEGER  Number of diagonals ignored in the contact matrix.
                  [default: 1]
  --perc FLOAT    Percentage (%) of rows/columns that interact the least to
                  filter out. Range from 0 to 100.  [default: 2]
  --mat TEXT      Allele flag of maternal-specific reads.  [default: ref]
  --pat TEXT      Allele flag of paternal-specific reads.  [default: alt]
  --amb TEXT      Allele flag of allele-ambiguous reads.  [default: both-ref]
  -h, --help      Show this message and exit.
```