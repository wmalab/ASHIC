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
  - t_mm.txt, t_mp.txt, t_pp.txt: diploid contact matrices in TEXT format (dimension: n * n)
  - structure.txt: predicted 3D structure (dimension: n * 3)