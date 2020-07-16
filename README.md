# ASHIC

Allele-specific modeling of diploid Hi-C data

## Installation

Create a conda env with `ASHIC` installed:

```bash
conda env create -f environment.yml
```

## ASHIC with real data

```bash
ashic run examples/sample_data/GSM2863686_chrX_1000000.pickle -o output --seed 0 --separate
```

## ASHIC with simulation data

### Generate simulation parameters

```bash
ashic paramshuman X chrX
```

### Generate simulation data pickle file

```bash
ashic simulatedata "chrX/params.json" chrX/data --seed 0
```

Simulated data will be stored as `chrX/data/simulate_data_0.pickle`

### Run ASHIC on simulation data

```bash
ashic run "chrX/data/simulate_data_0.pickle" -o output --paramsfile "chrX/params.json"  --simulated --seed 0 --separate
```

default options: `--maxiter=30`, `--tol=1e-4`, `--inix=MDS`, `--max-func=200`
