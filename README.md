# ASHIC

Allele-specific modeling of diploid Hi-C data

## ASHIC with simulation data

### Generate simulation parameters

```bash
ashic paramshuman chrom outdir
```

### Generate simulation data pickle file

```bash
ashic simulatedata "path/to/params.json" outdir --seed seed
```

Simulated data will be stored at outdir as `simulate_data_${seed}.pickle`

### Run ASHIC on simulation data

```bash
ashic run "path/to/xxx.pickle" -o outdir --paramsfile "path/to/params.json"  --simulated --seed 0
```

default options: `--maxiter=30`, `--tol=1e-4`, `--inix=MDS`, `--max-func=200`
