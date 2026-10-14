# Replication and scale measurements

Measured on the execution host, Linux, Python 3.12; 48 dimensions, four resamples, one BLAS thread. Peak RSS is Linux ru_maxrss of each fresh process (includes Python and imported libraries). Timings cover map construction, not the validation suite or HTML export. These are descriptive timings, not scaling guarantees.

| Rows | Build seconds | Peak RSS MiB |
|---:|---:|---:|
| 500 | 0.159 | 124.9 |
| 1500 | 0.513 | 132.3 |
| 5000 | 2.674 | 159.8 |

## Cross-seed planted mechanism diagnostics

No training labels are supplied to the map. The same predeclared contrasts are evaluated for three independently generated corpora. These are synthetic rank-separation checks, not semantic accuracy or human task success.

| Contrast | Seed 7 AUC | Seed 42 AUC | Seed 101 AUC |
|---|---:|---:|---:|
| High-dimensional patch vs basins | 0.975 | 0.964 | 0.979 |
| Redundant patch vs basins | 1.000 | 1.000 | 1.000 |
| Connector vs basins | 0.938 | 0.951 | 0.789 |
| Outliers vs basins | 1.000 | 1.000 | 1.000 |

The exact neighbor backend still requires quadratic corpus work. A small-dimensional timing cannot establish feasibility at 8 million rows or 1024 dimensions. Reference-subset maps change the corpus-relative object being measured; an ANN replacement must report recall, topology changes, and profile stability.
