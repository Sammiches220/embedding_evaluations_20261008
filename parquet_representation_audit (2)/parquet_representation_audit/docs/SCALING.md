# Scaling and explicit budgets

## What happens before fitting

The validator checks JSON structure, metadata semantics, physical Arrow schema, Parquet footer counts, asset headers, sample limits and approximate working costs before fitting. Run then validates every source identity and declared source feature through projected batches. It stores full-file uniqueness on temporary disk SQLite, selects reproducible evaluation positions from the footer row count, and gathers only the needed projected row-group batches. It never loads all vectors into a full RAM matrix first.

A Parquet row group containing sampled positions is decoded in bounded batches; decoding may still read unsampled vector cells in that batch. This is not a promise of O(sample_rows) disk I/O for embedded vectors. Source passes exclude embedding columns. Source-only gathering also excludes them.

External NPY matrices remain memory mapped. Their full ID arrays and lookup structures are population-sized and capped by `max_external_ids`. The matrix row gathers are batched; comparison dimensions may differ. A typical string lookup has Python/index overhead well beyond the raw NPY ID bytes. Raising the ID cap to 8 million needs several GB of realistic memory headroom. No benchmark or sublinear lookup-memory claim is made. Precompose external composite keys outside this release.

## Concrete example/default budgets

The runnable examples use at most 120 evaluation rows; project schemas accept deliberately budgeted larger runs. The examples explicitly set:

| Limit | Example value | What it bounds |
|---|---:|---|
| Evaluation rows | 5,000 | Retained vector/graph/fit cohort; examples actually sample 120 |
| Embedding working bytes | 512,000,000 | Conservative vector+normalization estimate |
| Exact pairs | 12,500,000 | About 5,000 complete-pool rows |
| Dense sampled tag bytes | 64,000,000 | Bounded sample × audited tag matrices |
| External IDs | 10,000,000 | Hard population ID-index row cap; separately budget actual memory |
| Source batch bytes | 128,000,000 | Decoded projected batch; lower batch_rows for large cells |
| Source batch cells | 1,000,000 | Rows × projected source columns |
| Source vocabulary | 10,000 | Counter entries per categorical/tag feature |
| Quantile reservoir | 1,000 | Uniform priority sample of known numeric values |
| Source features | 100 | Selected feature columns per run |
| Geometry work units | 200,000,000 | Coarse rows × max-k × targets × null work estimate |
| Distance relationship rows | 100 | Complete pair-distance correlation subset |
| Distance relationship pairs | 2,000 | Sampled distance–difference Spearman draws |
| Topology rows / repeats | 32 / 2 | Exact complete metric sample experiments |
| H1 rows / simplices | 12 / 5,000 | Small exact GF(2) reduction including triangles |
| Explorer rows / JSON bytes | 500 / 8,000,000 | Optional row snapshot; aggregate views survive disabling |

The inherited topology defaults cap H0 at 2,000 rows, H1 at 128 rows and 500,000 simplices, repetitions at 20 and permutations at 1,000. Explicit example settings are much smaller. Disable or reduce expensive experiments before raising caps. Failed or disabled analyses report their status; requested metrics/scales are never silently replaced.

## 8 million rows

Streamed counts, coverage and mergeable numeric moments are a reasonable design target. Their actual duration depends on storage, Parquet compression, Python conversion, tag vocabulary and disk identity indexing; **no 8-million-row benchmark was run**. Full ranks and exact quantiles are not streamed: full-source numeric quantiles use a labeled bounded reservoir, while Spearman/co-occurrence use a disclosed evaluation sample. A very large/high-cardinality target will hit vocabulary caps; explicitly preprocess or split features rather than accepting uncontrolled counters.

Exact neighborhood search costs roughly O(n²d) in the evaluation cohort. Reuse one graph per model across targets. Feature/null work scales with target count, k and repetitions. Dense tag matrices exist only within the explicit sample/audited-label cap; source tag counting uses sparse sets/counters. Production population ANN candidate pools and incremental million-row training are not implemented. Supplied graphs describe an explicitly bounded exported cohort.

Rips simplex growth is combinatorial. Full-population persistence at 8 million rows is outside this implementation. Use modest matched samples and keep sample/metric/cutoff dependence visible. A subset's persistent structures do not certify population topology.

## Source-only mode

This mode still validates stable source identity, unless explicit fingerprint-position identity is chosen. It streams the requested source statistics and saves a bounded evaluation sample for supplied predictions/probabilities. It does not fabricate vectors, graphs, probes, PCA or topology. Optional full-file source detail can be disabled per feature while keeping coverage. All results are unweighted.
