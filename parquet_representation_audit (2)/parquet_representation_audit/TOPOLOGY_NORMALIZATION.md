# Direction, magnitude and filtration (0.6.0)

This edition adds unit-normalization experiments and sampled topological
analysis to the feature-level report. All evaluated neighborhood k values and
filtration radii remain visible together. The feature checklist applies to the
new tables; no categorical-label or single-radius gate is required.

## Run on your Parquet

Inside your virtual environment, from the extracted project directory:

```bash
python -m pip install -e .
embedding-dataframe-audit \
  --dataframe ../data/posts-sample-populated.parquet \
  --config examples/enhanced_config.json \
  --cluster-metadata-dir ../data/clusters/metadata \
  --drop-invalid-embeddings \
  --out posts_sample_enhanced
```

Open `posts_sample_enhanced/report.html`. This config enables the existing
neighborhood audit, four normalization-probe pipelines, and sampled H0/H1 plus
feature filtrations. It retains the original vectors for the main analysis.
Use a fresh output folder. All measurements are computed from your real input;
the bundled demonstration is synthetic.

New CLI overrides are `--normalization-analysis`, `--topology` and
`--normalize-embeddings l2`. The first two enable their configured experiments.
The last changes the **main** analysis to unit vectors. Existing configs retain
the previous preprocessing and leave the new experiments off until enabled.

These are different operations: comparing normalized probes does not force all
main measurements onto the sphere. Main L2 preprocessing applies to both A/B
vectors after ID alignment and common valid-row filtering. Original vector norms
are saved before that transformation. Zero vectors are undefined for directional
analysis and require explicit exclusion, like nonfinite rows.

Cosine ranks should remain the same under positive row normalization, apart
from numerical ties. Euclidean geometry changes. The CLI rejects supplied
Euclidean graphs when main normalization is requested; rebuild from normalized
vectors and verify IDs/order. Cosine supplied graphs retain the caller's usual
search-fidelity responsibility.

## Four same-split linear pipelines

| Variant | Input and fitting | Question |
|---|---|---|
| `raw_standardized` | Original coordinates → training-fitted StandardScaler → linear model | What can the established pipeline recover? |
| `unit` | Row-wise unit vectors → linear model | What can direction alone recover while inputs remain on the sphere? |
| `unit_standardized` | Unit vectors → training-fitted StandardScaler → linear model | Does coordinate rescaling help access direction-only information? |
| `norm_only` | Original vector length as one coordinate → training-fitted scaler → linear model | Does magnitude carry recoverable target information? |

Each pipeline includes a dummy baseline. Logistic regression predicts categories
or independent binary tags; Ridge predicts numeric features. All variants and
A/B models share the feature-specific train/test rows, tag vocabulary and group
policy. No encoder is retrained. Scaling statistics are fitted on training rows.
Tag probes use the existing `max_probe_labels` cap; unsupported training tags
are excluded and test support is displayed.

Balanced accuracy and numeric R² appear across variants. Numeric cells also show
MAE; numeric color compares relative MAE improvement against the dummy baseline.
Categorical color uses balanced-accuracy gain. A change describes a tested pipeline,
not an inherent universal improvement in the embedding. Unit plus StandardScaler
usually leaves the sphere. Intercepts and removal of magnitude can change linear
separability; normalization cannot guarantee it.

Inspect the saved norm distribution before interpreting magnitude-only results.
If lengths are already nearly one, very small differences can reflect stored
floating-point roundoff rather than meaningful magnitude information.

Fit seconds, maximum solver iterations and convergence-warning counts are
reported. For tags, fit time and warnings sum across fitted binary targets, while
iterations use the maximum. They provide fit diagnostics, not a benchmark of
encoder training or proof that future training is stable. Fitting time excludes
input preprocessing and depends on hardware and cache state.

Configuration:

```json
{
  "embedding_normalization": "none",
  "normalization_analysis": {
    "enabled": true,
    "features": true,
    "variants": ["raw_standardized", "unit", "unit_standardized", "norm_only"],
    "max_rows": 10000,
    "regularization_grid": [1.0]
  }
}
```

A one-value grid fixes regularization. A multi-value grid, such as
`[0.01, 0.1, 1.0, 10.0]`, performs training-only three-fold CV (fewer folds when
necessary), independently per pipeline. Grid values are Ridge **alpha** for
regression and logistic **C** for classification; C is inverse penalty strength.
CV scores use negative MAE or balanced accuracy. The test set never selects the
parameter. Group folds are used when `group_column` is configured. Insufficient
training-fold class/group support causes an explicit variant skip; no leaking
fallback split is substituted. Per-tag chosen values are exported separately
inside `selected_regularization_by_label` JSON.

Set `features` to a feature list to limit fitting work or `false` to fit none.
`max_rows` is an explicit ceiling: oversized runs skip these experiments rather
than silently sample them. For large inputs, prepare a deliberate evaluation
sample. These supplemental scores describe the full experiment; browser row
brushing does not refit or rescore them. Feature selection still filters the view.

## Radius filtration and feature organization

The topology engine samples evaluated row positions uniformly with a recorded
seed. Every embedding and geometry uses the same sampled IDs. Each sample uses
complete pairwise distances, independently of the audit's supplied neighbor
graph. No projection or ANN approximation is substituted.

Two geometry choices are available:

- `unit_chordal`: Euclidean distance between unit vectors, measuring direction.
- `raw_euclidean`: Euclidean distance on vectors as loaded, retaining magnitude.

Unless explicit radii are given, thresholds are pooled pair-distance quantiles
across all models and repeated samples **within each geometry**. A/B therefore
use the same numeric radii. Radius zero is included. The raw and unit spaces have
different distance scales, so their barcode lengths are not directly interchangeable.

At each radius, edges with distance at most the radius are added. Union-find
records exact H0 component mergers, finite intervals, and component membership
at every evaluated threshold. Finite H0 deaths correspond to accepted
minimum-spanning-forest edge weights in this sample's threshold graph. One H0
component is essential; other survivors at a finite cutoff are right-censored.

For each feature, the engine measures:

- Known-row weighted component entropy, normalized by the fixed full-run category vocabulary.
- Mean binary component entropy across the fixed audited tag vocabulary.
- Known-row weighted within-component variance divided by full-run source variance for numeric fields.
- Known support, retained annotation share and the number of measured components.

Only components with at least `min_component_known` known annotations contribute;
retained coverage is explicitly exported. Unknown annotations can remain geometric
connecting points, while contributing no known labels to entropy/variance.

The null shuffles whole **audited** known label/tag rows inside each sample,
preserving counts, audited tag-set co-occurrence and the known mask. The default
is ordinary permutation. `null_mode: "stratified"` requires `strata_column` and
shuffles within those strata. Related-post/group dependence is not automatically
removed by ordinary permutation; choose conditioning appropriate to your question.

The main displayed **organization gap** is mean shuffled variation minus observed
variation. Positive values suggest more concentrated annotations within components
than that null expects. Singleton and fully merged extremes can yield zero
mechanically. A/B mode shows the change in this gap. These are descriptive
comparisons: colors are not p-values, and correlated radii are not independent
replications. Component entropy alongside H0 is a feature-aware scale diagnostic,
not itself a persistence diagram.

Repeated samples report mean results and per-sample min/max gap ranges. Those
ranges are not confidence intervals. If the sample includes the entire evaluation
cohort, the engine uses one actual sample and reports the requested repetition
count separately, avoiding false replication.

The scope remains global to the recorded samples. Browser brushing does not
recompute topology on a region. Doing that would change both the candidate pool
and the meaning of the filtration; use an explicit separate audit for that study.

## Bounded H1 with triangle filling

H1 uses a separately recorded uniform subset of the first H0 sample: 64 points
by default, configurable up to a hard limit of 128. It builds edges and triangles
through the largest evaluated radius and performs standard boundary-matrix
reduction over GF(2), using integer bitsets. Including triangles lets loops die
when they fill; this is not ordinary graph-cycle counting. H2 is not computed.

The calculation is exact for those sampled points and that cutoff. It is not
exact for the whole population. Size limits are explicit. Exceeding the simplex
budget skips H1 for that model/geometry while preserving H0 and feature summaries.
Right-censored H1 intervals have no observed death; finite-point full Rips
filtrations eventually fill loops, so a cutoff survivor is not an infinite semantic
feature.

The barcode export includes a representative **cycle at birth** as local H1 edge
pairs. Representatives are not unique or guaranteed shortest. Join endpoints to
`topology_sample_rows.csv` using `repeat=0` and `h1_position`. H0 representatives
refer to `sample_position`. This provides item-level provenance for investigation,
without automatically assigning semantic meaning to a loop.

Scientific PNGs show component counts at evaluated thresholds, largest component
share, and up to 50 longest H0/H1 intervals for the first sample. Complete intervals
remain in CSV. H0 and H1 can use different row counts; captions and status disclose
that distinction. Diagram distances, formal persistence confidence sets and
multiparameter persistence are not implemented.

## Topology configuration and limits

```json
{
  "topology": {
    "enabled": true,
    "features": true,
    "spaces": ["unit_chordal", "raw_euclidean"],
    "sample_size": 256,
    "repetitions": 2,
    "quantiles": [0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
    "radii": {},
    "permutations": 10,
    "null_mode": "permutation",
    "min_component_known": 1,
    "min_retained_share": 0.5,
    "h1": true,
    "h1_sample_size": 64,
    "max_simplices": 100000,
    "max_distance_bytes": 256000000
  }
}
```

`radii` can map a space to explicit nonnegative thresholds, for example
`{"unit_chordal": [0, 0.25, 0.5, 0.75, 1, 1.5, 2]}`. It overrides quantiles for
that space. Keep them fixed to compare separate audits; automatic pooled quantiles
can change when the cohort or participating models changes.

`sample_size` is capped at 2000; repetitions at 20. The memory guard estimates
stored distance matrices plus pooled distance values and can skip topology before
allocation. It is not a total process-memory guarantee. The simplex guard bounds
H1 expansion. Numerical known/support colors use `overview.min_known`, the
configured change margin and topology's `min_retained_share`. These are heuristics,
not power analyses. Ordinary topology permutations produce descriptive null means
and standard deviations; no multiple-testing claim is added.

At 8m rows, counts and moments can be streamed, but exact dense pair distances and
full Rips complexes remain unsuitable. This project still loads the audit dataset
in memory before sampling. The new sample cap does not make loading 8m × D vectors
cheap. A production streaming/ANN backend and an encoder-training pipeline remain
separate work. Normalization itself is row-local and could be streamed in such a
backend; it is not an 8m implementation here.

## New outputs

| File | Contents |
|---|---|
| `embedding_norms.csv` | Original loaded norms before main preprocessing |
| `normalization_probe_scores.csv` | Per-feature/model/variant scores, baseline, tuning and fit telemetry |
| `normalization_probe_predictions.csv` | Held-out predictions and variant provenance |
| `normalization_probe_splits.csv` | Exact train/test row positions for every variant |
| `topology_feature_summary.csv` | Mean feature measurements across samples and descriptive gap ranges |
| `topology_feature_trials.csv` | Individual sample/radius/null measurements |
| `topology_structure.csv` | H0 counts, component dominance and processed edge counts |
| `topology_barcodes.csv` | H0/H1 births, deaths, censoring and H1 cycle edges |
| `topology_merges.csv` | Accepted H0 merger edges and component sizes |
| `topology_sample_rows.csv` | Population positions, IDs and H0/H1 local-position mapping |
| `enhancement_measurements.json` | Report evidence, settings, skipped-analysis reasons and scopes |
| `enhancements/*.png` | Component curves and persistence barcode plots |

Keep the complete output folder for the offline report. Supplemental experiments
remain available as aggregate tables when the browser row snapshot exceeds its
limits. Unsupported or disabled analyses show their status instead of fabricated
scores. Original feature CSVs, readable metadata, neighborhood plots, missingness
experiments and A/B geometry remain available.

## Interpretation and references

Use persistent structure to form questions about useful information, then check
labels, known relationships, held-out probes or the actual task. A stable ring
could represent viewpoint or a sampling artifact. Different topology can coexist
with similar task performance; similar topology can coexist with very different
semantic label arrangements.

The longer conceptual guide is [Topology background](docs/TOPOLOGY_BACKGROUND.md).
Primary resources: [GUDHI Rips definitions](https://gudhi.inria.fr/python/latest/rips_complex_user.html),
[Ripser](https://ripser.scikit-tda.org/en/latest/reference/stubs/ripser.ripser.html),
[Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html),
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

Runtime dependencies are unchanged. Ripser is used only for optional independent
validation of the built-in small-sample reducer, not required for running this edition.
