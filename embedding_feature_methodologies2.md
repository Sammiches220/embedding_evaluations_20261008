# Understanding feature fidelity in embeddings

*Meaning, interpretation, and explicit full-population feasibility at eight million rows.*

An **embedding** represents an item as a numeric vector. A **feature** is something you know about that item: artist, rating, publication year, popularity, or a metadata-derived category. These analyses ask three different questions:

1. **Geometry:** Are items with similar features close together?
2. **Recoverability:** Can a model predict the feature from the vectors on unseen items?
3. **Prediction distortion:** Do inferred labels preserve the true feature structure, or merely make the map look cleaner?

These are different questions; none establishes causality or business value.

## 1. Source statistics: establish the reference

| Measure | Meaning and interpretation |
| --- | --- |
| Coverage | Fraction of items with usable observed values. High coverage helps, but does not guarantee representative annotation. |
| Label count and prevalence | How many categories exist, and how frequently each occurs. These establish the denominator for enrichment. |
| Largest-label share | Dominance of the most common category. A large share can hide poor minority-category performance. |
| Entropy | Diversity of label occurrences: $H=-\sum_c p_c\log p_c$. Greater entropy means less concentration. |
| Normalized entropy | $H/\log C$, where $C$ counts nonzero labels; conventionally zero for one label. Near 1 means evenness **among those labels**, not completeness or correctness. |
| Numeric summaries | Mean, variance, quantiles, and distributions expose spread, skew, and extremes. |
| Tag cardinality/co-occurrence | Labels per item and counts of labels occurring together. These reveal multilabel structure. |
| Conditional statistics | Repeat summaries within time periods, domains, or other groups to expose differences hidden by an overall average. |

For tags, **item prevalence** can sum above 1 because items carry multiple labels. Entropy and largest-label share based on **occurrence shares** describe tag assignments, not a partition of posts.

Missing values are not automatically a category. Likewise, numeric category IDs have no implied order: label `6` need not be more similar to `5` than to `1`.

**Scale:** Stream counts, coverage, means, and variance. Exact quantiles/ranks require more work. Use sparse or selected-pair co-occurrence counts for large tag vocabularies.

Read: [SciPy entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html).

### Can these source statistics use all eight million rows?

**Yes—most source summaries are practical to calculate exactly over the full population on a normal workstation.** They do not require embeddings, a neighbor index, or eight million rows to be loaded into pandas. The important differences are how much state each metric retains and how many tag occurrences it processes.

Here $N=8$ million, $C$ is the number of distinct labels, and $T$ is the total number of tag assignments. These assessments assume batched, vectorized, or database processing; a Python operation per cell can be much slower.

| Statistic | Full-population feasibility at 8 million rows | Work and retained state |
| --- | --- | --- |
| Row count, coverage, missing share | **Routine; use all rows.** Respect feature-specific missing IDs. | One scan; a few counters per feature. |
| Categorical counts/prevalence | **Routine for cluster IDs, ratings, and flags.** High-cardinality strings need more dictionary memory. | $O(N)$ work per feature; $O(C)$ counters, not $O(N)$ stored rows. |
| Number of labels | **Practical exactly.** Count distinct keys in the frequency table. | Same counting pass; memory grows with distinct labels. |
| Largest-label share | **Routine exactly.** | Maximum count divided by the appropriate total; $O(C)$ after counting. |
| Entropy/normalized entropy | **Routine exactly.** Eight million items do not require eight million entropy terms. | Sum over the $C$ label probabilities after counting. |
| Mean, variance, standard deviation, min/max | **Routine exactly, up to floating-point precision.** | One scan; mergeable running summaries per numeric feature. |
| Fixed-bin histogram | **Routine exactly.** Choose bin edges first. | One scan; one counter per bin. |
| Median and numeric quantiles | **Usually practical exactly for a limited set of numeric columns.** Approximation is an option, not a requirement at this size. | One float64 column occupies **64 MB**; selection/sorting and copies require additional memory. Process columns separately. Many per-group quantiles are harder. |
| Tag frequencies and mean tags per item | **Practical, but potentially a major parsing job.** | Work depends on $T$, not just $N$. With 20 tags per row, there are **160 million assignments**. Count unique tags per post for item prevalence. |
| All tag-pair co-occurrences | **Potentially expensive; do not assume routine.** Selected tags/pairs are much easier. | Naively enumerating pairs with 20 tags per row produces **1.52 billion pair updates**. Distinct pair state can also grow dramatically. |
| Conditional summaries by year/domain | **Practical for a modest number of groups.** | Counters per observed group/label combination. Millions of groups can dominate memory. |
| Exact source-feature Spearman correlation | **Practical for a few numeric features; more expensive than simple summaries.** | Requires ranks/sorting. Two float64 value arrays plus two rank arrays occupy **256 MB**, before sort buffers. Many feature pairs add substantial work. |

For example, source counts across 46 cluster columns involve **368 million label values**, not billions of 1,024-dimensional distance calculations. Their frequency tables remain small when each column has only a handful of bins. Once those tables exist, prevalence, dominance, and entropy are cheap derived calculations; they do not need separate population scans.

Use a column-selective Parquet reader or an analytical database so you never read the embedding column for these summaries. [PyArrow supports streamed batches and selected columns](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html). [DuckDB provides exact and approximate aggregates](https://duckdb.org/docs/current/sql/functions/aggregates.html); its [larger-than-memory support has operator-specific limits](https://duckdb.org/docs/current/guides/performance/how_to_tune_workloads.html), so disk spilling does not guarantee every grouped quantile or large intermediate aggregate will fit.

## 2. Neighborhood purity: practical concentration

A neighborhood contains the $k$ nearest **other** items. Around an item carrying label $c$, purity is the fraction of neighbors also carrying $c$; per-label purity averages this fraction across eligible anchors carrying $c$.

For example, **purity 0.30** means that, around members of a segment, roughly three in ten evaluated neighbors share that segment. This is useful when users navigate related items or retrieve similar customers/content.

This measures neighborhoods, rather than entire clusters. For tags, count shared positive presence, not agreement on the absence of thousands of unrelated tags.

Exclude unannotated neighbors from the denominator and report how many remain. Two known neighbors provide much less evidence than 100; annotation gaps can still bias purity.

**Scale:** Once a graph exists, a categorical same-label comparison can run in roughly $O(Nk)$ operations. Per-label loops and multilabel comparisons add cost. Batch edges and avoid a dense item × neighbor × label array.

## 3. Lift: concentration relative to prevalence

$$
\operatorname{Lift}_k(c)=\frac{\operatorname{Purity}_k(c)}{P(c)}.
$$

Suppose a segment is **5% of the reference population**, but **30% of its members' neighbors** belong to it. Its purity is 0.30 and its lift is **6**.

| Lift | Interpretation |
| --- | --- |
| Around 1 | Approximately the concentration expected from prevalence alone. |
| Greater than 1 | Enrichment: the segment concentrates in the neighborhoods of its members. |
| Less than 1 | Depletion: its members tend to have neighbors from other segments. |

Read **purity, lift, prevalence, and support together**. Rare categories can achieve enormous lift with few matches; common categories can have high purity and modest lift. Lift alone does not establish financial benefit.

The usual baseline is prevalence. In a finite sample with $n_c$ members among $n$ known items, excluding the anchor gives chance probability $(n_c-1)/(n-1)$. Consequently, an exact permutation-null lift need not equal exactly 1.

**Scale:** Computing lift from purity is cheap. Getting trustworthy neighborhoods and the correct population prevalence is the difficult part.

### Micro versus macro

**Micro** weights evaluated positive anchor–label pairs equally; **macro** gives each audited label equal weight. For single-label data, an item contributes one pair. For multilabel data, it contributes once per audited label it carries.

Micro answers “how does the evaluated population behave?” Macro asks “how does a typical audited label behave?” Macro can reveal minority failures, but becomes sensitive to tiny labels. Micro lift is a weighted mean of label lifts, **not** aggregate purity divided by aggregate chance purity.

## 4. Multiple neighborhood sizes: local versus broader organization

Plot purity and lift across $k=5,10,25,50,100,\ldots$.

- Strong lift at small $k$ only suggests very local organization.
- Enrichment that survives larger $k$ suggests organization extending into broader neighborhoods.
- Falling purity at large $k$ can be natural: a small category has few members to retrieve.

Fixed $k$ implies larger radii in sparse regions. These curves are not persistent-homology barcodes.

**Scale:** Retrieve the maximum $k$ once and reuse prefixes. Larger $k$ costs more storage/work; changing dataset size also changes the scale represented by a fixed $k$.

## 5. Continuous features: smoothness and association

| Method | What the result means |
| --- | --- |
| Mean neighbor absolute difference | Average abs(y_i-y_j) across eligible neighbor pairs. Smaller differences suggest local smoothness; units remain the feature's units. |
| Local/global variance ratio | Average variance within neighborhoods divided by population variance. Smaller ratios indicate locally narrower values. At $k=1$, local variance is mechanically zero; use null comparisons. |
| Moran's $I$ | Graph autocorrelation. Positive values indicate neighboring deviations from the global mean tend to align; negative values indicate opposing deviations. Interpret relative to a graph-specific null. It is not universally bounded by $[-1,1]$. |
| Spearman correlation between source features | Rank association between two observed features; useful for identifying potential confounders such as popularity and age. It does not measure embedding geometry. |
| Distance–difference Spearman correlation | Association between embedding distance and absolute feature difference across sampled item pairs. Positive values suggest farther items differ more; local organization can exist without strong global correlation. |
| Distance correlation / centered-distance association | Detects relationships that simple linear correlation can miss. Stronger values suggest association, not necessarily useful neighborhood retrieval. |

Classical distance correlation uses Euclidean distances and has theoretical independence guarantees under suitable assumptions. A statistic built by double-centering **cosine-distance** matrices should be interpreted as a distance-association diagnostic; do not automatically transfer the classical independence theorem to it. Finite-sample estimates also need a reference distribution.

**Scale:** Local differences, variance, and Moran's $I$ are graph computations. Full pairwise distance statistics are generally quadratic: estimate them on repeated representative subsamples or sampled pairs. A 500-item estimate is not an eight-million-item exact result. Rank calculations may need external sorting or sampling.

Read: [Moran's I](https://pysal.org/esda/stable/user-guide/global_morans_i.html), [distance-correlation theory](https://dcor.readthedocs.io/en/latest/theory.html), [Spearman correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).

## 6. Probes: what can be predicted from the vectors?

A **probe** trains a model with embedding coordinates as inputs and an observed feature as its target, then evaluates on held-out items.

| Probe | Interpretation | Feasibility at 8 million rows |
| --- | --- | --- |
| Constant/majority baseline | Establishes what can be achieved without using embeddings. | Easy. |
| Linear/logistic regression or ridge | Tests recoverability through a simple global relationship. | Practical with representative training samples or incremental linear alternatives; standard batch solvers still have substantial costs. |
| k-nearest-neighbor prediction | Tests whether nearby training items have informative labels/values. | Exact large-scale querying is expensive; use a validated index or a smaller training reference. |
| Random forest | Tests a richer nonlinear relationship. | Full-data, high-dimensional training can be costly in time and memory. Start with sampled learning curves rather than assuming all rows are necessary. |

Linear success supports simple recoverability; nonlinear success may reveal more complex dependence. Model choice and training quality matter, so failure does not prove information is absent.

Separate duplicate/family groups across train and test. Time splits test future generalization; artist splits test unseen artists. Fit preprocessing and select models using permitted training information only.

Read: [scikit-learn incremental learning](https://scikit-learn.org/stable/computing/scaling_strategies.html).

### Reading prediction metrics

| Metric | Interpretation |
| --- | --- |
| Accuracy | Fraction of correct labels; can look good under severe imbalance. |
| Balanced accuracy | Mean recall across observed classes; exposes classes missed by a majority predictor. |
| Precision / recall / F1 | Precision measures correctness of positive predictions; recall measures recovery of true positives; F1 balances the two. Inspect per-label support. |
| AUROC | Ranking quality across positive/negative examples; 0.5 is random ranking for a nondegenerate binary task. High AUROC alone can coexist with poor practical precision for rare labels. |
| Average precision (AP) | Summarizes precision–recall over score thresholds. Prevalence matters; it is not trapezoidal PR-AUC. |
| Multilabel subset accuracy | Requires the entire predicted label set to match; stricter than per-label scores. |
| $R^2$ | Relative squared-error performance: 1 is perfect, 0 matches the test-set-mean baseline, negative is worse. Constant targets are degenerate. |
| MAE | Mean absolute prediction error in original units; smaller is better. |
| Prediction Spearman | Agreement of predicted and true rankings; high rank agreement does not guarantee calibrated magnitudes. |

AUROC/AP require scores, not hard category IDs. Missing or unsupported classes can make metrics undefined.

Read: [balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html), [F1 and multilabel averaging](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html), [AUROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), [AP](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [$R^2$](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html), [MAE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html).

## 7. Null experiments: is the structure more than chance?

Keep embeddings and their neighbor graph fixed, then change feature assignments:

- **Permutation:** shuffle known assignments, preserving frequencies exactly. For tags, shuffle whole tag sets to preserve co-occurrence.
- **Prevalence-matched random assignment:** resample observed assignments; preserves frequencies in expectation rather than exactly.
- **Stratified permutation:** shuffle only within selected groups, such as year. This asks about organization beyond that grouping; homogeneous groups provide little randomization.

Compare with the null mean, a descriptive z-score, and an empirical tail probability. A z-score does not automatically imply a Gaussian p-value. With $B$ randomizations, the plus-one-corrected minimum p-value is $1/(B+1)$.

Adjust for multiple tests across features/scales. A q-value is not an individual finding's falsehood probability. Benjamini–Hochberg has dependence assumptions; arbitrary dependence can require a more conservative adjustment.

**Scale:** Reuse the graph; do not rebuild it for every shuffle. Cost still multiplies by permutations × features × labels × edges. Use a fixed anchor sample and limited hypotheses. With population neighbors, shuffle reference-item labels as well as anchor labels.

Read: [permutation testing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html), [false discovery control](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).

## 8. Extrinsic predictions: accuracy and induced distortion

Keep observed and predicted features separate. Evaluate correctness only where both exist, using the same items for observed-versus-predicted geometry.

Then compare prevalence, dominant share, entropy, label count, and neighborhood statistics. **Jensen–Shannon divergence** compares normalized occurrence distributions: with base-2 logarithms it ranges from 0 (identical) to 1. It cannot establish item-level correctness: swapping labels among posts can preserve the distribution perfectly.

Deltas are predicted minus observed; ratios become unstable near zero. Lift changes combine changes in purity and prevalence: report both.

Embedding-derived predictions naturally look organized in that embedding: higher lift is not independent proof of correctness. External models can also share training data or biases.

**Scale:** Counts and error totals stream easily. Exact binary AUROC/AP is often practical for eight million scored items; many-label versions need more storage and sorting. See Section 11 for concrete costs. Predicted geometry reuses the graph. Align tables by stable item ID.

Read: [Jensen–Shannon distance versus divergence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html) — SciPy returns distance; square it to obtain divergence.

## 9. Missingness experiments: reconstruct what you deliberately hide

Hide increasing fractions of **known** labels, predict only from permitted information, and score against the concealed truth.

Random removal tests uniform annotation loss. Rare-label-first removal tests minority vulnerability; numeric-extreme-first removal tests tail vulnerability. A priority order can simulate old-items-first, particular domains, or sparse regions.

Plot hidden-item prediction performance alongside changes in reconstructed entropy, prevalence, purity, and lift. Rising reconstructed lift while hidden-label accuracy falls is a warning that the predictor is imposing structure. Nonrandom loss can be much harder than random loss at the same percentage.

**Scale:** Training repeats for each mode/fraction. Use representative cohorts and fewer initial settings, then expand important cases. These experiments cannot validate naturally missing labels whose truth is unknown, or fully reproduce unobserved annotation mechanisms.

## 10. PCA and plots: inspection aids

PCA projects vectors onto directions explaining large amounts of variance. A 2D view can reveal broad patterns, but relevant feature information can reside in discarded directions. Overlapping points in a plot do not prove overlap in the original space.

Plots summarize statistics rather than add independent evidence. Keep label support visible.

**Scale:** Plot samples or density aggregates, not eight million individual markers. Fit a projection on a sample or use an incremental method; transform in batches. Compute fidelity metrics in the original representation unless projected-space fidelity is specifically the question.

Read: [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

## 11. What eight million rows changes

### Can the other metrics use all eight million rows?

Separate **producing predictions/neighbors** from **scoring outputs that already exist**. A difficult model does not make its accuracy calculation difficult.

| Metric or computation | Concrete full-population assessment |
| --- | --- |
| Accuracy; balanced accuracy; precision/recall/F1; multilabel subset accuracy | **Practical in batches once predictions and truth exist.** Maintain per-label TP/FP/FN counts or sparse confusion counts; do not allocate a huge dense confusion matrix. Work is proportional to rows or tag assignments. |
| MAE and $R^2$ | **Routine in batches.** Accumulate error sums and stable true-value moments. Training the predictor is a separate expense. |
| Exact binary AUROC/AP | **Often practical, but needs sorting rather than only counters.** Eight million float64 scores plus uint8 labels occupy **72 MB** before sorting buffers. Thousands of label-score columns change the problem; process selected labels separately or approximate score distributions. |
| Prediction Spearman correlation | **Practical for a small number of targets**, with ranking/sorting costs similar to source Spearman. |
| Predicted-vs-observed prevalence, entropy, dominant share, label count, Jensen–Shannon divergence | **Practical exactly.** Stream matched-item counts, then compare small label-distribution tables. Keep all-prediction prevalence separate from matched-truth accuracy. |
| Categorical purity/lift on an existing $k=100$ graph | **Feasible in batches, but a substantial job:** **800 million edges per feature**. Efficient label comparison and accumulation can handle this; dense per-label loops multiply the cost. Lift itself is cheap after purity is known. |
| Multilabel purity/lift | **More expensive and vocabulary-dependent.** Use sparse memberships or selected tags; a dense eight-million × 50,000 uint8 tag matrix alone would occupy **400 GB**. |
| Neighbor differences, local variance, Moran's $I$ | **Feasible graph reductions**, with approximately the same 800-million-edge workload per feature at $k=100$. Reuse neighbors and batch intermediate arrays. |
| Multiple-$k$ curves | **Feasible with graph reuse.** Prefix counts/sums can reduce repeated scans; extra scales need not rebuild the graph. |
| Permutation/null comparisons | **Expensive on the full graph.** Fifty permutations imply roughly **40 billion edge evaluations per feature** at $k=100$, before multilabel overhead. Prefer sampled anchors unless full-population randomization is justified. |
| Distance correlation over all item pairs | **Generally impractical with a straightforward quadratic implementation.** Use repeated subsamples. Sampled distance–difference correlations are feasible, but remain estimates. |
| Exact exhaustive neighbor-graph construction | **Generally impractical as a routine desktop audit.** Approximate indexing or sampled anchor queries against the population are the practical alternatives. |
| Linear probes; forests; kNN probes; missingness reconstructions | **Depends on model and repetition count, not only row count.** Scoring all available predictions is feasible; full-data training/search can remain costly. Use the model assessments in Section 6 and sampled learning curves. |

The achievable result is therefore often **exact source statistics + exact scoring of available predictions + approximate or sampled geometry**, rather than sampling everything. A full graph produced by approximate search gives an exact calculation *on that approximate graph*, not guaranteed exact-neighbor purity. Sampled anchors yield an estimate, even when source prevalence is exact.

Illustrative storage, assuming **1,024-dimensional float32 embeddings**:

| Object | Approximate size, before overhead |
| --- | --- |
| Eight million embeddings | 32.8 GB (30.5 GiB) |
| Full float32 distance matrix | 256 TB; 64 trillion entries |
| 100 neighbor IDs per item, int32 | 3.2 GB; 800 million directed edges |
| Same IDs, int64 | 6.4 GB |
| Optional float32 distances for those edges | Another 3.2 GB |

These are arithmetic estimates, not performance benchmarks. Index structures, copies, labels, and temporary buffers add memory. Exact exhaustive graph construction costs roughly $O(N^2D)$: smaller query batches reduce peak memory, **not the total comparisons**. GPU execution changes speed, not that quadratic scaling.

A practical design is:

1. Compute source counts over the full population in batches.
2. Build or reuse a suitable approximate-nearest-neighbor index. For cosine search, normalize nonzero vectors and use inner-product search where appropriate.
3. Sample representative **anchors**, but retrieve their neighbors from the **full population**. A graph built only within a 10,000-item sample measures different neighborhoods.
4. Validate approximate-search recall on an exact reference for selected queries, then check whether purity/lift conclusions remain stable.
5. Include rare groups deliberately; weight population aggregates to account for unequal anchor sampling. Train expensive probes on learning-curve samples and evaluate a fixed held-out cohort.
6. Reuse neighbors for geometry, predictions, and nulls; avoid dense all-tag matrices and full pairwise computations.

These strategies require suitable implementations; they do not make an ordinary in-memory pipeline automatically scale. Feasibility depends on dimensions, label count, $k$, hardware, and sampling design.

Read: [Faiss similarity search](https://faiss.ai/), [cosine search and normalization](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances), [out-of-core learning](https://scikit-learn.org/stable/computing/scaling_strategies.html).

## A practical reading order

Start with **coverage and prevalence**, then read **per-label purity/lift with support**. Add null comparisons and multiscale curves. Use held-out probes to test recoverability, then prediction and missingness comparisons to detect distortion. Check time/domain groups and annotation gaps before generalizing to the full population.
