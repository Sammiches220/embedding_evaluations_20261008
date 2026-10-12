# Interpretation and formulas

Every measurement states its source, evaluation and candidate-pool scope. Annotation unknowns are excluded from denominators; known absence stays an observation. There is no universal composite score across targets, units, pipelines or geometries.

## Source

Coverage is known rows / scoped rows. Categorical Shannon entropy is `H = -Σ p_l log(p_l)` (natural-log source exports), normalized by `log(L)` when L > 1. Largest label share measures imbalance. Tag occurrence entropy uses a label's fraction of all positive occurrences; item prevalence uses positives / known rows and can sum above one across tags. These are different denominators. The overview and topology use mean binary entropy over a fixed audited tag vocabulary, in bits. It differs from occurrence entropy. Numeric moments use float64 and mergeable centered second moments. Population variance divides by known row count. Quantiles are exact only when all known values fit the reservoir; capped source quantiles are explicitly approximate.

The report's evaluation source diversity uses the frozen run vocabulary. Full-source counts/prevalences are separate exports and do not silently replace a sample reference. Source relationships and tag co-occurrence are measured on the bounded evaluation sample, not claimed as full-file ranks/co-occurrences.

## Neighborhoods

For a known anchor i and label l present at i, purity is `P_il = Σ_{j in N_k(i), known(j)} 1[l present(j)] / #known(N_k(i))`. Unknown anchors and anchors with no known neighbors are ineligible. Known-empty tag anchors contribute support but no positive anchor-label pairs. Matching negatives never drives positive-tag agreement.

Per-label purity averages P_il over eligible anchors containing l. Per-label lift divides this average by the declared reference prevalence p_l. Micro purity and lift weight eligible positive anchor-label pairs; macro weights eligible audited labels equally. **Mean lift is the weighted mean of per-label ratios**, not a ratio of aggregate purity and aggregate chance. Per-label CSVs expose support, reference prevalence, effective neighbors and leave-one-out finite-pool chance. Prevalence baseline and fixed-graph permutation null answer different questions.

Local entropy is plug-in entropy of known neighbors, divided by a frozen categorical `log2(L)` or averaged binary entropy across audited tags. Small k has mechanical entropy bias. A curve's rise does not alone grade representation quality. k is a neighbor count, not a fixed physical radius.

Numeric edge absolute difference is `Σ eligible directed edges |y_i-y_j| / #eligible edges`, in the target's units. Local variance is the unweighted anchor mean of population variances among each anchor's known neighbors. At k=1 it is mechanically zero. The variance ratio divides by the fixed global variance of known evaluation rows. Constant targets have undefined ratios. Centered calculations avoid unnecessary cancellation for large offsets.

For the full known-row graph, Moran's I is `n / W × Σ directed known edges (y_i-mean)(y_j-mean) / Σ known rows (y_i-mean)^2`. I is graph-dependent and not universally constrained to [-1,1]. Selected anchors retain full-run centering/variance for a scoped exploratory contrast; that is not an independent regional global-I estimate.

## Distance relationships

Feature–feature Spearman ranks observed source measurements on pairwise-known sample rows; it suggests association/confounding, not embedding geometry. Distance–difference Spearman compares sampled item-pair embedding distance with absolute numeric feature difference. Pair sampling may repeat pairs, excludes self pairs, and records actual support.

Classical Euclidean distance correlation double-centers complete sampled Euclidean distance matrices A and B and computes `sqrt(max(0, mean(A*B)) / sqrt(mean(A*A)*mean(B*B)))`. It is a biased finite-sample estimator with a configured quadratic cap; no hypothesis-test independence claim is made from a finite observed value. Using cosine dissimilarity produces a **cosine centered-distance association**, named separately; general Euclidean independence guarantees are not transferred. Exact sampled row positions are recorded in the relationship JSON.

## Nulls

Fix graph and known mask, then permute **whole known feature rows**, including entire audited tag sets. Stratified permutations keep rows within a declared stratum. Unknown rows do not become known. Monte Carlo one-sided p is `(1 + #null results at least as extreme) / (1 + #estimable null repetitions)`. Lower absolute differences/variance and higher agreement/lift/Moran's I are the specified tails. Undefined observed/null metrics remain not estimable; effective repetition counts are exported. z scores are descriptive rather than guaranteed Gaussian.

BH correction is enabled only by request. The declared implementation family combines features, k values and modes **separately per metric**. Its adjusted q values and the family are recorded. Correlated scales are not independent corroborations; a configurable badge margin is not a significance threshold or confidence interval.

## Probes and supplied pipelines

Training-only dummy baselines share the same held-out rows with logistic/Ridge, optional kNN and forest. Standardization and CV tuning occur only in training data. Groups never cross outer train/test or inner CV folds; explicit leaked splits error. Defaults use a reproducible 25% test split when no explicit split exists. Vocabularies frozen from declared source support are disclosed; test support and per-label outcomes remain visible.

Categorical primary score is balanced accuracy (mean recall over observed test classes); secondary F1, accuracy and probability metrics are available. Multilabel probes fit independent binary targets and macro-average supported label results; unsupported training tags are explicitly skipped. Telemetry sums fit time/warnings across independent models and uses the maximum solver iteration count, rather than averaging a misleading total.

Numeric R² measures squared error against variance in held-out truth and can be negative; constant held-out truth has undefined R². MAE remains in original units. The overview numeric baseline gain uses relative MAE improvement, whereas displayed primary score is R². Targets with different units/difficulty cannot be ranked using raw MAE or color.

Normalization compares raw+training-standardized, unit direction, unit+training-standardized, and original norm-only+training-standardized inputs. Each feature's split is identical across variants and A/B. Unit direction preserves unit inputs; coordinate standardization generally leaves the sphere. Ridge uses alpha; logistic regression uses inverse-strength C. Fixed or training-only-tuned values, timings, warnings, test support and split signatures are exported. These describe a fit; they do not validate encoder-training stability. No encoder is retrained and no training objective is imposed.

Supplied predictions are evaluated only on matched known truth. Their distributions on missing truth are descriptive. Probability metrics require a real label-order contract. Geometry agreement does not independently certify accuracy, especially when the predictor uses the embedding. Artificially hidden known truth tests a configured recovery strategy; naturally missing truth does not provide measured recovery accuracy.

## Topology

Each repeated sample uses exact complete distances. A/B and raw/unit spaces use identical sampled IDs. Radii are explicit or pooled distance quantiles across models and repeats within each geometry. Filtrations are nested. Unit-chordal Euclidean is a directional metric; cosine dissimilarity itself need not be a genuine metric. Numeric barcode lifetimes change under distance reparameterization.

H0 records component mergers at actual pair distances, dominance at evaluated thresholds, and birth/death/censoring. H1 uses a bounded tested GF(2) Rips boundary reducer including edges **and triangles**; graph cycle rank alone is not H1. Alive intervals at the cutoff are right-censored. One H0 essential component follows the eventual connectivity of a finite complete metric; additional components at the cutoff are censored. H2 is not computed. H1 uses a separately recorded bounded subset, simplex cap and cycle-edge provenance. Empty H1, disabled H1 and a budget skip are different statuses.

Within-component categorical entropy/binary-tag entropy or numeric variance ratio is weighted by retained known rows. Full-run audited vocabulary/variance stays fixed. Whole known-row sample shuffles produce null-minus-observed organization gaps on the same geometry and masks. Singleton and fully merged extremes can mechanically give zero gap. Minimum component support, retained coverage and sample-specific trials are exposed.

Repeated-sample min/max ranges are descriptive variability, not confidence intervals. Positive organization gaps can reflect annotation/group/source artifacts. More loops, longer bars or more components are not representation quality scores. Small-sample topology does not establish whole-population topology or acquire ANN approximation guarantees.

## A/B and selection

A/B use identical identity order, annotations, k values, preprocessing and candidate pool. Main model differences use shared eligible anchors; paired purity weights positive anchor-label pairs. The conditional bootstrap samples anchors or whole related-item groups, with fixed graphs and saved annotations. It does not account for encoder retraining, ANN error or all neighborhood-induced dependence. Numeric paired-anchor difference weights anchors, unlike edge-weighted global absolute difference; the metric name records that distinction.

Region brushing changes anchors/source summaries and may rescore stored held-out predictions. It leaves candidate neighbors, references, training fits, normalization experiments and topology samples fixed. PCA is navigation, not the primary geometry. Exports retain exact selected IDs and displayed scope.
