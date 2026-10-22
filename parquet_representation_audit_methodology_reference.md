# Embedding–Feature Audit: A Plain-Language Methodology Reference

*A brief conceptual companion to `parquet_representation_audit`.*

## 1. What is the audit trying to establish?

An **embedding** represents an item as a vector of numbers, such as `[0.2, -0.7, …]`. The dimensions are learned coordinates, not necessarily individually interpretable properties. A separate dataset supplies **features** (scores, labels, tags, counts, etc.) for the same items.

The project asks three different questions:

1. **Organization:** Are similar feature values found near one another in embedding space?
2. **Recoverability:** Can a model learn to predict a feature from embeddings for items whose feature values it did not see during training?
3. **Robustness and comparison:** Does the pattern survive changes in scale, baselines, embedding models, vector normalization, and sampled structure?

A strong result is evidence *about that feature and that test*. It is **not** proof of causation, universal embedding quality, or lossless reconstruction of the source data.

## 2. The basic objects: rows, vectors, features, and neighborhoods

- **Item/row:** One record with a stable ID. The ID makes sure its feature values and embedding refer to the same object.
- **Embedding vector:** Its location in a potentially high-dimensional space. Two models may use different vector dimensions; their coordinates need not match.
- **Feature:** Something known about the item. Numeric values express amounts; categorical labels are mutually exclusive; multilabel tags can coexist; missing values are **unknown**, not automatically zero, false, or an empty tag set.
- **Distance:** A rule for deciding which embedding vectors are close. **Euclidean** measures straight-line separation; **cosine distance** compares their directions (ignoring positive differences in vector length).
- **k-nearest neighbors (kNN graph):** For each item, connect it to its `k` closest *other* items by embedding distance. The graph here is a measurement instrument; it does not learn from the features being tested.

**Example:** If two pictures have nearby embedding vectors, the audit checks whether they also tend to share a tag such as `forest` or have similar popularity scores. It does not assume the vectors' axes explicitly represent “forest” or “popularity.”

### Why test several neighborhood sizes (`k`)?

`k = 5` asks about an item's immediate vicinity; `k = 100` asks about a broader region. A feature can be locally smooth yet poorly organized at larger scales, or form broad regions with messy immediate neighborhoods. **A curve over `k` shows how organization changes with scale**, not how a model improves during training. The different `k` values reuse overlapping neighbors and are **not independent experiments**.

## 3. PCA scatterplots: what the dots actually show

**Principal component analysis (PCA)** finds directions through the vectors along which the evaluated embeddings vary most. It projects each high-dimensional vector onto the first two directions, producing an ordinary 2D scatterplot.

- **One dot = one item.** Its horizontal and vertical positions are its scores on principal components 1 and 2.
- **Axes are combinations of embedding dimensions**, not automatically human-interpretable traits.
- **Color/labels are annotations placed on top.** If high-score items cluster visually, that is an observed correspondence; PCA did not receive the score as a target to optimize.
- **The picture loses information.** Distances and apparent clusters in 2D can differ from the original high-dimensional geometry. PCA favors variance, not necessarily feature predictability or neighborhood faithfulness.

**What to say:** “This is a two-dimensional summary of how the embedding vectors vary. The colors help us see whether a particular feature is visibly organized, but the audit's neighborhood metrics are computed from the original vectors, not from the PCA picture.”

The project fits PCA separately for each embedding being compared. **Do not interpret matching PC1/PC2 positions across separate model plots as corresponding coordinates.**

## 4. Measuring feature organization without training a predictor

These methods *hold the embedding neighborhood graph fixed* and inspect the feature values along its connections.

| Method | Intuitive question | How to interpret it |
| --- | --- | --- |
| **Purity** (categorical/tags) | How many known neighbors share an anchor item's label? | Higher means more similar labels nearby. For tags, the project counts **shared presence**, not agreements that both items lack a tag. |
| **Prevalence** | How common is the label among known items? | A necessary reference: a common label can appear frequently in neighborhoods even by chance. |
| **Lift** | Is local label agreement greater than its base rate? | `lift = purity / reference prevalence`. A 5%-prevalent tag with 30% local purity has lift **6×**. `1×` means no enrichment over that reference. |
| **Micro vs. macro averaging** | Whose experience dominates the summary? | **Micro** weights label-positive anchor instances; **macro** averages audited labels equally. A strong micro score can conceal weak rare-label performance. |
| **Local normalized entropy** | How mixed are the labels in an item's neighborhood? | Low = neighbors are comparatively uniform; high = mixed. For exclusive categories, category entropy is normalized by the fixed audited vocabulary size. For multilabel tags, the project averages binary present/absent entropy across audited tags. |
| **Mean neighbor absolute difference** (numeric) | How far apart are linked numeric values? | Lower means neighbors have more similar numbers. It keeps the numeric feature's units. |
| **Local/global variance ratio** (numeric) | Is the local variation small relative to overall variation? | Less than 1 generally indicates smoother neighborhoods than the global feature spread. |
| **Moran's I** (numeric) | Do linked items have jointly high or jointly low values? | Positive suggests positive graph autocorrelation; negative suggests dissimilar values connect. It is not a causal effect. |
| **Distance–feature relationship** (numeric) | Are items that are farther apart in embedding space also more different in the feature? | Spearman compares pairwise distance rankings. The project also computes a centered-distance association on sampled pairs/items. For Euclidean distances, its distance-correlation statistic has the usual interpretation; the **cosine version must not be treated as a general independence test**. |

**Example:** A rare label can have only 30% purity yet impressive `6×` lift: neighbors share it much more often than random population prevalence suggests. Check the **actual number of positive anchors**, however—an extreme score for six examples is fragile.

**Important scope detail:** Unknown labels do not count as negatives. The project excludes unknown neighbors from the purity denominator, while the kNN search itself still uses the fixed evaluated embedding pool. Thus coverage affects how much usable evidence remains.

## 5. Feature “reconstruction” using prediction probes

A **probe** is a separately trained supervised model that takes an embedding as **input** and tries to predict a feature as **output**. The term *reconstruct* means **infer the declared target from the representation**; it does **not** mean decoding the embedding back into an original file or reversing the embedding model.

**A concrete numeric example:** Suppose 10,000 items each have a 512-number embedding and a known score.

1. Collect training pairs `(embedding vector, known score)`.
2. Fit a regression model that produces one numeric output for every vector. For a simple linear model, the predicted score has the form `w₁z₁ + w₂z₂ + … + w₅₁₂z₅₁₂ + b`. Training adjusts the weights so predictions approximately match the *training* scores.
3. Hide the score labels of held-out items from model fitting. Pass **only their embeddings** through the fitted model.
4. Compare those predictions with the held-out scores, which are used **only for evaluation**.

For a **categorical** target, the model instead predicts a class or class probabilities. For **multilabel** targets, this project fits separate yes/no predictors for selected tags. This is how the probe is told *what* to predict: its training examples supply both the input vectors and the correct output values.

### Which probes does the project offer?

| Probe | What it tests |
| --- | --- |
| **Dummy baseline** | How well a trivial rule works: predict the training mean for numbers, or the most frequent training class for labels. It does **not** use embeddings meaningfully. |
| **Linear** (Ridge / logistic regression) | Whether a relatively simple weighted combination of embedding coordinates exposes the feature. Standardization and optional regularization tuning are part of the configured pipeline. |
| **kNN probe** | Whether the target can be inferred from the labeled training vectors nearest to a new vector. This is a *predictive model*, distinct from simply measuring the full sample's neighborhood purity. |
| **Random forest** | Whether nonlinear decision rules involving embedding coordinates predict the feature. |
| **XGBoost** | Whether boosted decision trees can recover additional nonlinear relationships, subject to the selected settings and data size. |

A nonlinear probe succeeding when a linear probe fails suggests the target is **accessible to the more expressive fitted model**, not that it was absent from the embedding under the linear test. Conversely, poor performance can mean insufficient data, noisy labels, unsuitable models, or a weakly encoded signal; it is **not a proof of information absence**.

**Fairness of the experiment:** The project uses a held-out train/test split, the same feature-specific test items across configured probe families and paired embeddings, and group-aware splitting where configured. Linear hyperparameter searches, if requested, occur within training data. The main held-out test is not supposed to teach the probe its answers.

## 6. How to read probe performance scores

| Score | Meaning | Preferred direction |
| --- | --- | --- |
| **R²** (numeric) | Improvement in squared prediction error over predicting the test mean; `1` is perfect, `0` matches that reference, and **negative values are possible**. | Higher |
| **MAE** (numeric) | Average absolute error in the target's original units. | Lower |
| **Spearman correlation** (numeric) | Whether predicted and actual rankings move together, even if exact values differ. | Higher |
| **Accuracy / balanced accuracy** | Fraction of class predictions correct; balanced accuracy gives classes more equal influence. | Higher |
| **Precision / recall / F1** | Precision: predicted positives that are correct. Recall: real positives found. F1 balances both. **Macro F1** weights classes/tags equally; **micro F1** pools decisions. | Higher |
| **ROC AUC / average precision** | Whether probability scores rank true positives above negatives, and how well predicted positives are prioritized; availability depends on test support. | Higher |
| **Exact-set accuracy** (multilabel) | Proportion of items whose **entire predicted tag set** is correct. | Higher |

**Distribution diagnostics** ask a different question: Do predicted category/tag frequencies resemble the actual frequencies? The project includes prevalence differences, entropy differences and Jensen–Shannon divergence. **Matching population proportions does not mean assigning correct predictions to individual items.** For example, a classifier could predict exactly 30% “forest” items while identifying the wrong 30%.

Always read probe performance **against its dummy baseline**, with held-out size, class/tag support, and the selected evaluation target. Some metrics are undefined if the test cohort contains only one class or insufficient positives.

## 7. Shuffled-feature controls: is the geometry more organized than chance?

An impressive observed neighborhood score may arise just because a category is common. A **null comparison** deliberately breaks the assignment of feature values to embedding positions while keeping the graph fixed, then recalculates the same statistic many times.

- **Permutation null:** Shuffle feature values among known items; for multilabel items, shuffle their entire tag sets together.
- **Stratified null:** Shuffle only within a declared grouping, asking whether organization exceeds what that grouping alone might explain.
- **Prevalence-random null:** Resample observed feature rows with replacement, preserving their empirical distribution *in expectation*.

The **null mean** says what the score typically looks like after this disruption. A **z-score** describes the observed-minus-null difference in units of the null's measured standard deviation. An **empirical p-value** describes how often shuffled trials are at least as extreme in the relevant direction. The project can apply Benjamini–Hochberg correction across tested comparisons to control the false discovery rate under its assumptions.

**Interpretation:** “The feature's observed neighborhoods are more organized than shuffled annotations under this specified null.” Not: “The encoder causally learned this feature,” nor “all spatial patterns are significant.” Few shuffle repetitions provide coarse p-value resolution; multiple `k` values are correlated.

## 8. Does embedding direction or vector length carry the signal?

A vector contains **direction** (where it points) and **norm/magnitude** (its length). These need not mean the same thing.

The optional normalization experiment runs **the same held-out probes on matched rows** using:

- **Raw standardized:** Original vectors, with per-coordinate standardization in the applicable probe pipeline.
- **Unit vectors:** Each vector divided by its own Euclidean norm; length information is removed.
- **Unit vectors + standardized coordinates:** Normalize length first, then standardize coordinates for the probe.
- **Norm only:** Give the predictor just one number per item: the original vector length.

If unit-vector probes still predict a feature, **direction contains recoverable signal**. If norm-only probes work, **length contains signal**. Differences between variants also reflect model choice and preprocessing, so they do not uniquely measure how much information *exists* in direction versus magnitude. For cosine neighborhoods, direction already controls proximity; raw Euclidean distance can reflect both direction and length.

## 9. Comparing two embedding models fairly

The project can align **Embedding A and B by item ID**, restrict evaluation to the same valid items, and build separate neighborhood graphs with the same distance convention and `k` values. Both models are judged against the **same source features**, and their probes share the same held-out split.

For neighborhood comparisons it can compute a **per-item/anchor score** in A and B and report a paired difference, `B − A`. That removes some noise from comparing different populations. The report can also bootstrap items or configured groups to show an uncertainty interval, **conditional on the fixed graph and sample**.

**Direction matters:** A positive difference is favorable for a higher-is-better score such as purity; a negative difference is favorable for a lower-is-better score such as numeric neighbor difference. Separately fitted PCA plots are not a substitute for these paired quantitative comparisons.

## 10. Topology: watching geometric structure appear across distance scales

Optional **topological analysis** takes a bounded sample of vectors and gradually increases a distance threshold (**radius**). At each radius, it connects items that are close enough in the selected distance space. This evolving family of connections is a **filtration**.

- **Connected components (`H₀`, or `β₀`):** How many disconnected groups remain? At tiny radii, most points are isolated. As the radius grows, groups merge. A **barcode** records how long a component remains separate before merging.
- **Loops (`H₁`):** Does the evolving geometric complex contain a hole-like cycle? The project's bounded Vietoris–Rips calculation includes filled triangles, so a simple three-edge triangle is **not** mistaken for a persistent hole. A long-lived interval means a loop-like feature persists across a range of tested radii, on that sample.
- **Component feature organization:** Within each connected component, how variable or mixed are the feature values? Compare a **component variance ratio** (numeric) or **component entropy** (labels/tags) against shuffled assignments. A positive *shuffled minus observed* gap suggests components are more feature-homogeneous than the null.

**Example:** A map may start as many islands, become a few continents, then merge into one connected landmass as the radius increases. Topology records those transitions rather than declaring which labels the islands ought to contain.

Here topology uses complete pairwise distances **within sampled rows**, not the PCA plot or the main nearest-neighbor graph. It can examine unit-chordal or raw-Euclidean space. `H₁` is deliberately much more tightly size-bounded than `H₀`; sampled persistence cannot establish the exact topology of the whole dataset. Repeated-sample ranges are descriptive, not automatically confidence intervals or significance tests.

## 11. Missing-label reconstruction experiments

The optional **missingness experiment** starts with *known* feature values, hides a chosen fraction, trains on the remaining known values, and predicts those deliberately hidden values from embeddings. By default the reconstruction demonstration uses an embedding-based kNN predictor; a supplied predictor can replace it.

It can hide labels at random or preferentially hide **rare** values/labels (and use an explicitly defined priority). It then measures both:

1. **Prediction on hidden truth:** How accurate was the inferred value?
2. **Geometry/distribution distortion:** If predicted values replace the hidden ones, how much do the resulting feature distributions and neighborhood statistics differ from the original known-truth analysis?

**Interpretation:** “If a specified amount of annotation disappeared in this controlled way, this prediction strategy would reconstruct it this well on the known cohort.” That does **not** establish accuracy on naturally missing annotations, whose missingness mechanism may be different. When the predictor itself uses embeddings, its geometric agreement is not independent validation of those embeddings.

## 12. Source populations, selections, and what is actually evaluated

The project can stream **source-level statistics** (counts, missingness, numeric quantiles, label frequencies and tag co-occurrence) from the full Parquet table. Computationally heavier operations—PCA, nearest neighbors, probes, pairwise relationships and topology—may operate on a **bounded, reproducible evaluation sample** instead.

An interactive region or subset in the report can have its observed feature composition summarized and its **stored held-out predictions rescored**. That does not magically refit the model, rebuild the graph using only that subset, or turn a selected group into an independent validation set. Selecting an attractive-looking region after viewing outcomes also introduces exploratory-selection bias.

The distinction between **exclusive categories**, **true multilabel sets**, **numerical targets**, and **missing values** matters throughout. A high average over common labels does not establish success for rare ones; a feature evaluated on 2,000 valid sampled items does not automatically describe 8 million source items.

## 13. A practical way to explain any result

Use this simple template:

> **Question:** What feature and population are being tested?  
> **Method:** Which embedding geometry, neighborhood size, probe, or filtration radius is used?  
> **Observation:** What does the result say numerically, with which direction meaning better?  
> **Reference:** Compared with what—overall prevalence, shuffled labels, dummy probe, or embedding B?  
> **Limits:** How many items/positive examples, what missingness, sampling, and model restrictions apply?

**Worked reading:** “At `k = 20`, items tagged `forest` have 30% of their known neighbors also tagged `forest`. Because only 5% of known items have the tag, that is **6× prevalence lift**. This suggests local organization of the tag in this embedding, although we still need to examine positive-anchor count, other neighborhood sizes, and shuffled-label results.”

**One-sentence takeaway:** **PCA shows a compressed picture; neighborhood statistics test geometric organization; probes test recoverability; nulls test whether organization exceeds a controlled baseline; normalization and topology probe the form and scale of that structure.** Each answers a different, complementary question.

---

### Where the corresponding methods live in this project

- `geometry.py`, `feature_overview.py`, `statistics.py`: neighborhoods, purity/lift, numeric geometry, entropy, null comparisons and predictive statistics.
- `modeling.py`: supervised probes, held-out splits, fitting and scoring.
- `normalization.py`: direction, magnitude and preprocessing variants.
- `comparison.py`: aligned A/B graph scores and paired bootstrap differences.
- `topology.py`: distance filtrations, component mergers, bounded loop persistence and feature organization by component.
- `experiments.py`: controlled hiding and reconstruction of known features.
- `dataframe_report.py`, `visuals.py`, `exploration.py`: orchestration, PCA overview, reporting and interactive cohort exploration.

*This reference describes the methodology of the attached implementation, not a guarantee about the behavior of all representation-learning systems.*
