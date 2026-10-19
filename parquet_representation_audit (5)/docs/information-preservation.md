# What is preserved, and what is actually audited?

The audit does **not** require engineered features to become disjoint labeled clusters. Keep each numeric feature numeric, each nominal label categorical, each disjoint clustering as its existing one-label-per-row column, and each genuine tag set as a multilabel column. Several clusterings can share the same rows. You do not need separate progressively larger labeled datasets.

There are three separate questions: whether the source was preserved, whether a transformation changes meaning, and how much of that source the analysis examines. A valid schema answers only part of the first question.

## Direct mappings and their limits

| Source information | Direct mapping | What is retained | What is not established |
| --- | --- | --- | --- |
| Engineered real measurement | `numeric` | The scalar column, without binning | All relationships with other columns |
| Nonnegative count | `count` | The observed integer values in the source | Exact arbitrary-precision arithmetic in downstream numeric metrics |
| Nominal category | `categorical` | Category identity; internal class codes are reversible | A numeric distance or ordering among categories |
| One disjoint clustering | `cluster_label` | Exactly one cluster ID per source row; zero can be a real cluster | Within-cluster distinctions that were already absent from the labels |
| Several clusterings | Separate `cluster_label` columns | Each complete assignment aligned to the same rows | Joint agreement among all clusterings as one target |
| Genuine tag set | `multilabel` | Presence/absence membership | Sequence, duplicate counts, confidence weights or tag positions |
| Ordered category | `ordinal` | Declared category identity and its order in the schema | Default classification scores do not exploit ordinal spacing; a separately declared numeric derivation is a different target |
| Free text, timestamps | Source-only `text` / `datetime`, optionally explicit derived targets | Source values remain available in the input | The audit does not evaluate general text semantics or every temporal relationship |

These are semantic mappings, not a universal serialization format for arbitrary nested or structured information. Numeric analysis uses floating-point arithmetic, generally float64: integers beyond its exact range (2^53), arbitrary-precision decimals and similar values need a deliberate measurement design. Do not claim that an integer becoming a float is always lossless. Exact IDs use the separately typed identity path and should never be converted through floats.

A list of `[a, a, b]` is equivalent to `[a, b]` only if it is truly a **set** of tags. If repetition, order, weights, missing entries within a sequence or positions are meaningful, forcing it into `multilabel` loses information. Keep the original column and design explicitly named targets that answer the intended question. General sequence/vector-valued targets are not implemented; separate scalar columns preserve those scalar values but do not supply a general joint-distribution test.

## Transformations that remove distinctions

- **Binning, rounding, collapsing categories:** non-invertible unless the original values are also retained. Never required just to run a numeric feature.
- **One-hot encoding:** reversible for a complete, well-defined vocabulary with preserved missingness; incomplete/drop-first encodings require the mapping and conventions. Existing indicator columns may be audited separately as booleans. Turning a mutually exclusive cluster assignment into indicators is unnecessary for this project.
- **Unit normalization:** removes vector length. Cosine geometry uses direction even when vectors were not normalized explicitly. Raw Euclidean topology and magnitude-only probes examine complementary information; they do not undo the restriction of a cosine neighborhood result.
- **Ordinal-to-number mappings:** impose spacing assumptions. An order alone does not tell you that the distance from low to medium equals the distance from medium to high.
- **Text length or calendar parts:** new, narrower targets. Success on text length or year does not establish preservation of text meaning or the full timestamp.

The project reads the source and leaves it unchanged. It can nevertheless examine a reduced view of that source. Keeping the original file is necessary for preservation but does not make a derived analysis comprehensive.

## Deliberate analytical reductions

| Mechanism | What it limits | Where to inspect or adjust it |
| --- | --- | --- |
| Evaluation sampling | Rows used by neighbors, probes and pairwise metrics | `sample` and `limits`; manifest and sampled-row outputs |
| Multilabel vocabulary | Tags used for geometry/encoding | Feature `vocabulary`, `min_count`, `max_labels`; feature source/spec outputs and audited labels |
| Probe label cap | Tags for which binary probes are fitted | `probes.max_labels`; reported probe labels and supported-label counts |
| Source quantile reservoir | Precision of approximate full-source quantiles | `source.quantile_sample_size`; source-statistic scope |
| Source vocabulary budget | Maximum exact source label counts | `source.max_vocabulary`; exceeding it fails rather than silently collapsing labels |
| Finite k / radius grid | Neighborhood scales and filtration thresholds measured | Ordered `geometry.ks`; `topology.radii` / quantiles |
| Topology sampling and simplex budgets | Which components/loops can be observed | `topology.sample_size`, repetitions, H1 subset size and budgets; exact sampled IDs/status files |
| Independent target probes | Recoverability of each declared target | They are not a test of the complete joint distribution of all source features |
| Limited model families and fits | Functions that the probes can learn with their training data | Configured probes, fixed split, training-only linear tuning, held-out support |
| Report summaries and plotting caps | Details visible at a glance | Raw CSV/JSON outputs retain computed measurements; barcode plots show only a bounded number of intervals |

Tag vocabulary selection and row sampling are different. A full-source prevalence summary may include a tag that was not probed or represented in the bounded geometry analysis. A strong average can hide a weak rare label. Read the exact audited label list and support counts alongside scores.

Missing annotations and invalid embedding exclusions also change the evaluated population. All compared embeddings share a common valid cohort, but results on that cohort need not generalize to missing/invalid rows. Treating unknown tags as absent would add false information; the schema keeps null distinct from a known empty set.

## What a strong probe result means

A held-out result shows that a particular model could recover a particular declared target to the observed accuracy under that split. XGBoost can test nonlinear functions that a linear probe misses. It does not certify that every source distinction is present, establish causation, or prove that failure means information is absent. Finite training data, regularization, noise and model limitations can all affect the score.

Separate good results for several columns do not exhaust their interactions. Conversely, an embedding can preserve a relationship among source columns while failing to recover each column alone. The report is evidence about selected questions, not a lossless reconstruction certificate.

## Practical conclusion for an engineered feature table

Map the existing scalar columns directly. Preserve raw values and stable IDs. Keep your original disjoint cluster columns exactly as they are. Do not manufacture a clustering or bin every measurement to satisfy this audit. Declare only genuine tag sets as multilabel. Choose the row/vocabulary/probe scope deliberately, and name any narrower derived feature explicitly.

If the information you care about is a structured sequence, weighted tags, uncertain memberships or a joint target, the current audit may not evaluate it faithfully without additional methods. That is a limitation of the questions implemented here, not a reason to silently force the data into a disjoint-cluster format.
