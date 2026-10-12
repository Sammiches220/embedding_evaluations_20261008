## BEGIN IMPLEMENTATION PROMPT

Implement a downloadable Python project that evaluates how well embedding
representations organize and expose information in **arbitrary Parquet datasets**.
Generalize the following evaluation workflow through explicit schema/configuration,
rather than through dataset-specific column names or hardcoded taxonomies.

If an existing project ZIP is attached, inspect it, reuse sound implementations,
and preserve useful existing functionality. Explain any compatibility changes.
If no source project is provided, implement from this specification. Deliver
working code, configuration schemas, synthetic examples, tests, documentation,
and a complete downloadable ZIP. Do not stop at a plan or scaffolding.

### 1. Purpose and engineering style

Users should be able to ask:

- Which features are organized locally, and over which neighborhood scales?
- How does feature diversity change across regions or connected components?
- Can a linear pipeline recover each feature on held-out items?
- What changes when vector magnitude is removed?
- Does embedding model B preserve a particular feature better than A?
- Which structures persist through a filtration, and what relevant annotations explain them?

Use plain functions, dictionaries, NumPy arrays, pandas DataFrames and sparse
matrices where appropriate. Keep data conversion, feature encoding, measurement,
model fitting and reporting understandable independently. Avoid introducing
abstract wrapper hierarchies or large dataclass systems. Use type hints,
docstrings and readable error messages. Install and run in a virtual environment.
A small dependency set and an offline report are preferred.

“Any Parquet” means schema-driven ingestion with explicit handling of unsupported
columns. It does not mean every column is automatically a suitable prediction
target. Do not fabricate embeddings, target labels or mappings when they are absent.

### 2. Separate physical types, column roles and feature kinds

Create versioned JSON Schemas and validate configuration before expensive work.
Maintain two documents: `dataset_schema.json` for dataset semantics and asset
contracts, and `audit_config.json` for analysis choices and resource limits.
Inspection may suggest assignments; explicit user declarations take precedence.

A physical type such as int64, string or list<float> does not determine semantics.
Every column must be accounted for by an explicit specification or an explicit
unmapped-column policy. Recognize these roles:

| Role | Meaning |
|---|---|
| `id` | Stable item identity, never a default evaluation target |
| `embedding` | A fixed-length numeric vector per row |
| `feature` | Observed information to describe or evaluate |
| `prediction` | Supplied predicted values for a declared target and pipeline |
| `probabilities` | Scores/probabilities with an explicit target label order |
| `group` | Related-item identity for leakage-safe evaluation splits |
| `split` | Explicit train/test assignment; define accepted values |
| `stratum` | Conditioning variable for configured permutations or source summaries |
| `display` | Names, links or other presentation-only values |
| `ignore` | Intentionally excluded, with an optional reason |

A column may also have explicit auxiliary uses, such as a numerical feature
that is a permutation stratum. Support this through declared auxiliary roles or
analysis references, without silently removing it from the feature inventory.
Sampling/observation weights must never be mistaken for ordinary target values;
if weighted analyses are unsupported, require an explicit ignore policy or reject
the requested weighting and label outputs as unweighted.

For `feature` columns, support these declared kinds:

| Kind | Semantics and default analysis |
|---|---|
| `cluster_label` | Unordered assignment IDs, even when integers; categorical evaluation |
| `categorical` | One unordered label per known row |
| `boolean` | Two known classes; false is a valid observation |
| `ordinal` | Declared ordered levels; categorical metrics plus explicitly configured order-aware analyses |
| `count` | Nonnegative event/item totals by default; numerical evaluation with count-aware validation and units |
| `numeric` | Finite scalar values with declared units; continuous evaluation |
| `multilabel` | A set of simultaneously present labels/tags per annotated row |
| `datetime` | Temporal information; source summaries and explicitly declared derived targets |
| `text` | Free text; coverage/length summaries or explicit preprocessing, not automatic tag splitting |

Internally these kinds can map to categorical, continuous and multilabel
measurement functions. Record that mapping. Arithmetic differences between
unordered cluster IDs are invalid. Counts are not cluster IDs just because both
use integers. Ordinal distances require a declared order and a justified numeric
mapping; no arbitrary lexical ordering.

Support primitive Arrow types, dictionary encoding and numeric list/fixed-size
list embeddings. Nested structs/maps, mixed values and unsupported containers
must have an explicit flattening/conversion rule, or be ignored/rejected with a
clear reason. Do not silently flatten nested values or execute strings as code.

### 3. Column specifications and readable labels

Each feature specification should allow:

- Role, kind, description, units and feature family.
- Explicit missing sentinels and null/empty-string policy.
- Permitted scalar conversion, validation ranges and invalid-value policy.
- Label metadata file and typed ID handling, where relevant.
- Multilabel encoding, delimiter/parser, case policy, vocabulary rules and support caps.
- Ordinal order or datetime timezone/derived-feature rules, when relevant.
- Whether the feature is a source-statistics target, geometric target or probe target.
- Provenance for derived features or externally constructed assignments.

Distinguish **unknown annotation** from **known absence**. For multilabel data,
null may mean unannotated, while an empty list may mean an annotated item with no
positive labels. Empty strings can have either meaning only when configured.
For booleans and counts, `False` and `0` are valid by default. Do not infer that
cluster label 0 is missing. Sentinel exclusions must be declared. Apply those rules before domain validation,
so a declared missing count sentinel is not mistaken for a valid negative count.
Avoid truthiness-based tests that classify zero or false as unknown.

Readable metadata must preserve computational IDs and allow names, criteria,
descriptions, source fields, reasoning and excluded labels. Duplicate display
names must not merge distinct IDs. Do not recompute assignments from descriptions
unless explicit executable binning rules were requested. A column called
“cluster” can be a semantic bin or an external partition; it is not automatically
a spatial region of the embedding.

A representative schema contract is:

```json
{
  "schema_version": 1,
  "status": "validated",
  "dataset": {
    "name": "example_items",
    "path": "../data/items.parquet",
    "unmapped_columns": "error"
  },
  "columns": {
    "item_key": {"role": "id", "canonicalization": "lossless_string"},
    "vector": {"role": "embedding", "format": "list", "expected_dimension": 768},
    "segment_code": {
      "role": "feature",
      "kind": "cluster_label",
      "id_type": "integer",
      "metadata_path": "metadata/segment_code.json",
      "missing_values": [-1],
      "description": "Externally assigned segment; IDs have no numeric order"
    },
    "category": {"role": "feature", "kind": "categorical", "missing_values": []},
    "event_total": {
      "role": "feature", "kind": "count", "units": "events",
      "missing_values": [], "negative_values": "error"
    },
    "measurement": {"role": "feature", "kind": "numeric", "units": "declared_units"},
    "active": {"role": "feature", "kind": "boolean", "missing_values": []},
    "severity": {
      "role": "feature", "kind": "ordinal",
      "order": ["low", "medium", "high"], "evaluate_as": "categorical"
    },
    "labels": {
      "role": "feature", "kind": "multilabel", "encoding": "list",
      "null_is_unknown": true, "empty_is_unknown": false,
      "min_count": 20, "max_labels": 100
    },
    "created_time": {
      "role": "feature", "kind": "datetime", "timezone": "UTC",
      "derive": [{"name": "creation_year", "operation": "year", "kind": "numeric"}]
    },
    "related_group": {"role": "group"},
    "evaluation_split": {"role": "split", "values": {"train": "train", "test": "test"}},
    "domain": {"role": "stratum"},
    "predicted_category": {"role": "prediction", "target": "category", "pipeline": "external_a"},
    "category_probabilities": {
      "role": "probabilities", "target": "category", "pipeline": "external_a",
      "encoding": "list", "label_order": ["a", "b", "c"]
    },
    "item_link": {"role": "display"},
    "misc_payload": {"role": "ignore", "reason": "Uninterpreted nested payload"}
  },
  "embeddings": [
    {"name": "encoder_a", "column": "vector", "main_normalization": "none"}
  ],
  "prediction_pipelines": {
    "external_a": {"uses_embedding": "unknown", "description": "Supplied predictor"}
  }
}
```

These names and values are illustrative. No column names, dimensions, delimiters,
missing IDs or namespaces may be hardcoded from this example.

Example label metadata:

```json
{
  "schema_version": 1,
  "name": "segment_code",
  "id_type": "integer",
  "origin": "external_assignment",
  "description": "Externally supplied grouping",
  "source_fields": [],
  "excluded_from_coverage": [-1],
  "labels": [
    {"id": -1, "name": "unknown", "criteria": "Assignment unavailable"},
    {"id": 0, "name": "Group A", "criteria": "Defined by the supplied assignment pipeline"},
    {"id": 1, "name": "Group B", "criteria": "Defined by the supplied assignment pipeline"}
  ]
}
```

If metadata and schema disagree about missing values, fail validation unless an
explicit documented override resolves it. Report unmapped observed IDs using raw
IDs; never invent names or collapse them to the fallback label.

### 4. Dataset, identity and asset alignment

Require stable unique nonmissing IDs for external joins and A/B comparisons.
Allow explicitly specified composite keys. Preserve leading zeros and exact
identifier values. Detect collisions introduced by canonicalization. Do not
convert large integer IDs through floating point. Do not silently deduplicate.

For source-only analysis of a file without IDs, a reproducible file-fingerprint
plus row-position key may be explicitly selected; document that it is not a
stable cross-file identity. Do not use it to assume alignment across exports.

Support embeddings inside Parquet and external numeric NPY matrices with an
explicit ID array. Gather external vectors by ID using memory mapping and
batched reads, rather than loading a 30 GB matrix unnecessarily. Reject ragged,
textual or nonnumeric vectors. Validate dimensionality and row alignment.
Second-model dimensions may differ; each model must be internally consistent.

Filter nonfinite embeddings and undefined zero directions only under an explicit
policy. Use the same common valid cohort for A/B and record excluded IDs/reasons.
Feature missingness usually changes a feature-specific known mask, not the whole
embedding cohort. Do not silently reindex supplied graphs after filtering.

Supplied neighbors must declare row positions or neighbor IDs, ordered distances,
metric, candidate-pool scope, self-exclusion and provenance. Validate duplicates,
range and ordering/alignment contracts. Supplied search fidelity remains the
caller's responsibility. For A/B, require comparable candidate pools and masks.

Support `source_only` mode when no embeddings are provided; label geometry,
probes and topology as unavailable. Never generate embeddings from preview text
or abbreviated arrays containing `...`.

Resolve each JSON path relative to its containing configuration file; document
CLI override paths relative to the working directory. Avoid hidden working-directory
assumptions. Never overwrite source Parquet. Reject stale nonempty output folders
unless an explicit safe replacement policy is selected.

### 5. Required measurements and correct scopes

Preserve the following capabilities, generalized by feature kind:

| Area | Required analyses |
|---|---|
| Source statistics | Known coverage, counts/prevalence, largest share, entropy, numeric moments/quantiles, selected tag co-occurrence and source feature relationships |
| Categorical/tag geometry | Per-label and aggregate micro/macro purity and lift, prevalence references, eligible-anchor counts and known-neighbor share across every configured k |
| Numeric geometry | Known-edge mean absolute difference, mean local variance, local/global variance ratio and graph-specific Moran's I |
| Distance relationships | Explicitly sampled distance–difference Spearman and capped Euclidean distance correlation or accurately named centered-distance association |
| Nulls | Fixed-graph label shuffles; optional strata; known masks and audited tag-set rows preserved; empirical p-values/corrections with declared testing family when requested |
| Probes | Training-only dummy baseline and linear logistic/Ridge probes; optional kNN/forest; held-out support and per-label results |
| External predictions | Matched known truth/prediction metrics, probability-based metrics where valid, distribution changes and clearly scoped geometric comparisons |
| Missingness experiments | Explicit artificial hiding and a replaceable recovery callback; no accuracy claims on naturally missing truth |
| A/B comparison | Identical IDs, annotation masks, scales and paired eligible anchors; scoped bootstrap intervals when requested |

Define mathematical formulas and weighting in code/docstrings. Micro positive-tag
agreement weights eligible positive anchor–label pairs; macro weights audited labels.
Tag agreement must not be dominated by matching negatives. Known-neighbor denominators
exclude unknown annotations while retaining the configured candidate pool.
Mean lift is an average of label-specific ratios, not automatically aggregate
purity divided by aggregate chance.

Count features retain their units; distinguish negative net measurements from
nonnegative counts through schema. Numeric absolute errors from different units
are not directly comparable. Treat constants and insufficient support as explicitly
not estimable where appropriate. Moran's I is graph-dependent and not universally
bounded by [-1, 1]. Do not describe every cosine-based centered-distance statistic
as classical Euclidean distance correlation with an independence guarantee.

Report source-population versus evaluation-sample scopes separately. Frozen
vocabularies and reference denominators must be shared across comparable subsets
and models. A full-Parquet prevalence can be an explicitly declared external
reference; do not silently mix it with sample-candidate-pool baselines.

### 6. Unit normalization and probe stability

Provide optional main L2 row normalization and four comparison pipelines:

1. Raw coordinates → training-fitted coordinate standardization → linear probe.
2. Unit-normalized vectors → linear probe, preserving spherical inputs.
3. Unit vectors → training-fitted coordinate standardization → linear probe.
4. Original vector norm as one input → standardized linear probe.

Keep feature-specific splits, groups, tag vocabularies and test rows identical
between variants and models. Preserve original norms before main preprocessing.
Use numerically safe norm calculations and explicit handling of zero/nonfinite
vectors. Do not claim magnitude removal necessarily improves separability.
StandardScaler after normalization generally leaves the sphere.

Offer fixed regularization and optional training-only tuning per pipeline.
Preserve groups in outer and inner folds, keep the test set untouched, and report
insufficient-fold failures. Distinguish Ridge alpha from logistic C. Record fit
seconds, solver iterations, selected parameters and convergence warnings, with
accurate aggregation for independent tag probes. These are fit diagnostics, not
proof of stable encoder training. No encoder retraining is required for post-hoc
normalization, and no encoder training objective should be falsely claimed.

### 7. Filtration and topology

Implement sampled, nested distance filtrations using exact complete distances
within explicitly bounded samples. Use identical sampled IDs across models and
spaces. Support raw Euclidean and unit-chordal geometry, explicit radii or shared
pooled distance-quantile thresholds, and reproducible repeated samples.

Required outputs:

- H0 component mergers, component dominance and birth/death/censoring records.
- Feature entropy/variance inside components at every evaluated radius.
- Whole audited known-row label/tag shuffles on the same geometry and masks.
- Null-minus-observed organization gaps, support and retained coverage.
- Descriptive repeated-sample variability, clearly distinguished from confidence intervals.
- Optional bounded H1 persistence that accounts for triangle filling, through a
  validated library or a tested small-sample GF(2) reducer.
- Recorded H1 subset IDs, resource limits, cutoff censoring and cycle provenance
  where the chosen implementation supports it.

H1 must not be ordinary graph-cycle counting. Do not infer semantic quality from
loop counts or long bars alone. A component-entropy curve accompanies H0 but is
not itself a persistence diagram. Small components and singleton extremes have
mechanical effects; compare with an appropriate null. Unordered category IDs
cannot form a meaningful numerical feature filtration by ID order.

Explain that metric choice, sampling and cutoff affect interpretation. Cosine
dissimilarity is not generally a genuine metric; unit-chordal Euclidean distance
is appropriate for directional metric filtrations. Numeric lifetimes change
under distance reparameterization. Do not assert whole-population topology from
a small sample or inherit formal approximation guarantees from an arbitrary ANN graph.

### 8. Human-facing report

**Reproduce the original project's human-interface audit report, generalized to
the configured dataset. Visual and interaction fidelity is an explicit requirement;
a different dashboard containing equivalent metrics is insufficient.** Use the
latest feature-first report with normalization and topology sections as the reference.
Retain its light theme, white cards, table-led exploration, scale tiles, colored
badges, short interpretations and expandable guidance. The generic column schema
changes the report's data and wording, not this design.

When the original project is attached, inspect these reference files before coding:

- `demo_human_interfaces_audit/report.html`, with its adjacent `assets/` directory.
- `embedding_feature_audit/visuals.py` for the page shell and detailed feature panels.
- `embedding_feature_audit/explorer_html.py` and `assets/explorer.css` for the main
  feature workspace; `assets/explorer.js` for controls and interpretation behavior.
- `embedding_feature_audit/enhancement_report.py` and `assets/enhancements.js` for
  normalization and filtration sections.

Prefer adapting the existing shell, CSS and reusable rendering functions through
a generic report-data adapter. Audit reused computation against this specification;
copying the appearance does not justify retaining dataset-specific assumptions.
If the reference is unavailable, implement the following concrete design contract
and disclose that direct visual comparison was unavailable.

#### 8.1. Page appearance and section order

Use system sans-serif text at 15px, dark slate `#17303f` on pale gray `#f5f7fa`.
Center the page in a main container of approximately 1250px maximum width with
30px padding. White cards have 24px padding, a 1px `#d9e2e9` border, 12px corners
and 20px vertical spacing. Use approximately 34px page headings and 26px section
headings, muted text `#586c7b` and links `#126084`. Inputs have restrained borders
and 7px corners. At narrow widths, reduce padding and allow wide tables to scroll
horizontally; preserve readable values and labels.

Keep this page order:

1. Configurable audit title and a brief subtitle describing geometry, source
   information and held-out prediction.
2. A compact introductory card defining purity/lift, reference population and
   interpretation limits, with expandable **Run settings and limitations**.
3. The prominent **Features at a glance** card described below.
4. A **Direction, magnitude and topology** card with normalization and filtration.
5. Expandable raw measurement/export links. An existing legacy single-scale table
   may stay folded when adapting the original; do not create one as the main view.
6. A **Find feature** search and selector leading to detailed feature panels.

Do not replace this layout with a sidebar application, a collection of disconnected
KPI cards, or a scatterplot as the primary report. Keep the overview useful when
row exploration is disabled for size or embedding availability.

#### 8.2. Features at a glance

Start with one sentence explaining that reading across a row connects source
diversity, neighborhood organization and held-out prediction. Then show the color
legend, scope/status text, and an expandable **Choose a subset of features** with
search, checkboxes, **Select all** and **Clear**. Apply this selection consistently
to the overview, normalization table and topology feature table.

Use broad controls for **Across-k measurement** and **Show**. Measurement choices
cover entropy/variance ratio, agreement, lift/autocorrelation, macro summaries and
known-neighbor support, with appropriate names for each feature kind. Display modes
are **Measured values**, **Selected rows minus all rows**, and **Embedding B minus A**
when available. Show every configured k together; no categorical-label selector or
single-k slider is required for the main exploration.

Place an export button and a pale blue `#edf3f8` scope/interpretation readout above
one wide table. Preserve these five columns:

| Column | Required content |
| --- | --- |
| Feature / source change | Readable feature name and kind, known count/coverage, source entropy or numeric moments, units and applicable source changes. |
| Neighborhoods across k | A wrapping strip of compact tiles, each with a small k label and bold measured value or signed delta; expose effective support and metric meaning. |
| Linear probe / baseline | Primary held-out score, training-only dummy baseline, baseline-relative gain, test support and expandable secondary metrics. |
| External pipeline | Matched prediction accuracy/error, relevant reference and support, or an explicit unavailable/skipped state. |
| Brief interpretation | One or two concise, number-backed sentences connecting the displayed evidence, followed by expandable full measurements and a link to feature details. |

Match the original's approximately 13px overview text, subtle row borders and
light header background `#eef3f7`. The table is approximately 1050px minimum width;
give the scale strip and interpretation columns extra space. Tiles use 6px corners,
a subtle border and two lines of text. Fade untrusted k tiles with a dashed border
and a textual/accessible warning; keep their measured values visible. Configurable
`trusted_ks`, minimum support and color margins must be reported, not silently
chosen as proof of reliability.

#### 8.3. Color and natural-language interpretation

Use the original badge palette and keep descriptive changes separate from quality
comparisons. Show signs, reference names and text as well as color:

| Meaning | Background | Text | Example legend wording |
| --- | --- | --- | --- |
| Better relative to a declared reference | `#ddf1e5` | `#165733` | Better relative to reference |
| Worse relative to a declared reference | `#fae2df` | `#8f3029` | Worse relative to reference |
| Descriptive increase/diversity | `#dcecf9` | `#184b75` | More diverse / increased |
| Descriptive decrease | `#e4e8fb` | `#354983` | Less diverse / decreased |
| Limited evidence | `#fff0cc` | `#75500f` | Limited evidence |
| Small change, unavailable or descriptive state | `#edf1f4` | `#435765` | Small change / descriptive |

Generate explanations deterministically from actual results and scope. For example,
an observed balanced accuracy of 0.74 against a dummy score of 0.50 can produce:
“This linear pipeline recovers the feature better than its baseline (+0.24 balanced
accuracy on 620 held-out rows).” These example numbers must never be inserted into
real results. Entropy explanations should describe mixing and its change across
scales; do not treat lower entropy as automatically better. Numeric interpretation
must retain units and explain the fixed reference used by a variance ratio.

Place an expandable **How to read intrinsic and extrinsic evidence** directly below
the overview, with intuitive sections on diversity, neighborhoods, probes, external
prediction, changes/confidence and scaling. Include a separate deeper explanation
for normalization and topology. More persistence or loops is not a quality grade;
targets have different difficulty and units. Avoid universal composite scores.

#### 8.4. Normalization and filtration cards

Within **Direction, magnitude and topology**, retain two initially expanded sections:

- **Unit-normalization comparisons**: an embedding A/B selector where applicable,
  scope text, and a table with **Feature**, **Raw + standardized**, **Unit direction**,
  **Unit + standardized**, **Magnitude only**, and **Brief interpretation**. Show
  scores and baselines, with fit time, convergence and test support accessible.
  Explain whether direction or magnitude appears useful for this tested pipeline.
- **Feature organization across a filtration**: broad geometry/model/measurement
  controls, a scope readout and a three-column table: **Feature / sample support**,
  **Every evaluated radius**, **Brief interpretation**. Show all radii as scale
  tiles, including observed/null comparisons and model deltas where supported.
  Include H0/H1 scientific plots, explicit H1 status/cutoff/sample limits, and links
  to feature summaries, trial tables, barcodes and exact sampled IDs.

Both sections must display informative disabled/skipped states when analyses are
off. Clearly state that row brushing does not recompute these global experiments.
Keep full scientific explanations and provenance expandable rather than crowding
the short overview sentences.

#### 8.5. Optional region selection and detailed evidence

Retain the original optional **Rows to summarize** control, **All rows** reset and
folded **Select a region in the embedding overview** PCA canvas for bounded snapshots.
Region selection recomputes source/anchor summaries and may rescore saved held-out
predictions; it must not silently refit probes or narrow the graph candidate pool.
PCA/UMAP is navigation; primary geometry uses the original configured representation.
With no bounded snapshot, show aggregate tables and a clear scope notice.

Detailed feature panels use the same white-card styling. Show **What these results
mean**, readable label names and criteria from metadata, plots in a responsive grid,
and folded source, geometry, null, probe, missingness and external-result tables.
Preview bounded table rows and link complete CSVs. Undefined values, low support,
audited vocabulary and evaluation population remain explicit throughout.

Export exact selected IDs, displayed values, settings and scope. Keep all assets
local and usable offline; safely escape untrusted names/labels and use no network
or model calls for routine explanations. Supply complete machine-readable tables.

#### 8.6. Report acceptance checks

Deliver a generated generic demo `report.html` with its local assets. Check the
same five-column overview, page order, palette, all-k tiles, normalization table,
all-radius topology table and expandable guidance against the original reference.
Test feature filtering, display-mode changes, export contents, unavailable states
and scoped interpretations; changing a control must update values and explanatory
text together. Confirm long names and mixed feature kinds remain readable, and
that optional data cannot break the report. When permitted, inspect desktop and
narrow-width rendering and provide comparison screenshots. Otherwise state the
visual-verification limitation and perform source/markup/interaction tests that
are available. Do not claim visual fidelity from metric tests alone.

### 9. Scale, defaults and resource contracts

Design source statistics to read selected Parquet columns in batches when practical.
Counts and moments can be streamed; full ranks, exact quantiles and large vocabularies
need explicit algorithms and budgets. Identify exact versus approximate outputs.
Use an explicitly recorded reservoir/group-aware sample for expensive geometry
or fitting. Different sampling strategies must disclose their estimands.

Reuse each neighbor graph across features. Cap quadratic distance calculations,
null repetitions, probe vocabularies, topology sample sizes and H1 simplex growth.
Use sparse tag representations/counters rather than an uncontrolled dense
population × tag matrix. Bound row snapshots by rows and bytes; aggregate
scorecards must remain available when row exploration is disabled.

State explicitly what can handle 8m rows: streamed source counts/moments are a
reasonable target; exact all-pairs graphs and full Rips persistence are not default
operations. If the first release only samples expensive analyses, ensure sampling
happens before loading every vector into RAM. Do not call an in-memory-only
loader “streaming.” Production ANN candidate pools may remain a separately
specified backend; do not fabricate support or benchmark guarantees.

Make default caps concrete in configuration and fail early with estimated cost
and a useful remedy when limits are exceeded. Keep sampling, graph scope, missing
policies, seed, versions, ID alignment and evaluated scales in a run manifest.
Do not silently substitute a different metric or dataset to finish a run.

### 10. CLI, APIs, deliverables and acceptance criteria

Provide a generic CLI, for example:

```bash
parquet-representation-audit inspect --parquet PATH --out dataset_inspection
parquet-representation-audit validate --schema dataset_schema.json --config audit_config.json
parquet-representation-audit run --schema dataset_schema.json --config audit_config.json --out audit_output
```

Names may vary, but inspection, semantic validation and execution must be distinct
and documented. `inspect` should report physical schema, footer row counts and
explicitly scoped sampled statistics without assigning uncertain semantics as facts.
Validation must work before model fitting. Provide equivalent plain Python entry points.

Deliver at least:

- Installable project ZIP, README and dependency/venv instructions.
- JSON Schemas, runnable dataset-schema and analysis-config examples.
- Label metadata template and external embedding/prediction/neighbor contracts.
- Synthetic examples from at least two unrelated domains with different column
  names and encodings, proving that source-domain assumptions were removed.
- Source-only and embedded-vector examples, plus ID-aligned external A/B assets.
- Feature report, output manifest, alignment/exclusion records, predictions/splits,
  normalization diagnostics and topology tables/plots.
- Metric interpretation and scaling documentation.
- The mandatory dataset-preparation prompt and example prepared dataset bundle
  described in the next section.

Tests must include integer cluster IDs versus counts, valid zero/false, null versus
known-empty tag sets, typed/duplicate metadata IDs, irregular strings, ragged/nonfinite
vectors, aligned/shuffled external IDs, group leakage, skipped targets and limits.
Verify micro/macro weighting, missing-neighbor denominators, numeric units,
normalization rank behavior, identical splits, triangle filling and cutoff censoring.
Use independent persistence checks when implementing a custom reducer.

Run a small audit from a fresh extraction/install of the deliverable. Check real
configuration/CLI paths, safe embedded JSON and report assets. Verify browser behavior
when a permitted browser is available; otherwise state the unverified scope.
Report only tests actually run and results actually measured. Input/setup failures
must be corrected before claiming an executable deliverable.

### 11. Mandatory reusable dataset-preparation prompt template

The finished project **must include `prompts/prepare_dataset.md`**. This is a prompt
for a separate chat, not merely a configuration tutorial. A user must be able to
paste it, supply Parquet details and optional files, and receive the exact files
needed by the generalized project's implemented contract.

Include the following starter, adapted to the project's actual schemas, commands,
field names and supported capabilities. Keep the template usable without the
original implementation conversation. Add a filled example and cross-reference
it from the README. Test the example bundle with the validation command.

#### BEGIN DATASET-PREPARATION PROMPT TEMPLATE

You are preparing a dataset bundle for the generalized Parquet representation
audit project. Read its attached JSON Schemas, preparation guide and example
configuration before generating files. If those contracts are unavailable, ask
for them rather than inventing an incompatible configuration format.

**Inputs I will provide:**

```text
Parquet file or dataset path: <PATH>
Actual file available to inspect: <YES / NO>
Physical schema / df.info / Arrow schema: <PASTE OR ATTACH>
Row count and representative values: <KNOWN DETAILS; MARK SAMPLE SCOPE>
Domain and row meaning: <WHAT ONE ROW REPRESENTS>
Goals / useful information / downstream task: <DETAILS>

Identity: <ID COLUMN OR COMPOSITE KEY; UNIQUE?; LEADING ZEROS?>
Embeddings: <COLUMN(S) OR EXTERNAL MATRIX + ID FILE(S); MODEL NAMES; DIMENSIONS>
Additional embeddings to compare: <DETAILS OR NONE>

Column meanings:
<FOR EACH COLUMN: ROLE; FEATURE KIND IF APPLICABLE; DESCRIPTION; UNITS;
 MISSING SENTINELS; KNOWN-EMPTY POLICY; ORDINAL ORDER OR DATETIME TIMEZONE>
Cluster/label metadata: <FILES OR LABEL-ID DESCRIPTIONS; WHICH IDs ARE MISSING?>
Tag/set encodings: <LIST / DELIMITED STRING / JSON; DELIMITER; CASE POLICY>
Supplied predictions and probabilities: <TARGET MAPPINGS; LABEL ORDER;
 PIPELINE INPUTS; TRAINING/EVALUATION PROVENANCE; FILES/IDs>
Groups, explicit splits and strata: <COLUMNS AND MEANINGS OR NONE>
Supplied neighbors / ANN assets: <FORMAT, METRIC, IDs, CANDIDATE POOL OR NONE>

Desired source-population scope: <FULL FILE / DECLARED SAMPLE>
Evaluation sample and split policy: <DETAILS OR REQUEST A CONSERVATIVE DEFAULT>
Hardware / memory budget / time budget: <DETAILS IF KNOWN>
Analyses to enable: <SOURCE, GEOMETRY, PROBES, NORMALIZATION, TOPOLOGY,
 EXTERNAL PREDICTIONS, MISSINGNESS; OPTIONAL PREFERENCE>
Output bundle directory and path conventions: <DETAILS>
```

Prepare the necessary files rather than only giving advice. If the actual
Parquet is available, inspect metadata and a bounded sample first. Clearly
separate full-file facts, sampled observations and user declarations. Never
scan/load a huge embedding matrix unnecessarily.

Account for every column using the implemented roles and kinds. Do not infer
semantic meaning solely from dtype or a suffix. Keep integer cluster IDs
categorical, counts numerical, and false/zero valid unless explicitly excluded.
Preserve unknown versus known-empty label sets. Detect possible target leakage,
ID collisions, embedding alignment issues, unsupported nested fields and
probability-label-order ambiguity.

Produce a downloadable bundle containing at least:

1. `dataset_schema.json` matching the project's actual semantic schema.
2. `audit_config.json` with conservative explicit resource limits, seeds, sample
   scope, splits, analysis selections and normalization/topology assumptions.
3. `metadata/*.json` only for mappings actually supplied or observed; leave
   unavailable names unmapped rather than inventing semantic descriptions.
4. `PREPARATION_NOTES.md` listing every column assignment, assumptions, unresolved
   issues, known missingness and population/alignment scope.
5. `run_audit.sh` and/or an equivalent Python entry point using the real CLI/API,
   activated-venv instructions and documented paths. Do not embed installation
   into system Python or destructive overwrite commands.
6. `VALIDATION.md` stating checks actually performed and the exact inspect,
   validate and run commands to use locally.
7. Small extraction/alignment scripts if needed for external embeddings,
   probabilities, IDs or an explicit evaluation sample. Preserve original IDs
   and log exclusions. Do not manufacture real data assets from schema-only text.

Validate generated JSON against the supplied project schemas. When data and
runtime are accessible, run the configuration validator and a small explicit
smoke audit into a fresh folder. Clearly identify sampling and verification scope.
When data is unavailable, validate configuration structure only and provide local
checks; do not claim row-level validation or measured results.

Ask concise questions for genuinely blocking semantics, such as missing-ID rules,
embedding alignment or tag emptiness meaning. Continue preparing independent
parts while waiting. Record low-impact assumptions explicitly. Unresolved critical
items must leave the bundle marked as draft and unable to run a misleading audit;
use the project's supported draft/unresolved mechanism rather than invalid fake
paths or invented labels. Do not endlessly ask about optional implementation details.

Return the bundle, a brief explanation of important assumptions, and the exact
command to run after extracting it.

#### END DATASET-PREPARATION PROMPT TEMPLATE

### 12. Complete the work

Implement the generalized project and the mandatory preparation template. Keep
scope claims honest: if a method is sampled, approximate, disabled or unsupported,
expose that fact in configuration, outputs and explanations. Summarize what was
implemented, what was verified, how to use the dataset-preparation template, and
any material remaining limits. Deliver the complete ZIP.

## END IMPLEMENTATION PROMPT
