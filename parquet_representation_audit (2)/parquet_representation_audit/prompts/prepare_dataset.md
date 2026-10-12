# Reusable prompt: prepare an audit dataset bundle

Paste this entire prompt into another chat. Attach the actual project's `parquet_audit/schemas/{dataset_schema,audit_config,label_metadata}.schema.json`, `docs/PREPARATION.md`, `docs/CONTRACTS.md`, and a runnable example config. The filled `examples/retail/` bundle shows the expected output. You do not need the original implementation conversation.

---

You are preparing a dataset bundle for **parquet-representation-audit 1.0.0**. Read its attached JSON Schemas, preparation guide, alignment contracts and example configurations before generating files. If those contracts are unavailable, ask for them rather than inventing a configuration format. Prepare downloadable files, not just advice.

**Inputs I will provide:**

```text
Parquet file or dataset path: <PATH>
Actual file available to inspect: <YES / NO>
Physical schema / df.info / Arrow schema: <PASTE OR ATTACH>
Row count and representative values: <KNOWN DETAILS; MARK SAMPLE SCOPE>
Domain and row meaning: <WHAT ONE ROW REPRESENTS>
Goals / useful information / downstream task: <DETAILS>

Identity: <ID COLUMN OR COMPOSITE KEY; UNIQUE?; LEADING ZEROS?>
Embeddings: <COLUMN(S) OR EXTERNAL NUMERIC NPY MATRIX + ID NPY FILE(S);
             MODEL NAMES; DIMENSIONS>
Additional embeddings to compare: <DETAILS OR NONE>

Column meanings:
<FOR EACH COLUMN: ROLE; FEATURE KIND IF APPLICABLE; DESCRIPTION; UNITS;
 MISSING SENTINELS; KNOWN-EMPTY POLICY; ORDINAL ORDER OR DATETIME TIMEZONE>
Cluster/label metadata: <FILES OR LABEL-ID DESCRIPTIONS; WHICH IDs ARE MISSING?>
Tag/set encodings: <LIST / DELIMITED STRING / JSON; DELIMITER; CASE POLICY>
Supplied predictions and probabilities: <TARGET MAPPINGS; LABEL ORDER;
 PIPELINE INPUTS; TRAINING/EVALUATION PROVENANCE; PARQUET FILES/IDs>
Groups, explicit splits and strata: <COLUMNS AND MEANINGS OR NONE>
Supplied neighbors / ANN assets: <NPY FORMAT, METRIC, IDs, CANDIDATE POOL,
                                  PREPROCESSING AND PROVENANCE OR NONE>

Desired source-population scope: <FULL FILE / DECLARED SAMPLE>
Evaluation sample and split policy: <DETAILS OR REQUEST A CONSERVATIVE DEFAULT>
Hardware / memory budget / time budget: <DETAILS IF KNOWN>
Analyses to enable: <SOURCE, GEOMETRY, PROBES, NORMALIZATION, TOPOLOGY,
 EXTERNAL PREDICTIONS, MISSINGNESS; OPTIONAL PREFERENCE>
Output bundle directory and path conventions: <DETAILS>
```

Inspect actual Parquet footer/physical schema and a bounded sample first when bytes are available. Use the real `parquet-representation-audit inspect` command where installed. Separate full-file facts, sampled observations and user declarations. Do not scan a huge external embedding matrix; inspect numeric NPY headers and ID contracts first. Never manufacture real data assets, embeddings, labels or alignment from text previews or abbreviated `...` arrays.

Produce at least:

1. **`dataset_schema.json`**, using `schema_version: 1`, `status`, `dataset`, `columns`, `embeddings`, and only supported optional fields. Each column needs a role or an explicit unmapped policy. Include `weights: "unweighted"`; weight columns must be ignored with a reason unless an external weighted method is separately requested. Define all asset paths relative to this schema file.
2. **`audit_config.json`**, matching every required section of the actual analysis schema: `sample`, `source`, `limits`, `geometry`, `nulls`, `probes`, `normalization`, `topology`, `missingness`, `report`, and the required top-level fields. Start from the supplied runnable example rather than guessing defaults. Keep caps concrete and conservative; `output_policy` must be `"fresh"`.
3. **`metadata/*.json`** only for supplied or actually observed mappings. Use typed IDs, preserve exact computational values and permit duplicate readable names without merging IDs. Do not invent meanings for unavailable labels. `excluded_from_coverage` must agree with feature `missing_values`, unless the user explicitly authorizes a reasoned `metadata_missing_override`.
4. **`PREPARATION_NOTES.md`**, accounting for every column assignment, assumptions, unresolved issues, missingness, source/evaluation scope, candidate pool, ID alignment and resource estimates. Explain that physical dtype does not establish semantic role.
5. **`run_audit.sh`** or an equivalent Python entry point, using the installed real CLI/API and a fresh output path. Document virtual-environment installation without altering system Python or including destructive overwrite commands.
6. **`VALIDATION.md`**, distinguishing structural validation from actual row/asset/runtime checks. Include exact commands and only claim checks actually performed.
7. Small extraction/alignment scripts if needed to build explicitly ID-aligned numeric/vector/prediction assets or a declared sample from real inputs. Preserve exact source identity and log exclusions. Composite keys work inside Parquet; external vectors, predictions and graphs currently require one stable scalar key, so explicitly precompose a typed composite string if necessary.

Roles: `id`, `embedding`, `feature`, `prediction`, `probabilities`, `group`, `split`, `stratum`, `display`, `ignore`. Feature kinds: `cluster_label`, `categorical`, `boolean`, `ordinal`, `count`, `numeric`, `multilabel`, `datetime`, `text`. Auxiliary `stratum`, `group` or `split` uses may be declared without removing a feature from the inventory.

Keep unordered integer cluster IDs categorical. Do not mark 0 or false missing unless explicitly declared. Apply declared missing sentinels before count/range validation. Counts default to nonnegative integers; negative net measurements must be numeric with units. String IDs preserve leading zeros; large integer IDs never pass through float. Identity must be stable, unique and nonmissing for external joins/A/B. For explicitly source-only data lacking stable IDs, fingerprint-position identity can be declared, with its cross-file limitation documented.

For tag sets declare `encoding` and any literal delimiter, stripping/case rules, empty-token policy and vocabulary/support caps. Null may be unknown while an empty list or configured empty string is known absence. Free text is not automatically a tag set. Ordinal geometry/probes are categorical unless an explicit numeric `ordinal_value` derived mapping and reasoning are supplied. Datetimes require a timezone; derived numeric targets must be explicit. Base text/time fields have coverage/source summaries and no direct embedding targets.

Embeddings must be genuine fixed-length real numeric lists or external numeric NPY matrices with unique aligned ID NPY arrays. The model entry needs `main_normalization`. A/B dimensions may differ; both models share the common valid identity cohort. Explicit `invalid_embeddings: "exclude"` handles nonfinite/zero directions and records reasons. Missing feature truth generally changes feature-known masks, not the whole embedding cohort. Supplied graphs cannot be silently reindexed after filtering.

Predictions/probabilities must declare target, pipeline and whether that pipeline uses embeddings. Probability vectors need an explicit typed label order and valid ranges. For external predictions use ID-aligned Parquet `prediction_assets`; NPY score arrays need an explicit conversion script. One supplied prediction pipeline per target per audit is currently supported. Supplied graphs require a bounded exported Parquet cohort, `sample.strategy: "all"`, exactly matching anchor IDs/order, candidate_scope `evaluation_cohort`, metric, preprocessing, sorted aligned distances, no self/duplicate neighbors and provenance. Do not claim arbitrary production ANN candidate-pool support.

For no embeddings choose `mode: "source_only"`, set `embeddings: []`, and turn off geometry, probes, normalization, topology and missingness; matched external prediction evidence may still be evaluated. Do not generate embeddings to avoid this state.

Use a bounded `uniform_rows` sample for expensive analyses. Groups protect held-out and tuning folds; explicitly supplied split values must be mapped to train/test and must not leak groups. Keep feature vocabulary/test rows identical between models and normalization variants. Strata are explicit. BH correction supports family `all_features_scales_modes_per_metric`; do not name an unimplemented testing family. Full-source tag/label prevalence requires full streamed source statistics; numeric variance references remain the evaluation cohort. Explain approximate reservoir quantiles and sampled source relationships/co-occurrence. Resource limits do not create 8-million-row benchmark guarantees.

Validate generated JSON against the actual schemas. When real data/runtime are accessible, run:

```bash
parquet-representation-audit inspect --parquet ACTUAL_PATH --out new_inspection
parquet-representation-audit validate --schema dataset_schema.json --config audit_config.json
parquet-representation-audit run --schema dataset_schema.json --config audit_config.json --out new_smoke_audit
```

The CLI output path is relative to the caller's working directory; asset paths in JSON are relative to `dataset_schema.json`. Use a fresh folder. When data are inaccessible, perform structure-only validation and give local checks; do not claim identity uniqueness, alignment, scalar domains, measured metrics or successful fitting.

Ask concise questions only for blocking semantics: stable identity, real embedding alignment/dimension, tag emptiness, missing-label rules or probability order. Continue independent preparation while waiting. Record low-impact assumptions. Unresolved critical questions require `status: "draft"` with an `unresolved` list; such a bundle must fail ordinary validation/run rather than producing a misleading audit. Draft inspection may use `validate --structure-only --allow-draft`. Do not fill unresolved fields with fabricated paths, IDs, dimensions or labels merely to pass a validator; request the missing contract or keep a structurally valid conservative source-only draft with documented unresolved analysis intent.

Return the complete downloadable prepared bundle, its material assumptions, actual verification scope, and the exact command to run after extraction.

---

## Filled example to supply with this prompt

Use the shipped `examples/retail/` directory as a fully specified example. One row is a **synthetic retail order**; `order_key` is a large exact integer ID, `encoded_basket` is an 8-dimensional embedded vector, `partition_code` is an unordered integer assignment with -1 missing and 0 known, `units_sold` is a count, `net_margin` is numeric USD, `returned` is boolean, `priority` has explicit ordinal order, and `basket_labels` has null unknown versus list-empty known absence. Date/text targets are derived explicitly. The customer group protects an explicit fit/holdout split. Supplied department predictions and probabilities have declared pipeline and label order. This bundle and its validation/run commands are included in the ZIP; replace those synthetic domain meanings deliberately for a real dataset.
