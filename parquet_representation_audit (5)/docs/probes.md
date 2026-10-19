# Prediction probes and report controls

The overview and direction/magnitude workspace share one metric vocabulary and the same configurable probe families. Topology is a geometric analysis and has its own measurement toggles; no prediction model is fitted by its controls.

## Select methods

Set `probes.models` to any combination of `baseline`, `linear`, `knn`, `random_forest`, `xgboost`. The embedding audit includes a dummy baseline for comparisons. The report's probe dropdown is populated from the methods actually computed.

- `linear`: Ridge regression or balanced logistic regression.
- `knn`: distance-weighted neighbor regression/classification.
- `random_forest`: random forest regression/classification.
- `xgboost`: XGBoost regression/classification, or independent binary classifiers for multilabel tags.

All methods for a feature use the same held-out rows. Normalization variants and A/B embeddings also use matching feature-specific splits. A local category-code mapping is fitted on training labels for XGBoost, then predictions and probability columns are mapped back to original labels. This mapping does not turn nominal categories into ordered numeric measurements.

## Install XGBoost

In a virtual environment, install the optional dependency with:

```bash
python -m pip install -e ".[xgboost]"
```

Without this extra, the default baseline/linear configuration still works. Selecting XGBoost without the dependency produces an actionable error before the audit starts. The supplied complete examples select it and require this extra.

`probes.trees` controls both forest tree count and XGBoost boosting rounds. The XGBoost factory uses CPU histogram trees, one worker, depth 4, learning rate 0.1, and the run seed. These are fixed demonstration settings, not an exhaustive model search. XGBoost uses its ordinary unweighted fit; logistic/forest class weighting differs, so these are comparisons of the configured pipelines. No outer-test-set early stopping or tuning is performed. `probes.regularization_grid` tunes only the linear family using training-only folds.

Normalization inherits `probes.models` and `probes.trees`. You can explicitly narrow those experiments with `normalization.models` and `normalization.trees` when fitting every family across all four variants is too expensive. The baseline is always included. `normalization.regularization_grid` governs its linear fits.

## Select measurements

Both workspaces show explicit metric checkboxes. A selected metric is displayed only when estimable for that target and probe; inapplicable metrics do not cause substitution with another metric. Default selections include the summary metrics that were actually computed. Numeric recovery includes R², MAE and Spearman correlation; categorical/tag recovery includes accuracy, F1, ROC AUC and average precision where defined. Distribution diagnostics are available separately in the same list.

Color compares each probe metric with the matching dummy baseline:

- Higher recovery scores are better; MAE uses relative error improvement.
- Divergence/prevalence errors improve downward.
- Signed prediction-minus-truth distribution differences improve toward zero, based on their absolute difference from zero.
- Insufficient support or undefined baseline comparisons use the limited-evidence color.

Distribution matching is not a substitute for correct row-level predictions: a classifier can reproduce label counts while assigning them to the wrong items. Entropy differences use natural-log units in probe diagnostics; the normalized neighborhood entropy display uses its separately documented normalization.

The shared descriptive threshold is a display choice, not a statistical test. R² may be negative; rank/probability scores may be undefined for small or constant cohorts. Region selection rescores stored held-out predictions without refitting. Direction/magnitude and topology controls operate on the full recorded experiment; brushing does not rerun them.

## Implementation map

- `embedding_feature_audit/modeling.py`: plain model factories, split/fitting/scoring, XGBoost label mapping.
- `embedding_feature_audit/normalization.py`: raw, unit, standardized-unit and norm-only inputs using those same probes.
- `embedding_feature_audit/assets/metric_options.js`: common metric labels and baseline color rules.
- `embedding_feature_audit/assets/explorer.js`: overview controls and selected-held-out-row rescoring.
- `embedding_feature_audit/assets/enhancements.js`: normalization selectors and topology measurement toggles.
- `embedding_feature_audit/enhancement_report.py`: section markup and embedding-specific topology plots.

These are functions and plain tables; no per-feature wrapper framework is needed to add a model factory. New probe names must also be registered with the shared validator so unsupported names fail early.

Upstream reference: [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
