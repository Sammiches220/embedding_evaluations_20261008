# Validation actually performed

The deliverable was produced and checked on Python 3.12. A virtual environment was used throughout. These results describe synthetic fixtures and bounded examples, not the user's real Parquet or an 8-million-row benchmark.

## Executed checks

- **80 Python tests passed**: 24 new generic contract/end-to-end tests and 56 retained computational/report regression tests. Real Parquet/NPY data and CLI commands were exercised.
- Generic contracts cover unordered integer IDs versus counts, valid zero/false, declared sentinels before domain checks, null versus known-empty tags, literal delimiters/JSON tags, exact large IDs/leading zeros/composites, metadata typed IDs/duplicate names/missing-policy disagreement, ragged/text/null/nonfinite vectors, shuffled/missing/duplicate external IDs, explicit group leakage, auxiliary strata, disabled/unsupported targets, resource caps, fresh-output protection, source-only and evaluation-scope reporting, explicit nested extraction, long/untrusted names and embedded JSON safety.
- Numerical tests cover positive-tag micro/macro weighting, unknown-neighbor denominators, numeric units and constant targets, offset-stable variance, safe extreme norms, normalization rank behavior, identical A/B/variant splits, training-only group tuning, H0/H1 mergers/triangle filling and cutoff censoring.
- **36 H1 diagrams** agreed with independent Ripser calculations across multiple dimensions, sample sizes and complete/cutoff filtrations.
- **376 JavaScript/Python numerical comparisons** passed on the generic telemetry A/B demo, including full/subset geometry, entropy and saved probe rescoring.
- The retained JavaScript overview/enhancement numerical checks passed.
- Actual serialized report scripts were executed in **jsdom** for all three generated demos. Five-column table/section structure, feature search/subset/clear/select-all, consistent feature filtering in normalization and topology, display-mode/readout changes, all-k tiles, brushing with fixed global experiments, exports with exact IDs/settings/scope, unavailable/source-only states, model/filtration controls and existing local links passed.
- Runnable retail, telemetry A/B and source-only examples passed semantic validation and actual audits. Retail selected 120 rows and retained **119** common valid vectors; telemetry selected 120 and retained **118** common valid A/B vectors. Source-only summarized the full 160-row source and retained a 120-row sample for matched supplied prediction evidence. All were explicitly synthetic.
- The project ZIP was freshly extracted, installed into a new **isolated venv without system-site-packages**, and a real 32-row telemetry A/B smoke audit ran through the installed CLI with real relative asset paths. Installation also exercised packaged JSON Schemas and report assets.

## Reproduce

From the extracted project root and activated venv:

```bash
python -m pip install '.[test]'
python -m pytest tests tests_legacy -q
python tests_legacy/verify_persistence.py
python tests_legacy/verify_feature_overview.py
node tests_legacy/test_explorer.js
node tests_legacy/test_enhancements.js
```

For the DOM checks, install the optional JavaScript test dependency separately (not a report runtime dependency):

```bash
npm install --prefix /tmp/parquet-audit-ui jsdom
NODE_PATH=/tmp/parquet-audit-ui/node_modules node tests/test_report_dom.js
```

Each prepared example's `VALIDATION.md` lists its exact inspect/validate/run commands. The root README explains venv installation and fresh output folders. Existing generated reports can be opened offline.

## Visual-verification limitation

The original page shell, CSS, feature-first markup and adapted JavaScript were inspected and reused directly. Layout, palette, section order and controls were checked through source/markup and DOM tests. The environment had no installed Chromium; its attempted download returned unusable bytes. **Desktop/narrow-width rendered screenshots and a direct rendered visual comparison were not verified.** Responsive CSS was retained/supplemented and long-name behavior was tested at the markup/interaction level, not claimed as pixel-level proof.

The code was not benchmarked on 8 million rows. Source algorithms stream; expensive graphs/probes/topology remain bounded sampled in-memory analyses. External ID lookup has a population-sized cap. No full-population ANN, encoder retraining or weight support is claimed.
