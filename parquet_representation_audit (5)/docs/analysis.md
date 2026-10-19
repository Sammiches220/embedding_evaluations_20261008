# Analysis and report guide

The HTML report describes how each feature relates to the embedding geometry. It is evidence about organization, not a single pass/fail score for embedding quality.

## Neighborhood scales

`geometry.ks` is an ordered sequence of neighbor counts. The same sequence is applied to every selected feature and appears in the report in that order. Smaller values describe local neighborhoods; larger values describe broader neighborhoods. Use the same ordered values and sampling settings when comparing runs.

## Feature families

- Numeric/count features use local differences, variance, and graph autocorrelation measures. Numeric values keep their units unless a metric is explicitly normalized.
- Categorical and disjoint cluster labels use prevalence-aware organization and neighborhood label measures.
- Multilabel features retain set membership; coverage and tag prevalence are computed per tag.
- Configured linear, neighbor, forest and XGBoost probes test recoverability from vectors on held-out data. They do not establish causal importance or encoder training stability.
- Null comparisons test whether observed organization exceeds a shuffled annotation baseline. Reported neighborhood scales are related measurements, not independent replications.

## Source population versus evaluation sample

Source statistics can be streamed across all Parquet rows. Expensive vector, neighborhood, and topology work uses either all rows within configured limits or one reproducible bounded sample. The report records the scope and row counts for each result. Null or invalid embedding rows may be excluded only under the explicit configured policy; all compared embeddings use the same valid cohort.

## Topology

Optional topology analysis samples points and follows connected components as distance thresholds increase. It can show scales at which regions merge and can calculate a tightly bounded exact H1 summary. This describes geometric structure; topology does not name clusters or determine whether a feature is useful. Keep these analyses disabled for a first run and enable them only when that question is useful.

## Reading conclusions

Read effect sizes, support counts, missingness, baseline comparisons, and the feature definition together. A strong result for one feature says that the geometry reflects that feature to some extent; it does not show that the embedding preserves every useful distinction. The report keeps numeric results and caveats together and retains the same ordered feature and scale organization for repeatable review.

Both report workspaces expose explicit embedding and metric choices. Direction/magnitude additionally selects among configured probes. Topology shows each selected measurement at every radius and orients paired differences as other embedding minus selected. See [probe controls](probes.md) and [information preservation](information-preservation.md).
