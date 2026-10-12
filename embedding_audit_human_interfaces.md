# Making embedding audits useful to people

## From a measurement to a decision

An embedding audit becomes useful when someone can move from **a question**, to **evidence**, to **examples**, to **an action**. A number such as neighborhood lift should help someone decide where to search, which embedding to use, what annotations to inspect, or which limitation matters for their task.

This guide proposes interfaces and report explanations around those decisions. It is a design guide, not a statement that all these interfaces are already implemented. Existing audit measurements can support several views; region discovery, paired model comparison, review workflows, and uncertainty estimates require additional work.

**All numerical examples below are hypothetical. They are not results from your dataset.**

Three meanings of “preserves this feature” should remain visible:

| Meaning | Human question | Relevant evidence |
| --- | --- | --- |
| Local similarity | Do nearby items share this annotation or have similar values? | Purity, lift, continuous neighbor differences, behavior across neighborhood sizes |
| Recoverability | Can a model infer this annotation from the vectors on unseen items? | Held-out probe results versus a baseline; prediction errors |
| Task usefulness | Does this help the search, browsing, or review workflow I care about? | Examples, task-specific retrieval measures, human judgments |

A feature can be recoverable without being useful for nearest-neighbor browsing. Strong local agreement can also reflect duplicates or a correlated feature. Keep these as separate findings rather than combining them into a universal “embedding quality” score.

## 1. Feature scorecards: “How well is this feature preserved?”

**Interface.** A searchable table of features and tags, opening into a readable scorecard. Each card shows source prevalence, annotation coverage, known-neighbor coverage, positive-anchor support, neighborhood agreement, enrichment above chance, and a small curve across neighborhood sizes. Continuous features get differences in their original units. Probe results appear in a separate panel.

Use metadata names and criteria throughout: `score: (25, 50]` is more useful than `cluster 6`. Keep the raw ID available in details. Distinguish a bin of post scores from an artist-level aggregate; their meanings and possible uses differ. A metadata label called “cluster” does not necessarily represent a spatial region of the embedding.

**Report explanation.**

> Items annotated with “water” have the same tag on 65% of their known neighbors at k = 20, compared with an 8% source prevalence. That is about 8.1 times the prevalence baseline. This suggests that the embedding brings water-tagged items together locally. The result uses 1,200 positive anchors; unknown neighbor annotations are excluded from the agreement denominator. Inspect the examples to see whether the shared pattern is water itself, a recurring character, or another associated property.

Here, the percentage is the average of each positive anchor's agreement among known neighbors. For multilabel tags, explicitly say the result concerns neighbors of **items carrying that tag**. Agreement between two items that both lack a rare tag is not evidence of preservation. Explain micro averages as weighting positive anchor–label cases, and macro averages as weighting labels equally; a favorable average can hide weak labels.

**Actions.** Inspect good and poor neighborhoods, choose a feature for browsing, compare embeddings, or save a finding with its settings. Sort by support and effect size as well as lift: a very rare tag can have enormous lift with little practical agreement.

**Interpretation limit.** “Above shuffled labels” is different from “useful enough.” A small effect can be statistically clear in a large dataset. Avoid universal thresholds such as “lift above 2 means good.” The right threshold depends on the use case, prevalence, coverage, and neighborhood size.

## 2. Neighborhood inspector: “What does this actually look like?”

**Interface.** Select an item by ID or click a plotted point. Show its nearest neighbors in a gallery or table, with the selected feature highlighted, distances, readable annotations, and unknown values. Provide a neighborhood-size control and an option to inspect other features in the same neighborhood.

Offer three entry points: typical neighborhoods, weak neighborhoods, and a random sample. A gallery of only the best examples gives an incomplete impression. If previews are unavailable, the table should remain fully usable.

**Report explanation.**

> Of this item's 20 nearest neighbors, 16 have known annotations and 12 carry “water”: 75% agreement among known neighbors. Four annotations are unknown. Many neighbors also share the same character, so the visible concentration may involve both subject matter and character identity. This is one neighborhood, not an estimate for the entire tag.

**Actions.** Open source records, save a neighborhood, compare the same anchor in another embedding, or add items to a review list. Distinguish “annotation might be missing” from “retrieval seems irrelevant”; those lead to different fixes.

**Interpretation limit.** Distances are similarity scores under the chosen metric, not probabilities. Distance values from two different embedding models need not be comparable. Do not invent natural-language reasons for similarity from distances alone: metadata and visible examples can suggest an explanation, but cannot establish the model's internal reasoning.

## 3. Cohort comparison: “What is different between here and there?”

**Interface.** Define two named sets of item IDs. They can come from filters, saved neighborhoods, time periods, or a map selection. Compare distributions with side-by-side bars, numeric summaries, and a ranked table of differences. Always display set sizes, overlap, known-value counts, and missingness.

For tags, show **percentage-point differences**, prevalence ratios, and counts together. For continuous features, show distributions and differences in units, rather than only a divergence score. Jensen–Shannon divergence can summarize categorical distribution differences, but the bars explain what changed.

**Report explanation.**

> “Water” appears on 60% of annotated items in Region A and 10% in Region B: a difference of 50 percentage points and a prevalence ratio of 6. Region A also contains more landscape images. Annotation coverage is 75% in A and 95% in B, so these percentages describe the annotated subsets and may be affected by which items are missing labels.

**Actions.** Save the comparison, export IDs, inspect the largest differences, or repeat the comparison within a shared character, artist, year, or other relevant stratum.

**Interpretation limit.** A difference does not establish why it exists. If A was selected by filtering for water, its high water prevalence is expected. If two sets overlap, expose the overlap; treat them as disjoint only after making that an explicit operation. Stratified comparisons can assess a proposed confounder, but do not remove every possible confounder.

## 4. Region profile: “What is special about this region?”

**Interface.** Selecting a region opens a profile with:

- Common annotations: what describes many items here.
- Distinctive annotations: what is enriched here relative to a named reference.
- Unusual numeric distributions and annotation gaps.
- A gallery of central examples, boundary examples, and varied examples.

The reference might be the rest of the eligible dataset, the whole eligible dataset, or a matched cohort. Name it in the heading; each answers a different question. “Rest of dataset” is often easy to interpret because the selected region is excluded from its reference.

**Report explanation.**

> This region contains 2,400 items, of which 1,800 have known tag annotations. “Water” occurs on 1,080 of those annotated items: 60%, compared with 10% outside the region. It is a useful descriptive label for this selection. A second tag is more enriched but occurs on only 18 items, so it describes a small subset rather than the region as a whole.

Use names such as **“water-rich selection”** as provisional descriptions, not as facts that every item shares. Prefer short labels supported by several compatible observations; allow people to edit and save them.

**Actions.** Name a selection, inspect subgroups, locate neighboring regions, export a cohort, or flag a pattern for later validation.

**Interpretation limit.** A region must have a reproducible membership definition: saved IDs, a specified neighborhood, a clustering assignment, or a polygon tied to a particular projection. Selecting a visually interesting region and then finding its strongest labels is exploratory. Repeated searching and many tested labels can produce striking chance findings; confirmation should use a fresh sample or a clearly specified validation procedure.

## 5. Tag and feature discovery: “Where is this most present?”

**Interface.** Let someone search for a tag, a readable numeric interval, or a tag combination, then view matching items and ranked neighborhoods or regions. Provide explicit controls for **all tags**, **any tag**, and exclusions. Unknown annotations require their own state; “not annotated with X” is not automatically “confirmed absent.”

Offer separate ranking modes:

| Ranking | Question it answers | Why another ranking may disagree |
| --- | --- | --- |
| Matching count | Where are the most annotated matches? | Large regions can dominate |
| Matching fraction | Where is a high proportion of items a match? | Small regions can look excellent |
| Enrichment relative to reference | Where is this unusually concentrated? | Rare annotations can have large ratios |
| Share of all known matches captured | How much of the annotated target set is here? | A pure region can contain very few of the total matches |

**Report explanation.**

> This region has 60% water-tagged items among those with known annotations and contains 12% of all known water-tagged items in the eligible dataset. It is a concentrated place to browse, but it does not contain most of the annotated water examples.

For combinations, calculate their **joint** counts and prevalence. Individual lifts for “water” and “night” cannot be multiplied to infer the strength of “water AND night.” Sparse combinations should display a support warning and may need broader queries.

**Actions.** Browse high-yield regions, collect examples, save a query, or expand from annotated seeds to nearby unannotated candidates.

**Interpretation limit.** Nearby unannotated items are candidates for review, not confirmed matches. A smoothed color map estimates a local pattern; it is not an exact count. Overlapping neighborhoods reuse items, so summing their counts would double-count matches. Search may find several separated pockets; avoid forcing every concept into one region.

## 6. Embedding comparison: “Is another set of vectors better for this?”

**Interface.** An A/B view with aligned item IDs, the same feature definitions, and paired anchor examples. Show changes by feature and neighborhood size, plus a trade-off table. Let people choose the task they care about: local browsing, feature prediction, or recovery under missing annotations.

Use the same evaluation cohort, candidate pool, missingness rules, distance convention, and probe splits. Primary comparisons should use items valid in both embeddings; separately report each model's excluded items. For approximate search, check search quality so indexing error is not mistaken for an embedding difference.

**Report explanation.**

> On the shared evaluation cohort at k = 20, water-tag agreement is 65% for Embedding B and 41% for A: an improvement of 24 percentage points. B shows stronger local preservation of the water annotation on this cohort, making it a promising choice for water-oriented browsing. Character agreement decreases, so B is not uniformly better across the tested features. Inspect the paired examples to assess whether the improvement matches your browsing needs.

Add uncertainty to the **paired difference**, not just separate intervals around A and B. A paired bootstrap over saved anchor contributions is one possible approach; related items may require group resampling. Such intervals describe anchor sampling variability conditional on the evaluated graphs. They do not automatically measure index error, dataset shift, or retraining variability.

**Actions.** Choose an embedding for a specified task, retain different embeddings for different tasks, or create a repeatable evaluation before accepting a new model.

**Interpretation limit.** Similar-looking maps are not a fair model comparison. Two independently fitted projections can rotate, rearrange, or distort neighborhoods. Compare original-vector measurements and the same item IDs; a side-by-side map is supporting navigation. A claim of “better overall” needs declared task priorities or weights, not an arbitrary average of incompatible metrics.

## 7. Review and recovery: “What should I inspect or annotate next?”

**Interface.** A review queue with separate reasons: low neighborhood agreement, prediction–annotation disagreement, missing annotations near annotated examples, and underrepresented groups. Show the evidence and counterexamples beside each candidate. Keep observed annotations and inferred suggestions visibly distinct.

**Report explanation.**

> This item lacks a water annotation, while 15 of its 20 known nearest neighbors carry it. It is a candidate for annotation review. Other explanations include a missing annotation, a loosely related scene, or a neighborhood organized around another shared property. No label has been changed.

A missingness report can support a narrower statement:

> With 30% of known annotations hidden at random, the reconstruction model recovers this tag with the reported held-out precision and recall. Performance is lower when rare examples are hidden first. These experiments suggest sensitivity to the pattern of missing annotations; they do not establish accuracy on naturally unannotated items.

**Actions.** Add a review decision, correct an annotation through an explicit workflow, prioritize collection, or compare recovery methods. Preserve the original value, proposed value, reason, reviewer decision, and timestamp.

**Interpretation limit.** Disagreement is not proof of an annotation error. High-confidence prediction is not proof of correctness. A queue ranked by disagreement can overlook uniformly wrong predictions; include random review and rare-group review. If suggestions later become training labels, keep them identifiable so evaluation does not unknowingly measure agreement with the system's own earlier guesses.

## 8. Explanations should be part of every report

Every view should answer four things in ordinary language: **what was measured, what it suggests, what limits the conclusion, and what someone can do next**. Put a short explanation next to the chart and expandable definitions below it. An optional question box should navigate these views and their computed evidence; it should not be the only way to access them.

Start with deterministic explanations generated from structured results. They are reproducible and work in an offline report. Optional language-model narration can improve phrasing, but should receive the same bounded evidence and must not invent counts, causes, confidence intervals, or uncomputed comparisons.

Useful explanation patterns include:

| Evidence | Report wording | Suggested next step |
| --- | --- | --- |
| High purity, high lift, adequate support | “This annotation is concentrated in nearby items relative to the stated baseline.” | Inspect examples; test browsing usefulness |
| High purity, lift near 1 | “Agreement is high largely because the annotation is common; enrichment is limited.” | Compare against the prevalence baseline |
| High lift, low purity | “This rare annotation is enriched locally, but most known neighbors do not carry it.” | Use narrower neighborhoods; inspect false matches |
| Strong probe, weak local agreement | “The tested predictor can recover the feature, although nearest-neighbor browsing groups it poorly.” | Consider a predictor or a different retrieval representation |
| Agreement declines as k increases | “The pattern is concentrated at short range and becomes more mixed in broader neighborhoods.” | Use local browsing; avoid broad region claims |
| Predicted prevalence rises substantially | “Predictions assign this annotation more often than the observed labels on the matched cohort.” | Check precision, thresholds, and missingness before interpreting spread |
| Few known labels or very few positives | “There is insufficient evidence for a stable conclusion.” | Collect annotations or broaden the evaluation |

Continuous-feature explanations should translate back into units: “neighbor scores differ by 12 points on average, versus 40 under the specified shuffle” is easier to use than an isolated autocorrelation value. A distance-association statistic can support “embedding distance and score differences are associated,” but not a claim about why a particular image was retrieved.

Each explanation needs an evidence record containing feature and label IDs/names, measurement and aggregation definition, value, baseline, support, coverage, cohort/reference, embedding version, metric, k, sampling/search method, and uncertainty if available. Undefined metrics should produce “not estimable” with a reason, not zero. Expose numerical evidence through a “Show evidence” table and export it with the narrative.

For tags, document when an absent tag is treated as a negative for calculation and when the annotation is unknown. An annotated tag set can still be incomplete. For probes, expose the held-out split and any grouping by artist, parent family, or another leakage risk. These policies materially change what the narrative can claim.

Use descriptive conclusions by default. Words such as “strong” or “reliable” should have declared criteria and reflect support, coverage, stability, and practical size—not only a p-value. Shuffling tests a specified null; it does not prove semantic understanding or causation.

## A map can help, but it should not carry the conclusion

A searchable table, distributions, and galleries can answer many questions before an interactive map is built. When a map is added, use it to navigate and define selections. Recompute or retrieve the selected IDs' evidence in the original embedding and source data.

Show which points were sampled, the projection method, and whether colors indicate labels, local estimates, or model predictions. Unknown values need a separate color. Do not treat visible blob size as a reliable estimate of population size. UMAP documentation specifically warns that density and apparent cluster splits can be distorted; a visually isolated island is not automatically a distinct semantic class.

Source-feature bins and embedding regions need different names and controls. Filtering for `score: (25, 50]` selects an annotation category; selecting an embedding neighborhood selects items by vector similarity. Combining them can be useful, provided the selection rule remains explicit.

## Keeping these interfaces usable at eight million rows

An interface does not need to display or recompute eight million items to answer a question about them. Separate full-dataset source summaries, indexed item retrieval, and costly geometric evaluation. This is a proposed scaling architecture; it is not a claim that the current in-memory audit already supports it.

| View | Practical strategy at 8 million rows | What the user should see |
| --- | --- | --- |
| Feature cards | Batch exact counts, coverage, and suitable numeric summaries; cache geometric results on a defined anchor sample | Source population size, evaluated anchors, and their different scopes |
| Item neighborhoods | Query a similarity index and fetch only requested IDs/annotations; cache frequently used neighborhoods | Exact or approximate search, candidate pool, and k |
| A/B comparison | Reuse the same anchors and candidate IDs; compute paired contributions offline | Common cohort, exclusions, and comparison uncertainty |
| Cohort/region profiles | Batch selected-column counts; store sparse tag counts for saved cohorts; fetch galleries separately | Exact counts or explicitly sampled estimates |
| Hotspot discovery | Predefine a manageable set of regions or anchors and precompute selected tag summaries | Discovery scope, support thresholds, and overlap |
| Map | Render aggregated tiles or a representative sample; fetch detail on demand | Display sample size; which selections include undisplayed items |
| Review queue | Score in batches and page a ranked list; limit expensive explanations to requested items | Queue reason, freshness, and evidence |

For perspective, a full k = 100 graph has 800 million directed edges. Repeatedly scanning it for every feature and every null trial is substantial work. A 100,000-anchor evaluation has 10 million edges at the same k, while those anchors can still search the full eligible candidate population. Validate approximate search and sample coverage; oversampling rare groups requires weighting for population summaries.

Exact source statistics do not make sampled geometry exact. Also, use a prevalence baseline appropriate to the neighbor candidate population and annotation policy: a full-dataset count can describe the source while being an unsuitable chance baseline for a filtered search pool. Label those roles separately.

For map selections, distinguish **select displayed points** from **select all items in the defined area**. The latter requires saved coordinates or another population-membership operation; it cannot be inferred from a sparse display. Precomputed regional summaries also cannot answer every arbitrary new polygon without additional computation.

## A useful implementation order

1. **Readable scorecards and explanations.** Add evidence-backed narrative to existing measurements, readable metadata labels, explicit scopes, and links to examples. This offers immediate value in a static report.
2. **Neighborhood inspection and saved cohorts.** Add per-anchor evidence, galleries/tables, reproducible ID selections, and cohort comparisons. People can investigate an aggregate result and export something actionable.
3. **Paired embedding comparison.** Align datasets, freeze evaluation settings, compare per-feature changes, and add appropriate uncertainty. This directly supports choosing a representation.
4. **Region discovery and review queues.** Add hotspot search, interactive navigation, and annotation-review support once region definitions and evidence records are dependable. Large-data versions need indexed retrieval and cached/batched statistics.

The first useful report could let someone select a tag, read an explanation, inspect ten varied neighborhoods, compare a second embedding, and export a review cohort. That is a complete human workflow even without a sophisticated map or a conversational assistant.

## Resources and design references

- [TensorBoard Embedding Projector](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin): a reference for exploring vectors with metadata. It offers navigation ideas; visual inspection remains separate from quantitative evaluation.
- [Facets Overview and Dive](https://pair-code.github.io/facets/): examples of feature distribution comparisons and moving from dataset summaries to individual items. Treat these as design references, not a promise of eight-million-row browser rendering.
- [UMAP: using projections for clustering](https://umap-learn.readthedocs.io/en/latest/clustering.html): explains why projected density and apparent regions require care.
- [SciPy paired bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html): describes paired resampling for confidence intervals; dependence and the evaluation target still need to be considered separately.
- [Faiss documentation](https://faiss.ai/): similarity indexing and accuracy–speed trade-offs relevant to on-demand neighborhood inspection at scale.

