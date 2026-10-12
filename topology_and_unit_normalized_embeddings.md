# Topology, filtration and unit-normalized embeddings

*A practical guide to evaluating representations — October 12, 2026*

A useful embedding preserves information that matters for a task in a form that
its downstream pipeline can access. Topology adds a question: **what structure
survives as the scale of observation changes?** Unit normalization adds another:
**how much of that structure and information lives in direction rather than
magnitude?** Neither produces a universal quality grade.

This guide discusses analysis ideas. The current audit has cosine neighborhoods
and linear probes; persistent homology and optional row normalization are not
implemented by this document.

## 1. What filtration gives you

A filtration is a nested sequence of spaces: increasing its parameter adds
structure without removing what was already included. For a distance filtration,
start with embedding points and gradually increase a distance threshold ε.
Connect pairs when their distance is at most ε. A Vietoris–Rips complex also fills
triangles when all three edges exist, and adds higher-dimensional simplices
according to the same all-pairs rule. [1]

Persistent homology tracks structures through this sequence. [2]

| Object | Intuitive meaning | Useful interpretation |
|---|---|---|
| H₀ | Connected components and their mergers | Which groups connect early, and which remain separated? |
| H₁ | Loops not yet filled by triangles | Is there a cyclic arrangement or a hole in the sampled structure? |
| H₂ | Enclosed cavities | A higher-dimensional structural diagnostic, often expensive to investigate |
| Barcode / persistence diagram | Birth and death scales | Over what interval does each structure exist? |
| Betti curve βₚ(ε) | Number of p-dimensional features alive at ε | How does structural complexity change with scale? |

Imagine points sampled around a circle. At very small thresholds they are
separate. At intermediate thresholds they can form a connected loop. At larger
thresholds enough simplices fill that loop. A long H₁ bar describes the interval
between its formation and filling. It does not identify what the circle means.

**Your existing entropy-versus-k and purity-versus-k curves are scale analyses,
but they are not themselves persistent homology.** To obtain persistence, define
nested complexes and track homology through their inclusion maps. A fixed ordered
neighbor list can yield nested symmetrized graphs as k rises, but that construction
has a different scale meaning from a radius-based Rips filtration.

For H₁, an ordinary graph cycle count is insufficient: graph edges alone cannot
fill a loop. A Rips calculation needs triangles, or an algorithm that accounts
for their effects implicitly. “Compute H₁” and “store only a graph” are different
computational choices.

## 2. Robustness has a specific meaning

Persistence has formal stability results. Under suitable assumptions, small
changes to filtration functions produce small changes to diagrams in bottleneck
distance. This is a mathematical reason to prefer a scale-dependent structural
summary over trusting one arbitrary threshold. [3]

It does **not** establish that every long bar is semantic signal. Stable
structure can reflect acquisition artifacts, duplicate families, background
colors or missing regions of the sample. Changing the sampling distribution,
metric or graph construction can exceed the perturbations covered by a theorem.
Even a stable diagram may give a different count at one threshold near a birth
or death event.

Treat long persistence as a candidate for explanation. Check repeated samples,
related-item groups, plausible perturbations and matched reference models. A
subsampling variability band is descriptive unless its construction supports a
formal inference claim. Landmark selection also matters: random samples favor
dense regions, while farthest-point selection can emphasize outliers. [4]

## 3. How topology can contribute to “good representations”

The following are proposed evaluation designs, not automatic quality rules.

| Question | Analysis | Evidence needed to call it useful |
|---|---|---|
| Are there persistent groups? | H₀ merger structure plus feature summaries | Relevant annotations or task differences explain the groups |
| Does a feature remain organized across scales? | Feature entropy inside filtration components | Improvement over label-shuffled controls, with adequate support |
| Does a known cycle survive? | H₁ plus an external cyclic variable | Independent evidence such as controlled camera rotation |
| Did a projection destroy structure? | Compare original-space and projected diagrams on identical IDs | A declared reference geometry and additional item-level checks |
| Does model B change useful structure? | Matched A/B filtrations, labels and probes | Task-relevant changes supported by held-out evaluation |

A viewpoint cycle is a plausible H₁ hypothesis: rotating a camera around an
object eventually returns to the starting view. An isolated ring in an arbitrary
image collection could instead reflect a sampling gap. Likewise, persistent
separation by watermark source may be strong structure with little value for
semantic retrieval.

Representation quality requires declaring what should be preserved and what
should be ignored. A useful semantic model might remove nuisance variation;
forcing it to preserve every topological feature of raw pixels could oppose
that goal. Research on topological autoencoders shows how input/latent topology
can be coupled through a training loss, but the choice of reference space remains
central to interpreting that preservation. [5]

## 4. A direct connection to feature entropy

For a feature-level dashboard, start with **component mergers and component
entropy**. This connects filtration to measurements people can already read.

At each radius ε:

1. Find connected components of the full evaluated proximity graph.
2. Measure each component's feature distribution among known annotations.
3. Average component entropy, weighting by its known-row count.
4. Compare with shuffled annotations on the same components and known-value mask.

For a categorical feature Y, this weighted quantity is H(Y | Cε), where Cε is
component membership. A practical descriptive contrast is:

**Mean shuffled component entropy − observed component entropy.**

A positive contrast means components are more label-concentrated than expected
under that shuffle. At the singleton extreme, both observed and shuffled entropy
are mechanically zero; after everything merges, both become source entropy.
The informative region is often between those extremes. Components with very
little support should not receive a strong interpretation.

For tags, use an explicitly chosen audited vocabulary and mean binary
presence/absence entropy. Shuffle whole observed tag sets if the null should
retain tag co-occurrence; preserve groups or strata when the question requires
it. For numeric features, track within-component variance relative to the full
source rather than inventing unordered entropy categories.

Unknown annotations remain unknown. Geometry may still include those rows as
connecting points; disclose that choice. Freeze vocabularies and denominators
across comparable views. Summarizing only large components changes the evaluation
population and must report the retained coverage.

Hypothetical report wording:

> Character information remains concentrated within connected regions across an
> intermediate range of radii, beyond the shuffled-label reference. This supports
> organization at several scales; it does not establish linear recoverability.

That sentence would need actual values, support and shuffle results. This is a
proposal, not a finding about your current dataset. The entropy curve accompanies
H₀ structure; it is not a persistence diagram itself.

## 5. Why topology and linear probes answer different questions

Persistent homology describes structural features under a chosen filtration.
Linear probes ask whether a weighted combination of coordinates predicts a
specific observed target on held-out data.

Two embeddings can have similar unlabeled persistence diagrams while assigning
semantic labels to points differently. Their probe or retrieval performance can
therefore differ. Similar diagrams do not guarantee correspondence between
particular items, faithful distances or preserved labels.

Even a smooth invertible deformation can change linear separability while
retaining underlying topological type. Metric-based persistence diagrams can still
change under that deformation; topological type and diagram equality differ.
Conversely, a representation can support
excellent classification without interesting H₁ structure. More loops or fewer
components is not inherently better.

Use complementary evidence: source coverage and diversity, neighborhood
agreement, filtration behavior, held-out probes and actual task performance.
Do not combine these into a universal score without a declared task and justified
weighting.

## 6. Comparing embeddings fairly

Use the same IDs, annotation masks, duplicate/group policy, sample sizes,
filtration definition and homology dimensions. Declare the distance convention:
Rips libraries can use different radius/diameter parameterizations. [1]

Distances and barcode lifetimes depend on scale. Multiplying every coordinate
by two doubles Euclidean birth/death scales without changing which structures
occur in which order. A distance-quantile parameterization can compare relative
edge-inclusion scales, but it removes absolute distance-spread information.
Report that transformation rather than treating it as harmless calibration.

Bottleneck or Wasserstein diagram distances can summarize structural differences;
a smaller distance is desirable only when the reference structure is itself
relevant. Distinct diagrams can also summarize genuinely different useful
representations. Neither comparison supplies a semantic verdict on its own.

Compute primary structural diagnostics in the original embedding space. A PCA
or UMAP visualization can introduce or remove apparent gaps and connections.
Analyzing the projection separately tests the visualization's structural fidelity;
it cannot establish the original embedding's quality.

Feature-value filtrations are another option: for a numeric score, include
vertices as a threshold rises and include a simplex after its required vertices
and faces appear. That asks how the score field is arranged on a fixed complex.
Unordered categorical IDs have no meaningful numerical filtration order.

## 7. Unit normalization and its consequences

For a nonzero embedding z:

    z_unit = z / ||z||₂

This maps each vector onto the unit hypersphere. It can be applied after embedding
creation. It retains direction and discards magnitude. Nonfinite and zero vectors
require explicit handling. [6]

To make the encoder **learn under normalization**, place normalization before
the training loss so gradients account for it. SimCLR uses normalized similarities
and a temperature parameter in its contrastive objective; its results support
that combination for that training setup, not a universal benefit for every
embedding. [7]

A mathematical counterexample explains the information trade-off: differently
labeled vectors (1, 0) and (2, 0) can be separated by x = 1.5, but both normalize
to (1, 0). For boundaries through the origin, positive normalization preserves
the sign of wᵀz. With an intercept, normalization can change separability.

For a frozen linear probe, equal input lengths may reduce variation in gradient
scale due to example magnitude. They do not remove coordinate correlations or
guarantee good conditioning. For an encoder, normalization is sensitive near
zero because its derivative involves inverse norm; epsilon-based implementations
avoid division by zero. A normalized input also does not bound classifier weights.

The audit currently uses coordinate-wise StandardScaler before logistic/Ridge
probes. This is different from row normalization: it learns coordinate means and
standard deviations from the training data. Applying it after unit normalization
generally moves vectors off the sphere again. [8]

| Pipeline to compare | Question |
|---|---|
| Raw embeddings → coordinate standardization → linear probe | What can the current pipeline recover? |
| Unit normalization → linear probe | What can direction alone recover with spherical inputs? |
| Unit normalization → coordinate standardization → linear probe | Does coordinate rescaling help access direction-only information? |
| Embedding norm alone → simple probe | Which targets have information in magnitude? |

Keep splits identical and tune regularization inside the training data separately
for each pipeline. Report metric changes per feature. Magnitude may contain useful
information or nuisance variation; test rather than assume which applies.

## 8. How normalization interacts with filtration

For unit vectors u and v:

    d_chord(u, v)² = 2 − 2(u · v)
    d_angle(u, v) = arccos(u · v)

Chordal distance is ordinary Euclidean distance on normalized vectors. Both
chordal and angular distance increase monotonically as cosine similarity falls.
They yield the same edge order, with different threshold coordinates. Their
filtrations correspond under that monotone reparameterization, but numerical
barcode lengths and diagram distances change.

Cosine dissimilarity, 1 − cos, is not generally a metric because it can violate
the triangle inequality. A finite dissimilarity matrix can still define a nested
Rips-style filtration. However, metric-based guarantees should be applied using
a genuine metric and their stated assumptions. Chordal distance is a natural
choice when examining directional structure.

Consequently, raw cosine analysis and unit-normalized chordal analysis inspect
the same directional relationships, while raw Euclidean analysis also uses
magnitude. Normalization can substantially change that latter filtration.
Centering or coordinate scaling before normalization changes directions too;
record the complete preprocessing order.

Being on a high-dimensional sphere does not imply that a finite embedding sample
has a meaningful H₁ loop. Sampling geometry and the occupied portion of the sphere
matter. Alignment of known positive pairs and distributional uniformity are
additional hyperspherical diagnostics studied in contrastive learning; neither
alone validates preservation of your specific features. [9]

## 9. Feasibility at eight million rows

Filtration avoids choosing one scale, but can multiply computation across many
scales and simplices. Practical feasibility depends on graph density, requested
homology dimensions, sampling and threshold limits. [1, 2]

| Analysis | Practical large-data approach | Main limitation |
|---|---|---|
| H₀ and component feature summaries | Sort weighted edges; union-find mergers; merge annotation counters | Building a suitable graph is still expensive |
| H₁ on the full population | Usually investigate samples, landmarks or selected local regions first | Triangle/clique structure can grow dramatically |
| H₂ and higher | Small controlled studies | Combinatorial cost and interpretation burden |
| Many features across one filtration | Reuse the geometric structure; attach counters/statistics | Per-feature labels, missingness and null repetitions add work |
| A/B comparison | Shared sampled IDs and repeated matched samples | Sampling error, graph approximation and runtime |

An 8m × 8m dense float32 distance matrix alone would require about **256 TB**
(decimal units), before any topological calculation. A 100-neighbor graph has
800 million directed entries, about 3.2 GB of int32 indices before distances,
symmetrization or higher-dimensional structure. These are storage arithmetic,
not implementation benchmarks.

For H₀, finite merger scales correspond to minimum-spanning-tree edge weights
of the supplied weighted graph, making union-find a useful starting point. A
truncated ANN graph can miss mergers and is not automatically equivalent to the
complete-metric filtration. Check neighbor recall against exact search on a
controlled sample. Missing edges can also create spurious H₁ cycles.
Even computing landmarks against every original point can be expensive. [10]

A structure that remains alive when a threshold-limited calculation stops may
be censored rather than permanently nontrivial. Report the cutoff explicitly.
Sparse Rips approximations and landmark methods have explicit approximation
settings; arbitrary ANN sparsification does not inherit their guarantees. [1, 10]
Validated edge-collapse reductions can instead preserve the input flag filtration’s
persistence exactly while reducing its size. [11]

For your use case, start with repeated matched samples, H₀ merger summaries and
feature entropy contrasts. Add H₁ where there is a question worth answering.
Use Ripser.py or GUDHI for controlled experiments, then benchmark the exact
sample size, graph density, maximum scale and dimensions before expanding.

## 10. A practical evaluation sequence

1. Declare useful targets, relevant invariances and the downstream task.
2. Compare raw and direction-only probe pipelines on identical held-out groups.
3. Choose a metric and a nested filtration; analyze original-space embeddings.
4. Pair H₀ mergers with per-feature entropy/variance and suitable label nulls.
5. Investigate H₁ only where labels, controlled variation or examples support a hypothesis.
6. Compare models with matched IDs, scale conventions and repeated samples.
7. Write conclusions that name the evidence: structural persistence, annotation
   organization, probe recoverability or task success.

A good report could say that a feature remains organized across several scales
and is recoverable by a linear pipeline. A persistent barcode alone cannot say
that the embedding is a good semantic representation.

## Resources

1. [GUDHI: Rips complex definition, sparse approximations and edge collapse](https://gudhi.inria.fr/python/latest/rips_complex_user.html).
2. [Otter et al.: A roadmap for the computation of persistent homology](https://link.springer.com/article/10.1140/epjds/s13688-017-0109-5).
3. [Cohen-Steiner, Edelsbrunner and Harer: Stability of persistence diagrams](https://math.uchicago.edu/~shmuel/AAT-readings/Data%20Analysis%20/Stability.pdf).
4. [Stolz: Outlier-robust subsampling techniques for persistent homology](https://jmlr.org/papers/v24/21-1526.html).
5. [Moor et al.: Topological autoencoders](https://arxiv.org/abs/1906.00722).
6. [scikit-learn: row-wise unit normalization](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html).
7. [Chen et al.: SimCLR](https://arxiv.org/html/2002.05709v3).
8. [scikit-learn: coordinate standardization](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), and [PyTorch: epsilon-based normalization](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html).
9. [Wang and Isola: Alignment and uniformity on the hypersphere](https://proceedings.mlr.press/v119/wang20k.html).
10. [Ripser.py: dimensions, thresholds, sparse inputs and landmark options](https://ripser.scikit-tda.org/en/latest/reference/stubs/ripser.ripser.html).

11. [GUDHI: persistence-preserving edge collapse](https://gudhi.inria.fr/python/latest/edge_collapse.html).

The feature-dashboard and comparison procedures in this guide are suggested
analysis designs. Hypothetical explanations are not measured findings, and the
links above support underlying methods rather than certify these designs for
your particular data.
