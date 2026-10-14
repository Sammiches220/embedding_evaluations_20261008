# Corpus-Relative Geometric Topology Methodology

## Purpose

This methodology treats a large corpus of embeddings as a corpus-relative geometric ecology. It asks whether known and new points occupy stable, non-interchangeable roles within that ecology, and how navigable or difficult different regions of the landscape are.

The method is deliberately validation-oriented. It does not use human labels, metadata, a learned detector, or a training objective. It first maps the geometry that already exists, then measures what can be inferred from that map.

The central asymmetry is:

```text
analyze a fixed corpus              expensive but feasible offline
score a known embedding              relatively cheap after mapping
score a new embedding                feasible through out-of-sample attachment
see whether terrain is difficult      relatively easy once the map exists
predict a precise relation            more expensive
classify an external property         target-dependent and not label-free
create a desired geometric role       difficult inverse problem
create a new persistent neighborhood  more difficult still
```

The word “soul” is used here only as shorthand for a point’s geometric role: its situatedness, distinctiveness, stability, and structural non-interchangeability within a particular corpus. The method does not claim to detect intrinsic meaning, authorship, intent, consciousness, or moral value.


## Earlier conceptual findings and scope

### The motivating problem

The broader motivation is to locate and preserve value-bearing complexity in an information environment increasingly filled with synthetic material that is fluent, polished, and superficially diverse but structurally interchangeable.

The concern is not that generated media is always bad, nor that human-made media is always valuable. The concern is that large-scale generation can make it difficult to distinguish work that carries a specific situated relation to the world from outputs that occupy a broad, easily replaceable region of a learned distribution.

The geometric methodology does not attempt to resolve that cultural question directly. It isolates one technically testable component:

> Can an embedding corpus reveal stable, nontrivial, non-interchangeable relational structure without using human judgments as training labels?

### Operational dimensions of “soul-like” structure

Earlier discussion identified several dimensions that motivate the geometric proxy:

* situatedness: the point occupies a particular position in a larger ecology rather than existing as an isolated style token;
* witness: the point appears to retain evidence of a specific encounter, process, or relation rather than only a generic category;
* specificity: the point has a narrow, structured relation to multiple other points;
* accountable complexity: the point’s distinctive character is distributed across relations and scales rather than reducible to one obvious feature;
* non-interchangeability: replacing the point changes some part of the corpus geometry rather than leaving the structure effectively unchanged.

Only the last three have direct geometric proxies. Situatedness can be approximated through corpus-relative position and persistence. Specificity can be approximated through multiscale relations and unusual but coherent neighborhoods. Accountable complexity can be approximated through stable high-dimensional structure and compression response. Witness and authorship, however, are not recoverable from embeddings alone.

The resulting methodology should therefore use “soul-like” as a disciplined operational metaphor, not as a claim that geometry observes the full human meaning of an artifact.

### Why the target is not generic novelty

A random outlier is not necessarily valuable. It may be noise, an encoder failure, or an adversarial point. The desired structure is better described as:

```text
coherent enough to belong somewhere,
distinct enough not to be interchangeable,
complex enough not to collapse to one visible feature,
stable enough to survive reasonable perturbations,
and relational enough to alter how neighboring regions connect.
```

This is why the signature must combine cohesion with novelty, persistence with compression response, and local distinctiveness with structural influence. No single statistic is sufficient.

### Relation to alignment and uniformity

A useful conceptual precedent is the distinction between alignment and uniformity in representation geometry. Alignment preserves meaningful proximity between related examples, while uniformity prevents the representation from collapsing into a small undifferentiated region. See [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/abs/2005.10242).

The present methodology takes a different direction. It does not train an encoder to optimize alignment or uniformity. It treats the resulting embedding geometry as a fixed object and asks what structure remains visible inside it.

The distinction can be summarized as:

```text
alignment/uniformity: properties used to shape a representation
corpus-relative topology: properties used to inspect an existing representation
```

Uniformity alone would not establish value: a cloud of random, evenly distributed points could be highly uniform. Alignment alone would not establish non-interchangeability: a smooth cluster of near-duplicates could be highly aligned. The proposed signature examines the interaction between local coherence, global placement, multiscale persistence, and resistance to trivial compression.

### The ecology rather than the isolated point

The relevant object is not an embedding considered in isolation. It is an embedding situated among other embeddings.

A point may be interesting because it:

* occupies a stable but sparsely populated branch;
* bridges two otherwise separate regions;
* preserves a high-dimensional relation lost by coarse compression;
* changes the connectivity or interpretation of nearby points;
* acts as a specific witness to a region without becoming a hub for everything.

Conversely, an apparently unusual point may be uninteresting if its geometry is unstable, disconnected, or easily replaced by many equivalent points.

This ecological framing also explains why the analysis is corpus-relative and nonstationary. Adding or removing data can change neighborhoods, persistence, boundaries, and structural influence. The map is therefore a versioned measurement of an ecology, not a permanent certificate attached to an item.

### Why synthetic smoothness is a relevant contrast

Large generative systems often produce outputs that are locally plausible, stylistically coherent, and easy to place near familiar regions of an embedding space. Such outputs can be high quality while still being geometrically redundant.

The methodology is not designed to detect “AI” as a category. It instead asks whether an item’s distinctive role survives tests that are difficult to satisfy through superficial similarity alone:

* relation to multiple anchors;
* persistence across neighborhood scales;
* stable boundary or bridge position;
* coherent local complexity;
* response to dimensional compression;
* measurable influence on nearby structure.

This is a stronger requirement than matching a reference style or target vector. It evaluates whether the output occupies a specific role in the ecology rather than merely resembling a recognizable mode.

### Validation rather than detector training

The system should remain a validation and navigation layer, not become a trained “soul detector.” This preserves the central asymmetry:

* the corpus is analyzed once using expensive global procedures;
* known and new embeddings are scored against the resulting map;
* no human label is required to establish the geometric profile;
* no scalar objective is presented as a universal value function;
* generation is treated as a separate, difficult inverse problem.

A learned classifier could eventually imitate any public finite benchmark. A validation protocol based on hidden resampling, held-out scales, compression conditions, and graph perturbations makes proxy optimization harder, but it cannot turn a corpus-relative geometric measurement into an absolute truth.

## 1. Mathematical object

Let the frozen corpus be:

\[
C=\{z_1,\ldots,z_n\}, \qquad z_i\in\mathbb{R}^d, \qquad \|z_i\|=1.
\]

The embeddings may represent images, text, audio, or other media. The encoder and normalization procedure are fixed before analysis.

For normalized embeddings, a natural distance is cosine distance:

\[
d(z_i,z_j)=1-z_i^\top z_j.
\]

The exact distance is less important than consistency. All measurements must use the same distance, normalization, and corpus version.

A point’s signature is corpus-relative:

\[
\phi_C(z_i),
\]

not an absolute property of the point. If the corpus changes, its neighborhood, component, boundary position, and even its apparent uniqueness may change.

## 2. Offline construction of the geometric map

### 2.1 Multiscale neighborhood graphs

Construct approximate or exact nearest-neighbor graphs at several neighborhood sizes:

\[
k\in\{k_1,k_2,\ldots,k_m\}.
\]

For each scale, retain:

* neighbor identities;
* distances and rank positions;
* reciprocal-neighbor status;
* local density;
* edge persistence across scales;
* connected-component membership.

This produces a graph family:

\[
G_{k_1},G_{k_2},\ldots,G_{k_m}.
\]

A relation that appears at only one scale is weak evidence. A relation that persists across many scales is stronger evidence of stable organization.

### 2.2 Persistent graph structure

At minimum, track persistent connected components. These reveal:

* dense basins;
* bridges between basins;
* emerging clusters;
* isolated regions;
* branches and boundaries;
* points that connect otherwise separate regions.

The operational map should not be only a two-dimensional visualization. A projection can distort distances and topology. The usable map is a sparse, multiscale graph containing stable edges, landmarks, components, branch points, and boundaries.

Higher-dimensional persistent homology can be added, but it becomes expensive quickly. Persistent connected components and sparse graph structure are a practical starting point.

### 2.3 Landmarks and navigation skeleton

Select landmarks that preserve:

* component coverage;
* boundary regions;
* branch points;
* high-dimensional regions;
* compression-sensitive regions;
* structurally influential points.

Connect landmarks using persistent graph paths. This creates a smaller navigation skeleton that supports coarse terrain assessment and routing without repeatedly processing the full corpus.

## 3. Per-point geometric signature

For each known point, compute a profile rather than a single score:

\[
\phi_C(z_i)=
[\text{cohesion},\text{density},\text{local dimension},
\text{persistence},\text{boundary position},\text{hubness},
\text{compression response},\text{stability},\text{influence}].
\]

### 3.1 Local cohesion

For neighborhood scale \(k\), define:

\[
C_k(z_i)=\frac{1}{k}\sum_{j\in N_k(i)}d(z_i,z_j).
\]

Low average distance indicates local cohesion. Cohesion alone is not evidence of non-interchangeability: a repetitive dense cluster may be cohesive but highly redundant.

### 3.2 Effective local dimension

Estimate how distances expand around a point. One family of estimators uses ordered neighbor distances:

\[
\widehat{D}(z_i)
\approx
\left[
\frac{1}{k-1}
\sum_{r=1}^{k-1}
\log\frac{T_k}{T_r}
\right]^{-1},
\]

where \(T_r\) is the distance to the \(r\)-th nearest neighbor.

This estimates local geometric complexity, not an intrinsic metaphysical dimension. Low dimension suggests a constrained local direction; high dimension suggests many independent ways to move while remaining locally plausible.

### 3.3 Persistence and component membership

Record whether a point remains in the same component or structural region across scales and resamples. A point may be:

* deeply embedded in a stable basin;
* visible only at a narrow scale;
* a bridge between regions;
* part of a branch or frontier;
* effectively isolated.

### 3.4 Boundary and bridge position

Use graph conductance, edge disagreement, local asymmetry, and approximate betweenness to distinguish:

* interior points with many interchangeable neighbors;
* boundary points with asymmetric neighborhoods;
* bridge points whose removal weakens connectivity.

### 3.5 Hubness

Define reverse-neighbor count:

\[
H_k(z_i)=\left|\{j:i\in N_k(j)\}\right|.
\]

High hubness can reflect high-dimensional distance concentration rather than meaningful centrality. It should normally be treated as a correction or warning signal, not as evidence of value.

### 3.6 Compression response

Apply controlled dimensional reductions or coordinate masking. Compare full and compressed neighborhoods:

\[
R_r(z_i)=
\operatorname{Jaccard}
\left(N_k^{\mathrm{full}}(i),N_k^{(r)}(i)\right).
\]

Important profiles include:

* robust structure that survives aggressive compression;
* fragile structure dependent on a few dominant directions;
* structure that is stable under mild compression but loses its distinctiveness under aggressive compression;
* points whose neighbors change substantially under compression.

The last pattern can indicate high-dimensional residual structure, although it can also indicate noise.

### 3.7 Structural influence and non-interchangeability

Estimate what changes when a point is removed:

\[
I_i=\Delta(\text{connectivity},\text{persistence},\text{neighbor assignments}).
\]

A point with high influence may:

* connect otherwise separate regions;
* support a persistent branch;
* represent a local direction with few substitutes;
* alter the neighborhood geometry around it.

Exact leave-one-out analysis may be expensive. A practical approximation uses local graph updates, landmark influence, and edge criticality.

## 4. Terrain and navigability layer

A key addition is that the broad difficulty of traversing the landscape can be easy to see once the global map exists. This is analogous to looking at physical terrain and immediately recognizing plains, mountains, cliffs, islands, bottlenecks, or mazes without actually traversing them.

### 4.1 Physical terrain analogies

| Physical terrain | Embedding analogue |
|---|---|
| Flat plain | Dense, low-dimensional, highly redundant region |
| Mountainous terrain | High local dimension or strong metric variation |
| Cliff | Abrupt gap or unstable neighborhood transition |
| Narrow pass | Low-conductance bottleneck between regions |
| Maze | High path ambiguity and route entropy |
| Isolated island | Disconnected or weakly connected component |
| Ridge or boundary | Persistent separation between neighborhoods |
| Fog or unstable ground | Geometry that changes under resampling or compression |

These are analogies for measurable graph properties, not claims that embeddings literally have physical terrain.

### 4.2 Traversability profile

For the persistent graph, estimate a corpus- or region-level profile:

\[
T(C)=
[\text{connectivity},\text{path stretch},\text{bottleneck strength},
\text{branching},\text{route entropy},\text{stability}].
\]

Useful measurements include:

* connectivity: what fraction of points can reach one another through persistent edges;
* path stretch: how much longer stable graph paths are than direct distances;
* bottleneck strength: how narrow the bridges are between regions;
* branching: how many persistent directions leave a region;
* route entropy: how many nearly equivalent paths exist;
* stability: whether terrain features survive subsampling and compression.

For a persistent edge \((i,j)\), a terrain-aware cost can be defined as:

\[
c(i,j)=d(i,j)\frac{1}{p(i,j)+\epsilon},
\]

where \(p(i,j)\) is the fraction of analyzed scales at which the edge remains valid.

Short, persistent paths are easy to traverse. Long or unstable paths are expensive. A region can therefore be visually or numerically identified as open, rugged, fragmented, bottlenecked, or maze-like without training a predictor.

### 4.3 Coarse terrain versus exact route planning

The distinction is important:

```text
seeing that a region is difficult: relatively easy
estimating broad terrain statistics: relatively easy
finding a stable route: moderately expensive
finding the exact best route: more expensive
predicting a precise destination: more expensive still
```

Humans can see that a mountain range is difficult to cross. They cannot necessarily determine the exact best route instantly, and they certainly cannot create a new mountain range with the same geology. The same asymmetry applies to embedding landscapes.

## 5. Out-of-sample inference for new embeddings

Given a new embedding \(z_{\mathrm{new}}\), use the already-built corpus map rather than rebuilding the entire analysis.

### Procedure

1. Normalize the new embedding using the frozen corpus procedure.
2. Query the approximate-nearest-neighbor index.
3. Attach the point to nearby persistent landmarks.
4. Estimate local cohesion, dimension, persistence, boundary position, hubness, and compression response.
5. Compare its signature with known points and regions.
6. Determine whether it is interior, boundary-like, bridge-like, novel, or isolated.

This produces:

\[
\phi_C(z_{\mathrm{new}}).
\]

The result means: how does the new point participate in the existing ecology?

A single point cannot establish a persistent new neighborhood by itself. It can be evaluated relative to existing structure, but new topology requires multiple coordinated points.

A full recomputation is only needed when:

* many new points arrive;
* the corpus changes substantially;
* the point may create a new persistent component;
* global rankings or topology must be exact.

## 6. Label-free prediction cost

The methodology can estimate how expensive prediction would be without training a predictor or using semantic labels.

### 6.1 Geometric navigation

Measure:

* graph expansions required to reach a target region;
* stable path length;
* number of plausible paths;
* landmark coverage;
* route sensitivity to perturbation.

### 6.2 Geometric prediction

Use held-out recovery tests. Temporarily hide:

* a point;
* an edge;
* a local neighborhood;
* a branch connection.

Then use the remaining graph to estimate where the hidden structure belongs. Record:

* rank of the correct re-entry location;
* number of candidate regions;
* graph expansions required;
* distance between predicted and actual location;
* stability of the recovered route.

This predicts geometric relations, not external meaning.

### 6.3 Ambiguity and effective candidate count

If a query leaves many possible destinations, represent that ambiguity with entropy:

\[
N_{\mathrm{eff}}=\exp(H),
\]

where \(H\) is the entropy of the candidate-location distribution.

High \(N_{\mathrm{eff}}\) means that many destinations remain plausible and prediction is geometrically expensive. Low \(N_{\mathrm{eff}}\) means that the geometry strongly constrains the answer.

### 6.4 Description length

A navigable region can often be described compactly:

```text
component 3 → branch B → local offset 4
```

A difficult region may require listing many unrelated relationships. Approximate description length therefore provides a label-free proxy for structural complexity:

\[
\text{navigation cost}
\approx
\text{bits required to identify the next valid region}.
\]

## 7. Relation to semantic classification

Without labels, geometry cannot determine whether a particular semantic classification task is easy. A future class boundary may be:

* aligned with one persistent basin;
* spread across many unrelated branches;
* independent of the embedding geometry.

The map can estimate potential task difficulty, but not actual semantic accuracy.

If labels are later introduced, task difficulty can be evaluated through:

* graph cut complexity;
* boundary length;
* label entropy within neighborhoods;
* number of persistent regions crossed;
* sample coverage required.

The label-free phase asks only:

> How much geometric complexity would a future classifier have to represent if its target aligned with this landscape?

## 8. Why generation is much harder

Inference asks:

\[
z\longrightarrow\phi_C(z).
\]

Generation asks for an input \(x\) satisfying:

\[
\phi_C(f(x))\approx\phi^*.
\]

The generator must produce an embedding with the desired relations to many existing points across multiple scales. A strong target may require matching:

\[
\begin{aligned}
&d(f(x),z_{j_1}),\ldots,d(f(x),z_{j_k}),\\
&\text{neighbor ranks},\\
&\text{component membership},\\
&\text{local dimension},\\
&\text{persistence},\\
&\text{compression response},\\
&\text{non-hubness},\\
&\text{structural influence}.
\end{aligned}
\]

Matching one target vector is much weaker than matching its geometric role.

A hypothetical generator objective might combine:

\[
\mathcal{L}(x)=
\lambda_1\mathcal{L}_{\mathrm{relations}}
+\lambda_2\mathcal{L}_{\mathrm{signature}}
+\lambda_3\mathcal{L}_{\mathrm{stability}}
+\lambda_4\mathcal{L}_{\mathrm{anti\text{-}hubness}}.
\]

This is difficult because:

* nearest-neighbor membership changes discontinuously;
* persistence changes at threshold events;
* local-dimension estimates are noisy;
* the target depends on the entire corpus;
* improving some relations can damage others;
* the embedding function may admit many-to-one and adversarial preimages;
* optimizing the evaluator may produce media that exploits the encoder rather than meaningfully occupying the intended role.

If the evaluator is public and fixed, a sufficiently capable generator can eventually imitate its proxies. Held-out anchors, scales, subsamples, and perturbations make this harder but do not create mathematical impossibility.

A single new point also cannot reproduce a whole persistent neighborhood. Recreating a rich local role may require generating a coordinated set of points whose mutual relations form the desired terrain.

## 9. Difference from style training

Style training generally targets local or distributional properties such as:

* color and texture statistics;
* surface appearance;
* composition;
* feature similarity to a reference;
* broad visual regularities.

The present target is a corpus-conditioned relational role:

```text
how the item relates to many other items,
across several scales,
under graph transformations,
under compression,
and under perturbation.
```

A stylistic imitation can look close to a reference while occupying an ordinary, interchangeable region of the corpus. A geometric-role imitation must reproduce a pattern of proximity, separation, dimensionality, persistence, and structural influence.

It is therefore similar to style training only superficially. The desired output is not merely a recognizable appearance; it is a difficult-to-substitute position in an existing semantic ecology.

## 10. Computational profile

### Offline corpus mapping

Exact pairwise distance construction costs approximately:

\[
O(n^2d)
\]

with quadratic storage if all distances are retained.

Practical systems use approximate-nearest-neighbor methods and retain a sparse graph with roughly \(O(nk)\) edges. Multiscale analysis multiplies graph work by the number of neighborhood scales. Persistent connected components over the sparse graph are tractable; higher-dimensional complexes can become combinatorial.

### Known-point lookup

After indexing, known-point lookup is effectively a table or graph query.

### New-point lookup

A new point requires approximate-nearest-neighbor search plus local calculations, approximately:

\[
\text{ANN lookup}+O(km),
\]

for \(k\) neighbors and \(m\) scales, assuming graph statistics are cached.

### Generation cost

If each candidate generated by a model must be evaluated against the corpus, the cost multiplies by the number of candidates and optimization iterations. If the target includes global topology rather than local relations, repeated corpus-wide or landmark-wide evaluation may be required.

The difficulty is not necessarily that every single calculation is impossible. It is that reliable high-dimensional coverage, multiscale stability, and corpus-wide relational matching create an unfavorable combination of sample complexity, search cost, and discontinuous objectives.

## 11. Validation protocol

A robust implementation should use the following sequence.

### Phase A: freeze the corpus

Record:

* encoder version;
* normalization procedure;
* distance function;
* corpus membership;
* numerical precision;
* graph-construction parameters.

### Phase B: construct the map

Build:

* approximate-nearest-neighbor index;
* multiscale kNN graphs;
* persistent sparse graph;
* landmarks;
* component and branch summaries.

### Phase C: compute signatures

For every known point, calculate:

* cohesion and density;
* effective local dimension;
* persistence;
* boundary and bridge measures;
* hubness;
* compression response;
* resampling stability;
* structural influence.

### Phase D: compute terrain summaries

For each region, calculate:

* connectivity;
* path stretch;
* bottleneck strength;
* branching;
* route entropy;
* stable reachability;
* compression and resampling sensitivity.

### Phase E: run null and stability controls

Repeat selected measurements under:

* random subsampling;
* neighborhood-scale changes;
* dimensional compression;
* matched null clouds;
* leave-one-out approximations;
* alternate compatible embedding views, if available.

### Phase F: evaluate new points

Attach new embeddings through the landmark graph and report their signatures with uncertainty intervals rather than a single verdict.

### Phase G: evaluate generation claims separately

If generation is tested, hold out:

* some anchors;
* some scales;
* some resampling conditions;
* some compression settings.

A generated point should not be considered successful merely because it matches one target embedding or one visible statistic.

## 12. Recommended outputs

The system should produce several artifacts rather than one “soul score.”

### Point profile

```text
cohesion
local dimension
persistence
boundary position
hubness
compression response
stability
structural influence
```

### Region profile

```text
terrain type
connectivity
branching
bottlenecks
route ambiguity
compression sensitivity
stability
```

### Navigation report

```text
entry cost
stable route cost
candidate ambiguity
landmark coverage
route sensitivity
```

### Generation-difficulty report

```text
number of relational constraints
number of scales involved
anchor coverage required
stability requirements
estimated evaluation cost
risk of encoder exploitation
```

## 13. Limitations

* The signature is relative to the corpus and encoder.
* High-dimensional geometry may reflect embedding artifacts.
* Compression sensitivity can indicate meaningful residual structure or noise.
* A graph can show structural novelty without showing human value.
* A persistent component is not automatically semantically meaningful.
* A new point can be scored relative to existing structure but cannot create rich topology alone.
* Label-free analysis cannot establish the difficulty of an arbitrary external classification task.
* Any fixed, public evaluator can eventually be optimized or overfit.
* “Non-interchangeable” is a geometric statement, not a claim about cultural, ethical, or historical importance.

## Conclusion

The methodology has a deliberate division of labor:

```text
map the corpus globally;
measure point-level geometric roles;
see broad terrain difficulty;
attach and score new embeddings;
estimate the cost of future prediction;
reserve generation as the difficult inverse problem.
```

The key result is not that a model can identify an objective metaphysical soul. It is that a frozen corpus can reveal stable differences between redundant points, boundary points, bridge points, high-dimensional points, and structurally influential points.

Once the global map is built, the broad navigability of the landscape can be easy to see—much as physical terrain can be judged from a map or a glance. Exact traversal, precise prediction, and especially deliberate generation of a new point with the same multiscale relational role remain progressively more expensive.
