# Prompt: Explain and work with the implemented Geometric Ecology methodology

You are helping me understand, assess, reproduce, or extend **Geometric Ecology**, a Python feasibility project for mapping a fixed corpus of embeddings as a readable geometric landscape.

Treat the following as the implemented baseline, **version 0.5.0**. It describes the implementation rather than promising every capability in the broader research proposal. You should be able to explain the concepts from this prompt alone. If project files are also supplied, inspect the relevant source before claiming exact implementation details or making changes.

**Requested follow-up task:** [Insert my question or requested extension here. If this is blank, explain the methodology and its concepts, using concrete examples and distinguishing established measurements from hypotheses. Do not automatically rebuild the project.]

## 1. The objective and the interface requirement

Treat embeddings as a **corpus-relative geometric ecology**: a point has a relational role because of its distances, neighbors, connected component, local variation, stability, and contribution to graph connectivity.

The purpose is to make these properties readable as a map. A person should be able to recognize concentrated regions, gaps, restricted crossings, alternative routes, boundaries, and unstable neighborhoods without first answering questions or supplying supporting evidence.

The interface is the **computed map itself**. It is not a questionnaire, evidence-entry form, or human evaluation workflow. Supporting measurements explain the map; they do not ask the person to construct its conclusions.

The default representation is **strictly two-dimensional**. Geometry must be communicated through measured color fields, labelled contours, planar neighborhood glyphs, and actual connections. A third display dimension must not be necessary. Optional relief and a rotatable 3D mesh provide additional reference views.

Keep four ideas separate:

1. **Measured geometry:** distances, neighbor identities, covariance, graph paths, cuts, and response to controlled changes.
2. **Cartography:** a schematic arrangement and interpolation that makes those measurements legible.
3. **Terrain vocabulary:** analogies that summarize particular measured properties.
4. **Human meaning:** a task- and context-dependent interpretation that geometry alone does not establish.

A geometrically complicated point is not automatically valuable, human-authored, culturally important, semantically unique, or difficult to reach. There is no single universal score for those properties.

## 2. Core vocabulary

| Concept | Meaning in this project |
|---|---|
| Corpus | The fixed collection of reference embedding vectors and aligned item identities. |
| Corpus-relative | Measurements depend on which reference items and encoder are used. Changing the corpus can change a point's apparent role. |
| Neighborhood | The nearest reference items under the declared original-space distance. It is not merely a group that looks close in the plotted map. |
| Local geometric signature | A collection of point measurements, rather than a single ranking: cohesion, dimension, support, boundary position, hubness, compression response, sensitivity, and structural influence. |
| Retained graph | The reciprocal-neighbor edges that meet the configured support threshold across neighborhood scales. |
| Connected component | A set of vertices mutually reachable through retained edges. |
| Navigation cell | A graph-distance Voronoi partition used for routing and summarization. It is not automatically a semantic cluster or persistent basin. |
| Landmark | A representative anchor used for coverage, graph-distance layout, or query attachment. |
| Density | A proxy derived from original-space neighbor radius. It is distinct from crowding on the displayed map. |
| Dimensionality | Several related but different notions: ambient vector length, local distance-growth dimension, and effective covariance directions. |
| Stability | Agreement under a stated change, such as reference resampling, vector perturbation, or compression. Each change defines a different test. |
| Structural influence | How removing a vertex changes connectivity in the fixed retained graph. |
| Traversability | Reachability, costs, stretch, bottlenecks, and route alternatives in the declared graph. |
| Geometric substitutability | Nearby alternatives under an explicit distance tolerance. It does not guarantee semantic interchangeability. |

## 3. Inputs, freezing, and distance

The input is a numeric matrix `X` with shape `(N, D)`, aligned unique item IDs, and optional display metadata. The project does not supply an encoder or train an embedding model.

Freeze and record the encoder identifier, corpus membership, ID alignment, normalization, precision, distance definition, graph parameters, and random seed. Save hashes of the normalized corpus and IDs. The supplied encoder name is recorded, not independently authenticated.

Normalize each vector to unit L2 norm and retain float32 embeddings:

\[
z_i = x_i / \|x_i\|_2.
\]

Reject nonfinite or zero-length input vectors. Duplicate vectors remain separate items; exclude a point's own **identity**, not every equal vector, when finding its reference neighbors. Ties and duplicates require explicit handling.

The original-space metric is Euclidean **chord distance** on the unit sphere:

\[
d(i,j)=\|z_i-z_j\|_2=\sqrt{2(1-z_i^\top z_j)}.
\]

This has the same neighbor ordering as cosine distance, but chord lengths, cosine distances, and squared distances are not interchangeable in radius-growth estimates or path-length calculations.

Metadata is for presentation. It does not determine the geometry, terrain labels, or layout. Synthetic mechanism labels are used only afterward for diagnostic validation.

## 4. Build multiscale reciprocal-neighbor graphs

1. Compute exact nearest neighbors in batches, excluding self identity.
2. Analyze several neighborhood sizes `k`.
3. At each size, keep an undirected edge only when each endpoint is among the other's neighbors: a **mutual/reciprocal kNN** edge.
4. For each edge, record the fraction of analyzed scales at which it occurs:

\[
s_{ij}=\frac{\#\{\text{analyzed scales containing edge }(i,j)\}}{\#\{\text{analyzed scales}\}}.
\]

5. Retain edges whose support meets the configured threshold.
6. Compute connected components and a weighted retained graph.

The route cost of a retained edge is:

\[
c_{ij}=\frac{d(i,j)}{s_{ij}+0.01}.
\]

This is a declared navigation cost that penalizes lower scale support. Keep it separate from the edge's raw chord length.

**Important distinction:** neighborhood sizes are nested. Recurrence across those sizes is **rank/scale support**, not independent evidence of robustness. Some source variables call these edges `stable_edges`; that name must not be treated as proof of resampling stability.

The core Python function has smaller defaults of scales `(6, 12, 24)` and four reference resamples. The shipped large example uses scales `(12, 24, 48)`, support threshold `0.5`, three reference resamples, and signature neighborhood size `24`. Distinguish configuration from a universal methodological constant.

## 5. What “topology” and “persistence” currently mean

The implementation has several distinct analyses:

- Connected components of the retained graph.
- Component memberships at the analyzed kNN scales.
- Sparse **H0** merge events from a chord-length filtration over the maximum-k mutual-edge set, using a Kruskal-style procedure.
- Separate neighborhood sensitivity tests under resampling, perturbation, and compression.

H0 tracks connected pieces merging as edges are admitted. The filtration is exact for the supplied sparse edge set; missing edges can overfragment the result relative to a complete distance complex.

Do not claim full Vietoris–Rips persistence, H1 loop persistence, independently resampled basin persistence, or a formal hierarchy of semantic branches. A visible ring or road circuit is not itself a computed persistent-homology loop.

## 6. Point measurements

### 6.1 Cohesion, density, and redundancy

For the signature neighborhood, store mean neighbor distance, k-neighbor radius, and inverse radius. The inverse-radius percentile is a corpus-relative density proxy, not a calibrated probability density.

Define a geometric near-substitute tolerance:

\[
\tau=0.1\,\operatorname{median}_{i}(r_k(i)).
\]

For each point, measure the fraction of its signature neighbors within `tau`. This is an explicit geometric tolerance, not a semantic equivalence relation.

A redundant-plain cell label requires all three conditions: median density percentile above `0.65`, median distance-growth dimension percentile below `0.65`, and median near-substitute fraction above `0.25`.

### 6.2 Local distance-growth dimension

For ordered chord radii `T_1, ..., T_k`, the implemented finite-sample-corrected estimate is:

\[
\widehat d_{\mathrm{radius}}=
\frac{k-2}{\sum_{r=1}^{k-1}\log(T_k/T_r)}.
\]

This estimates local distance growth, with assumptions and finite-sample limitations. Degenerate radii, duplicates, ties, curvature, and small neighborhoods can make it unreliable or undefined.

It is **different from** the covariance participation ratio below.

### 6.3 Local covariance directions: the main visual dimensionality measure

For a unit-normalized query `q` and its neighbors `z_j`, construct offsets and remove their sphere-normal components:

\[
u_j=z_j-q,
\qquad
v_j=u_j-(u_j^\top q)q.
\]

Center the tangent offsets over the neighborhood. Compute their singular values, or equivalent eigenvalues of their local Gram matrix. If their variance energies are `lambda_a`, define:

\[
p_a=\frac{\lambda_a}{\sum_b\lambda_b}.
\]

The implementation computes these spectra for every corpus point in batches. It also stores richer local views at navigation-cell representatives.

Three useful summaries are:

\[
d_{\mathrm{eff}}=\frac{1}{\sum_a p_a^2},
\qquad
h_2=1-(p_1+p_2),
\qquad
m_{90}=\min\left\{m:\sum_{a=1}^{m}p_a\ge0.9\right\}.
\]

- `d_eff` is the **effective number of covariance directions**, or participation ratio. It need not be an integer.
- `h_2` is the fraction of local covariance omitted by the first two axes. It is a variance fraction, not a dimension count.
- `m_90` counts the axes needed to retain 90% of the measured variation.

For example, two equally weighted directions give effective count 2; twelve equally weighted directions give effective count 12. Unequal weights can produce intermediate values.

The observable centered tangent covariance rank is at most `min(k-1, D-1)`. A neighborhood sample cannot establish dimensions it does not resolve. A zero-variance neighborhood has no defined normalized spectrum; do not invent directions for it.

Local PCA axes are statistical directions, not named semantic concepts. Different neighborhoods' local PCA frames have no common global semantic orientation.

### 6.4 Other signature measurements

The project also records:

- **Boundary position:** fraction of forward neighbors assigned to other navigation cells; partition-dependent.
- **Hubness:** reverse-neighbor count; popularity as a neighbor is not a value score.
- **Sampled connector centrality:** shortest-hop-path betweenness estimated from up to 24 source vertices; not exact weighted all-pairs centrality.
- **Structural influence:** connectivity loss from vertex deletion in the fixed graph; no neighbor refill or full topology rebuild.
- **Local next-hop ambiguity:** an effective count from normalized inverse incident-edge costs; not entropy over complete routes.

## 7. Stability and compression are separate tests

### Reference resampling

Repeatedly retain about 80% of reference items, query with the original vectors, exclude self identity, and compare neighbor identities.

Let `A` be the original neighbor set, `R` the retained reference set, and `B` the new k-neighbor set within that reference subset. The backend comparison is:

\[
J_{\mathrm{subset}}=\frac{|(A\cap R)\cap B|}{|(A\cap R)\cup B|}.
\]

This conditions the baseline on retained references, but refilling a k-neighbor list can still change the denominator. It is not a population confidence interval or a fully corrected probability of geometric failure.

The persistent neighborhood strip uses a separate **raw-set Jaccard** for its representative subset comparison; that comparison includes missing original references. Do not silently equate the strip value with the backend conditional field.

### Dimensional compression

Fit corpus-only PCA at approximately 25%, 50%, and 85% of the available rank, plus a coordinate half-mask control. Recompute neighbors and compare their identities with the original neighbors.

Compressed coordinates are **not renormalized**. Their Euclidean distances describe discarded-coordinate effects in the frozen metric. Reuse the fitted transformation for new queries.

The flat compression-disagreement field uses the middle available PCA setting, recorded in the exported data.

### Vector perturbation

Apply independently sampled vector noise of chord-scale magnitude approximately `0.01`, renormalize, recompute neighbors, and measure overlap. A fixed absolute perturbation can affect dense neighborhoods more strongly.

### Combined disagreement

Define losses as one minus the corresponding neighbor overlaps. The visual fog field is:

\[
L_{\mathrm{fog}}=\max(L_{\mathrm{subset}},L_{\mathrm{compression}}).
\]

Sensitivity can reveal structure, insufficient sampling, noise, or encoder artifacts. Stability does not establish semantic meaning.

## 8. Construct a readable graph atlas

The display is a **schematic atlas**, not a metric-faithful projection.

1. Partition retained components into graph-distance Voronoi navigation cells using farthest-point seeds. Preserve small components separately. Partition construction uses retained-edge chord lengths.
2. Choose representative and coverage landmarks.
3. Use landmark graph-distance multidimensional scaling and a Nyström extension for component layouts.
4. Magnify populated basins with a monotone per-axis blend: approximately 55% normalized linear position and 45% empirical rank position on nondegenerate axes.
5. Pack substantial components into separate, nonoverlapping chart areas. Place tiny components in a labelled islet area.
6. Draw coasts, measured fields, retained-edge roads, restricted passages, and supporting local views.

Do not jitter duplicate points to manufacture geometry. Disconnected chart spacing and drawn land area are arbitrary cartographic quantities, not original embedding distances.

Coastlines summarize displayed occupancy. They do not independently discover high-dimensional decision boundaries. Different named components cannot have a retained-graph route, but multiple occupied patches of one component can be connected by actual road bridges.

Projection distortion is exposed through original-neighbor versus atlas-neighbor overlap. Raw global PCA remains available as a reference view.

## 9. Connectivity, bottlenecks, and routes

### Roads and crossing strength

A road follows paths inside two adjacent navigation cells plus one actual inter-cell crossing. Every step must be a retained point-graph edge. Its accumulated cost is recorded.

The visible regional network contains a minimum-cost spanning backbone plus selected stronger alternatives. The full regional adjacency graph remains available for ambiguity analysis. Road width summarizes the number of graph edges crossing the cell pair.

### Conductance and marked passes

For a cell pair, crossing conductance is crossing-edge count divided by the smaller incident edge volume of the two cells. Keep this pairwise quantity distinct from a cell-versus-complement cut.

The prototype marks a stricter pass when pair conductance is below `0.04`, median path-centrality percentile exceeds `0.6`, and the crossing is a bridge in the complete cell adjacency graph. Adjacent qualifying crossings are grouped into one pass label.

A regional critical crossing can contain many parallel original edges. Its marking does not imply that removing one original point or edge disconnects the corpus.

### Bounded route-set ambiguity

Route ambiguity is conditioned on a source and destination. It is not interchangeable with local graph degree.

On the complete weighted regional adjacency graph:

1. Use Yen enumeration for up to 12 shortest simple regional paths.
2. Keep paths with cost at most `1.25 × best_cost`.
3. Weight those paths by excess cost:

\[
w_r\propto\exp\left(-\frac{c_r-c_*}{0.1c_*}\right).
\]

4. Normalize the weights, then compute:

\[
H=-\sum_r w_r\log_2w_r,
\qquad
N_{\mathrm{eff}}=2^H.
\]

Report when the 12-route cap is reached. This is entropy over the enumerated bounded set, not every possible route, and it does not imply a calibrated human choice distribution.

For the continuous route field, summarize each source cell by the median effective count to up to three geometrically selected reference destinations: the largest cell, a farthest cell, and a cell farthest from those references. The displayed bundle identifies its actual source and target.

Expand regional routes into actual retained point-edge walks. A regional path is simple, but its expansion may revisit points through shared within-cell segments.

Path stretch compares accumulated raw chord length along the chosen route with direct chord distance. Weighted route cost is a different quantity. Many similarly priced routes can be useful redundancy rather than difficulty.

## 10. The eight terrain concepts

Preserve the following meanings, with their implementation qualifications:

| Physical terrain | Intended embedding analogue | Implemented evidence and qualification |
|---|---|---|
| Flat plain | Dense, low-dimensional, highly redundant region | Original density, dimension, and near-substitute fraction must be considered together. Density alone is insufficient. |
| Mountainous terrain | High local dimension or strong metric variation | Direction spectra, effective-direction contours, omitted-variance fields, and metric-variation fields; optional measured relief. Complexity is not automatic value or travel cost. |
| Cliff | Abrupt gap or unstable neighborhood transition | The present cliff detector tests unusually large original-space crossing steps. Observation-condition instability is separately visible as fog; no general unstable-transition classifier is claimed. |
| Narrow pass | Low-conductance bottleneck between regions | Actual paths converge through limited crossings; marked passes use the stricter critical-crossing rule above. |
| Maze | High path ambiguity and route entropy | Actual bounded alternative-route bundles to stated destinations, plus their entropy-equivalent count. Not local degree or all-path entropy. |
| Isolated island | Disconnected or weakly connected component | Separate retained-component charts and reachability; condition-sensitive connections can be distinguished separately. Water means no route in this retained graph, not absolute separation in the full embedding space. |
| Ridge or boundary | Persistent separation between neighborhoods | Candidate seams supported across graph scales, with crossing-distance contrast and a density dip. Navigation cells define the candidate cuts; semantic boundaries are not inferred. |
| Fog or unstable ground | Geometry changing under resampling or compression | Separate disagreement fields, combined hatching, and changed-neighbor comparisons. These are sensitivities to declared conditions. |

Prototype detection thresholds, rather than universal terrain definitions:

- **Cliff:** a retained crossing step above `2.5 ×` the geometric mean of its endpoints' first-four-neighbor median radii, with crossing/internal median-length contrast above `1.8`.
- **Ridge:** pair conductance below `0.06` at at least two-thirds of analyzed scales, crossing/internal length contrast above `1.25`, and crossing-vertex median radius above `1.5 ×` the two cells' median radius.

A uniform ring is a negative control against turning arbitrary navigation-cell cuts into ridges. A corpus is allowed to contain no examples of a terrain. Do not force all eight categories onto a map or make them a mutually exclusive classification.

## 11. Default human interface: a flat field atlas

### Nine independently selectable measurements

Either color or contours can show any of:

| Field | Scale |
|---|---|
| Variation hidden by two axes | Omitted local covariance fraction, 0–1 |
| Effective local directions | Covariance participation ratio, with the sample rank bound recorded |
| Original-space density rank | Inverse-radius percentile, 0–1 |
| Near-substitute fraction | Fraction within the declared geometric tolerance, 0–1 |
| Regional route alternatives | Median bounded effective route count, 1–12 |
| Reference-subset disagreement | One minus conditional resampling overlap, 0–1 |
| Compression disagreement | One minus moderate-PCA neighbor overlap, 0–1 |
| Metric-variation rank | Percentile of local neighbor-distance coefficient of variation, 0–1 |
| Combined geometry disagreement | Maximum subset/compression loss, 0–1 |

The local distance-growth estimator remains a separate backend measurement and optional height channel; do not relabel covariance participation ratio as that estimator.

### Four coordinated presets

| Preset | Color | Contours | Additional cue |
|---|---|---|---|
| Dimensionality | Omitted variance | Effective local directions | Planar ellipses and spectral rings |
| Density & redundancy | Original-space density rank | Near-substitute fraction | Rings retain dimensionality information |
| Routes & passes | Regional route alternatives | Original-space density rank | Actual destination-labelled route bundle and marked passes |
| Stability | Subset disagreement | Compression disagreement | Combined disagreement hatching |

Color fill, isolines, contour labels, stepped bands, and spectral rings are independently toggleable. Road, boundary, point, and connection layers remain available. Manually changing the field combination creates a custom configuration.

### Field construction and contour semantics

- Interpolate original-space measurements in displayed atlas coordinates on a 54×54 grid for each substantial component, using occupancy-weighted Gaussian smoothing.
- Mask unsupported regions; never interpolate one component's terrain across another component's gap.
- Render the flat field through bilinear interpolation in supported quads. Use shared measurement scales across components, not independent per-island normalization.
- Trace ContourPy isolines with masked arrays and `corner_mask=False`: every traced quad must have four supported corner samples.
- Label actual values. Fractions and percentiles use percent labels; effective directions and route counts use numerical labels.
- Step color bands between declared thresholds when requested.
- Let constant or unsupported fields produce no contours. Label placement can suppress crowded numerals without changing the underlying lines.
- Display undefined covariance regions neutrally and suppress unsupported covariance contours.

A contour labelled 4 for effective directions means participation ratio 4. Close contours mean rapid change in that selected field **across chart coordinates**; they do not establish an original-space gradient magnitude, a cliff detector result, or a navigation cost.

Changing flat layers never changes point positions or adds elevation, lighting, perspective, or surface normals. Plot-point opacity is not a calibrated density measure; the density field supplies the original-space measurement.

### Planar neighborhood glyphs

At selected cell representatives, draw:

1. An inner ellipse with semiaxis ratio `sqrt(p_2 / p_1)`, describing first-two-axis anisotropy in a canonical local PCA frame.
2. A surrounding ring with sector angles proportional to actual covariance variance fractions.
3. Teal sectors for the first two directions, purple for further displayed directions, and one gray aggregate for the tail beyond the first 12.
4. A numeric effective-direction count.

The glyph's display size is normalized for legibility. Its ellipse orientation is not a global semantic direction, its sector count is not the full dimension count, and it adds no graph edge. Screen-space collision filtering reduces glyph clutter while all corpus points and all-point fields remain included.

### Persistent neighborhood strip

Hover a ring or item, or select a point, to update a nonmodal strip containing actual neighbor offsets, the variance spectrum, and parallel-coordinate direction ribbons.

Each ribbon is a measured neighbor's centered score across local PCA axes. A few-direction neighborhood becomes nearly flat beyond those axes; a many-direction neighborhood continues varying.

Detailed views are precomputed at cell representatives. When another item is selected, identify the representative explicitly rather than inventing an exact spectrum for the selected item. Changed-neighbor overlays reuse the original local axes; membership changes are not fabricated motion of embeddings.

## 12. Optional relief and 3D reference

The project also retains:

- A relief view with selectable measured height channels.
- An orbitable WebGL mesh with shaded terrain triangles, wire edges, retained-graph roads, and local neighborhood envelopes.
- Local PCA-3 envelopes made from actual projected neighbor positions: 3D convex hulls for rank-three clouds, planar triangulations for rank-two clouds, and no invented surface facets for lines or coincident points.

Surface tessellation count is a rendering choice, not a dimension estimate. Local envelope facets are not traversable graph edges. Their footprints and placement above anchors are normalized display conventions, not globally aligned physical volumes.

A 4D and a 12D neighborhood can both look volumetric in three axes. Spectra, omitted variance, and ribbons are needed to reveal the difference. The flat atlas works without WebGL; 3D remains optional.

## 13. Attach new embeddings without changing the map

Normalize new embeddings with the frozen procedure and query the existing reference corpus. Reuse fitted PCA transformations, neighbor radii, regions, and landmark data.

The current scorer reports local cohesion and radius-growth dimension, compression response, reverse-neighbor insertion diagnostics, heuristic region weights, possible landmark entry paths, and reference-subsample sensitivity ranges.

Candidate-region weights are heuristic geometric masses derived from neighbor distances. They are not calibrated semantic probabilities. Inherited support or stability is explicitly a proxy. Do not claim freshly established persistence, new-point deletion influence, or a persistent new basin from one attached point.

A landmark path enters through a compatible reference anchor. Do not invent routes to disconnected landmarks. Attachment paths are heuristics relative to the frozen graph, not exact shortest routes in a fully rebuilt corpus.

New-query scoring currently uses exact reference scans, not an implemented ANN index. Rebuild when corpus changes require updated global topology or rankings. Sensitivity ranges are not confidence intervals.

## 14. Validation and demonstration evidence

The project includes a shipped **20,000-point, 64-dimensional synthetic corpus**. It is a geometric feasibility demonstration, not a real-content semantic benchmark. Its retained graph has six substantial components and 41 tiny components, and its navigation partition has 113 cells after preserving small components.

All 20,000 points remain represented. Glyph decluttering is separate from point inclusion. In 3D, depth testing can occlude individual points; an in-frame position count is not a per-pixel visibility count.

The controlled Geometry Lab separates vector fixtures from explicit graph fixtures:

- Vector fixtures use 2,000 reference vectors plus a query in 32 ambient dimensions, with 80 query neighbors and matched radius about `0.025` chord.
- In the planted 2D case, two axes retain about 100% of measured covariance; in the planted 12D case, they retain about 28.3%. Their participation ratios are approximately 2.00 and 10.02. Planted dimension and measured effective dimension are different quantities.
- A 9×9 grid and its spanning-tree comparison use identical displayed positions and shortest source-to-target cost 16. The grid reaches the 12-route enumeration cap with entropy about 3.585 bits; the tree has one route and zero bits. This isolates ambiguity from crowding and shortest cost.
- Separate controlled graphs illustrate a large crossing step, a restricted gate, disconnection, and a multiscale separator. They are explicitly constructed graph examples, not inferred kNN discoveries from the large vector corpus.
- The shipped corpus has two ridge candidates and zero cliff candidates under the implemented thresholds; those numbers are observations about that fixed example, not required output categories.

Additional checks include held-out query recovery with queries excluded from the fitting corpus, perturbation sensitivity, sparse H0 controls, graph-deletion influence checks, and null-cloud comparisons in smaller reference runs. The large run omits the additional large null builds.

Version 0.5 records **25 numerical tests**, browser checks for the planar and 3D interfaces, and an installed-package smoke test. Flat checks include exact isolines on a known field, masked-gap separation, no artificial contours for constant/invalid fields, and unchanged atlas positions and graph edges after preparing flat layers.

These checks establish numerical and interface behavior. They do not by themselves measure human comprehension, semantic quality, or task performance.

## 15. Implemented scope versus broader research ideas

| Status | Scope |
|---|---|
| Implemented | Exact batched neighbors, multiscale mutual graphs, sparse H0, point signatures, fixed-graph deletion influence, reference/compression/perturbation sensitivity, graph atlas, actual roads, pass/ridge/cliff heuristics, bounded regional route ambiguity, all-point covariance, flat fields/contours/glyphs, persistent strip, optional 3D, frozen-map query attachment, controlled validation. |
| Limited proxies | Boundary measures depend on navigation cells; sampled centrality is approximate; near-substitution is geometric; entropy depends on the bounded route set; attached-query support/stability can be inherited; higher dimension may reflect noise. |
| Not implemented | An ANN backend, full Vietoris–Rips persistence, H1 persistence, independent persistence of entire basins under resampling, exhaustive all-path entropy, alternate-encoder drift validation, full leave-one-out graph rebuilding, broad hidden-branch recovery, or a generation/inverse solver. |
| Conceptual motivation only | Matching a complete corpus-relative relational role can impose more constraints than matching one vector. The project does not prove a generation lower bound, impossibility, or calibrated difficulty score. |

The analysis/prediction/generation asymmetry is a research motivation: computing a profile of an existing point is a different problem from generating an input with a desired collection of relations. It is not a demonstrated theorem that one is universally easy and the other hard.

## 16. Computation, files, and implementation style

The backend is an exact-search reference implementation. An all-corpus neighbor pass costs approximately `O(N²D)`; multiple compression and sensitivity passes repeat work. Retained data are roughly `O(ND + Nk)`, with temporary batch-to-reference distance workspaces. No full N×N distance matrix is intentionally retained, and that does not make exact search linear-time.

The added local covariance analysis costs roughly `O(N k² D + N k³)` through batched Gram matrices. PCA fitting, model loading, graph layouts, output size, and browser rendering have separate costs.

There is no demonstrated million- or eight-million-point production capability. An ANN replacement would need explicit recall, topology, and stability validation against the exact baseline.

The implementation uses ordinary NumPy arrays, sparse matrices, lists, dictionaries, and direct-purpose functions. Preserve human readability; do not add unnecessary wrapper abstractions or application dataclasses.

Relevant source modules:

| Module | Responsibility |
|---|---|
| `core.py` | Normalization, exact neighbors, graphs, signatures, query scoring, and point-level routing. |
| `atlas.py` | Graph-based layout, component packing, coast cartography, roads, and pass structure. |
| `terrain.py` | Measured terrain channels, local lenses, candidate boundaries/gaps, and bounded regional route sets. |
| `mesh.py` | All-point covariance spectra and representative measured neighborhood envelopes. |
| `flat.py` | Extra field channels and per-component masked isolines. |
| `flat_view.js` | Planar field rasters, contours, spectral glyphs, presets, and legends. |
| `geometry_views.js`, `terrain_overlay.js` | Shared local graphics and the persistent neighborhood strip. |
| `mesh_gl.js`, `mesh_view.js` | Optional offline WebGL reference view. |
| `terrain_lab.py`, `terrain_lab.html` | Controlled vector/graph demonstrations. |
| `validation.py` | Independent geometric recovery, controls, and diagnostics. |
| `io.py`, `cli.py` | Safe persistence, offline export, reports, and commands. |

Primary outputs include `explorer.html`, `terrain_lab.html`, `atlas.json`, numeric `map.npz` without pickle, a configuration/hash manifest, per-item profiles, region summaries, sparse H0 events, aligned IDs/metadata, and validation reports when enabled. The default entry point in the supplied project is `examples/large_map/explorer.html`.

## 17. Instructions for your response or subsequent work

When addressing my requested follow-up:

1. Start with the purpose and a concrete example, then explain the measurements needed for that example.
2. Keep covariance effective directions, radius-growth dimension, omitted variance, density, stability, and route ambiguity distinct.
3. For each terrain or visual cue, state what is measured, what the person can infer, and its practical qualification.
4. Distinguish exact calculations on the declared graph from heuristics, display conventions, and proposed future methods.
5. Keep the flat map capable of communicating its analysis without requiring a popup or 3D view. Retain optional 3D references when useful.
6. Do not infer semantic value, authorship, metaphysical properties, or arbitrary external classification accuracy from geometry alone.
7. If proposing changes, preserve the frozen-corpus identity contract, point/ID alignment, graph integrity, explicit units, and representative-versus-selected-point distinction.
8. If implementing, use readable functions and basic data structures, meaningful numerical controls, and visual checks appropriate to the change. Do not claim a capability is implemented merely because the original methodology proposed it.

The guiding idea is: **measure original-space relations, construct an honest schematic atlas, and make the resulting geometric distinctions directly visible to a human.**
