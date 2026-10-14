# The map is the interface

The map supplies its connectivity, terrain and route cues automatically. It
solicits no answers, confidence scores, evidence or annotations. The design
comparison in VISUAL_DESIGN.md explains why the overview and local lenses use
different encodings.

## Atlas construction

1. Compute exact multiscale reciprocal kNN graphs from normalized embeddings.
2. Retain edges meeting the nested-scale support threshold; keep independent
   resampling, perturbation and compression measurements separate.
3. Partition retained components into graph Voronoi navigation cells.
4. Fit landmark graph-distance MDS, then extend to every point.
5. Blend 55% normalized linear position with 45% empirical rank position along
   each nondegenerate axis to magnify populated basins. Exact duplicates are not
   jittered. Pack substantial components into disjoint discs and tiny components
   into separate islets.
6. Draw measured fields within each component, real graph roads, restricted
   passages and labels. The raw PCA view remains available as a reference.

Layout, island spacing and land area are schematic. Original/atlas neighbor
Jaccards expose projection loss. Coastlines are smoothed displayed occupancy;
their shape is not a high-dimensional decision boundary. Named membership and
actual road crossings supply the connectivity information.

## All eight terrain meanings

| Physical terrain | Embedding analogue | Visible encoding | Qualification |
|---|---|---|---|
| Flat plain | Dense, low-dimensional, highly redundant region | Low relief, concentrated spectrum, overlapping near-neighbor geometry | Requires all three qualities, not green color alone |
| Mountainous terrain | High local dimension or strong metric variation | Measured relief, independent-direction spectrum and unfolded neighbor ribbons | Complexity is not a value score or an automatic travel barrier |
| Cliff | Abrupt gap or unstable transition | Large crossing step relative to local spacing; candidate gap ticks | Current detector tests gaps; condition instability is shown separately as fog |
| Narrow pass | Low-conductance bottleneck between regions | Approach paths funnel into a restricted crossing, then diverge | Marked atlas passes use a stricter critical-crossing criterion |
| Maze | High path ambiguity and route entropy | Destination-labelled real route bundles, effective count and cap | Bounded near-shortest regional route-set entropy, not degree or all-path entropy |
| Isolated island | Disconnected or weakly connected component | Disjoint charts; condition-sensitive roads distinguished by dashes | No retained route at the selected support threshold, not absolute separation in embedding space |
| Ridge / boundary | Persistent separation between neighborhoods | Seam plus crossing gates and scale evidence | Reference cells define the candidate cut; this is not semantic clustering |
| Fog / unstable ground | Geometry changing under resampling or compression | Disagreement tint; original and changed neighbor-link ghosts | Compression can discard structure or noise; missing references affect raw overlap |

## Fields, height and local lenses

The 54×54 fields are occupancy-weighted Gaussian interpolations of original-space
measurements in atlas coordinates, separately within each substantial component.
The field mask comes from displayed occupancy. Points and roads follow the same
bilinearly interpolated height as the surface. Empty bins inherit the nearest supported height for line interpolation,
while the surface remains masked; individual measurements remain in
the inspector. This suppresses misleading estimator-noise spikes in connecting
lines. Field color and light depend on the selected measured height and its slope.

- **Dimension height:** `clip(log1p(dimension)/log1p(q95(dimension)),0,1)`.
- **Metric variation height:** percentile of the coefficient of variation of the
  first signature-k original neighbor distances.
- **Ambiguity height:** `log(effective route count)/log(12)`, summarized by the
  median over up to three reference destinations in the component.
- **Density:** inverse-k-radius percentile. Contours appear in the flat view;
  they use original-space measurements, not raw plot density.
- **Fog:** maximum of `1-resample_jaccard` and `1-moderate_PCA_jaccard`. The backend
  resampling overlap conditions on retained eligible original neighbors.

The local lens computes actual tangent-neighborhood covariance directions at
navigation representatives. Its first-two-axis view, variance spectrum, and
first-12-axis parallel-coordinate ribbons share those measured directions.
The exact representative is stated when a different selected point refers to it.
Changed neighbors are projected into the original axes and do not fabricate
embedding motion. Lens Jaccards are **raw set** comparisons, so its 80% subset
comparison includes removed observations; this differs from the conditional
backend resampling field. No confidence interval is implied.

A near-substitute is a neighbor within `0.1 * median(corpus k-radius)` chord.
A cell is labelled redundant plain only when median inverse-radius rank > .65,
dimension rank < .65, and near-substitute fraction > .25. This tolerance is
explicit and geometric; it does not establish semantic substitutability.

## Roads, passes, cliffs and ridges

Every road uses paths inside its two endpoint cells and one actual crossing
edge. The visible network is a minimum-cost cell backbone plus strong alternative
crossings; the complete adjacency graph remains available for route analysis.
Width encodes crossing-edge count logarithmically. Each drawn road step is a
retained graph edge. Pale space on an island does not imply unrestricted movement.

A marked pass requires pair conductance < .04, median path-centrality percentile
> .6 and a bridge in the complete cell adjacency graph. Adjacent qualifying pairs
are grouped as one pass. The cut concerns all crossings of the pair; it does not
claim deleting one point or one original edge disconnects the map.

A cliff candidate requires a retained crossing edge > 2.5 times the geometric mean
of its endpoints' first-four-neighbor median radii, and median crossing length
> 1.8 times nearby within-cell edge length. Zero candidates is an allowed result.

A ridge candidate requires pair cut conductance < .06 at >= two-thirds of the
analyzed scales, crossing/internal median length contrast > 1.25, and crossing
vertex median radius > 1.5 times the two cells' median radius. The density dip
prevents a uniform ring's arbitrary cell cuts from becoming artificial ridges;
that negative control is tested. These are inspectable prototype thresholds,
not universally calibrated terrain definitions or independent basin persistence.

## Destination-conditioned route ambiguity

Yen enumeration uses the complete weighted cell adjacency graph. For each source,
up to three declared targets are chosen geometrically: largest cell, farthest
from it, and farthest from those two. Keep up to 12 shortest simple regional paths
whose cost <= 1.25 times the best. Weights are proportional to
`exp(-(cost-best)/(0.1*best))`. Entropy is `-sum(p*log2(p))`; effective count is
`2**entropy`. A 12-route cap is exposed. The median effective count supplies the
regional field; a displayed bundle always states its actual start and target.

Each displayed regional path is expanded into real point-edge walks. Shared
within-cell segments can revisit point vertices, even though the cell path is
simple. Counts are regional alternatives, not an enumeration of every original
point route. Alternative routes may be useful redundancy, rather than difficulty.
A sampled real-edge fabric appears at zoom > 1.5, capped at 18,000 drawn segments;
all points remain included.

## Scale and interpretation

The included atlas has 20,000 embeddings in 64 dimensions, six substantial
components and 41 tiny components. All points are included in the offline HTML.
Canvas clips offscreen points rather than display-sampling them. Precomputed
fields and local lenses trade file size for offline use. The separate lab labels
its vector fixtures and explicit graph fixtures clearly.

A corpus-relative geometric landscape cannot determine human authorship,
cultural importance or semantic coherence. Visible complexity and traversability
are separate measured properties. Human meaning needs a chosen task or context.

## v0.4 3D presentation

The default is now the orbitable WebGL mesh described in MESH_VIEW.md. Its default
height is covariance participation ratio scaled to the finite neighborhood rank
bound; the older radius-growth dimension remains a separate height option. Hue
shows the fraction of local covariance omitted by two axes. Actual local PCA-3
envelopes and spectra are attached to representative anchors, with a persistent
unfolding strip. These hull facets are not roads or additional graph edges.
The eight terrain interpretations and original graph connectivity remain as above.

## v0.5 planar presentation

The flat atlas is now the initial view. FLAT_VIEW.md specifies the independent
field/contour channels, masked per-component tracing, spectral rings and preset
combinations. Colors and lines carry the measurements without moving any point
out of the plane or adding lighting to imply elevation. The 3D and relief views
remain available as references.
