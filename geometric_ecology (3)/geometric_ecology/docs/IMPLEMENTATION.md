# Measurement decisions and methodology coverage

## Coverage

| Methodology element | Implementation | Scope / qualification |
|---|---|---|
| Frozen corpus | Normalized float32 array, encoder string, SHA-256 of vector bytes and IDs | Encoder identity is supplied by user, not independently verified |
| Multiscale graphs | Exact, identity-excluded kNN; reciprocal edges at k=6/12/24 | Nested edge frequency is rank support, not independent stability |
| Persistent components | Kruskal H0 merge events over max-k mutual-edge distance filtration; separate components at each k | Exact for this sparse filtration; missing edges can overfragment; no H1 |
| Local cohesion/density | Mean k-distance, k-radius, inverse radius | Inverse radius is a proxy, not a calibrated density estimate |
| Local dimension | (k-2) / sum(log(Tk/Tr)), chord radii | Finite-sample corrected local estimator; invalid on exact duplicates/ties; not a semantic dimension |
| Boundary | Fraction of forward neighbors outside navigation cell | Partition-dependent proxy, not an intrinsic decision boundary |
| Hubness | Reverse kNN count | Diagnostic only; not evidence of value |
| Compression | Corpus-fitted PCA at ~25%,50%,85% rank; coordinate half-mask | Euclidean projected coordinates without renormalization; measures discarded directions |
| Resampling | Four 80% reference subsets, original query points, identity removal | Jaccard against retained eligible original neighbors; not a population CI |
| Perturbation | Independently sampled 0.01-chord vector noise, renormalize and rebuild neighbors | Fixed absolute noise scale can disproportionately affect dense regions |
| Structural influence | Tarjan vertex articulation, separated remaining-pair fraction | Exact fixed-graph connectivity ablation; no kNN refill or H0 recomputation |
| Connector centrality | Brandes from up to 24 random sources, shortest hop paths | Sample-dependent proxy, not weighted route betweenness |
| Navigation cells | Farthest-point seeds and graph-distance Voronoi within support components | A navigation partition, not semantic clustering or a persistent basin claim |
| Landmarks | Cell representatives + extreme profiles + farthest Euclidean coverage | Budget expands to cover every cell/component; not an optimal skeleton |
| Navigation skeleton | Landmark spanning tree under support-graph path distances, drawn along actual graph paths | It connects landmarks per component; never bridges disconnected components |
| Connectivity | Component IDs, sizes and route reachability | Support graph changes when threshold changes |
| Conductance | Unweighted cell cut / min(volume(cell), volume(complement)) | Partition and retained-graph dependent |
| Stretch | Original chord length accumulated along weighted-shortest graph route / direct chord | Weighted cost ratio also reported separately |
| Branching / ambiguity | Local next-hop effective count plus Yen enumeration of up to 12 regional simple paths within 25% of shortest cost | Destination-conditioned, cost-weighted truncated route-set entropy; not all-path entropy |
| Route sensitivity | Rebuild support edges under independent vector noise; route-edge retention | Not equivalent to whole-route survival |
| Hidden recovery | 20% vectors excluded from map/PCA fitting; recover nearest training anchor after query noise | Direct retrieval baseline; no hidden-block/branch recovery |
| Nulls | Uniform sphere and approximate mean/covariance-matched Gaussian | One draw each, no significance testing, no density-matched null |
| New points | Frozen-corpus neighbors, dimension, compression, insertion reverse count, landmark entry paths | Uses exact scans; inherited stability/support labelled proxies; influence unavailable |
| Query uncertainty | p10–p90 of cohesion under reference subsampling | Sensitivity range, not confidence interval or semantic posterior |
| Human interface | Automatic terrain atlas: separate components, shores, roads, regional passes, original-density contours, measured relief, original-neighbor direction spectra/ribbons, bounded route bundles and resampling/compression fog | Map encodings provide the computed information directly; no questionnaire |
| Generation difficulty | Number of possible landmark-distance constraints and scales | Inventory only; no generation, inverse solver, or lower-bound estimate |
| Alternate encoder / drift | Not implemented | Rebuild a separate map; corpus changes change the object |

## Metric consistency

All reference and query neighbors, edge lengths, H0 thresholds, and unweighted path
lengths use unit-normalized Euclidean chord distance:

`||u-v|| = sqrt(2*(1-u·v))`.

Neighbor ordering matches cosine; the local radius is not squared and raw path stretch
has a metric denominator. Terrain route edges cost `chord/(scale_support + 0.01)`.
These costs are explicitly distinct from geometric length. Projected/compressed
coordinates are not renormalized: their Euclidean distance represents a controlled
loss of coordinates of the fixed full-space metric.

## Independence and controls

The first k neighbors are necessarily contained in a larger k list. Their recurrence
cannot establish robustness. We therefore separate nested-scale support from reference
subsampling, independently noisy embeddings, compressed-space neighbor changes, and
held-out query recovery. We do not claim resampling persistence for entire basins;
currently the resampling signature concerns neighbors. The null clouds show how
geometry can produce structure and stability even in the absence of the planted
mechanisms. This is not evidence that one cloud has more human value.

## Navigation-cell budget

Every connected component gets at least one seed. Components containing at least
`max(20, floor(0.02*N))` points share the requested large-component cell budget,
with extra seeds allocated by population per existing seed. Small components get
one separate cell each, so the final count can exceed `--regions`. Seeds are then
chosen by farthest graph distance within each component. A Voronoi cell is simply a
region closer to one seed than the others in the original graph. Cell boundaries,
conductance and attachment ambiguity depend on this partition and its granularity.

## Cartographic presentation

See CARTOGRAPHY.md for the graph-distance MDS layout, monotone population
magnification, non-overlapping component packing, original-space terrain
interpolation, actual crossing paths, and grouped critical-pass definitions.
The raw PCA is retained as a reference; the default is the computed terrain atlas.

## New-point outputs

Forward neighbors exclude no reference identity because a new query is not a member.
A duplicate query can therefore retrieve its identical known embedding. Reverse
insertion count scans reference k-radii with strict `<`; ties are not counted. Candidate
cell weights are neighbor softmax weights using the query median neighbor distance as
temperature. They are heuristic geometric masses, not learned posteriors or calibrated
probabilities. Effective cell count is exp(entropy of those masses).

Nearest landmarks are selected by direct chord distance. Each landmark with a query
neighbor in its component gets a path via the nearest eligible entry anchor and the
frozen support graph. A landmark without a compatible anchor gets no invented path.
Costs include the direct query-to-anchor entry edge plus terrain-weighted graph cost;
this is an attachment heuristic, not the exact shortest route in a rebuilt graph.

## Complexity and reproducibility

The reference backend uses batched exact search. It retains O(Nk) graph/neighbor
storage plus O(ND) embeddings and transformed views, but each all-corpus neighbor
pass costs O(N²D). There are several passes for compression/resampling. A batch can
use O(batch_size*N) temporary distances; implementation details depend on scikit-learn.
Full PCA uses an in-memory SVD; dimensionality is another significant cost.
Landmark skeleton export solves paths from each component's landmarks. Many tiny
components expand landmark count. This prototype needs resource budgeting before
large-scale use; the CLI's 20,000-row guard is not a performance guarantee.

Hashes identify normalized values and aligned IDs, with shape and precision recorded
in the manifest. Timings and dependency versions reflect the actual run. Exact ties,
BLAS implementations and floating-point details can affect graph edges. The shipped
run used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0 and
threadpoolctl 3.6.0. Dependency ranges are broader for installation; use these recorded
versions for closest reproduction. Runtime imports use no remote services.

## Primary implementation references

- [scikit-learn: unsupervised nearest neighbors and algorithm costs](https://scikit-learn.org/stable/modules/neighbors.html)
- [scikit-learn: NearestNeighbors API](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)
- [SciPy: sparse connected components](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html)
- [SciPy: sparse shortest paths](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.dijkstra.html)
- [Levina and Bickel: Maximum Likelihood Estimation of Intrinsic Dimension](https://papers.nips.cc/paper_files/paper/2004/hash/74934548253bcab8490ebd74afed7031-Abstract.html)
- The supplied methodology, preserved in `docs/methodology.md`, motivates the profile
  dimensions and the strict separation between corpus-relative roles and human value.

## v0.3 terrain channels

See VISUAL_DESIGN.md and CARTOGRAPHY.md. The overview distance-growth dimension
and the lens covariance participation ratio measure different properties. Neither
is calibrated as semantic value or human travel difficulty. Route alternatives
use the complete cell adjacency graph, rather than only the displayed road
backbone. Expanded route walks follow actual point edges; their regional cell
paths are loopless, but shared within-cell segments can repeat point vertices.

- [SciPy: Yen shortest simple paths](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.yen.html)
- [scikit-learn: PCA and explained variance](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

## v0.4 direct geometric visualization

Batched all-point tangent covariance spectra now support omitted-variance color
and effective-direction height fields. Measured representative PCA-3 convex
envelopes and compact spectra are drawn directly on an orbitable mesh. A
persistent strip replaces modal inspection. These are local sample statistics,
with rank <= min(k-1,D-1), and do not expand the backend's semantic claims.
See MESH_VIEW.md for the geometric construction and rendering qualifications.

## v0.5 flat fields and contours

Independent color and contour channels now expose nine measured quantities in
a strictly planar view. Masked per-component ContourPy isolines carry actual
values; constant fields produce no decorative contours. Canonical covariance
ellipses and variance-partition rings complement continuous fields. Flat layers
never change atlas positions or graph edges. See FLAT_VIEW.md for definitions,
scale choices and interpolation qualifications.
