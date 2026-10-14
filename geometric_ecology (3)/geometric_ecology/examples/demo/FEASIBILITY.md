# Measured feasibility report

This is a geometric feasibility demonstration, not a semantic or human-value detector.
The atlas presents its computed analyses directly. All numbers below come from the shipped run.

## Corpus and construction

- Items: 1100; dimensions: 48; encoder: synthetic-mechanisms-v1/seed-42.
- Corpus SHA-256: `c1597b6be3ba14d94ce24dc14d2b7c238b6d68eb6b12ec4f8567bf61212b8658`.
- Metric: unit-normalized Euclidean chord; exact batched neighbor backend.
- Scales: [6, 12, 24]; support threshold: 0.5.
- Build time: 0.370 s. Validation time: 1.300 s.
- Support graph components: 23; largest share: 0.487.
- Navigation cells: 25. These are graph Voronoi cells, not semantic classes.
- Sparse H0 infinite bars: 16. H1/loops are not measured.

## Does the representation survive changes?

- Median eligible-universe subsample Jaccard: 0.792.
- Median 0.01-chord perturbation Jaccard: 0.846.
- Median original-vs-display neighbor Jaccard: 0.091.
- PCA display variance explained: 0.671.

Low display Jaccard is a reason to inspect original neighbors and graph routes rather
than reading projected proximity as evidence. Nested k-edge support is a rank statistic;
it is never used as independent perturbation evidence. Resampling ranges are empirical
sensitivity ranges over 4 replicates, not confidence intervals.

## Held-out geometric recovery

The map and PCA are fit only to 880 training vectors; 220
vectors are held out. The target is the nearest training anchor under the original
query. Query vectors are then perturbed by 0.05 chord units.

| Measurement | Result |
|---|---:|
| Direct retrieval top-1 anchor recovery | 0.559 |
| Direct retrieval top-10 anchor recovery | 0.864 |
| Random anchor top-10 expectation | 0.011 |
| Compressed retrieval top-10 recovery | 0.727 |
| Centroid navigation-cell recovery | 0.709 |
| Majority navigation-cell baseline | 0.232 |
| Median correct-anchor rank | 1.000 |

The direct-neighbor baseline already has the embedding vectors. The map adds inspection,
terrain and routing; these measurements do not establish better retrieval accuracy.
The cell baseline can expose misleading accuracy from a dominant cell.

## Navigation and structural influence

- Reachable sampled pairs: 8/32.
- Median raw graph-length / direct-chord stretch: 1.143.
- Median support-weighted path-cost / direct-chord ratio: 1.201.
- Median Dijkstra expansions, including unreachable attempts: 138.000.
- Reachable fraction after one middle-edge ablation: 1.000.
- Mean route-edge retention after independent geometry noise: 0.887.
- Fixed-graph articulation vertices: 24.

Deletion influence is exact on the retained graph, with no neighbor refill. A redundant
path can make deletion influence zero even at an obvious connector. Sampled hop-path
betweenness complements it. Local effective next-hop count is not global route entropy.

## Null controls

One realization per null; these are descriptive comparisons, with no p-values.
The Gaussian control matches mean/covariance in expectation before renormalization,
not local density. Uniform-sphere structure is not a model of all irrelevant data.

| Cloud | Components | Median local dimension | Perturbation Jaccard | Display Jaccard |
|---|---:|---:|---:|---:|
| Observed | 23 | 2.883 | 0.846 | 0.091 |
| uniform_sphere | 1 | 26.299 | 1.000 | 0.000 |
| moment_matched_gaussian | 52 | 10.350 | 1.000 | 0.091 |

## Planted mechanism checks

Labels below are synthetic evaluation metadata, never geometry inputs. AUC is an
oriented rank-separation diagnostic, not learned detector performance. 0.5 is no
rank separation; 1 is complete separation. These checks are not semantic validation.

| Contrast | Measurement | AUC |
|---|---|---:|
| High-dimensional patch vs basins | local_dimension | 0.964 |
| Redundant patch vs basins | neighbor_radius | 1.000 |
| Connector vs basins | sampled_betweenness | 0.951 |
| Outliers vs basins | neighbor_radius | 1.000 |

## Cartographic interface

The interface is the computed map, with no questionnaire or evidence-entry workflow.

- 1,100 points are embedded and represented; no display sampling.
- 6 substantial connected landmasses and 17 tiny components.
- 2 mapped road paths and 1 marked regional passes.
- Cartographic construction: 0.162 seconds.
- Median original-neighbor / atlas-neighbor Jaccard: 0.263.
- Separate component discs prevent visual overlap between disconnected major islands.
- Every road follows real retained edges. Passes require a critical cell-level crossing.

Measured dimension relief, flat-map density contours, narrow crossings and
resampling/compression disagreement are drawn automatically. The persistent inline strip
unfolds actual covariance directions without a neighborhood popup, and `terrain_lab.html` supplies eight
controlled comparisons. Route ambiguity uses up to 12 regional near-shortest
paths to stated destinations; it is distinct from the local next-hop statistic.
Dimension height, metric variation and bounded route ambiguity are separate
selectable channels. Landmark graph-distance MDS is followed by monotone population
magnification so populated basins remain legible beside long passages. Island spacing
and drawn areas are schematic, not metric distances. Terrain interpolation uses original
measurements within each component; synthetic labels do not enter cartography.


## 3D dimensionality view

The optional orbitable mesh shows local neighbor envelopes and ordered spectra
directly on the map. Hue is the variance omitted by the first two local covariance
axes. Default height is covariance participation ratio, scaled to the neighborhood
rank bound 11. It is distinct from radius-growth dimension.
All-point spectral analysis and representative hull construction: 0.027 s.

Surface facets are rendering geometry, not a dimension count or additional graph
edges. Representative meshes have normalized footprints in local PCA frames.
Color and spectra expose variation beyond their three displayed axes. Without
WebGL, relief and inline spectra remain available.


## Flat field atlas

The default view keeps every position strictly planar. Nine measured fields can
independently drive color and labelled isolines. Four presets combine
dimensionality, density/redundancy, route alternatives and stability. Spectrum
rings and first-two-axis ellipses show original local covariance at representatives.
Fields and contours are masked per component; unsupported gaps are not bridged.
Flat-field preparation and contour tracing: 0.046 s.


## Practical limits and omitted parts

This exact reference backend is appropriate for hundreds to low tens of thousands of
points, depending on dimension and hardware. Search remains quadratic over the corpus;
the retained graph is sparse but the full workflow is not linear. PCA and npz loading
are in memory. New-query scoring scans the reference corpus and reuses fitted PCA.
No million-point performance claim is made. An ANN backend needs a separate recall
benchmark before deployment. Sparse H0 omits edges absent from the max-k graph; no H1,
global route entropy, alternate encoder validation, block/branch recovery, corpus-drift
study, full leave-one-out topology, or generation experiment is implemented. Generation
output is only a constraint inventory, not a measured difficulty estimate.

The project presents corpus-relative geometry as a directly readable atlas. Whether
its regions correspond to meaningful content depends on the corpus, encoder and task.
