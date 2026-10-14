# Larger terrain demonstration · 20,000 points

The primary included map is `large_map/explorer.html`. All 20,000 points are represented, without display sampling. Computed analyses appear as cartographic layers, with no question or answer workflow.

| Measurement | Result |
|---|---:|
| Points / dimensions | 20,000 / 64 |
| Exact geometry construction | 28.19 s |
| Cartographic construction | 2.43 s |
| Additional fields, lenses and bounded regional route analysis | 3.59 s |
| Independent geometric checks | 29.61 s |
| Substantial connected landmasses | 6 |
| Tiny disconnected components / points | 41 / 44 |
| Actual mapped road paths | 82 |
| Grouped narrow passes / critical crossing segments | 1 / 6 |
| Median original / atlas neighbor Jaccard | 0.116 |
| Median original / raw PCA neighbor Jaccard | 0.043 |
| Held-out vectors excluded from fit | 4,000 |

Observed browser startup was about 1.0–1.4 seconds, with roughly 50 ms overview redraws at 1680×1080 on the execution host. All 20,000 point positions fit inside the initial view; depth testing can occlude individual points in 3D. These are observations, not performance guarantees. The HTML is large because it embeds all points, profiles and graph edges.

## Integrity of what is drawn

Every mapped road was checked against the retained graph and its accumulated cost. Each marked narrow segment was independently checked by removing its cell-pair crossing edges from the original graph and confirming endpoint disconnection. Adjacent qualifying segments are grouped into one pass label rather than presenting navigation-cell subdivisions as separate passages.

The marked pass anchor's post-hoc planted group is narrow connector. This is an evaluation diagnostic only: planted names did not determine the layout, terrain or pass selection.

## What the larger map makes visible

- Connected basins linked by a narrow passage have a continuous, explicitly drawn road network.
- Disconnected components occupy separate chart areas with their own shorelines.
- A ring has a closed road circuit; its uniform arcs are not labelled critical passes.
- High-dimensional patches have measured relief and local direction spectra/ribbons.
- A dense redundant patch can still show resampling/compression disagreement, distinguishing density from robustness.

## Limits of a map of high-dimensional geometry

Graph-distance layout is followed by monotone population magnification, so drawn distances and areas are schematic. Neighbor fidelity remains limited: this is a connectivity and terrain atlas, not an exact 2D reconstruction. Roads provide explicit traversability evidence. Density contours and dimension colors interpolate original-space measurements. Water between named islands means no retained-graph route, not absolute impossibility in the complete embedding space.

The earlier 1,100-point example remains under `demo/`, re-exported with the same terrain interface. Its null controls and the three-seed results remain small-corpus references. The large run omits two extra large null-cloud builds but includes held-out, perturbation, route and planted-mechanism checks.

## v0.4 mesh presentation

The map now defaults to orbitable WebGL, with all-point covariance color/height
fields, representative measured neighborhood envelopes and adjacent spectra.
The persistent neighborhood strip replaces the popup. Additional spectral
analysis on this fixed corpus took approximately 0.65 seconds. See
TERRAIN_DEMONSTRATION.md and ../docs/MESH_VIEW.md for the refinement.
