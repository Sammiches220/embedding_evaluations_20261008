# Terrain demonstration · v0.5

Open `large_map/terrain_lab.html` for the controlled comparisons, then
`large_map/explorer.html` for the 20,000-point corpus.

## Measured directional contrast

Each fixture uses 2,000 normalized reference vectors in 32 ambient dimensions and
80 query neighbors. Query-neighbor radius is equalized to 0.025 chord. Spectra
are measured from the actual neighbor vectors, with no fabricated peaks/spokes.

| Planted local directions | Variance retained by first two axes | Directions retaining 90% | Covariance participation ratio |
|---|---:|---:|---:|
| 2 | 100.0% | 2 | 2.00 |
| 4 | 60.0% | 4 | 3.80 |
| 8 | 34.4% | 7 | 7.40 |
| 12 | 28.3% | 10 | 10.02 |

Radius and point count alone do not explain the difference. The unfolded ribbons
show measured variation continuing along the additional axes in the complex case.

## Connectivity contrasts

The 9×9 grid and its spanning-tree comparison use the same point positions,
start and destination, and shortest cost 16. The grid reaches the 12-route cap,
with 3.585 bits and effective count 12 over the enumerated equal-cost routes;
the tree has one route and zero bits. This isolates path alternatives from plot
crowding and shortest travel cost. It is not entropy over every possible path.

The cliff has a crossing five times the typical local edge length. The pass has
short steps but one restricted crossing. The island has no route. The ridge has
several gates and reports cut conductance under three declared graph conditions.
These are controlled weighted graphs, explicitly distinct from inferred kNN maps.

## Actual corpus integration

All 20,000 points remain represented. Relief, local direction lenses, instability
fields and regional route bundles are computed from the retained corpus geometry.
Each route bundle expands to actual retained point edges, independently checked.
There are two boundary candidates satisfying the stated density-dip, distance-
contrast and multiscale-cut criteria, and zero cliff candidates. The demonstration
does not force all eight metaphors to occur in this corpus.

This is a reproducible interface feasibility demonstration. It establishes that
these measured distinctions can be made visible; it does not establish semantic
importance or quantify human comprehension from geometry alone.

## 3D mesh refinement

The main map now opens directly into an orbitable mesh with shaded facets,
actual local neighbor envelopes and compact spectra. An always-visible bottom
strip shows unfolded directions without a modal. Covariance spectra are computed
for all 20,000 points; representative envelopes are decluttered on screen.
The lab's vector comparison includes a rotatable planar/volumetric mesh pair.
See ../docs/MESH_VIEW.md for exact encodings and projection qualifications.

## Planar field refinement

Open the main map for the default 2D atlas. Omitted-variation color, direction-
count contours and representative spectral rings expose complexity without
height. Density/redundancy, routes/passes and stability presets expose other
relations, with separate color and contour choices across nine fields. The lab
opens to its planar spectral-ring comparison and retains the 3D reference.
