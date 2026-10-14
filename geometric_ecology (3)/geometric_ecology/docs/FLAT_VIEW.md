# Geometry carried by a flat map · v0.5

Open `examples/large_map/explorer.html`. Every point stays in the atlas plane.
Color fields, value-labelled contours and planar neighborhood glyphs show the
analysis directly. The 3D mesh and relief remain optional reference views.

## Presets and independent layers

| Preset | Color | Contours | Additional cue |
|---|---|---|---|
| Dimensionality | Variance omitted by two axes | Effective covariance directions | Spectral rings and first-two-axis ellipses |
| Density & redundancy | Original-space inverse-radius percentile | Near-substitute fraction | Rings retain an independent dimensionality cue |
| Routes & passes | Regional effective route alternatives | Original-space density percentile | Actual alternative-route bundle with a stated destination; orange passes |
| Stability | Conditional reference-subset disagreement | Moderate-PCA compression disagreement | Combined disagreement hatching |

The two dropdowns can combine any of nine measurements independently. Changing
one creates a custom combination. The legend states each measure and its units.
Color fill, isolines, stepped bands, contour labels and spectral rings have their
own switches. Roads, real-edge fabric, boundary seams, points and disagreement
hatching retain their layer switches. Fields can be compared without moving a
point or changing the retained graph.

## What the contours mean

Contours trace equal values of the selected measured field. A direction-count
line labelled 4 means covariance participation ratio 4; a disagreement line
labelled 50% means disagreement .5. Fraction and percentile fields use percent
labels, while direction and route counts use their numerical values. All islands
share a measurement scale. No per-island color normalization fabricates contrast.

Lines are traced from the existing occupancy-weighted field grid, using ContourPy
with masked arrays and `corner_mask=False`. Every retained quad therefore has
four supported corners. Disconnected components are processed separately, and
unsupported gaps interrupt contours. A constant field can correctly have no
contours at the chosen levels; the interface reports when no selected level
crosses the visible regions. It does not add decorative hill contours.

The raster uses bilinear interpolation over those supported quads, without
lighting, perspective, surface normals or z displacement. Stepped bands group
values between the same declared contour thresholds. Label placement suppresses
crowding, so not every visible line gets a numeral at every zoom level. As with
any field map, contour spacing shows change across the **chart coordinates**;
it is not an original-space gradient magnitude or a route-cost estimate.

## The nine measurements

| Field | Meaning and scale |
|---|---|
| Variation hidden by two axes | `1 - sum(first two covariance variance fractions)`, 0–1 |
| Effective local directions | Covariance participation ratio `1/sum(p_i²)`; finite-sample rank bound is recorded |
| Original-space density rank | Inverse neighbor-radius percentile across the corpus, 0–1 |
| Near-substitute fraction | Neighbor fraction within `0.1 × median corpus neighbor radius`, 0–1 |
| Regional route alternatives | Median effective count over bounded routes to up to three reference targets, 1–12 |
| Reference-subset disagreement | One minus backend conditional resampling neighbor overlap, 0–1 |
| Compression disagreement | One minus overlap under the recorded moderate PCA compression, 0–1 |
| Metric-variation rank | Percentile of neighbor-distance coefficient of variation, 0–1 |
| Combined geometry disagreement | Maximum resampling/compression disagreement, 0–1 |

These remain distinct: dense regions can be high-dimensional or unstable; many
routes need not be hard routes. A high-dimensional region is not automatically
a meaningful destination. Coincident neighborhoods with no defined covariance
are gray in covariance fields and do not generate dimension contours.

## A 2D neighborhood glyph with measured parts

The inner ellipse has semiaxis ratio `sqrt(variance_PC2 / variance_PC1)`.
It shows first-two-axis anisotropy in a canonical local PCA frame, not a global
orientation. The surrounding ring partitions the actual covariance energy:
first two directions are teal, subsequent directions are purple, and variance
beyond the first 12 directions is collected into a gray tail. The ring pieces
have angles proportional to their variance, rather than invented spokes.

A thin region can produce an almost line-shaped inner ellipse and a dominant
first ring sector. A neighborhood with many directions spreads variance over
many sectors. The number below is its effective direction count; it is not the
number of rendered ring pieces. Glyphs have equal display size and screen-space
collision filtering; more representatives appear when zooming makes room.
They are supplementary local readouts, not new nodes or graph edges.

Hover a ring or point to update the persistent original-neighborhood strip.
When a point is not a precomputed representative, the strip explicitly identifies
the representative being shown. No popup or supplied answer is required.
Geometry Lab opens with a measured 2D ring/ellipse comparison; its 3D reference
can still be selected.

## Connectivity, scale and implementation

Named islands still reflect retained connected components. Roads still follow
actual retained edges; restricted crossings, boundary candidates and bounded
route bundles retain their earlier definitions. The coast is a cartographic
outline, not a high-dimensional boundary. All 20,000 points remain included.
Points are drawn in grouped canvas paths for speed; opacity overlap is not a
calibrated density measure. Use the original-space density field for density.

`flat.py` builds the extra field channels and masked isolines using ordinary
arrays and dictionaries. `flat_view.js` renders cached field rasters, vector
contours and planar glyphs. No additional Python dependency was needed; ContourPy
was already used for cartography. The flat renderer uses Canvas2D and requires no WebGL or network access;
the optional 3D module still checks WebGL availability.

Primary contour reference:
[ContourPy: masks and corner masking](https://contourpy.readthedocs.io/en/v1.3.3/api/contourpy/contour_generator.html).
