# Making terrain legible through geometry

The useful physical-map analogy is an inference from visible structure: paths
converge at a pass; a gap needs a large crossing; a route maze has alternatives.
An icon alone supplies a label. No single 2D projection can expose every local
direction, so this implementation combines an overview with local unfolding.

## Techniques considered

| Technique | Strength | Limitation | Decision |
|---|---|---|---|
| Terrain icons / categorical colors | Fast vocabulary | Does not show why a region has that label | Replace mountain symbols with measured fields |
| Density field / contours | Reveals concentration and separation | Projected density can confuse crowding with true density | Interpolate original-space inverse-radius ranks within each component; retain flat-map contours |
| Shaded 2.5D relief | Immediate peaks, slopes and basins | Scalar height can imply nonexistent difficulty or value | Use explicit selectable measurements, identical height rules for surface, points and roads |
| Rotating a 3D projection | Reveals a third direction | Still hides dimensions 4 onward | Favor covariance spectra and unfolded direction ribbons |
| Local PCA spectrum + parallel coordinates | Reveals how many directions carry variation | Requires a local view; axes/signs are not semantic concepts | Implement linked representative lenses and matched-radius controls |
| Edge fabric / crossing gates | Makes neighborhood connectivity visible | Dense graphs become hairballs | Zoom-dependent sampled fabric, simplified roads and controlled gate examples |
| Actual route bundles | Reveals alternatives to a stated target | Enumerating every route is infeasible and target-dependent | Up to 12 near-shortest regional routes; show cap, entropy and destination |
| Before/after neighbor ghosts | Makes instability observable | Missing references cause unavoidable membership changes | Fixed-position membership ghosts and separately labelled raw/conditional Jaccards |

## A mountain is directional complexity

In the default view, height interpolates `log1p(local_dimension)` divided by its
95th-percentile display cap, clipped to [0,1]. Dimension comes from original chord
radius growth, not atlas point crowding. A separate metric-variation mode shows
neighbor-distance coefficient-of-variation ranks. In the local lens, covariance
spectra and ribbons reveal the actual directions supporting that complexity.

The lens removes the query's sphere-normal component, computes a centered tangent
SVD, and displays neighbor offsets relative to the actual query in the resulting
axes. Spectrum variance is centered; ribbons show centered direction scores with
a common scale. These covariance measures differ from the radius-growth estimator.
The first two spectrum bars are green; remaining directions are orange. The
matched-radius 2D control becomes flat beyond its second ribbon axis; the 12D
condition continues varying across many axes. No extra spokes or noise are invented.

A physical peak has both elevation and a travel cost governed by physics. A
high-dimensional embedding neighborhood has no equivalent universal cost law.
High dimension can mean signal, noise, many independent distinctions or poor
compression. It does not by itself establish a meaningful destination, cultural
value, or difficulty reaching it. Actual graph costs, restricted gates and route
alternatives supply separate navigational information.

## Different structures require different cues

Cliff and pass are not synonyms: one is a large local step, the other a scarce
crossing. A maze is conditional on a destination; a high-degree star still has
one simple route between any two leaves. A persistent boundary can have several
usable gates and need not disconnect the corpus. Fog concerns changes across
observation conditions, not inherently high dimension.

The atlas is the readout. The lab explains the encodings through controlled
comparisons; it asks no questions and needs no supplied evidence or annotations.
The complete eight-part mapping is in CARTOGRAPHY.md.

## v0.4: the local geometry stays on the map

The subsequent refinement implements an orbitable, depth-tested triangle mesh
with original-neighborhood envelopes and covariance spectra at map anchors.
A persistent strip replaces the neighborhood modal. All-point batched covariance
measurements now supply the color and default height fields; local meshes expose
actual PCA-3 shape. This combines an overview, repeated local shapes and inline
unfolding. MESH_VIEW.md documents the measurements, glyph normalization,
projection limits, rendering controls and WebGL fallback.

## v0.5: encode complexity without height

A continuous color field reveals regional differences; labelled isolines expose
value changes and nested regions; a planar ellipse and spectral ring expose
anisotropy and the distribution of independent variation. These cues are
complementary: density alone cannot establish dimension, and an ellipse alone
cannot expose all omitted axes. Four presets coordinate nine independently
selectable measurements, with field, isoline, stepped-band and ring toggles.
The flat renderer has no surface lighting or vertical displacement.
