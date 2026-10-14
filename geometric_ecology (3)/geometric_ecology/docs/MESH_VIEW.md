# Reading dimensionality directly on the map · v0.4

Open `examples/large_map/explorer.html`. Choose **3D mesh** for an orbitable WebGL
view, with local neighborhood shapes and spectra already on the map. There is
no neighborhood popup. A persistent strip follows hovered neighborhood meshes
and selected items; it displays the identified cell representative's actual
neighbors, covariance spectrum and unfolded direction ribbons.

## Three complementary cues

| Cue | Measurement | What becomes visible |
|---|---|---|
| Landscape height | Effective covariance directions by default; radius-growth dimension, metric variation and bounded route ambiguity remain selectable | Regions with more independent variation rise above simpler regions |
| Teal → amber → purple | `1 - (variance_PC1 + variance_PC2) / total_variance` | How much a two-axis view omits, independently of plot density |
| Local faceted mesh | Convex envelope of actual neighbor offsets in their first three tangent PCA axes | A thin structure, sheet or volume, with its measured anisotropy |
| Bars beside each mesh | Actual ordered PCA variance, up to 12 directions | Whether variation concentrates in one or two directions or continues across many axes |
| Persistent bottom strip | Exact representative neighborhood, spectrum and direction scores | The underlying geometry without opening an inspection window |
| Roads / zoomed connection fabric | Actual retained graph edges | Traversability, separately from local envelope facets |

The hue reports an omitted **fraction**, rather than a calibrated dimension
number. The displayed effective-direction count is the covariance participation
ratio `1 / sum(p_i**2)`, where `p_i` are normalized local eigenvalues. Neither is
the radius-growth dimension estimator retained in the earlier view.

## Measurements and mesh construction

`mesh.py` computes covariance spectra for **every point**, not only representative
anchors. For a query q with unit-normalized neighbors x, form offsets x-q, remove
the component normal to q's sphere tangent plane, and center the offsets. Batched
neighbor Gram eigendecompositions recover the same nonzero eigenvalues as a full
local SVD. The neighborhood sample bounds the observable covariance rank by
`min(k-1, D-1)`; there is no claim to recover hidden dimensions beyond this bound.
The height field scales `(effective_count-1)/(rank_bound-1)` with a protected
zero denominator. Degenerate coincident neighborhoods have an explicit invalid
spectrum rather than invented directions.

The added analysis costs O(N k² D + N k³) using batched local Gram matrices; it
does not add another all-corpus pair-distance pass. Original exact kNN remains
the overall scaling constraint.

Original-space covariance statistics are interpolated in atlas coordinates,
within component masks, for surface height and color. This remains a schematic
graph atlas, not a 3D embedding reconstruction. Surface triangles have a fixed
resolution; lighting follows their geometric normals. No random roughness or
extra peaks are added to suggest complexity.

Representative local meshes use the actual stored PCA-3 neighbor offsets.
Rank-three clouds use a 3D convex hull; rank-two clouds use a planar boundary
triangulation. Rank-one or coincident data receive no fabricated surface facets.
No jitter is used to force a hull. Each shape has a normalized screen footprint
and is lifted above its anchor for legibility, with a leader line. This is an
attached local geometric view in its own PCA frame, not a globally oriented
volume at its original spatial scale. Different local PCA frames have no common
semantic orientation. Hull edges are **not extra graph connections**.

Representative meshes are spread across navigation cells, with screen-space
collision filtering. More appear when zooming makes room. Their count is a
legibility budget, not display sampling of the corpus: all 20,000 item positions
and all-point measurement fields remain included. Tiny components remain point
islets. On-map labels show covariance participation ratio, not integer rank.

The first-three-axis envelope can look volumetric for both 4D and 12D data.
Color, ordered variance bars and ribbons expose the remaining difference. The
Geometry Lab now includes a rotatable matched-radius 2D-sheet/12D-volume pair,
with the exact spectra directly below it. The demonstration does not infer
semantic importance or navigation difficulty from a convex hull.

## Controls

- Drag to orbit; right-drag or Alt-drag to pan; wheel or +/− buttons to zoom.
- With the mesh focused, arrow keys rotate, +/− zoom, and R resets the camera.
- Change relief scale, wireframe, local neighborhood meshes or surface coloring.
- Hover a local shape to update the persistent strip, or click to select its anchor.
- Shift-click a destination for the original retained-graph route.
- Minimize the strip when more map area is useful; no popup is required.

Original/subset/compressed conditions in the strip compare actual neighbor
membership. These raw set Jaccards include missing references. The map's
conditional resampling field retains its distinct definition.

## Rendering and fallback

The offline HTML embeds the mesh data and two small JavaScript renderer modules;
there are no CDN assets, network calls or JavaScript package installs. WebGL 1
renders shaded triangles, depth-tested wire edges and all point positions.
The ordinary 2D relief and flat-map views retain inline spectra and the strip.
If WebGL is unavailable, the interface keeps the flat atlas available and explains
why the 3D button is disabled.

Primary rendering references:

- [MDN: WebGLRenderingContext](https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext)
- [MDN: depth testing](https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext/depthFunc)

The v0.5 default is now the independent flat-field atlas (FLAT_VIEW.md). This
guide describes the optional 3D reference; its measurements remain available.
