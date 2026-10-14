# Verification record · v0.5

This version verifies the computed map itself: connectivity, terrain encodings,
actual road paths and larger-corpus rendering. There is no questionnaire workflow.

## Numerical and cartographic checks

`python -m unittest discover -s tests -v`: 25 tests.

The geometry checks cover invalid vectors, identity exclusion with duplicates,
known sparse H0 merges, exact deletion influence against brute-force checks,
connected/disconnected routes, model persistence, immutable query attachment and
held-out-reference separation.

The atlas checks cover finite positions for every point, nonoverlapping component
discs, minor-component placement, real retained edges for every road step, exact
road costs, one correct cell-pair crossing per mapped road, embedded numeric
terrain rasters, and valid fidelity ranges. Regional-cut flags are independently
checked by deleting their crossing edges from the original graph and verifying
that the road endpoints become disconnected.

The five added terrain checks verify matched-radius dimensional contrast, actual
route entropy rather than node degree, equal-position/equal-shortest-cost maze
versus tree controls, separate gap/gate/isolation cases, and a uniform-ring
negative control against fabricated persistent ridges. Every expanded route
bundle step in the shipped large example is also checked against retained edges.

## 20,000-point browser check

The self-contained large example was loaded in headless Chromium at 1680×1080.
The automated check verified all 20,000 point positions are represented and project inside the
initial view, default flat atlas, no question/answer form, island focus,
connected route drawing, immediate disconnection handling, item search, layer
switches, four height channels, persistent original/changed-neighbor strip, all eight lab
cases, the maze/tree comparison, PCA reference, measurement panel, and a
420-pixel mobile layout with
no horizontal overflow. No JavaScript page errors were observed.

Observed cached WebGL overview redraws were roughly 5–15 milliseconds, with
startup around 1.8–2.0 seconds on the execution host using software WebGL; startup and draw details are printed by the browser check. These are illustrative observations, not
browser/device guarantees. No point display sampling is used.

The optional check is `tests/browser_smoke.cjs`. It requires Playwright separately;
that library is not a Python runtime dependency. Desktop overview and focused
views were inspected locally. Screenshots are intermediates, not deliverables.

## Larger geometry run

The shipped large run contains 20,000 vectors in 64 dimensions, k=12/24/48,
three 80% reference resamples, 72 requested navigation cells for substantial
components and 96 requested landmarks. It uses exact reference search.

Map construction, cartographic construction and independent geometric validation
are timed separately. Held-out recovery excludes 4,000 queries from the map fit,
and the route/perturbation and planted-mechanism checks are saved. Large null-cloud
builds are omitted; the smaller reference retains its two null controls.

No participant response is needed to supply the map's analyses. Geometry alone
still does not establish the semantic or cultural importance of its regions.

## Installation smoke check

Built a v0.5 wheel with both HTML templates and all five JavaScript renderers,
installed it into an isolated virtual environment with system scientific
dependencies, and generated a new 600-point map through the installed CLI.
The generated atlas and geometry lab operate offline.

## Added mesh controls

Four numerical controls verify batched Gram spectra against independent local
SVDs, exact measured hull vertices with planar/volumetric rank, duplicate-point
degeneracy without fabricated facets, and finite all-point fields. Browser checks
exercise WebGL rendering without errors, orbit controls, mesh layers, inline
spectra without a neighborhood modal, strip minimization, and automatic fallback
when a test disables WebGL. Main-map and controlled-comparison views were also
visually inspected. Browser screenshots are internal QA intermediates.

In 3D, depth testing naturally occludes some points. The legacy draw-stat field
`visible_points` counts projected positions inside the canvas, not per-pixel
visibility after occlusion. All item positions are submitted when points are
enabled; representative neighborhood glyphs are separately decluttered.

## Added flat-atlas checks

Four additional numerical checks verify exact levels on a linear test field,
masked-gap line separation, constant/invalid fields without fabricated contours,
and preservation of atlas positions and original graph edges. Browser checks
exercise all nine color and contour choices, four presets, independent fill/line/
band/ring controls, ring hover, and unchanged screen positions across layer
changes. The 2D lab is the initial reading; its 3D reference remains accessible.
The default flat atlas runs when WebGL is disabled. Observed cached flat redraws
were around 45 ms on the execution host, without point display sampling.
