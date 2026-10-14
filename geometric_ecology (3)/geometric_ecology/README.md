# Geometric Ecology · Terrain Atlas v0.5

A Python implementation of the supplied corpus-relative geometric topology
methodology. The interface is the **computed map itself**: shorelines, terrain,
roads, narrow passes and instability overlays reveal the analyses directly.

**Open `examples/large_map/explorer.html` after extracting this project.**
The included, self-contained map shows **20,000 embeddings at 64 dimensions**,
about eighteen times as many points as the original demo. No installation is
needed to open it, and all points are represented rather than display-sampled.

The default **flat atlas** communicates geometry without elevation. Independent
color and contour fields, labelled isolines, stepped bands and planar spectral
rings reveal the measurements directly. Four presets emphasize dimensionality,
density/redundancy, routes/passes, or stability. All nine fields can be combined
manually. No neighborhood popup is needed.

A persistent bottom strip follows hovered rings and selected items. The 3D mesh
and relief views remain available alongside the flat atlas. Geometry Lab now
opens with a 2D covariance-ellipse/spectral-ring comparison and retains its
rotatable 3D reference and eight terrain cases.

See [the flat atlas guide](docs/FLAT_VIEW.md) for presets, units and encodings,
and [the 3D view guide](docs/MESH_VIEW.md) for the optional mesh. Open
`examples/large_map/terrain_lab.html` for the controlled comparisons.

## How the geometry becomes visible

| Terrain | Computed meaning | Visible evidence |
|---|---|---|
| Flat plain | Dense, low-dimensional, redundant neighborhood | Low relief, concentrated direction spectrum, overlapping nearby points; explicit near-substitute tolerance |
| Mountain | High local dimension or strong metric variation | Interpolated measured relief; local variance spectra and direction ribbons expose hidden axes |
| Cliff | Abrupt original-space gap | Long crossing compared with local edge lengths; a separate controlled gap example |
| Narrow pass | Low-conductance bottleneck | Approach paths converge through a restricted crossing and fan out again |
| Maze | Destination-conditioned route ambiguity | Actual near-shortest routes split and rejoin; entropy and enumeration cap are stated |
| Isolated island | Disconnected or condition-sensitive component | Disjoint charts, absent crossings, immediate reachability information |
| Ridge / boundary | Separation across analyzed graph scales | A seam backed by low cut conductance, distance contrast and a density dip |
| Fog / unstable ground | Changes under reference resampling or compression | Disagreement tint plus original and changed neighbor links |

The flat view uses unlit measured fields and value contours. The optional
**3D mesh** and **relief** views show the same geometry with height. Height can also show **route ambiguity** or **metric variation**. The
flat view retains original-density contours. At closer zoom, the connection
fabric reveals sampled real graph edges. Roads always follow retained edges.

The persistent neighborhood strip shows actual neighbor offsets, their PCA
variance spectrum and parallel-coordinate direction ribbons. Flat ribbons beyond two axes distinguish
a low-dimensional neighborhood from one whose directions disappear in projection.
Lenses are precomputed at cell representatives; the inspector explicitly identifies
that anchor when a different point is selected.

The geometry lab holds radius and neighbor count fixed across vector cases.
Graph cases are declared controlled fixtures that isolate gates, gaps, alternative
routes and separators. They are not presented as discoveries from the large corpus.
See [the visualization design comparison](docs/VISUAL_DESIGN.md) and
[the measured demonstration results](examples/TERRAIN_DEMONSTRATION.md).

Components are laid out using graph distances, with basin areas magnified for
legibility and disconnected components packed separately. This is a **schematic
atlas**, not a metric projection: spacing between islands and drawn land area are
not embedding distances or measures of human value. Coast shapes interpolate
point locations; named component membership supplies the connectivity guarantee.

The entire landscape is visible at first load. Click an island in the briefing to
zoom into it. Scroll to zoom, drag to pan, and click an item for local inspection.
The map highlights everything reachable from the selected location. Shift-click a
destination to draw its computed route, if desired. Selecting different islands
cannot invent a route across the water. Raw PCA is provided as a reference view,
where disconnected components may overlap.

## Install and run

Python 3.10+ in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# Optional localhost server; direct file opening works too.
geometric-ecology serve --map examples/large_map --open

# Reproduce the larger demo and its independent geometric validation.
# Nulls are omitted here to avoid two additional large-corpus builds.
geometric-ecology demo --out runs/large_map \
  --n 20000 --dimensions 64 --scales 12 24 48 \
  --resamples 3 --regions 72 --landmarks 96 --skip-nulls

# A smaller quick run, including null controls.
geometric-ecology demo --out runs/small_map --n 1100

# Numerical and atlas-integrity verification.
python -m unittest discover -s tests -v
```

`--skip-validation` generates the map without held-out/null/route experiments.
`--skip-nulls` retains the other validation checks. The default demo now uses
20,000 points, 64 dimensions, k=12/24/48, three resamples, 72 navigation cells and
96 landmarks. Build commands for user data retain their smaller default graph
settings; all can be changed explicitly. `--overwrite` permits replacing an
existing generated output directory. The local server binds to 127.0.0.1:8765;
choose another port with `--port`.

## Measured larger example

The included run builds a 20,000-point, 64-dimensional map using exact neighbors.
It contains six substantial connected components and 41 tiny disconnected
components. Cartography follows these actual memberships, without using planted
synthetic group labels. The number and placement of marked passes are computed
from the regional graph and recorded in `atlas.json`.

See [the larger demonstration record](examples/LARGE_DEMO.md),
[its complete geometry report](examples/large_map/FEASIBILITY.md), and
[three-seed diagnostics and earlier scaling measurements](examples/REPLICATION.md).
Those earlier 500/1,500/5,000-point timings concern the geometry backend, not the
new atlas renderer. The included large-run report distinguishes the stages.

The map tests geometric structure. It does not establish semantic relevance,
authorship or cultural significance. Higher-dimensional regions can contain
noise; stable random clouds remain a useful counterexample to a single score.

## Bring your own embeddings

Supply a numeric `.npy` matrix of shape **(N,D)**, aligned unique IDs, and optional
display metadata. No content encoder is included.

```bash
geometric-ecology build \
  --embeddings /path/to/embeddings.npy \
  --ids /path/to/item_ids.npy \
  --metadata /path/to/items.jsonl \
  --encoder 'encoder-version-and-preprocessing' \
  --scales 12 24 48 --regions 72 --landmarks 96 \
  --skip-nulls --out runs/my_atlas

geometric-ecology serve --map runs/my_atlas --open
```

IDs can be a flat Unicode `.npy` array or JSON list. Metadata can be an aligned
JSON list or JSONL stream, for example:

```json
{"title":"Archive item 123","text":"Human-readable description","url":"https://example.org/items/123"}
```

IDs and metadata are optional; absent IDs become row numbers. Metadata does not
enter the geometry or cartographic layout. The reserved `demo_group` field is
used only for synthetic validation after mapping. Invalid/NaN/Inf/zero rows are
rejected. Filter embeddings, IDs and metadata identically if removing rows.
Duplicates remain; ties may be order-dependent and local dimension undefined.

Distances throughout are **unit-normalized Euclidean chord lengths**,
`sqrt(2*cosine_distance)`. This preserves cosine neighbor ordering, supports
metric path-stretch measurements, and avoids using squared radii in the local
intrinsic-dimension estimator. Compressed coordinates are not renormalized.

## Frozen-map queries and routes

```bash
geometric-ecology score --map runs/my_atlas \
  --queries /path/to/new_embeddings.npy --out runs/attachments.json

geometric-ecology route --map examples/large_map \
  --start demo-00040 --end demo-09000
```

The scorer reports local distances/dimension, candidate-region ambiguity,
insertion reverse-neighbor count, corpus-fitted compression response, landmark
entry paths and subsample sensitivity ranges. Inherited stability/support are
explicit proxies; new persistence and structural influence are not inferred.
The frozen map is unchanged. Exact reference scans remain O(ND) per query.

The small `examples/new_queries.npy` and `examples/attachments.json` files belong
to the original 1,100-point reference run under `examples/demo`, which is also
re-exported with the terrain interface. They are separate from the larger demo.

## Outputs and source

| File | Purpose |
|---|---|
| `explorer.html` | Flat field atlas, independent labelled contours, spectral rings, persistent strip and optional 3D mesh |
| `terrain_lab.html` | Eight offline controlled demonstrations, including actual local spectra and route sets |
| `atlas.json` | Graph layout, disjoint component discs, numeric terrain textures/contours, actual road paths, pass definitions and cartographic fidelity |
| `map.npz` | Numeric frozen geometry and fitted compression transformations; no pickle |
| `manifest.json` | Corpus/ID hashes, encoder, parameters, metric and dependency versions |
| `profiles.csv` | Per-item measurements |
| `regions.json` | Navigation-cell summaries and cut conductance |
| `h0.json` | Sparse H0 filtration merge events |
| `evaluation.json`, `FEASIBILITY.md` | Independent numerical checks, when enabled |
| `ids.json`, `roles.json`, `metadata.json` | Aligned IDs, descriptive flags and optional metadata |

The source uses arrays and plain dictionaries, with direct-purpose modules:
`core.py` for geometry, `atlas.py` for cartography, `terrain.py` for measured
fields/lenses/routes, `terrain_lab.py` for controlled demonstrations, `mesh.py` for all-point covariance
fields and measured hulls, `flat.py` / `flat_view.js` for masked isolines and
planar layers, `mesh_gl.js` / `mesh_view.js` for offline 3D rendering, `validation.py` for independent
checks, `demo.py` for synthetic data, `io.py` for saving/reporting, `cli.py` for
commands, and `explorer.html` for the map. No database, application dataclasses,
model training or external services are required at runtime.

Detailed map semantics and thresholds are in [CARTOGRAPHY.md](docs/CARTOGRAPHY.md).
Methodology coverage is in [IMPLEMENTATION.md](docs/IMPLEMENTATION.md), and checks
are recorded in [VERIFICATION.md](docs/VERIFICATION.md). The supplied methodology
is preserved as [methodology.md](docs/methodology.md).

## Practical limits

This is an exact-search feasibility implementation, with O(N²D) all-corpus neighbor
passes and O(ND + Nk) retained embedding/graph storage. It does not allocate an
N-by-N distance matrix, but search, compression and resampling repeat substantial
work. Normalization, PCA and model loading are in memory. The CLI guards user
builds above 20,000 rows unless `--allow-slow` is supplied. More points require a
separately validated ANN backend; no million-point claim is made.

The 20,000-point HTML embeds all profiles, edges and cartography, making the file
large. Rendering is canvas-based and clips offscreen points; it does not sample
points away. Sparse H0 is not full Vietoris–Rips persistence. Route entropy concerns up to
12 near-shortest regional paths to declared destinations, not all paths. H1,
alternate encoders, full leave-one-out rebuilding, and generation are not implemented. None of these limits is hidden by the terrain analogy.

The flat atlas works without WebGL. The optional 3D mesh uses WebGL 1 when
available; its button is disabled when unavailable. Surface facet count is fixed for drawing and is
not a dimensionality measurement. Local hulls show three projected axes; their
color and spectra make further hidden directions explicit.
