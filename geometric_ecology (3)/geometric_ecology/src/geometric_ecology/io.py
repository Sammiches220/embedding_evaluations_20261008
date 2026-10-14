"""Safe array persistence and self-contained, offline explorer export."""
import csv
import json
import platform
import os
import tempfile
import zipfile
from pathlib import Path
import importlib.metadata
import numpy as np
from .core import sparse_graph


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return value


def dump_json(path, data):
    Path(path).write_text(json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save_model(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    keys = ["x", "neighbors", "distances", "edges", "lengths", "frequency", "stable_edges", "stable_lengths", "stable_frequency", "component", "scale_components", "region", "landmarks", "xy", "pca_mean", "pca_axes", "coordinate_mask", "display_mean", "display_axes"]
    arrays = {key: model[key] for key in keys}
    arrays.update({f"profile__{key}": value for key, value in model["profiles"].items()})
    # Stage and validate the archive before replacing an existing frozen map.
    # A partial write must never become the only map in an output directory.
    handle, name = tempfile.mkstemp(prefix=".map-", suffix=".npz", dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Frozen map archive failed its integrity check.")
        os.replace(temporary, directory / "map.npz")
    finally:
        temporary.unlink(missing_ok=True)
    versions = {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "scikit-learn", "threadpoolctl", "contourpy"]}
    dump_json(directory / "manifest.json", model["manifest"] | {"python": platform.python_version(), "dependencies": versions})
    dump_json(directory / "ids.json", model["ids"])
    dump_json(directory / "roles.json", model["roles"])
    dump_json(directory / "h0.json", model["h0"])
    with (directory / "profiles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        keys = list(model["profiles"])
        writer.writerow(["id", "component", "navigation_cell", "roles"] + keys)
        for i, item_id in enumerate(model["ids"]):
            writer.writerow([item_id, model["component"][i], model["region"][i], "; ".join(model["roles"][i])] + [clean(model["profiles"][key][i]) for key in keys])


def load_model(directory):
    directory = Path(directory)
    with np.load(directory / "map.npz", allow_pickle=False) as data:
        model = {key: data[key].copy() for key in data.files if not key.startswith("profile__")}
        model["profiles"] = {key.split("__", 1)[1]: data[key].copy() for key in data.files if key.startswith("profile__")}
    for key in ["manifest", "ids", "roles", "h0"]:
        model[key] = json.loads((directory / f"{key}.json").read_text(encoding="utf-8"))
    if model["manifest"]["schema_version"] != 1:
        raise ValueError("Unsupported map schema version.")
    model["graph"] = sparse_graph(len(model["x"]), model["stable_edges"], model["stable_lengths"] / (model["stable_frequency"] + .01))
    return model


def region_summaries(model):
    result = []
    for label in np.unique(model["region"]):
        members = np.flatnonzero(model["region"] == label)
        row = {"id": int(label), "n": len(members), "representative": int(members[0]), "component": int(model["component"][members[0]])}
        for key, values in model["profiles"].items():
            subset = values[members]
            subset = subset[np.isfinite(subset)]
            row[key] = float(np.median(subset)) if len(subset) else None
        mask = model["region"][model["stable_edges"]] == label
        crossing = np.sum(mask[:, 0] != mask[:, 1])
        volume = np.sum(mask)
        complement = 2 * len(mask) - volume
        row["cut_conductance"] = float(crossing / max(1, min(volume, complement)))
        row["crossing_edges"] = int(crossing)
        row["landmarks"] = int(np.sum(np.isin(model["landmarks"], members)))
        row["terrain"] = "isolated cell" if crossing == 0 else ("narrow connection" if row["cut_conductance"] < .1 else "open connection")
        result.append(row)
    return result


def export_explorer(model, directory, evaluation=None, metadata=None, atlas=None):
    directory = Path(directory)
    metadata = metadata or [{} for _ in model["ids"]]
    if len(metadata) != len(model["ids"]): raise ValueError("Metadata must align to every embedding row.")
    points = []
    for i, item_id in enumerate(model["ids"]):
        points.append({"id": item_id, "xy": model["xy"][i], "region": int(model["region"][i]),
                       "component": int(model["component"][i]), "roles": model["roles"][i],
                       "metrics": {key: value[i] for key, value in model["profiles"].items()},
                       "neighbors": model["neighbors"][i, :model["manifest"]["signature_k"]],
                       "distances": model["distances"][i, :model["manifest"]["signature_k"]], "metadata": metadata[i]})
    from .atlas import build_atlas
    if atlas is None:
        atlas = build_atlas(model)
    if "terrain" not in atlas:
        from .terrain import enrich_terrain
        atlas["terrain"] = enrich_terrain(model, atlas)
    if "mesh" not in atlas:
        from .mesh import enrich_mesh
        atlas["mesh"] = enrich_mesh(model, atlas)
    if "flat" not in atlas:
        from .flat import enrich_flat
        atlas["flat"] = enrich_flat(model, atlas)
    data = {"manifest": model["manifest"], "points": points, "landmarks": model["landmarks"],
            "edges": np.column_stack([model["stable_edges"], np.maximum(model["stable_lengths"] / (model["stable_frequency"] + .01), 1e-8)]),
            "atlas": atlas, "regions": region_summaries(model), "h0": model["h0"],
            "evaluation": evaluation}
    dump_json(directory / "atlas.json", atlas)
    template = Path(__file__).with_name("explorer.html").read_text(encoding="utf-8")
    encoded = json.dumps(clean(data), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    shared = Path(__file__).with_name("geometry_views.js").read_text(encoding="utf-8")
    template = template.replace("__GEOMETRY_JS__", shared)
    template = template.replace("__TERRAIN_JS__", Path(__file__).with_name("terrain_overlay.js").read_text(encoding="utf-8"))
    template = template.replace("__MESH_GL_JS__", Path(__file__).with_name("mesh_gl.js").read_text(encoding="utf-8"))
    template = template.replace("__MESH_VIEW_JS__", Path(__file__).with_name("mesh_view.js").read_text(encoding="utf-8"))
    template = template.replace("__FLAT_VIEW_JS__", Path(__file__).with_name("flat_view.js").read_text(encoding="utf-8"))
    (directory / "explorer.html").write_text(template.replace("__ECOLOGY_DATA__", encoded), encoding="utf-8")
    from .terrain_lab import build_lab_data
    lab = json.dumps(clean(build_lab_data()), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    lab_template = Path(__file__).with_name("terrain_lab.html").read_text(encoding="utf-8")
    (directory / "terrain_lab.html").write_text(lab_template.replace("__LAB_DATA__", lab).replace("__GEOMETRY_JS__", shared).replace("__MESH_GL_JS__", Path(__file__).with_name("mesh_gl.js").read_text(encoding="utf-8")), encoding="utf-8")
    dump_json(directory / "regions.json", data["regions"])
    if evaluation is None:
        for filename in ["evaluation.json", "FEASIBILITY.md"]:
            (directory / filename).unlink(missing_ok=True)
    if evaluation:
        dump_json(directory / "evaluation.json", evaluation)
        (directory / "FEASIBILITY.md").write_text(feasibility_report(model, evaluation, atlas), encoding="utf-8")


def fmt(value, digits=3):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def feasibility_report(model, e, atlas=None):
    m, observed, held, routes = model["manifest"], e["observed"], e["heldout"], e["routes"]
    text = f'''# Measured feasibility report

This is a geometric feasibility demonstration, not a semantic or human-value detector.
The atlas presents its computed analyses directly. All numbers below come from the shipped run.

## Corpus and construction

- Items: {m['n']}; dimensions: {m['dimensions']}; encoder: {m['encoder']}.
- Corpus SHA-256: `{m['corpus_sha256']}`.
- Metric: {m['metric']}; exact batched neighbor backend.
- Scales: {m['scales']}; support threshold: {m['edge_support_threshold']}.
- Build time: {fmt(m['timings']['build_total_s'])} s. Validation time: {fmt(e['elapsed_s'])} s.
- Support graph components: {m['components']}; largest share: {fmt(observed['largest_component_share'])}.
- Navigation cells: {m['regions']}. These are graph Voronoi cells, not semantic classes.
- Sparse H0 infinite bars: {model['h0']['infinite_bars']}. H1/loops are not measured.

## Does the representation survive changes?

- Median eligible-universe subsample Jaccard: {fmt(observed['resample_jaccard'])}.
- Median 0.01-chord perturbation Jaccard: {fmt(observed['perturbation_jaccard'])}.
- Median original-vs-display neighbor Jaccard: {fmt(observed['projection_jaccard'])}.
- PCA display variance explained: {fmt(m['display_explained_variance'])}.

Low display Jaccard is a reason to inspect original neighbors and graph routes rather
than reading projected proximity as evidence. Nested k-edge support is a rank statistic;
it is never used as independent perturbation evidence. Resampling ranges are empirical
sensitivity ranges over {m['resamples']} replicates, not confidence intervals.

## Held-out geometric recovery

The map and PCA are fit only to {held['train_n']} training vectors; {held['heldout_n']}
vectors are held out. The target is the nearest training anchor under the original
query. Query vectors are then perturbed by 0.05 chord units.

| Measurement | Result |
|---|---:|
| Direct retrieval top-1 anchor recovery | {fmt(held['anchor_top1'])} |
| Direct retrieval top-10 anchor recovery | {fmt(held['anchor_top10'])} |
| Random anchor top-10 expectation | {fmt(held['random_anchor_top10_expectation'])} |
| Compressed retrieval top-10 recovery | {fmt(held['compressed_anchor_top10'])} |
| Centroid navigation-cell recovery | {fmt(held['cell_centroid_top1'])} |
| Majority navigation-cell baseline | {fmt(held['majority_cell_baseline'])} |
| Median correct-anchor rank | {fmt(held['anchor_median_rank'])} |

The direct-neighbor baseline already has the embedding vectors. The map adds inspection,
terrain and routing; these measurements do not establish better retrieval accuracy.
The cell baseline can expose misleading accuracy from a dominant cell.

## Navigation and structural influence

- Reachable sampled pairs: {routes['reachable']}/{routes['attempted']}.
- Median raw graph-length / direct-chord stretch: {fmt(routes['median_geometric_stretch'])}.
- Median support-weighted path-cost / direct-chord ratio: {fmt(routes['median_support_weighted_cost_ratio'])}.
- Median Dijkstra expansions, including unreachable attempts: {fmt(routes['median_expansions_all_attempts'])}.
- Reachable fraction after one middle-edge ablation: {fmt(routes['middle_edge_ablation_survival'])}.
- Mean route-edge retention after independent geometry noise: {fmt(routes['mean_route_edge_retention_under_noise'])}.
- Fixed-graph articulation vertices: {observed['articulations']}.

Deletion influence is exact on the retained graph, with no neighbor refill. A redundant
path can make deletion influence zero even at an obvious connector. Sampled hop-path
betweenness complements it. Local effective next-hop count is not global route entropy.

## Null controls

One realization per null; these are descriptive comparisons, with no p-values.
The Gaussian control matches mean/covariance in expectation before renormalization,
not local density. Uniform-sphere structure is not a model of all irrelevant data.

| Cloud | Components | Median local dimension | Perturbation Jaccard | Display Jaccard |
|---|---:|---:|---:|---:|
| Observed | {observed['components']} | {fmt(observed['local_dimension'])} | {fmt(observed['perturbation_jaccard'])} | {fmt(observed['projection_jaccard'])} |
'''
    for name, values in e["nulls"].items():
        text += f"| {name} | {values['components']} | {fmt(values['local_dimension'])} | {fmt(values['perturbation_jaccard'])} | {fmt(values['projection_jaccard'])} |\n"
    if e["mechanism_checks"]:
        text += "\n## Planted mechanism checks\n\nLabels below are synthetic evaluation metadata, never geometry inputs. AUC is an\noriented rank-separation diagnostic, not learned detector performance. 0.5 is no\nrank separation; 1 is complete separation. These checks are not semantic validation.\n\n| Contrast | Measurement | AUC |\n|---|---|---:|\n"
        for item in e["mechanism_checks"]:
            text += f"| {item['question']} | {item['measurement']} | {fmt(item['auc'])} |\n"
    if atlas:
        text += f"""
## Cartographic interface

The interface is the computed map, with no questionnaire or evidence-entry workflow.

- {m['n']:,} points are embedded and represented; no display sampling.
- {len(atlas['islands'])} substantial connected landmasses and {len(atlas['minor_components'])} tiny components.
- {len(atlas['roads'])} mapped road paths and {atlas['passes']} marked regional passes.
- Cartographic construction: {atlas['elapsed_s']:.3f} seconds.
- Median original-neighbor / atlas-neighbor Jaccard: {np.median(atlas['fidelity']):.3f}.
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

"""
    if atlas and 'mesh' in atlas:
        mesh=atlas['mesh']
        text += f'''
## 3D dimensionality view

The optional orbitable mesh shows local neighbor envelopes and ordered spectra
directly on the map. Hue is the variance omitted by the first two local covariance
axes. Default height is covariance participation ratio, scaled to the neighborhood
rank bound {mesh['covariance_cap']:.0f}. It is distinct from radius-growth dimension.
All-point spectral analysis and representative hull construction: {mesh['elapsed_s']:.3f} s.

Surface facets are rendering geometry, not a dimension count or additional graph
edges. Representative meshes have normalized footprints in local PCA frames.
Color and spectra expose variation beyond their three displayed axes. Without
WebGL, relief and inline spectra remain available.

'''
    if atlas and 'flat' in atlas:
        text += f'''
## Flat field atlas

The default view keeps every position strictly planar. Nine measured fields can
independently drive color and labelled isolines. Four presets combine
dimensionality, density/redundancy, route alternatives and stability. Spectrum
rings and first-two-axis ellipses show original local covariance at representatives.
Fields and contours are masked per component; unsupported gaps are not bridged.
Flat-field preparation and contour tracing: {atlas['flat']['elapsed_s']:.3f} s.

'''
    text += '''
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
'''
    return text
