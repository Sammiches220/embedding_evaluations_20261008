"""Turn measured graph geometry into an automatically readable terrain atlas.

Layout is schematic. Components are packed into disjoint discs. Inside each
component, landmark graph distances drive a classical-MDS/Nystrom layout.
Terrain rasters and contours interpolate ORIGINAL-space measurements, rather
than calling density in the 2D drawing the density of the embedding corpus.
"""
import base64
import struct
import time
import zlib
from collections import defaultdict

import contourpy
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata
from scipy.sparse.csgraph import dijkstra
from .core import percentile, neighbor_overlap, knn


def png_data(rgba):
    """Encode a generated numeric RGBA grid without a plotting dependency."""
    height, width, _ = rgba.shape
    def chunk(tag, body):
        return struct.pack('!I', len(body)) + tag + body + struct.pack('!I', zlib.crc32(tag + body) & 0xffffffff)
    scan = b''.join(b'\x00' + row.tobytes() for row in rgba)
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 6, 0, 0, 0))
    data += chunk(b'IDAT', zlib.compress(scan, 8)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')


def geodesic_layout(graph, representatives):
    """Landmark Isomap-style layout; no semantic or synthetic labels enter it."""
    n = graph.shape[0]
    if n == 1:
        return np.zeros((1, 2)), {"landmarks": 1, "positive_variance_2d": 1.0}
    reps = list(dict.fromkeys(map(int, representatives)))
    # A minimum of four landmarks reveals two-dimensional structure even in
    # components assigned only one navigation cell.
    distances = dijkstra(graph, directed=False, indices=reps)
    nearest = distances.min(axis=0)
    while len(reps) < min(12, n):
        nearest[reps] = -1
        j = int(np.argmax(nearest))
        reps.append(j)
        dj = dijkstra(graph, directed=False, indices=j)
        distances = np.vstack([distances, dj])
        nearest = np.minimum(nearest, dj)
    squared = distances[:, reps] ** 2
    mean = squared.mean(axis=1)
    overall = squared.mean()
    gram = -.5 * (squared - mean[:, None] - mean[None, :] + overall)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    positive = np.maximum(eigenvalues[order], 0)
    count = min(2, int(np.sum(positive > 1e-10)))
    xy = np.zeros((n, 2))
    if count:
        basis = eigenvectors[:, order[:count]] / np.sqrt(positive[:count])
        cross = -.5 * (distances ** 2 - mean[:, None] - (distances ** 2).mean(axis=0)[None, :] + overall)
        xy[:, :count] = cross.T @ basis
    # Degenerate equivalent points are permitted to overlap. No jitter is used
    # to manufacture geometry where the graph distances have none.
    return xy, {"landmarks": len(reps), "positive_variance_2d": float(positive[:2].sum() / max(positive.sum(), 1e-10))}


def _contours(xcenters, ycenters, field, levels, valid=None):
    z = np.ma.array(field, mask=~valid) if valid is not None else field
    generator = contourpy.contour_generator(x=xcenters, y=ycenters, z=z, name='serial')
    return [{"level": float(level), "paths": [p[::max(1, len(p)//220)].tolist() for p in generator.lines(level) if len(p) >= 4]} for level in levels]


def terrain_surface(xy, density, dimension, stability, compression, grid=136):
    """Kernel-interpolated original-space measurements on a bounded island."""
    low, high = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(high - low, .05)
    low, high = low - .09 * span - .025, high + .09 * span + .025
    limits = [(low[0], high[0]), (low[1], high[1])]
    def hist(weights=None):
        return np.histogram2d(xy[:, 1], xy[:, 0], bins=grid, range=[limits[1], limits[0]], weights=weights)[0]
    occupancy = gaussian_filter(hist(), 2.0)
    maximum = occupancy.max()
    threshold = max(.006 * maximum, .004)
    visible = occupancy > threshold
    def field(values):
        return gaussian_filter(hist(np.nan_to_num(values, nan=.5)), 2.0) / np.maximum(occupancy, 1e-9)
    den, dim, fog, comp = field(density), field(dimension), field(1 - stability), field(1 - compression)
    # Green plains and ochre ruggedness are independent from brightness/density.
    green, ochre = np.array([132, 175, 128]), np.array([192, 155, 104])
    rugged = np.clip((dim - .4) / .55, 0, 1)
    rgb = green[None, None] * (1 - rugged[:, :, None]) + ochre[None, None] * rugged[:, :, None]
    rgb = rgb * (.96 + .09 * den[:, :, None]) + (1 - den[:, :, None]) * 15
    alpha = np.clip((occupancy - threshold) / max(.14 * maximum, .03), 0, 1)
    rgba = np.dstack([np.clip(rgb, 0, 255), 225 * alpha]).astype(np.uint8)
    # Row zero in PNG is the top, whereas Cartesian y grows upwards.
    image = png_data(rgba[::-1])
    xe, ye = np.linspace(low[0], high[0], grid + 1), np.linspace(low[1], high[1], grid + 1)
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    return {"bounds": [float(low[0]), float(low[1]), float(high[0]), float(high[1])], "image": image,
            "coasts": _contours(xc, yc, occupancy, [threshold]),
            "density_contours": _contours(xc, yc, den, [.25, .5, .75, .9], visible),
            "fog_contours": _contours(xc, yc, gaussian_filter(((fog > .55) & visible).astype(float), 1.0), [.45]),
            "compression_contours": _contours(xc, yc, gaussian_filter(((comp > .65) & visible).astype(float), 1.0), [.45])}


def build_atlas(model):
    started = time.perf_counter()
    n = len(model['x'])
    components = model['component']
    labels, counts = np.unique(components, return_counts=True)
    order = labels[np.argsort(counts)[::-1]]
    threshold = max(30, int(.0015 * n))
    major = [int(c) for c in order if np.sum(components == c) >= threshold]
    minor = [int(c) for c in order if int(c) not in major]
    # All-small corpora still get useful cartography of their largest component.
    if not major:
        major, minor = [int(order[0])], list(map(int, order[1:]))
    graph, region, x = model['graph'], model['region'], model['x']
    xy = np.zeros((n, 2))
    islands, circles = [], []
    largest = int(max(counts))
    metrics = model['profiles']
    density = percentile(-metrics['neighbor_radius'])
    dimension = percentile(metrics['local_dimension'])
    centrality = percentile(metrics['sampled_betweenness'])
    compression_key = sorted([k for k in metrics if k.startswith('pca_')], key=lambda k:int(k.split('_')[1]))[0]
    stability = metrics['perturbation_jaccard']
    cells, cell_members, medoids = [], {}, {}
    for rid in np.unique(region):
        inds = np.flatnonzero(region == rid)
        centroid = x[inds].mean(axis=0)
        representative = int(inds[np.argmin(np.linalg.norm(x[inds] - centroid, axis=1))])
        rid = int(rid)
        cell_members[rid], medoids[rid] = inds, representative
    for rank, comp in enumerate(major):
        inds = np.flatnonzero(components == comp)
        local_lookup = {int(j): i for i, j in enumerate(inds)}
        representatives = [local_lookup[medoids[int(r)]] for r in np.unique(region[inds])]
        local, info = geodesic_layout(graph[inds][:, inds], representatives)
        local -= (local.max(axis=0) + local.min(axis=0)) / 2
        # Cartographic area magnification: dense basins should remain legible
        # beside long thin corridors. This is a monotone axis transform of the
        # graph-distance layout, not a claim to preserve metric distances.
        span = np.ptp(local, axis=0)
        longest = max(float(span.max()), 1e-8)
        for axis in range(2):
            if span[axis] > longest * 1e-6:
                linear = local[:,axis] / (span[axis]/2)
                quantile = 2 * (rankdata(local[:,axis], method='average')-.5) / len(local) - 1
                local[:,axis] = (.55*linear+.45*quantile) * max(span[axis]/longest, .55)
        radius = 1.0 + 5.0 * np.sqrt(len(inds) / largest)
        extent = max(np.linalg.norm(local, axis=1).max(), 1e-8)
        local *= .80 * radius / extent
        if not circles:
            center = np.array([0., 0.])
        else:
            # Deterministic non-overlapping circle packing. Empty water expresses
            # disconnection, not a measured inter-component distance.
            candidates = []
            for existing, er in circles:
                for angle in np.linspace(0, 2*np.pi, 72, endpoint=False):
                    pos = existing + (er + radius + .75) * np.array([np.cos(angle), np.sin(angle)])
                    if all(np.linalg.norm(pos - p) >= r + radius + .74 for p, r in circles):
                        candidates.append(pos)
            center = min(candidates, key=lambda p: np.linalg.norm(p + [.8, -.5])) if candidates else np.array([sum(r*2+1 for _,r in circles), 0])
        circles.append((center, radius))
        xy[inds] = local + center
        islands.append({"component": comp, "name": f"Island {rank+1}", "n": len(inds), "center": center.tolist(), "radius": radius,
                        "density": float(np.nanmedian(density[inds])), "dimension": float(np.nanmedian(dimension[inds])),
                        "stability": float(np.nanmedian(stability[inds])), "layout": info,
                        "surface": terrain_surface(xy[inds], density[inds], dimension[inds], stability[inds], metrics[compression_key][inds])})
    # Small components are drawn in an explicitly labelled disconnected-islet bay.
    small_box = None
    if minor:
        right = max(c[0][0] + c[1] for c in circles)
        top = max(c[0][1] + c[1] for c in circles)
        columns = max(3, int(np.ceil(np.sqrt(len(minor)))))
        rows = int(np.ceil(len(minor) / columns))
        small_box = [right + .8, top - max(2.6, rows*.45), right + .8 + max(3.0, columns*.45), top]
        for slot, comp in enumerate(minor):
            inds = np.flatnonzero(components == comp)
            row, column = divmod(slot, columns)
            cx = small_box[0] + (column+.5)*(small_box[2]-small_box[0])/columns
            cy = small_box[3] - (row+.5)*(small_box[3]-small_box[1])/rows
            residual = model['xy'][inds] - model['xy'][inds].mean(axis=0)
            extent = max(np.linalg.norm(residual, axis=1).max(), 1e-8)
            xy[inds] = np.array([cx,cy]) + .12 * residual/extent
    for rid, inds in cell_members.items():
        comp = int(components[inds[0]])
        if comp not in major: continue
        dp, ip, sp = float(np.nanmedian(density[inds])), float(np.nanmedian(dimension[inds])), float(np.nanmedian(stability[inds]))
        terrain = 'dense plain' if dp > .75 else ('rugged ground' if ip > .75 else ('sparse frontier' if dp < .2 else 'connected terrain'))
        if sp < .45: terrain = 'unstable ground'
        cells.append({"id": rid, "component": comp, "n": len(inds), "center": xy[medoids[rid]].tolist(),
                      "representative": medoids[rid], "terrain": terrain, "density_percentile": dp,
                      "dimension_percentile": ip, "perturbation_stability": sp,
                      "compression_jaccard": float(np.nanmedian(metrics[compression_key][inds]))})
    crossings = defaultdict(list)
    degrees = np.zeros(int(region.max())+1)
    for edge_index, (u, v) in enumerate(model['stable_edges']):
        a,b = int(region[u]),int(region[v])
        degrees[a] += 1
        degrees[b] += 1
        if a!=b: crossings[tuple(sorted((a,b)))].append(edge_index)
    quotient = defaultdict(set)
    for a,b in crossings:
        quotient[a].add(b)
        quotient[b].add(a)
    def critical_crossing(a,b):
        # Delete every edge between this cell pair in the quotient. If the
        # regions cannot reconnect, the crossing is a real regional cut.
        seen, queue = {a}, [a]
        while queue:
            u = queue.pop()
            for v in quotient[u]:
                if (u==a and v==b) or (u==b and v==a): continue
                if v==b: return False
                if v not in seen: seen.add(v);queue.append(v)
        return True
    roads = []
    local_routes = {}
    for rid, inds in cell_members.items():
        if int(components[inds[0]]) not in major: continue
        lookup = {int(j):i for i,j in enumerate(inds)}
        start = lookup[medoids[rid]]
        dd, pred = dijkstra(graph[inds][:,inds], directed=False, indices=start, return_predecessors=True)
        local_routes[rid] = (inds, lookup, dd, pred)
    def trace(rid, endpoint):
        inds, lookup, dd, pred = local_routes[rid]
        j = lookup[int(endpoint)]
        if not np.isfinite(dd[j]): return []
        path = [int(endpoint)]
        while path[-1] != medoids[rid]:
            j = int(pred[j])
            if j < 0: return []
            path.append(int(inds[j]))
        return path[::-1]
    # Each road follows a real shortest path between region medoids. Its cut
    # count/support comes from ALL graph edges crossing the two cells.
    for (a,b), indices in crossings.items():
        if int(components[medoids[a]]) not in major: continue
        comp = int(components[medoids[a]])
        candidates = []
        for ei in indices:
            u,v = map(int, model['stable_edges'][ei])
            if int(region[u]) != a: u,v = v,u
            da = local_routes[a][2][local_routes[a][1][u]]
            db = local_routes[b][2][local_routes[b][1][v]]
            w = float(graph[u,v])
            candidates.append((da+w+db,u,v))
        cost,u,v = min(candidates)
        if not np.isfinite(cost):
            # A numerical Voronoi tie could leave a cell disconnected. Never
            # invent a road or a pass through such a cell.
            continue
        path = trace(a,u) + trace(b,v)[::-1]
        support = float(np.mean(model['stable_frequency'][indices]))
        crossing_count = len(indices)
        conductance = float(crossing_count/max(1,min(degrees[a],degrees[b])))
        bc = float(np.nanmedian(centrality[path]))
        critical = critical_crossing(a,b)
        narrow = conductance < .04 and bc > .6 and critical
        roads.append({"a":a,"b":b,"component":comp,"path":path,"coordinates":xy[path],
                      "cost":float(cost),"crossing_edges":crossing_count,"conductance":conductance,
                      "scale_support":support,"stability":float(np.nanmedian(stability[path])),
                      "centrality_percentile":bc,"regional_cut":bool(critical),"narrow":bool(narrow)})
    # Distill the network into a spanning backbone plus the strongest alternate
    # corridors. This is always labelled a simplified network, not every edge.
    parent = {rid:rid for rid in medoids}
    def root(i):
        while parent[i]!=i: parent[i]=parent[parent[i]];i=parent[i]
        return i
    backbone = []
    for road in sorted(roads,key=lambda r:r['cost']):
        a,b = root(road['a']),root(road['b'])
        if a!=b:
            parent[a]=b
            road['backbone']=True
            backbone.append(road)
    selected = {(r['a'],r['b']) for r in backbone}
    alternatives = [r for r in sorted(roads,key=lambda r:r['crossing_edges'],reverse=True) if (r['a'],r['b']) not in selected]
    for r in alternatives[:max(4,len(backbone)//4)]: r['backbone']=False
    visible_roads = backbone + alternatives[:max(4,len(backbone)//4)]
    candidates = [r for r in visible_roads if r['narrow']]
    # A long pass may cross several navigation cells. Group adjacent critical
    # segments so cell granularity does not manufacture repeated pass labels.
    groups = []
    remaining = list(candidates)
    while remaining:
        group = [remaining.pop(0)]
        vertices = {group[0]['a'],group[0]['b']}
        changed = True
        while changed:
            changed = False
            for road in remaining[:]:
                if road['a'] in vertices or road['b'] in vertices:
                    group.append(road)
                    vertices.update([road['a'],road['b']])
                    remaining.remove(road)
                    changed = True
        groups.append(group)
    groups.sort(key=lambda group:min(r['conductance'] for r in group))
    for i, group in enumerate(groups[:8]):
        representative = min(group,key=lambda r:r['conductance'])
        points = np.unique(np.concatenate([r['path'] for r in group]))
        center = xy[points].mean(axis=0)
        midpoint = int(points[np.argmin(np.linalg.norm(xy[points]-center,axis=1))])
        representative['pass_name'] = f"Pass {i+1}"
        representative['pass_point'] = midpoint
        representative['pass_road_count'] = len(group)
    passes = groups[:8]
    # Fidelity is evaluated separately; the atlas does not pretend that graph
    # distance can be embedded without loss in the plane.
    _, nn2 = knn(xy, model['manifest']['signature_k'], exclude_self=True)
    fidelity = neighbor_overlap(model['neighbors'][:,:model['manifest']['signature_k']],nn2)
    return {"schema_version":3,"xy":xy,"islands":islands,"minor_components":minor,
            "small_component_box":small_box,"small_component_n":int(sum(np.sum(components==c) for c in minor)),
            "cells":cells,"roads":visible_roads,"corridors":roads,"all_corridor_count":len(roads),"passes":len(passes),
            "density_percentile":density,"dimension_percentile":dimension,
            "fidelity":fidelity,"compression_key":compression_key,
            "elapsed_s":time.perf_counter()-started,
            "notes":"Schematic graph-distance layout with monotone population magnification for readable basins. Water means no graph connection. Island spacing is arbitrary. Contours interpolate original-space density ranks; ochre indicates high original local-dimension rank. Fog denotes median neighbor Jaccard below 0.45 under 0.01-chord perturbation. Roads simplify real graph paths; unshown edges still exist."}
