"""Verify the shipped large atlas and refresh its measured demonstration record."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import connected_components
from geometric_ecology.io import load_model
from geometric_ecology.core import sparse_graph


def main():
    examples=Path(__file__).resolve().parent
    root=examples/'large_map'
    model=load_model(root)
    atlas=json.loads((root/'atlas.json').read_text())
    evaluation=json.loads((root/'evaluation.json').read_text())
    metadata=json.loads((root/'metadata.json').read_text())
    for road in atlas['roads']:
        path=road['path']
        assert all(float(model['graph'][u,v])>0 for u,v in zip(path,path[1:]))
        assert len(set(model['component'][path]))==1
        cost=sum(float(model['graph'][u,v]) for u,v in zip(path,path[1:]))
        assert abs(cost-road['cost'])<1e-6
    for bundle in atlas['terrain']['route_ambiguity']['examples']:
        for path in bundle['point_paths']:
            assert all(float(model['graph'][u,v])>0 for u,v in zip(path,path[1:]))
    cuts=[r for r in atlas['roads'] if r['narrow']]
    endpoints=model['region'][model['stable_edges']]
    for road in cuts:
        a,b=road['a'],road['b']
        keep=~(((endpoints[:,0]==a)&(endpoints[:,1]==b))|((endpoints[:,0]==b)&(endpoints[:,1]==a)))
        graph=sparse_graph(len(model['x']),model['stable_edges'][keep],np.ones(int(keep.sum())))
        _, labels=connected_components(graph,directed=False)
        assert labels[road['path'][0]]!=labels[road['path'][-1]]
    passes=[r for r in atlas['roads'] if 'pass_name' in r]
    info={'all_points':len(model['x']),'roads_checked':len(atlas['roads']),
          'narrow_segments_checked':len(cuts),'grouped_passes':len(passes),
          'pass_anchor_planted_groups':[metadata[r['pass_point']]['demo_group'] for r in passes]}
    (root/'atlas_integrity.json').write_text(json.dumps(info,indent=2)+'\n')
    lines=['# Larger terrain demonstration · 20,000 points','',
        'The primary included map is `large_map/explorer.html`. All 20,000 points are represented, without display sampling. Computed analyses appear as cartographic layers, with no question or answer workflow.','',
        '| Measurement | Result |','|---|---:|',
        f"| Points / dimensions | {len(model['x']):,} / {model['x'].shape[1]} |",
        f"| Exact geometry construction | {model['manifest']['timings']['build_total_s']:.2f} s |",
        f"| Cartographic construction | {atlas['elapsed_s']:.2f} s |",
        f"| Additional fields, lenses and bounded regional route analysis | {atlas['terrain']['elapsed_s']:.2f} s |",
        f"| Independent geometric checks | {evaluation['elapsed_s']:.2f} s |",
        f"| Substantial connected landmasses | {len(atlas['islands'])} |",
        f"| Tiny disconnected components / points | {len(atlas['minor_components'])} / {atlas['small_component_n']} |",
        f"| Actual mapped road paths | {len(atlas['roads'])} |",
        f"| Grouped narrow passes / critical crossing segments | {atlas['passes']} / {len(cuts)} |",
        f"| Median original / atlas neighbor Jaccard | {np.median(atlas['fidelity']):.3f} |",
        f"| Median original / raw PCA neighbor Jaccard | {evaluation['observed']['projection_jaccard']:.3f} |",
        f"| Held-out vectors excluded from fit | {evaluation['heldout']['heldout_n']:,} |",'',
        'Observed browser startup was about 1.0–1.4 seconds, with roughly 50 ms overview redraws at 1680×1080 on the execution host. All 20,000 point positions fit inside the initial view; depth testing can occlude individual points in 3D. These are observations, not performance guarantees. The HTML is large because it embeds all points, profiles and graph edges.','',
        '## Integrity of what is drawn','',
        'Every mapped road was checked against the retained graph and its accumulated cost. Each marked narrow segment was independently checked by removing its cell-pair crossing edges from the original graph and confirming endpoint disconnection. Adjacent qualifying segments are grouped into one pass label rather than presenting navigation-cell subdivisions as separate passages.','',
        f"The marked pass anchor's post-hoc planted group is {', '.join(info['pass_anchor_planted_groups'])}. This is an evaluation diagnostic only: planted names did not determine the layout, terrain or pass selection.",'',
        '## What the larger map makes visible','',
        '- Connected basins linked by a narrow passage have a continuous, explicitly drawn road network.',
        '- Disconnected components occupy separate chart areas with their own shorelines.',
        '- A ring has a closed road circuit; its uniform arcs are not labelled critical passes.',
        '- High-dimensional patches have measured relief and local direction spectra/ribbons.',
        '- A dense redundant patch can still show resampling/compression disagreement, distinguishing density from robustness.','',
        '## Limits of a map of high-dimensional geometry','',
        'Graph-distance layout is followed by monotone population magnification, so drawn distances and areas are schematic. Neighbor fidelity remains limited: this is a connectivity and terrain atlas, not an exact 2D reconstruction. Roads provide explicit traversability evidence. Density contours and dimension colors interpolate original-space measurements. Water between named islands means no retained-graph route, not absolute impossibility in the complete embedding space.','',
        'The earlier 1,100-point example remains under `demo/`, re-exported with the same terrain interface. Its null controls and the three-seed results remain small-corpus references. The large run omits two extra large null-cloud builds but includes held-out, perturbation, route and planted-mechanism checks.','']
    (examples/'LARGE_DEMO.md').write_text('\n'.join(lines))
    print(json.dumps(info))

if __name__=='__main__':main()
