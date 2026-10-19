"""Small scientific plots and report markup for normalization / topology."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def enhancement_plots(out, enhancements, embedding_names=None):
    """Save shared-geometry plots for each embedding and a combined comparison."""
    topo=enhancements['topology']
    if topo['status']!='ok': return
    folder=Path(out)/'enhancements';folder.mkdir(exist_ok=True);plots=[];by_embedding={}
    names=embedding_names or {}
    models=sorted(set(row['embedding'] for row in topo['structure']))
    for index,space in enumerate(topo['radii']):
        for selected in [None]+models:
            visible=models if selected is None else [selected]
            figure,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
            for model in visible:
                color='#1c6c9c' if model=='A' else '#b06518'
                for repeat in range(topo['actual_repetitions']):
                    rows=[r for r in topo['structure'] if r['space']==space and r['embedding']==model and r['repeat']==repeat]
                    x=[r['radius'] for r in rows]
                    axes[0,0].step(x,[r['beta0'] for r in rows],where='post',color=color,alpha=.65,label=names.get(model,model) if repeat==0 else None)
                    axes[0,1].step(x,[r['largest_component_share'] for r in rows],where='post',color=color,alpha=.65)
            axes[0,0].set_title('H0: components at evaluated thresholds');axes[0,0].set_ylabel('Connected components');axes[0,0].legend()
            axes[0,1].set_title('Largest component share');axes[0,1].set_ylabel('Share of sampled rows');axes[0,1].set_ylim(0,1.03)
            for dimension,ax in [(0,axes[1,0]),(1,axes[1,1])]:
                intervals=[r for r in topo['barcodes'] if r['space']==space and r['dimension']==dimension and r['repeat']==0 and r['embedding'] in visible]
                cutoff=topo['radii'][space][-1]
                intervals.sort(key=lambda row:(float('inf') if not np.isfinite(row['death']) else row['death']-row['birth']),reverse=True)
                for y,row in enumerate(intervals[:50]):
                    end=row['death'] if np.isfinite(row['death']) else cutoff
                    color='#1c6c9c' if row['embedding']=='A' else '#b06518'
                    ax.hlines(y,row['birth'],end,color=color,linewidth=2)
                    if not np.isfinite(row['death']): ax.plot(end,y,marker='>',color=color,markersize=4)
                ax.set_title(f'H{dimension}: up to 50 longest bars, first sample');ax.set_ylabel('Intervals')
                if not intervals:
                    computed=any(r['space']==space and r['status']=='ok' and r['embedding'] in visible for r in topo.get('h1_status',[]))
                    message=('No nonzero H1 intervals recorded' if computed else 'H1 disabled or skipped; inspect status') if dimension else 'No intervals'
                    ax.text(.5,.5,message,ha='center',transform=ax.transAxes)
            for ax in axes.flat: ax.set_xlabel('Pairwise distance threshold');ax.grid(alpha=.15)
            figure.suptitle(f'{space} · '+', '.join(names.get(m,m) for m in visible)+f' · H0 n={topo["n_sample"]}')
            suffix='' if selected is None else '_'+selected.lower()
            path=folder/f'topology_{index+1:02d}{suffix}.png';figure.savefig(path,dpi=140);plt.close(figure)
            relative=str(path.relative_to(out))
            if selected is None: plots.append(relative)
            else: by_embedding[space+':'+selected]=relative
    topo['plots']=plots;topo['plots_by_embedding']=by_embedding


def enhancement_markup():
    return '''<section class="card" id="ea-workspace"><h2>Direction, magnitude and topology</h2>
<p>These analyses complement the neighborhood overview. The feature selection above applies here. Choose the embedding, probe and metrics for normalization, or the embedding, geometry and measurements for topology. Topology has no prediction probe; row brushing does not recalculate the sampled topology or normalization experiments.</p>
<details open><summary>Unit-normalization comparisons</summary><p id="ea-normalization-scope" class="muted"></p><label for="ea-normalization-model">Embedding </label><select id="ea-normalization-model"></select> <label for="ea-normalization-probe">Probe </label><select id="ea-normalization-probe"></select>
<fieldset class="ex-metric-toggles"><legend>Color-coded probe metrics</legend><div id="ea-normalization-metrics" class="ex-metric-options"></div></fieldset>
<div class="ex-table-wrap"><table class="ex-overview"><thead><tr><th>Feature</th><th>Raw + standardized</th><th>Unit direction</th><th>Unit + standardized</th><th>Magnitude only</th><th>Brief interpretation</th></tr></thead><tbody id="ea-normalization-table"></tbody></table></div>
<p>Green/red compares each pipeline with its own training-only dummy baseline. The unit pipeline keeps input norm one; StandardScaler generally moves unit inputs off the sphere. Magnitude-only scores test whether length carries recoverable information. Timing, solver iterations and convergence warnings describe this fit; they do not certify encoder-training stability.</p>
<p><a href="normalization_probe_scores.csv">Normalization scores CSV (when enabled)</a> · <a href="normalization_probe_splits.csv">Identical-split provenance</a> · <a href="embedding_norms.csv">Original vector norms</a></p></details>
<details open><summary>Filtration and feature organization</summary><p id="ea-topology-scope" class="muted"></p>
<label for="ea-topology-space">Geometry </label><select id="ea-topology-space"></select> <label for="ea-topology-model">Embedding </label><select id="ea-topology-model"></select> <fieldset class="ex-metric-toggles"><legend>Topology measurements</legend><div id="ea-topology-metrics" class="ex-metric-options"></div></fieldset>
<p id="ea-topology-readout" class="ex-readout"></p><div class="ex-table-wrap"><table class="ex-overview"><thead><tr><th>Feature / sample support</th><th>Every evaluated radius</th><th>Brief interpretation</th></tr></thead><tbody id="ea-topology-table"></tbody></table></div><p id="ea-h1-status" class="muted"></p><div id="ea-topology-plots" class="figures"></div>
<p><a href="topology_feature_summary.csv">Feature filtration summaries</a> · <a href="topology_feature_trials.csv">Per-sample measurements</a> · <a href="topology_structure.csv">Components and edge counts</a> · <a href="topology_barcodes.csv">H0/H1 intervals</a> · <a href="topology_sample_rows.csv">Exact sampled IDs</a></p></details>
<details><summary>How these findings relate to representation quality</summary><p>A radius filtration adds edges and fills cliques in a nested sequence. H0 records component mergers; H1 records loops that survive triangle filling. This bounded H1 calculation uses exact sample distances and GF(2) boundary reduction. A loop alive at the cutoff is right-censored, not proof of an infinitely persistent semantic feature. H2 is not computed.</p>
<p>Component entropy is known-row weighted conditional entropy using the fixed audited vocabulary. Tags use mean binary entropy; numeric fields use within-component variance divided by full-source variance. Whole known label/tag rows are shuffled within each sample, preserving its label counts and tag sets. At singleton and fully merged extremes, the permutation contrast is mechanically zero. Positive gap in between supports annotation organization for that null, with no automatic significance claim.</p>
<p>Different samples use the same radii; A/B use identical IDs. Unless radii are explicit, thresholds are pooled pair-distance quantiles across models and repeated samples in each geometry. Repeated-sample min/max ranges are descriptive rather than confidence intervals. Correlated posts, incomplete annotations and source artifacts can still explain structure.</p>
<p>Chordal distance uses unit-normalized directions; raw Euclidean distance also uses magnitude. Their barcode scales differ. Topology can remain meaningful while linear separability changes. Stable loops or components require task-relevant labels or examples before being called useful; the number of loops is not an embedding-quality score.</p>
<p>The same selected probe is compared across normalization variants. Linear probes compare fixed regularization unless a multi-value grid enables training-only CV. Ridge uses alpha; logistic regression uses C, its inverse penalty strength. The outer split and group policy are identical between variants. No encoder is retrained and no hyperspherical training objective is imposed.</p>
<p>At millions of rows, use bounded matched samples or an independently validated sparse/ANN backend. Exact all-pairs persistence is not supplied for that scale. Limits and skipped analyses are reported explicitly; the main feature overview remains available.</p></details></section>'''
