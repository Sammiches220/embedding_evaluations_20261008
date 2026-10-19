"""Standalone figures and an offline HTML report with a feature selector."""
from pathlib import Path
import base64
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .cluster_metadata import display_label
from .explorer_html import explorer_markup
from .enhancement_report import enhancement_markup

plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':.15,'font.size':10})


def save(fig,path):
    fig.tight_layout(); fig.savefig(path,bbox_inches='tight'); plt.close(fig)


def feature_plots(out,source,summary,labels,null_score,probes,missing_summary,missing_geometry,
                  values,valid,kind,coords=None,names=None,prediction_geometry=None,cooccurrence=None,label_metadata=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if kind!='continuous' and not summary.empty:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for prefix in ['micro','macro']:
            axes[0].plot(summary.k,summary[prefix+'_purity'],'o-',label=prefix)
            axes[1].plot(summary.k,summary[prefix+'_lift'],'o-',label=prefix)
        axes[0].plot(summary.k,summary.chance_purity,'--',color='gray',label='prevalence baseline')
        axes[1].axhline(1,color='gray',linestyle='--',label='population baseline')
        if not null_score.empty and 'null_macro_lift_mean' in null_score:
            axes[1].plot(null_score.k,null_score.null_macro_lift_mean,':',label='permutation mean')
        for a,title in zip(axes,['Purity: fraction sharing a label','Lift: enrichment over prevalence']):
            a.set(title=title,xlabel='Neighborhood size k'); a.legend(); a.set_xscale('log')
        save(fig,out/'purity_lift.png')
        if not labels.empty:
            k=summary.iloc[(summary.k-50).abs().argmin()].k
            top=labels[labels.k==k].sort_values('lift',ascending=False).head(20).iloc[::-1]
            fig,ax=plt.subplots(figsize=(9,max(4,len(top)*.27)))
            ax.barh(top['label_display'] if 'label_display' in top else top.label.astype(str),top.lift,color='#276a83'); ax.axvline(1,color='gray',linestyle='--')
            ax.set(title=f'Per-label lift at k={int(k)} (see CSV for support)',xlabel='Lift')
            save(fig,out/'per_label_lift.png')
    elif not summary.empty:
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,col,title in zip(axes,['mean_neighbor_abs_difference','variance_ratio_to_global','morans_i'],
                               ['Neighbor absolute difference','Local/global variance','Moran’s I']):
            ax.plot(summary.k,summary[col],'o-'); ax.set(title=title,xlabel='k'); ax.set_xscale('log')
        save(fig,out/'continuous_geometry.png')
    fig,ax=plt.subplots(figsize=(8,4))
    if kind=='continuous':
        ax.hist(values[valid],bins=40,color='#276a83'); ax.set(xlabel='Observed value',ylabel='Items',title='Source distribution')
    elif not source.empty:
        top=source.sort_values('count',ascending=False).head(20)
        ax.bar(range(len(top)),top.prevalence,color='#276a83'); ax.set_xticks(range(len(top)),top['label_display'] if 'label_display' in top else top.label.astype(str),rotation=60,ha='right')
        ax.set(ylabel='Item prevalence',title='Most prevalent source labels')
    save(fig,out/'source.png')
    if not probes.empty and 'model' in probes:
        cols=[c for c in ['balanced_accuracy','macro_f1','average_precision','r2','mae'] if c in probes]
        if cols:
            fig,axes=plt.subplots(1,len(cols),figsize=(max(6,3*len(cols)),4),squeeze=False)
            ok=probes[probes.status=='ok'] if 'status' in probes else probes
            for ax,col in zip(axes[0],cols):
                ax.bar(ok.model,ok[col],color='#276a83'); ax.tick_params(axis='x',rotation=50); ax.set(title=col)
            save(fig,out/'probes.png')
    if missing_summary is not None and not missing_summary.empty:
        col=next((c for c in ['macro_f1','r2','mae'] if c in missing_summary),None)
        if col:
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for mode,part in missing_summary.groupby('mode'):
                axes[0].plot(part.missing_fraction,part[col],'o-',label=mode)
            axes[0].set(ylabel=f'Hidden-label {col}',xlabel='Hidden fraction'); axes[0].legend()
            delta='macro_lift_delta' if kind!='continuous' else 'morans_i_delta'
            if missing_geometry is not None and delta in missing_geometry:
                k=sorted(missing_geometry.k.unique(),key=lambda v:abs(v-50))[0]
                for mode,part in missing_geometry[missing_geometry.k==k].groupby('mode'):
                    axes[1].plot(part.missing_fraction,part[delta],'o-',label=mode)
                axes[1].axhline(0,color='gray'); axes[1].set(ylabel=delta,xlabel='Hidden fraction',title=f'Reconstruction distortion at k={k}'); axes[1].legend()
            save(fig,out/'missingness.png')
    if prediction_geometry is not None and not prediction_geometry.empty:
        col='macro_lift' if kind!='continuous' else 'morans_i'
        fig,ax=plt.subplots(figsize=(7,4))
        for suffix in ['true','pred']:
            ax.plot(prediction_geometry.k,prediction_geometry[col+'_'+suffix],'o-',label=suffix)
        ax.set(xlabel='k',ylabel=col,title='Matched-item observed vs predicted geometry'); ax.legend()
        save(fig,out/'prediction_geometry.png')
    if coords is not None:
        fig,ax=plt.subplots(figsize=(7,5))
        if kind=='continuous':
            p=ax.scatter(coords[valid,0],coords[valid,1],c=values[valid],s=7,cmap='viridis',rasterized=True)
            fig.colorbar(p,ax=ax,label='Value')
        else:
            ax.scatter(coords[:,0],coords[:,1],c='#dadfe3',s=5)
            if kind=='multilabel':
                mask=valid & (values[:,0]==1)
                ax.scatter(coords[mask,0],coords[mask,1],s=7,label=names[0]); ax.legend()
            else:
                top=pd.Series(values[valid]).value_counts().head(8).index
                for name in top:
                    mask=valid & (values==name)
                    ax.scatter(coords[mask,0],coords[mask,1],s=7,label=display_label(name,label_metadata) if label_metadata else str(name))
                ax.legend(markerscale=2,fontsize=8)
        ax.set(title='PCA overview (metrics use original embeddings)',xlabel='PC 1',ylabel='PC 2')
        save(fig,out/'pca_overview.png')
    if cooccurrence is not None and not cooccurrence.empty:
        top=source[source.audited].head(15).label.tolist()
        mat=np.zeros((len(top),len(top)))
        lookup={v:i for i,v in enumerate(top)}
        for row in cooccurrence.itertuples():
            if row.label_a in lookup and row.label_b in lookup:
                i,j=lookup[row.label_a],lookup[row.label_b]; mat[i,j]=mat[j,i]=row.count
        fig,ax=plt.subplots(figsize=(8,6)); im=ax.imshow(mat,cmap='Blues')
        ax.set_xticks(range(len(top)),top,rotation=70,ha='right'); ax.set_yticks(range(len(top)),top)
        ax.set(title='Tag co-occurrence counts (diagonal omitted)'); fig.colorbar(im,ax=ax)
        save(fig,out/'cooccurrence.png')


def image_tag(path):
    data=base64.b64encode(Path(path).read_bytes()).decode('ascii')
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(Path(path).stem)}">'


def write_report(out,manifest,overview,results):
    out=Path(out)
    panels=[];options=[]
    for name,result in results.items():
        folder=result['folder']; number=len(panels)
        options.append(f'<option value="feature-{number}">{html.escape(name)}</option>')
        body=f'<h2>{html.escape(name)}</h2><p>Type: {html.escape(result["kind"])} · {html.escape(result["status"])}</p>'
        if result.get('error'):
            body+=f'<p>{html.escape(result["error"])}</p>'
        if result.get('explanation'):
            body+='<div class="explanation"><h3>What these results mean</h3>'
            body+=''.join('<p>'+html.escape(text)+'</p>' for text in result['explanation']['paragraphs'])
            body+='</div>'
        metadata=result.get('metadata')
        if metadata:
            body+='<p>'+html.escape(metadata.get('description',''))+'</p>'
            body+='<details><summary>Cluster meaning and label definitions</summary>'
            body+='<p>'+html.escape(metadata.get('reasoning',''))+'</p>'
            body+='<p>Source fields: '+html.escape(', '.join(metadata.get('source_fields',[])))+'</p>'
            path=folder/'tables'/'labels_metadata.csv'
            if path.exists():
                body+=pd.read_csv(path,dtype={'label_name':'string','label_criteria':'string'}).to_html(index=False,escape=True)
            body+=f'<p><a href="{(folder/"cluster_metadata.json").relative_to(out).as_posix()}">Original metadata JSON</a></p></details>'
        body+='<div class="figures">'+''.join(image_tag(p) for p in sorted((folder/'plots').glob('*.png')))+'</div>'
        for title,file in [('Source summary','source_summary.json'),('Prediction validity','prediction_metrics.json')]:
            if (folder/file).exists():
                data=json.loads((folder/file).read_text())
                body+=f'<details><summary>{title}</summary><pre>{html.escape(json.dumps(data,indent=2))}</pre></details>'
        for title,file in [('Source label prevalence','source_prevalence.csv'),('Source tag co-occurrence','cooccurrence.csv'),('Declared source-population prevalence','population_source_prevalence.csv'),('Purity/lift by k','geometry.csv'),('Null comparison','null_comparison.csv'),
                           ('Per-label purity/lift (first 50)','per_label_geometry.csv'),('Probe results','probes.csv'),
                           ('Missingness','missingness.csv'),('Prediction comparison','prediction_geometry.csv'),
                           ('Local entropy across k','local_entropy.csv'),('Second embedding probes','comparison_probes.csv'),('Second embedding geometry','comparison_geometry.csv'),('Second embedding per-label geometry','comparison_per_label_geometry.csv')]:
            path=folder/'tables'/file
            if path.exists() and path.stat().st_size>2:
                try:
                    table=pd.read_csv(path,dtype={c:'string' for c in ['label_name','label_criteria','label_display','true_name','predicted_name']})
                except pd.errors.EmptyDataError:
                    continue
                body+=f'<details><summary>{title}</summary><a href="{path.relative_to(out).as_posix()}">Complete CSV</a>'+table.head(50).to_html(index=False,float_format=lambda x:f'{x:.4g}',escape=True)+'</details>'
        panels.append(f'<section id="feature-{number}" class="panel">{body}</section>')
    title=manifest.get('config',{}).get('report_title','DataFrame embedding audit')
    overview_html=overview.to_html(index=False,float_format=lambda x:f'{x:.4g}',escape=True) if len(overview) else '<p>No successful features.</p>'
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DataFrame embedding audit</title><link rel="stylesheet" href="assets/explorer.css"><style>
body{font:15px system-ui,sans-serif;color:#17303f;background:#f5f7fa;margin:0}main{max-width:1250px;margin:auto;padding:30px}
h1{font-size:34px}h2{font-size:26px}.card,.panel{background:white;padding:24px;border:1px solid #d9e2e9;border-radius:12px;margin:20px 0}
.panel{display:none}.panel.active{display:block}.figures{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:20px}img{width:100%;height:auto}
table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}th,td{padding:7px;border-bottom:1px solid #d9e2e9;text-align:left}summary{cursor:pointer;font-weight:650;padding:12px 0}pre{overflow:auto}select,input{padding:10px;font:inherit;max-width:100%;border:1px solid #acbcc8;border-radius:7px}a{color:#126084}.muted{color:#586c7b}</style>
<main><h1>DataFrame embedding audit</h1><p class="muted">Original embedding geometry · source features · held-out prediction · reconstruction distortion</p>
<div class="card"><p><b>Purity</b> is the mean fraction of known neighbors sharing the anchor’s label. <b>Lift</b> divides that fraction by reference prevalence. A 5% segment with 30% purity has lift 6.</p>
<p>Micro weights positive anchor/label pairs; macro gives each audited label equal weight. Large lift with very little support needs scrutiny. Tag metrics count shared presence, and unannotated rows remain unknown.</p>
<p>Reference: the supplied sample unless explicit population prevalence is provided. This report does not establish whole-population fidelity from a potentially selected sample. PCA is only an overview. An embedding-based predictor’s geometric coherence is not independent validation.</p>
<details><summary>Run settings and limitations</summary><pre>MANIFEST</pre></details></div>
WORKBENCH
ENHANCEMENTS
<details class="card"><summary>Legacy single-scale table and raw CSV exports</summary><p class="muted">This older table uses k nearest 50. The main view above shows every configured scale.</p><p><a href="feature_measurements.json">Feature measurements JSON</a> · <a href="feature_overview.csv">Legacy overview CSV</a> · <a href="embedding_comparison.csv">Paired model comparison CSV (when configured)</a></p>OVERVIEW</details>
<div class="card"><label for="filter">Find feature </label><input id="filter" placeholder="artist, score, rating…" oninput="filterOptions()"> <select id="feature" onchange="showFeature()">OPTIONS</select></div>PANELS
<script>function showFeature(){let s=document.getElementById('feature');document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id===s.value));if(s.selectedOptions.length)window.dispatchEvent(new CustomEvent('audit-feature-change',{detail:s.selectedOptions[0].text}))}function filterOptions(){let q=document.getElementById('filter').value.toLowerCase();let s=document.getElementById('feature');for(let o of s.options)o.hidden=!o.text.toLowerCase().includes(q);let first=[...s.options].find(o=>!o.hidden);if(first){s.value=first.value;showFeature()}}showFeature()</script></main></html>'''
    doc=doc.replace('DataFrame embedding audit',html.escape(title)).replace('artist, score, rating…','Find a configured feature…')
    doc=doc.replace('MANIFEST',html.escape(json.dumps(manifest,indent=2))).replace('OVERVIEW',overview_html).replace('OPTIONS',''.join(options)).replace('PANELS',''.join(panels)).replace('WORKBENCH',explorer_markup())
    doc=doc.replace('ENHANCEMENTS',enhancement_markup())
    payload=json.loads((out/'explorer_data.json').read_text(encoding='utf-8'))
    safe=json.dumps(payload,ensure_ascii=True,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    # Append JSON only after template replacements; script-breaking labels are escaped.
    scripts='<script id="explorer-payload" type="application/json">'+safe+'</script><script src="assets/metric_options.js"></script><script src="assets/explorer.js"></script><script src="assets/enhancements.js"></script>'
    doc=doc.replace('</html>',scripts+'</html>')
    (out/'report.html').write_text(doc,encoding='utf-8')
