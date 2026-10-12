"""Generic data adapter preserving the original feature-first HTML/CSS/controls."""
import html
import json
from pathlib import Path
import re
import uuid
from .semantics import KINDS
import numpy as np
import pandas as pd
from embedding_feature_audit.dataframe_report import write_json,save_table
from embedding_feature_audit.visuals import write_report
from embedding_feature_audit.exploration import copy_explorer_assets
from embedding_feature_audit.feature_overview import overview_settings


def empty_record(source: dict,spec: dict,folder: Path) -> dict:
    return dict(kind=KINDS.get(spec['kind']) or 'summary',semantic_kind=spec['kind'],units=spec.get('units',''),description=spec.get('description',''),labels=[],source=source,
        geometry=[],comparison_geometry=[],probes=[],comparison_probes=[],probe_status='Unavailable for this source-only target',probe_labels=[],
        external=None,paired=[],folder=folder.name,status='source_only',entropy=[],comparison_entropy=[],paired_geometry_a=[],paired_geometry_b=[],paired_entropy_delta=[],diversity=source.get('normalized_entropy'))


def source_report(out: Path,schema: dict,cfg: dict,summaries: dict,tables: dict,frame: pd.DataFrame) -> dict:
    """All aggregate source targets remain available without fabricated embeddings."""
    results={};overview={};rows=[]
    for i,(name,summary) in enumerate(summaries.items(),1):
        spec=schema['columns'].get(name,{'kind':summary['kind']});folder=out/'features'/f'{i:03d}'
        (folder/'tables').mkdir(parents=True);(folder/'plots').mkdir()
        write_json(summary,folder/'source_summary.json')
        metadata=None;table=tables.get(name,pd.DataFrame())
        if spec.get('metadata_path'):
            from .config import read_metadata
            from embedding_feature_audit.cluster_metadata import annotate_labels,metadata_key
            metadata=read_metadata(spec['metadata_path'],spec);table=annotate_labels(table,metadata)
            write_json(metadata,folder/'cluster_metadata.json')
            count_lookup={} if table.empty else dict(zip(table.label,table['count']))
            save_table(pd.DataFrame([{'label':row['id'],'label_name':row['name'],'label_criteria':row.get('criteria',''),'n_known_source':count_lookup.get(metadata_key(row['id'],metadata),0)} for row in metadata['labels']]),folder/'tables'/'labels_metadata.csv')
        save_table(table,folder/'tables'/'source_prevalence.csv')
        explanation={'paragraphs':[f'{name}: {summary["n_known"]:,} known of {summary["n_items"]:,} source rows. Geometry and held-out recovery are unavailable in source-only mode.']}
        result=dict(folder=folder,kind=spec['kind'],status='source_only',source=summary,explanation=explanation,metadata=metadata)
        results[name]=result;overview[name]=empty_record(summary,spec,folder)
        rows.append({'feature':name,'kind':spec['kind'],'coverage':summary['coverage'],'status':'source_only'})
    view=dict(ks=[],settings=overview_settings(cfg['geometry']['ks'],{'ks':cfg['geometry']['ks'],'overview':cfg['report']['overview']}),features=overview,
        embedding_name='Source-only analysis',comparison_name=None,scope='Source population only; no embedding geometry, probes, or topology.')
    payload=dict(available=False,reason='Source-only mode. No embeddings were supplied for this analysis.',run_id=uuid.uuid4().hex,
        overview=view,enhancements={'normalization':{'status':'unavailable','reason':'Source-only mode'},'topology':{'status':'unavailable','reason':'Source-only mode'}})
    manifest=dict(run_id=payload['run_id'],mode='source_only',config={'report_title':cfg['report']['title']},n_rows=len(frame),features=schema['columns'])
    write_json(payload,out/'explorer_data.json');write_json(view,out/'feature_measurements.json');write_json(payload['enhancements'],out/'enhancement_measurements.json')
    copy_explorer_assets(out);write_json(manifest,out/'manifest.json');table=pd.DataFrame(rows);save_table(table,out/'feature_overview.csv')
    write_report(out,manifest,table,results)
    return dict(out_dir=out,report=out/'report.html',overview=table,features=results,manifest=manifest)


def refresh_report(out: Path,result: dict,schema: dict,cfg: dict,summaries: dict,tables: dict,specs: dict) -> None:
    """Add text/time inventory, scopes, units and source-population tables to the reused shell."""
    payload=json.loads((out/'explorer_data.json').read_text());view=payload['overview'];results=result['features']
    for name,spec in specs.items():
        if name in view['features']:
            record=view['features'][name];record.update(semantic_kind=spec['kind'],units=spec.get('units',''),description=spec.get('description',''))
            # Surface fresh generic probe/probability metrics after adapter-side analyses.
            for field in ['probes','comparison_probes']:
                if field in results[name]: record[field]=results[name][field].to_dict('records')
            if 'prediction_metrics' in results[name]: record['external']=results[name]['prediction_metrics']
            record['probe_status']=results[name].get('probe_status',record.get('probe_status'))
            if payload.get('available') and name in payload['features']:
                payload['features'][name].update(semantic_kind=spec['kind'],units=spec.get('units',''))
        elif name in summaries:
            folder=out/'features'/f'{len(results)+1:03d}';(folder/'tables').mkdir(parents=True);(folder/'plots').mkdir()
            summary=summaries[name];write_json(summary,folder/'source_summary.json')
            results[name]=dict(folder=folder,kind=spec['kind'],status='source_only',source=summary,
                explanation={'paragraphs':[f'{name} is {spec["kind"]} information. The source summary reports {summary["n_known"]:,} known rows. Only explicitly declared derived targets enter embedding measurements.']})
            view['features'][name]=empty_record(summary,spec,folder)
            if payload.get('available'):
                payload['features'][name]=dict(kind='summary',values=[None]*len(payload['ids']),labels=[],audited_labels=[],per_label=[],explanation=results[name]['explanation'])
        if name in view['features'] and name in summaries:
            view['features'][name]['population_source']=summaries[name]
            write_json(summaries[name],results[name]['folder']/'population_source_summary.json')
            save_table(tables.get(name,pd.DataFrame()),results[name]['folder']/'tables'/'population_source_prevalence.csv')
    write_json(payload,out/'explorer_data.json');write_json(view,out/'feature_measurements.json')
    manifest=json.loads((out/'manifest.json').read_text());manifest['config']['report_title']=cfg['report']['title']
    manifest['generic_dataset']=schema['dataset'];manifest['semantic_inventory']=specs
    manifest['feature_folders']={n:str(r['folder'].relative_to(out)) for n,r in results.items()}
    result['manifest']=manifest;write_json(manifest,out/'manifest.json')
    write_report(out,manifest,result['overview'],results)
    doc=(out/'report.html').read_text()
    source_scope='Full source-population tables are exported separately; embedding measurements use the bounded evaluation cohort.' if cfg['source']['scope']=='full' else 'Source summaries and embedding measurements describe the declared evaluation sample.'
    doc=doc.replace('Reference: the supplied sample unless explicit population prevalence is provided.',html.escape(source_scope)+' Reference prevalence: '+html.escape(cfg['geometry']['reference'])+'.')
    if cfg['mode']=='source_only':
        doc=doc.replace('Original embedding geometry · source features · held-out prediction · reconstruction distortion','Source information · schema-driven coverage and distributions · no embedding analyses')
        doc=doc.replace('<p><b>Purity</b>', '<p>Source-only mode: geometry, probes and topology are unavailable. The tables describe observed source information.</p><p><b>Purity</b>')
    # Add projected-source scope to every detail panel, without changing the shell.
    doc=doc.replace('Legacy single-scale table and raw CSV exports','Raw measurements and CSV exports')
    doc=doc.replace('<p><a href="feature_measurements.json">', '<p><a href="source_population/summaries.json">Declared source summaries JSON</a> · <a href="feature_measurements.json">')
    doc=doc.replace('This older table uses k nearest 50. The main view above shows every configured scale.','The main feature view shows every configured scale. Full source and evaluation scopes are exported separately.')
    doc=doc.replace('Filtration and feature organization','Feature organization across a filtration')
    doc=doc.replace('Enable normalization_analysis.enabled in config.', 'Enable normalization.enabled in audit_config.json.')
    def link(match):
        url=match.group(1);label=match.group(2)
        if url.startswith(('#','http:','https:','data:')): return match.group(0)
        if not (out/url).exists(): return '<span class="muted" title="Unavailable for this run">'+label+' (unavailable)</span>'
        return match.group(0)
    doc=re.sub(r'<a href="([^"]+)">([^<]*)</a>',link,doc)
    (out/'report.html').write_text(doc)
