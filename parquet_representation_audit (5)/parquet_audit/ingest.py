"""Projected Parquet batches, bounded sample gathering, and ID-aligned NPY reads."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .semantics import is_null, missing, convert, normalize_series, typed_key
from .config import roles


def inspect_parquet(path: str | Path,*,sample_rows: int=100) -> dict:
    """Physical schema/footer facts plus explicitly bounded observations; no roles inferred."""
    file=pq.ParquetFile(path)
    batch=next(file.iter_batches(batch_size=max(1,sample_rows)),None)
    records={}
    for field in file.schema_arrow:
        values=[] if batch is None else batch.column(batch.schema.get_field_index(field.name)).to_pylist()[:sample_rows]
        sample=[]
        for value in values[:3]:
            text=repr(value)
            sample.append(text[:300]+(' [preview truncated; not a data asset]' if len(text)>300 else ''))
        sequences=[v for v in values if isinstance(v,(list,tuple))]
        lengths=Counter(map(len,sequences))
        element_types=Counter(type(item).__name__ for row in sequences for item in row[:20])
        records[field.name]={'physical_type':str(field.type),'sample_n':len(values),
             'sample_nulls':sum(is_null(v) for v in values),'representative_previews':sample,
             'sample_list_lengths':dict(sorted(lengths.items())),
             'sample_list_element_types':dict(sorted(element_types.items())),
             'semantic_assignment':'unassigned; explicitly declare role/kind'}
    return {'path':str(Path(path).resolve()),'footer_rows':file.metadata.num_rows,'row_groups':file.num_row_groups,
            'sample_scope':f'First at most {sample_rows} rows; descriptive, not population estimates','columns':records}


def canonical_identity(values: list,specs: list[dict]) -> str:
    """Lossless scalar IDs. Composite serialization preserves each primitive's type."""
    output=[]
    for value,spec in zip(values,specs):
        if missing(value,spec): raise ValueError('Item identity must be nonmissing')
        if isinstance(value,np.generic): value=value.item()
        typ=spec.get('id_type')
        if typ is None:
            if isinstance(value,str): typ='string'
            elif isinstance(value,bool): typ='boolean'
            elif isinstance(value,int): typ='integer'
            else: raise ValueError('IDs must be exact strings/integers/booleans; floating IDs are rejected')
        key=typed_key(value,typ);output.append([typ,key])
    return output[0][1] if len(output)==1 else json.dumps(output,ensure_ascii=True,separators=(',',':'))


def fingerprint(path: str | Path) -> str:
    """Full byte fingerprint for explicitly selected row-position source identity."""
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()


def identity_columns(schema: dict) -> list[str]:
    return schema['dataset'].get('identity_columns',roles(schema,'id'))


def derive_columns(frame: pd.DataFrame,columns: dict) -> tuple[pd.DataFrame,dict]:
    """Derive only explicitly declared time/order/text targets, retaining provenance."""
    specs={}
    for name,spec in columns.items():
        if spec['role']!='feature': continue
        for rule in spec.get('derive',[]):
            output=[]
            for value in frame[name]:
                if is_null(value): output.append(None);continue
                op=rule['operation']
                if op=='year': output.append(value.year)
                elif op=='month': output.append(value.month)
                elif op=='dayofweek': output.append(str(value.dayofweek) if rule['kind']=='categorical' else value.dayofweek)
                elif op=='unix_seconds': output.append(value.timestamp())
                elif op=='ordinal_value': output.append(rule['mapping'][str(value)])
                elif op=='text_length': output.append(len(value))
            if rule['kind']=='categorical': output=[None if v is None else str(v) for v in output]
            frame[rule['name']]=pd.Series(output,index=frame.index,dtype=object)
            specs[rule['name']]={'role':'feature','kind':rule['kind'],'id_type':'string',
                'units':rule.get('units',''),'description':f'Derived {op} from {name}',
                'provenance':{'source':name,'rule':rule},'geometry':True,'probe':True,'source_statistics':True}
    return frame,specs


def prepare_batch(batch, schema: dict) -> tuple[pd.DataFrame,dict]:
    # Arrow to_pylist avoids float promotion of nullable large integer IDs.
    frame=pd.DataFrame({n:pd.Series(batch.column(batch.schema.get_field_index(n)).to_pylist(),dtype=object) for n in batch.schema.names})
    for name in frame:
        spec=schema['columns'][name]
        if spec['role']=='feature': frame[name]=normalize_series(frame[name],spec)
        elif spec.get('struct_path') or spec.get('conversion','none')!='none':
            frame[name]=pd.Series([convert(v,spec) for v in frame[name]],index=frame.index,dtype=object)
    return derive_columns(frame,schema['columns'])


def source_statistics(schema: dict,cfg: dict,*,progress=None) -> tuple[dict,dict,dict]:
    """Stream source counts/moments and validate all identities with disk-backed uniqueness.

    Numeric quantiles use a bounded uniform priority reservoir and are labeled
    approximate when capped. Counters and row batches have explicit budgets.
    Identity index is disk-backed; embeddings never enter this source pass.
    """
    file=pq.ParquetFile(schema['dataset']['path']);columns=schema['columns'];ids=identity_columns(schema)
    features=roles(schema,'feature');projected=list(dict.fromkeys(ids+features))
    states={};specs={n:columns[n] for n in features};rng=np.random.default_rng(cfg['seed'])
    cap=cfg['source']['quantile_sample_size'];vocab_cap=cfg['source']['max_vocabulary'];count=0
    fp=fingerprint(schema['dataset']['path']) if not ids else None
    with tempfile.TemporaryDirectory(prefix='parquet-audit-identities-') as temp:
        db=sqlite3.connect(str(Path(temp)/'identities.sqlite'))
        db.execute('CREATE TABLE identities (id TEXT PRIMARY KEY) WITHOUT ROWID')
        for raw in file.iter_batches(batch_size=cfg['source']['batch_rows'],columns=projected):
            if raw.nbytes>cfg['limits']['max_source_batch_bytes'] or raw.num_rows*raw.num_columns>cfg['limits']['max_source_batch_cells']:
                raise ValueError('Source batch budget exceeded; lower source.batch_rows or project fewer features')
            frame,derived=prepare_batch(raw,schema);specs.update(derived)
            if ids:
                keys=[canonical_identity(list(row),[columns[c] for c in ids]) for row in frame[ids].itertuples(index=False,name=None)]
                try: db.executemany('INSERT INTO identities VALUES (?)',[(x,) for x in keys]);db.commit()
                except sqlite3.IntegrityError as exc: raise ValueError('Duplicate IDs or identity canonicalization collision; no rows deduplicated') from exc
            for name,spec in specs.items():
                state=states.setdefault(name,{'n':0,'known':0,'counts':Counter(),'mean':0.,'m2':0.,'min':float('inf'),'max':float('-inf'),'reservoir':np.array([]),'priorities':np.array([])})
                values=[v for v in frame[name] if not is_null(v)];known=len(values)
                state['n']+=len(frame);state['known']+=known
                kind=spec['kind']
                if not spec.get('source_statistics',True): continue
                if kind in ['numeric','count'] and known:
                    a=np.asarray(values,dtype=float);mean=float(a.mean());m2=float(np.sum((a-mean)**2));previous=state['known']-known
                    delta=mean-state['mean'];state['m2']+=m2+delta*delta*previous*known/state['known'];state['mean']+=delta*known/state['known']
                    state['min']=min(state['min'],float(a.min()));state['max']=max(state['max'],float(a.max()))
                    joined=np.r_[state['reservoir'],a];priority=np.r_[state['priorities'],rng.random(known)]
                    keep=np.argpartition(priority,min(cap,len(priority))-1)[:cap]
                    state['reservoir']=joined[keep];state['priorities']=priority[keep]
                elif kind=='text':
                    state['counts']['characters']+=sum(map(len,values));state['counts']['empty_strings']+=sum(v=='' for v in values)
                elif kind=='datetime':
                    if values:
                        lo=min(values).isoformat();hi=max(values).isoformat()
                        state['time_min']=min(lo,state.get('time_min',lo));state['time_max']=max(hi,state.get('time_max',hi))
                else:
                    for value in values:
                        state['counts'].update(value if kind=='multilabel' else [value])
                    if len(state['counts'])>vocab_cap: raise ValueError(f'{name}: source vocabulary exceeds {vocab_cap}; configure preprocessing/ignore or raise budget')
            count+=len(frame)
            if progress and count%max(cfg['source']['batch_rows']*10,1)==0: progress(f'Source pass: {count:,} rows')
        db.close()
    summaries={};tables={}
    for name,state in states.items():
        spec=specs[name];kind=spec['kind'];n=state['n'];known=state['known']
        summary={'kind':kind,'units':spec.get('units',''),'n_items':n,'n_known':known,'coverage':known/n if n else None,
            'scope':'Full Parquet source population','algorithm':'exact batched counts/moments','weighted':False}
        if not spec.get('source_statistics',True):
            summary.update(details_status='disabled',reason='source_statistics=false; coverage retained')
            summaries[name]=summary;continue
        if kind in ['numeric','count']:
            summary.update(mean=state['mean'] if known else None,variance=state['m2']/known if known else None,
                min=state['min'] if known else None,max=state['max'] if known else None,
                quantile_algorithm='exact' if known<=cap else 'uniform priority reservoir; approximate',quantile_sample_n=len(state['reservoir']))
            if known: summary.update({f'q{q:02d}':float(np.quantile(state['reservoir'],q/100)) for q in [5,25,50,75,95]})
        elif kind=='text':
            summary.update(mean_characters=state['counts']['characters']/known if known else None,empty_strings=state['counts']['empty_strings'],geometry='unavailable without explicit derived target',probe='unavailable without explicit derived target')
        elif kind=='datetime': summary.update(min=state.get('time_min'),max=state.get('time_max'),timezone=spec['timezone'],geometry='unavailable without explicit derived target',probe='unavailable without explicit derived target')
        else:
            counts=state['counts'];total=sum(counts.values());p=np.array(list(counts.values()),float)/total if total else np.array([])
            h=float(-np.sum(p*np.log(p))) if total else 0.
            summary.update(n_labels=len(counts),entropy=h,normalized_entropy=h/np.log(len(counts)) if len(counts)>1 else 0.,largest_label_share=max(counts.values(),default=0)/total if total else 0.,largest_item_prevalence=max(counts.values(),default=0)/known if known else 0.)
            tables[name]=pd.DataFrame([{'label':label,'count':c,'prevalence':c/known if known else None,'occurrence_share':c/total if total else None} for label,c in counts.most_common()],columns=['label','count','prevalence','occurrence_share'])
        summaries[name]=summary
    return summaries,tables,{'fingerprint':fp,'identity_validation':'Exact full-file disk-backed uniqueness' if ids else 'File-byte fingerprint plus row position; not cross-file identity','source_rows':count,'feature_specs':specs}


def sample_positions(n: int,cfg: dict) -> np.ndarray:
    if cfg['sample']['strategy']=='all': return np.arange(n,dtype=np.int64)
    return np.sort(np.random.default_rng(cfg['seed']).choice(n,min(n,cfg['sample']['max_rows']),replace=False))


def gather_parquet(schema: dict,cfg: dict,positions: np.ndarray,*,source_fingerprint: str | None=None) -> tuple[pd.DataFrame,dict]:
    """Read only projected row groups containing the chosen positions, in bounded batches.

    Parquet list cells in those batches are decoded; only selected vectors are
    retained. Sampling occurs before any full matrix is constructed in RAM.
    """
    file=pq.ParquetFile(schema['dataset']['path']);columns=schema['columns']
    projected=[n for n,s in columns.items() if s['role']!='ignore' and (cfg['mode']!='source_only' or s['role']!='embedding')];frames=[];offset=0;derived={}
    for group in range(file.num_row_groups):
        length=file.metadata.row_group(group).num_rows
        wanted=positions[(positions>=offset)&(positions<offset+length)]
        if not len(wanted): offset+=length;continue
        within=offset
        for raw in file.iter_batches(batch_size=cfg['source']['batch_rows'],row_groups=[group],columns=projected):
            if raw.nbytes>cfg['limits']['max_source_batch_bytes']: raise ValueError('Sample batch byte limit exceeded; lower source.batch_rows')
            local=wanted[(wanted>=within)&(wanted<within+raw.num_rows)]-within
            if len(local):
                frame,d=prepare_batch(raw.take(local),schema);derived.update(d)
                frame['_audit_source_position']=within+local;frames.append(frame)
            within+=raw.num_rows
        offset+=length
    if not frames: raise ValueError('No sampled rows; source file is empty')
    frame=pd.concat(frames,ignore_index=True);idcols=identity_columns(schema)
    internal='_audit_item_identity'
    if internal in columns or '_audit_source_position' in columns: raise ValueError('Reserved internal audit column collision')
    if idcols:
        frame[internal]=[canonical_identity(list(row),[columns[c] for c in idcols]) for row in frame[idcols].itertuples(index=False,name=None)]
    else:
        frame[internal]=[f'{source_fingerprint}:{p}' for p in frame['_audit_source_position']]
    # Group/stratum values preserve missingness and exact scalar types.
    for name,spec in columns.items():
        if spec['role'] in ['group','stratum']:
            frame[name]=[None if is_null(x) else json.dumps([type(x).__name__,str(x)]) for x in frame[name]]
        if spec['role']=='split':
            values=spec['values'];frame[name]=['train' if type(x)==type(values['train']) and x==values['train'] else 'test' if type(x)==type(values['test']) and x==values['test'] else None for x in frame[name]]
            if frame[name].isna().any(): raise ValueError(f'{name}: unknown explicit split values')
    return frame,derived


def external_vectors(asset: dict,ids: np.ndarray,id_spec: dict,cfg: dict) -> tuple[np.ndarray,np.ndarray]:
    """Mmap external matrix; gather batched selected rows by unique exact IDs.

    Only the ID index is population-sized and capped. The vector matrix stays
    memory mapped; rows absent by explicit exclude policy become nonfinite.
    """
    matrix=np.load(asset['matrix_path'],mmap_mode='r',allow_pickle=False)
    raw=np.load(asset['ids_path'],mmap_mode='r',allow_pickle=False)
    keys=[canonical_identity([value],[id_spec]) for value in raw]
    if len(set(keys))!=len(keys): raise ValueError('External asset has duplicate/colliding IDs')
    positions=pd.Index(keys).get_indexer(ids)
    absent=positions<0
    if absent.any() and asset.get('missing_ids','error')=='error': raise ValueError(f'{asset["name"]}: {absent.sum()} sampled IDs absent from external embeddings')
    result=np.full((len(ids),matrix.shape[1]),np.nan,dtype=np.float64)
    batch=cfg['limits']['external_batch_rows']
    for start in range(0,len(ids),batch):
        rows=np.arange(start,min(len(ids),start+batch));rows=rows[positions[rows]>=0]
        result[rows]=matrix[positions[rows]]
    return result,absent


def embedded_vectors(frame: pd.DataFrame,asset: dict,spec: dict) -> np.ndarray:
    vectors=[]
    for i,v in enumerate(frame[asset['column']]):
        if is_null(v):
            vectors.append(np.full(spec['expected_dimension'],np.nan));continue
        a=np.asarray(v)
        if a.ndim==1 and a.dtype==object and all(is_null(x) or isinstance(x,(int,float,np.number)) and not isinstance(x,(bool,np.bool_)) for x in a):
            a=np.array([np.nan if is_null(x) else x for x in a],dtype=np.float64)
        if a.ndim!=1 or len(a)!=spec['expected_dimension'] or not np.issubdtype(a.dtype,np.number) or np.iscomplexobj(a):
            raise ValueError(f'{asset["name"]}: row {i} must be a real numeric vector of dimension {spec["expected_dimension"]}')
        vectors.append(a.astype(np.float64))
    return np.stack(vectors)


def supplied_graph(asset: dict,ids: np.ndarray,id_spec: dict,cfg: dict) -> tuple[np.ndarray,np.ndarray]:
    from embedding_feature_audit.geometry import validate_neighbors
    keys=np.load(asset['ids_path'],allow_pickle=False)
    keys=[canonical_identity([x],[id_spec]) for x in keys]
    if keys!=list(ids): raise ValueError('Supplied graph anchor IDs must exactly match final cohort order; no implicit graph reindexing')
    raw=np.load(asset['indices_path'],allow_pickle=False);dist=np.load(asset['distances_path'],allow_pickle=False)
    if asset['index_kind']=='ids':
        flat=[canonical_identity([x],[id_spec]) for x in raw.ravel()];positions=pd.Index(keys).get_indexer(flat)
        if (positions<0).any(): raise ValueError('Neighbor IDs outside declared candidate pool')
        raw=positions.reshape(raw.shape)
    graph=validate_neighbors(raw,len(ids),max(cfg['geometry']['ks']))
    if dist.shape!=graph.shape or not np.isfinite(dist).all() or (dist<0).any() or (np.diff(dist,axis=1)<-1e-12).any(): raise ValueError('Supplied distances must align, be finite/nonnegative, and sorted')
    return graph,dist


def sample_source_summaries(frame: pd.DataFrame,specs: dict) -> tuple[dict,dict]:
    """Exact source summaries on the explicitly retained evaluation cohort."""
    from embedding_feature_audit.dataframe import encode_feature
    from embedding_feature_audit.statistics import feature_source
    from .semantics import KINDS,computation_spec
    summaries={};tables={}
    for name,spec in specs.items():
        kind=spec['kind'];series=frame[name];values=[v for v in series if not is_null(v)]
        base={'kind':kind,'units':spec.get('units',''),'n_items':len(frame),'n_known':len(values),'coverage':len(values)/len(frame) if len(frame) else None,'scope':'Retained evaluation sample','weighted':False,'algorithm':'exact sample summaries'}
        if KINDS[kind] is not None:
            cs=computation_spec(spec);y,known,names=encode_feature(series,cs);summary,table,_=feature_source(series,y,known,names,cs['kind'],cs)
            base.update(summary);tables[name]=table
        elif kind=='text': base.update(mean_characters=np.mean([len(v) for v in values]) if values else None,empty_strings=sum(v=='' for v in values))
        else: base.update(min=min(values).isoformat() if values else None,max=max(values).isoformat() if values else None,timezone=spec['timezone'])
        summaries[name]=base
    return summaries,tables
