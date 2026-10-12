"""JSON structure, semantic contracts and early resource validation."""
import copy
import json
from pathlib import Path
from importlib.resources import files
import jsonschema
import pyarrow as pa
import pyarrow.parquet as pq
from .semantics import typed_key, KINDS


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding='utf-8') as stream: return json.load(stream)


def validate_json(document: dict,name: str) -> None:
    schema=json.loads(files('parquet_audit').joinpath('schemas',name+'.schema.json').read_text())
    errors=sorted(jsonschema.Draft202012Validator(schema).iter_errors(document),key=lambda x:str(list(x.path)))
    if errors:
        raise ValueError(name+': '+ '; '.join(f'{"/".join(map(str,e.path)) or "root"}: {e.message}' for e in errors[:8]))


def read_metadata(path: str | Path,spec: dict | None=None) -> dict:
    data=read_json(path); validate_json(data,'label_metadata')
    keys=[typed_key(row['id'],data['id_type']) for row in data['labels']]
    if len(set(keys))!=len(keys): raise ValueError(f'{path}: duplicate typed metadata IDs')
    excluded={typed_key(x,data['id_type']) for x in data['excluded_from_coverage']}
    if excluded-set(keys): raise ValueError(f'{path}: excluded IDs must be declared in labels')
    if spec is not None:
        typ='boolean' if spec['kind']=='boolean' else spec.get('id_type','string')
        if typ!=data['id_type']: raise ValueError(f'{path}: feature and metadata ID types disagree')
        declared={typed_key(x,typ) for x in spec.get('missing_values',[])}
        if declared!=excluded and not spec.get('metadata_missing_override'):
            raise ValueError(f'{path}: metadata missing values disagree; declare metadata_missing_override with a reason')
    return data


def load_bundle(schema_path: str | Path,config_path: str | Path) -> tuple[dict,dict]:
    schema_path=Path(schema_path).resolve();config_path=Path(config_path).resolve()
    schema=read_json(schema_path); cfg=read_json(config_path)
    validate_json(schema,'dataset_schema');validate_json(cfg,'audit_config')
    schema=copy.deepcopy(schema)
    def path(value):
        p=Path(value).expanduser();return str(p if p.is_absolute() else (schema_path.parent/p).resolve())
    schema['dataset']['path']=path(schema['dataset']['path'])
    for spec in schema['columns'].values():
        if 'metadata_path' in spec: spec['metadata_path']=path(spec['metadata_path'])
    for asset in schema.get('embeddings',[])+schema.get('neighbors',[]):
        for key in ['matrix_path','ids_path','indices_path','distances_path']:
            if key in asset: asset[key]=path(asset[key])
    for asset in schema.get('prediction_assets',[]):
        asset['path']=path(asset['path'])
    schema['_schema_path']=str(schema_path);cfg['_config_path']=str(config_path)
    return schema,cfg


def roles(schema: dict,role: str) -> list[str]:
    return [name for name,s in schema['columns'].items() if s['role']==role]


def validate_bundle(schema_path: str | Path,config_path: str | Path,*,check_data: bool=True,allow_draft: bool=False) -> dict:
    """Validate without fitting. Footer/type checks and bounded asset header checks only.

    Full identity uniqueness, scalar domains, every source annotation and sampled
    embedding rows are validated during run. No claim of full-vector validation.
    """
    schema,cfg=load_bundle(schema_path,config_path)
    if schema['status']=='draft' or schema.get('unresolved'):
        if not allow_draft: raise ValueError('Draft/unresolved dataset bundle cannot run; resolve critical semantics first')
    columns=schema['columns']; names=set(columns)
    for name,spec in columns.items():
        if spec['role']=='id' and spec.get('conversion','none')!='none': raise ValueError(f'{name}: identity conversion is unsupported; supply exact stable primitive IDs')
    idcols=schema['dataset'].get('identity_columns',roles(schema,'id'))
    if idcols and (set(idcols)-names or any(columns[c]['role']!='id' for c in idcols)): raise ValueError('identity_columns must reference declared id columns')
    if not idcols:
        if cfg['mode']!='source_only' or schema['dataset'].get('row_identity')!='fingerprint_position':
            raise ValueError('Stable ID columns are required, or explicit source_only fingerprint_position identity')
    elif schema['dataset'].get('row_identity'): raise ValueError('Choose stable IDs or row identity, not both')
    for role,key in [('group','group_column'),('split','split_column')]:
        declared=roles(schema,role);chosen=cfg['probes'].get(key)
        if len(declared)>1 and not chosen: raise ValueError(f'Choose {key} when several columns have role={role}')
        if chosen and (chosen not in columns or role not in [columns[chosen]['role']]+columns[chosen].get('auxiliary_roles',[])): raise ValueError(f'{key} must declare role/auxiliary role {role}')
    stratum=cfg['nulls'].get('stratum_column')
    if stratum and (stratum not in columns or 'stratum' not in [columns[stratum]['role']]+columns[stratum].get('auxiliary_roles',[])): raise ValueError('stratum_column must declare stratum use')
    if ('stratified' in cfg['nulls']['modes'] or cfg['topology'].get('null_mode')=='stratified') and not stratum: raise ValueError('Stratified permutations require an explicit stratum_column')
    derived=set()
    for name,spec in columns.items():
        if spec['role']=='feature':
            kind=spec['kind']
            if kind not in KINDS: raise ValueError(f'{name}: unsupported feature kind')
            if kind in ['text','datetime'] and (spec.get('geometry',False) or spec.get('probe',False)): raise ValueError(f'{name}: declare derived targets for geometry/probes')
            if kind=='ordinal':
                keys=[typed_key(x,spec.get('id_type','string')) for x in spec['order']]
                if len(set(keys))!=len(keys) or not keys: raise ValueError(f'{name}: ordinal order must be nonempty and unique')
            if 'metadata_path' in spec:
                if kind not in ['cluster_label','categorical','boolean','ordinal']: raise ValueError(f'{name}: label metadata requires an unordered/ordinal scalar label target')
                read_metadata(spec['metadata_path'],spec)
            if kind=='multilabel' and len(set(spec.get('vocabulary',[])))!=len(spec.get('vocabulary',[])): raise ValueError(f'{name}: vocabulary contains duplicates')
            for rule in spec.get('derive',[]):
                d=rule['name']
                if d in names or d in derived: raise ValueError(f'Derived feature collides: {d}')
                derived.add(d)
                valid_ops={'datetime':{'year','month','dayofweek','unix_seconds'},'ordinal':{'ordinal_value'},'text':{'text_length'}}
                if rule['operation'] not in valid_ops.get(kind,set()): raise ValueError(f'{name}: unsupported derived operation {rule["operation"]}')
                if rule['operation']=='ordinal_value':
                    if not rule.get('mapping') or not rule.get('reasoning'): raise ValueError('Ordinal numeric targets require mapping and reasoning')
                    if set(rule['mapping'])!=set(typed_key(x,spec.get('id_type','string')) for x in spec['order']): raise ValueError('Ordinal mapping must cover every declared level')
    pipelines=schema.get('prediction_pipelines',{})
    predicted=[]
    for name,spec in list(columns.items())+[(n,s) for a in schema.get('prediction_assets',[]) for n,s in a['columns'].items()]:
        if spec['role'] in ['prediction','probabilities']:
            target=spec['target']
            if target not in columns or columns[target]['role']!='feature' or KINDS[columns[target]['kind']] is None: raise ValueError(f'{name}: target must be an evaluable feature')
            if spec['pipeline'] not in pipelines: raise ValueError(f'{name}: pipeline is undeclared')
            if spec['role']=='prediction':
                if target in predicted: raise ValueError('First release supports one prediction pipeline per target; use separate runs')
                predicted.append(target)
            else:
                typ='string' if columns[target]['kind']=='multilabel' else 'boolean' if columns[target]['kind']=='boolean' else columns[target].get('id_type','string')
                keys=[typed_key(x,typ) for x in spec['label_order']]
                if len(set(keys))!=len(keys) or not keys: raise ValueError('Probability label_order must have unique typed IDs')
                if columns[target]['kind'] in ['numeric','count']: raise ValueError('Probability arrays require categorical or multilabel truth')
    for asset in schema.get('prediction_assets',[]):
        if set(asset['columns']) & names: raise ValueError('External prediction column names must not collide with Parquet columns')
        if any(s['role'] not in ['prediction','probabilities'] for s in asset['columns'].values()): raise ValueError('External prediction assets contain only declared prediction/probability columns')
        if len(idcols)!=1: raise ValueError('External prediction tables currently require one stable ID column; precompose composite keys explicitly')
    embeddings=schema['embeddings']
    if len({e['name'] for e in embeddings})!=len(embeddings): raise ValueError('Embedding model names must be unique')
    if cfg['mode']=='embedding' and not embeddings: raise ValueError('Embedding mode requires real declared embedding assets')
    if cfg['mode']=='source_only' and (cfg['geometry']['enabled'] or cfg['probes']['enabled'] or cfg['normalization']['enabled'] or cfg['topology']['enabled'] or cfg['missingness']['enabled']): raise ValueError('source_only requires geometry/probes/normalization/topology/missingness disabled')
    if cfg['mode']=='embedding' and not cfg['geometry']['enabled']: raise ValueError('Embedding mode currently requires geometry enabled; use source_only otherwise')
    if len({e['main_normalization'] for e in embeddings})>1: raise ValueError('A/B models must use the same main normalization')
    for e in embeddings:
        if 'column' in e and (e['column'] not in columns or columns[e['column']]['role']!='embedding'): raise ValueError('Embedding asset column must have embedding role')
        if 'matrix_path' in e and len(idcols)!=1: raise ValueError('External assets require a single stable ID column')
    if schema.get('neighbors') and len(idcols)!=1: raise ValueError('Supplied graphs require one stable ID column; precompose composite keys explicitly')
    for graph in schema.get('neighbors',[]):
        if graph['embedding'] not in {e['name'] for e in embeddings}: raise ValueError('Graph references undeclared model')
        if graph['metric']!=cfg['geometry']['metric'] or graph['preprocessing']!=embeddings[0]['main_normalization']: raise ValueError('Supplied graph metric/preprocessing disagrees with requested geometry')
        if cfg['sample']['strategy']!='all': raise ValueError('Supplied graphs require strategy=all for their explicitly exported cohort; make a sampled Parquet first')
    if len({g['embedding'] for g in schema.get('neighbors',[])})!=len(schema.get('neighbors',[])): raise ValueError('Only one supplied graph per model')
    if cfg['geometry']['reference']=='full_source' and (cfg['source']['scope']!='full' or any(s['role']=='feature' and s.get('geometry',True) and s['kind'] in ['multilabel','categorical','cluster_label','ordinal','boolean'] and not s.get('source_statistics',True) for s in columns.values())):
        raise ValueError('Full-source prevalence requires full streamed source statistics for each label/tag target')
    if cfg['nulls']['correction']=='bh' and cfg['nulls']['testing_family']!='all_features_scales_modes_per_metric': raise ValueError('Implemented BH family is all_features_scales_modes_per_metric; use none for another family')
    from embedding_feature_audit.normalization import normalization_options
    from embedding_feature_audit.topology import topology_options
    from embedding_feature_audit.feature_overview import overview_settings
    normalization_options({'normalization_analysis':cfg['normalization']});topology_options({'topology':cfg['topology']})
    overview_settings(cfg['geometry']['ks'],{'ks':cfg['geometry']['ks'],'overview':cfg['report']['overview']})
    result={'status':'draft' if schema['status']=='draft' else 'valid','checks':'configuration and asset headers; full row checks occur during run','data_checked':check_data,'schema':schema,'config':cfg}
    if not check_data: return result
    file=pq.ParquetFile(schema['dataset']['path']);physical=file.schema_arrow
    if set(names)-set(physical.names): raise ValueError(f'Configured columns absent: {sorted(set(names)-set(physical.names))}')
    unassigned=set(physical.names)-names
    if unassigned and schema['dataset']['unmapped_columns']=='error': raise ValueError(f'Unmapped columns: {sorted(unassigned)}')
    for name,spec in columns.items():
        if spec['role']=='ignore': continue
        typ=physical.field(name).type
        if pa.types.is_dictionary(typ): typ=typ.value_type
        if pa.types.is_nested(typ) and spec['role']!='embedding' and spec.get('encoding')!='list' and not spec.get('struct_path'):
            raise ValueError(f'{name}: nested Arrow type needs explicit struct_path/encoding or ignore')
        if spec['role']=='embedding':
            if not (pa.types.is_list(typ) or pa.types.is_fixed_size_list(typ) or pa.types.is_large_list(typ)): raise ValueError(f'{name}: embeddings must be numeric Arrow lists')
            if not (pa.types.is_integer(typ.value_type) or pa.types.is_floating(typ.value_type)): raise ValueError(f'{name}: textual/nonnumeric embeddings rejected')
            if pa.types.is_fixed_size_list(typ) and typ.list_size!=spec['expected_dimension']: raise ValueError(f'{name}: expected_dimension disagrees with Arrow vector length')
    if len(roles(schema,'feature'))>cfg['source']['max_feature_columns']: raise ValueError('Feature column budget exceeded; split the run or raise the explicit limit')
    n=file.metadata.num_rows; cap=cfg['sample']['max_rows']; count=n if cfg['sample']['strategy']=='all' else min(n,cap)
    if cfg['sample']['strategy']=='all' and n>cap: raise ValueError(f'{n:,} rows exceed sample.max_rows={cap:,}; choose uniform_rows before loading vectors')
    if count>cfg['limits']['max_evaluation_rows']: raise ValueError('Evaluation-row limit exceeded; lower sample.max_rows')
    if cfg['mode']=='embedding' and any(k>=count for k in cfg['geometry']['ks']): raise ValueError(f'Every requested k must be < evaluation rows ({count}); no scales are silently discarded')
    if cfg['mode']=='embedding' and count*(count-1)//2>cfg['limits']['max_exact_pairs']: raise ValueError('Exact graph pair budget exceeded; reduce evaluation sample')
    import numpy as np
    byte_estimate=0;dimensions=[]
    for e in embeddings:
        if 'matrix_path' in e:
            matrix=np.load(e['matrix_path'],mmap_mode='r',allow_pickle=False);ids=np.load(e['ids_path'],mmap_mode='r',allow_pickle=False)
            if matrix.ndim!=2 or not np.issubdtype(matrix.dtype,np.number) or np.iscomplexobj(matrix) or matrix.shape[1]!=e['expected_dimension']: raise ValueError('External matrix must be a real numeric N×declared D array')
            if ids.ndim!=1 or len(ids)!=len(matrix): raise ValueError('External embedding IDs must match matrix rows')
            if len(ids)>cfg['limits']['max_external_ids']: raise ValueError('External ID-index budget exceeded; raise max_external_ids after budgeting memory')
            dimension=matrix.shape[1]
        else: dimension=columns[e['column']]['expected_dimension']
        byte_estimate+=count*dimension*8;dimensions.append(dimension)
    if byte_estimate*3>cfg['limits']['max_embedding_bytes']: raise ValueError(f'Estimated vector/normalization working memory {byte_estimate*3:,} bytes exceeds limit; lower sample size')
    labels=sum(min(columns[c].get('max_labels',100),cfg['source']['max_vocabulary']) for c in roles(schema,'feature') if columns[c]['kind']=='multilabel')
    if count*max(labels,1)>cfg['limits']['max_dense_feature_bytes']: raise ValueError('Bounded tag sample matrix budget exceeded; reduce sample/max_labels')
    feature_count=len(roles(schema,'feature'))+len(derived)
    work=count*max(cfg['geometry']['ks'])*max(feature_count,1)*(1+cfg['nulls']['permutations'])*max(len(embeddings),1)
    if work>cfg['limits']['max_work_units']: raise ValueError(f'Estimated geometry work {work:,} exceeds max_work_units; reduce rows/k/features/permutations')
    result.update(n_rows=n,evaluation_rows=count,unmapped_columns=sorted(unassigned),estimated_embedding_working_bytes=byte_estimate*3,estimated_geometry_work=work)
    return result
