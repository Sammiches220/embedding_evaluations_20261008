"""Read cluster JSON; preserve IDs for computation and add human-facing labels."""
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import pandas as pd


def cluster_id(value):
    """Canonicalize 1, '1', and 1.0 without merging distinct text labels."""
    text=str(value)
    try:
        number=Decimal(text)
        if number.is_finite() and number==number.to_integral_value():
            return str(int(number))
    except InvalidOperation:
        pass
    return text


def read_cluster_metadata(path):
    path=Path(path)
    data=json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data,dict) and 'id_type' in data:
        from parquet_audit.config import read_metadata
        return read_metadata(path)
    if not isinstance(data,dict) or not isinstance(data.get('labels'),list) or not isinstance(data.get('name'),str):
        raise ValueError(f'{path}: expected a cluster object with name and labels[]')
    seen=set()
    for label in data['labels']:
        if not isinstance(label,dict) or 'id' not in label or not isinstance(label.get('name'),str):
            raise ValueError(f'{path}: each label needs id and a string name')
        key=cluster_id(label['id'])
        if isinstance(label['id'],bool) or not key.isdigit() or key in seen:
            raise ValueError(f'{path}: cluster IDs must be unique nonnegative integers; got {label["id"]!r}')
        seen.add(key)
    excluded=data.get('excluded_from_coverage',[])
    if not isinstance(excluded,list) or any(cluster_id(v) not in seen for v in excluded):
        raise ValueError(f'{path}: excluded_from_coverage must list declared label IDs')
    return data


def resolve_config_metadata_paths(config,base):
    """For CLI config files only: resolve metadata paths relative to that file."""
    config=dict(config); base=Path(base)
    def resolve(value):
        p=Path(value).expanduser()
        return str(p if p.is_absolute() else (base/p).resolve())
    if config.get('cluster_metadata_dir'):
        config['cluster_metadata_dir']=resolve(config['cluster_metadata_dir'])
    config['cluster_metadata']={name:resolve(path) for name,path in config.get('cluster_metadata',{}).items()}
    if isinstance(config.get('features'),dict):
        config['features']={name:dict(spec) if isinstance(spec,dict) else spec for name,spec in config['features'].items()}
        for spec in config['features'].values():
            if isinstance(spec,dict) and spec.get('metadata_file'):
                spec['metadata_file']=resolve(spec['metadata_file'])
    return config


def configure_cluster_metadata(specs,config):
    """Return feature specs and {column: metadata}; explicit files beat discovery."""
    specs={name:dict(spec) for name,spec in specs.items()}
    paths={}; explicitly=dict(config.get('cluster_metadata',{}))
    for name,spec in specs.items():
        if spec.get('metadata_file'):
            explicitly[name]=spec['metadata_file']
    directory=config.get('cluster_metadata_dir')
    if directory:
        directory=Path(directory).expanduser()
        if not directory.is_dir():
            raise ValueError(f'Cluster metadata directory does not exist: {directory}')
        for path in sorted(directory.rglob('*.json')):
            data=json.loads(path.read_text(encoding='utf-8'))
            # Ignore other valid JSON objects such as pipeline config files.
            if not isinstance(data,dict) or not isinstance(data.get('labels'),list) or not isinstance(data.get('name'),str):
                continue
            column=data['name'] if data['name'].endswith('_cluster') else data['name']+'_cluster'
            if column in specs and column not in explicitly:
                if column in paths:
                    raise ValueError(f'Duplicate metadata for {column}: {paths[column]} and {path}. Set cluster_metadata explicitly.')
                paths[column]=path
    for column,path in explicitly.items():
        if column not in specs:
            raise ValueError(f'Metadata configured for inactive/missing feature: {column}')
        paths[column]=Path(path).expanduser()
    resolved={}
    for column,path in paths.items():
        if specs[column]['kind']!='categorical':
            raise ValueError(f'Cluster metadata requires a categorical feature: {column}')
        data=read_cluster_metadata(path)
        spec=specs[column]
        # An explicit missing_values list, including [], overrides metadata policy.
        if 'missing_values' not in spec:
            spec['missing_values']=list(data.get('excluded_from_coverage',[]))
        spec['cluster_ids']=not spec.get('normalized_ids',False)
        spec['metadata_file']=str(Path(path).resolve())
        resolved[column]=data
    return specs,resolved


def label_lookup(metadata):
    return {metadata_key(row['id'],metadata):row for row in metadata['labels']} if metadata else {}


def display_label(value,metadata):
    key=metadata_key(value,metadata); row=label_lookup(metadata).get(key)
    return f'{key} · {row["name"]}' if row else f'{key} · unmapped'


def annotate_labels(table,metadata):
    """Add names/criteria to label or prediction columns; metrics stay unchanged."""
    if not metadata or table.empty:
        return table
    table=table.copy(); lookup=label_lookup(metadata)
    for col in ['label','label_a','label_b','true','predicted']:
        if col not in table:
            continue
        keys=table[col].map(lambda v:metadata_key(v,metadata))
        table[col+'_id']=keys
        table[col+'_name']=keys.map(lambda key:lookup.get(key,{}).get('name','unmapped'))
        table[col+'_criteria']=keys.map(lambda key:lookup.get(key,{}).get('criteria',''))
        table[col+'_display']=keys.map(lambda key:display_label(key,metadata))
        table[col+'_metadata_known']=keys.isin(lookup)
    return table


def metadata_label_table(series,metadata,spec):
    raw=series[series.notna()].map(lambda v:metadata_key(v,metadata))
    counts=raw.value_counts()
    excluded={metadata_key(v,metadata) for v in spec.get('missing_values',[])}
    table=pd.DataFrame([dict(label=metadata_key(row['id'],metadata),label_name=row['name'],
        label_criteria=row.get('criteria',''),n_items=int(counts.get(metadata_key(row['id'],metadata),0)),
        excluded_from_evaluation=metadata_key(row['id'],metadata) in excluded,
        metadata_excluded_from_coverage=metadata_key(row['id'],metadata) in {metadata_key(v,metadata) for v in metadata.get('excluded_from_coverage',[])})
        for row in metadata['labels']])
    return table


def metadata_key(value,metadata):
    """Preserve declared text IDs; internal normalized integer/bool strings stay exact."""
    if 'id_type' not in metadata: return cluster_id(value)
    typ=metadata['id_type']
    if isinstance(value,str): return value
    from parquet_audit.semantics import typed_key
    return typed_key(value,typ)
