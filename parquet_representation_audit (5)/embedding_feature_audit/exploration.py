"""Row values, cohort descriptions, and bounded offline exploration payloads."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .dataframe import tag_sets, encode_feature, add_derived_features, default_features
from .cluster_metadata import display_label, configure_cluster_metadata


def row_feature(series, spec, metadata=None):
    """Represent unknown as None; retain all tags, including unaudited tags."""
    y, valid, names = encode_feature(series, spec)
    if spec['kind'] == 'multilabel':
        sets, _ = tag_sets(series, spec.get('empty_is_missing', True))
        values = [sorted(tags) if known else None for tags, known in zip(sets, valid)]
        labels = sorted(set().union(*(tags for tags, known in zip(sets, valid) if known)))
    else:
        values = [v.item() if isinstance(v, np.generic) else v for v in y]
        values = [value if known else None for value, known in zip(values, valid)]
        labels = names
    return dict(kind=spec['kind'], values=values, labels=labels, audited_labels=names,
                label_display={label: display_label(label, metadata) if metadata else str(label) for label in labels},
                metadata=metadata, spec=spec)


def clause_value(feature, clause, row):
    """Three-valued query logic: True, False, or unknown (None)."""
    value = feature['values'][row]
    if clause.get('known') == 'unknown':
        return value is None
    if clause.get('known') == 'known':
        return value is not None
    if value is None:
        return None
    if feature['kind'] == 'continuous':
        return bool((clause.get('min') is None or value >= clause['min']) and (clause.get('max') is None or value <= clause['max']))
    members = set(value) if feature['kind'] == 'multilabel' else {str(value)}
    include = set(map(str, clause.get('include', [])))
    exclude = set(map(str, clause.get('exclude', [])))
    included = not include or (include <= members if clause.get('mode', 'all') == 'all' else bool(include & members))
    return included and not bool(exclude & members)


def query_dataframe(df, clauses, *, mode='all', features=None, config=None):
    """Return a Boolean row mask from plain clause dictionaries, without eval.

    Example: [{'feature': 'score', 'min': 25},
              {'feature': 'tag_string_general', 'include': ['water'], 'mode':'all'}].
    Unknown logical results do not match. Explicit known/unknown clauses can
    select annotation gaps. Tag absence in a known set is an operational negative.
    """
    if mode not in ['all', 'any'] or not clauses:
        raise ValueError('Provide clauses and mode all or any')
    df = add_derived_features(df)
    specs = features if features is not None else (config or {}).get('features', default_features(df))
    specs = {k: {'kind': v} if isinstance(v, str) else dict(v) for k, v in specs.items()}
    specs, metadata = configure_cluster_metadata(specs, config or {})
    records = {}
    for clause in clauses:
        name = clause['feature']
        if name not in specs or name not in df:
            raise ValueError(f'Unknown query feature: {name}')
        if clause.get('mode', 'all') not in ['all', 'any']:
            raise ValueError('Tag mode must be all or any')
        if clause.get('known','values') not in ['values','known','unknown']:
            raise ValueError('Annotation condition must be values, known or unknown')
        if any(not isinstance(clause.get(key,[]),list) for key in ['include','exclude']):
            raise ValueError('Include/exclude must be lists of labels or tags')
        if clause.get('min') is not None and clause.get('max') is not None and clause['min']>clause['max']:
            raise ValueError('Minimum must not exceed maximum')
        if name not in records:
            records[name] = row_feature(df[name], specs[name], metadata.get(name))
    matches = []
    for row in range(len(df)):
        values = [clause_value(records[c['feature']], c, row) for c in clauses]
        if mode == 'all':
            result = False if False in values else None if None in values else True
        else:
            result = True if True in values else None if None in values else False
        matches.append(result is True)
    return np.asarray(matches, dtype=bool)


def cohort_profile(feature, rows):
    """Counts and numeric summaries among known values in a reproducible cohort."""
    values = [feature['values'][i] for i in rows if feature['values'][i] is not None]
    out = dict(n_items=len(rows), n_known=len(values), coverage=len(values)/len(rows) if len(rows) else None)
    if feature['kind'] == 'continuous':
        out.update(mean=float(np.mean(values)) if values else None,
                   median=float(np.median(values)) if values else None)
    else:
        counts = {}
        for value in values:
            for label in value if feature['kind'] == 'multilabel' else [value]:
                counts[label] = counts.get(label, 0) + 1
        out['counts'] = counts
    return out


def compare_cohorts(df, ids_a, ids_b, *, features=None, config=None):
    """Return per-label and numerical cohort differences; missingness stays visible.

    IDs may overlap; overlap is reported, not silently removed. No significance
    test is implied for exploratory selections. Raises for unknown/duplicate IDs.
    """
    cfg = config or {}
    df = add_derived_features(df)
    ids = df[cfg.get('id_column', 'id')].astype(str).to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError('Source IDs must be unique after string normalization')
    def rows_for(selected):
        selected = [str(v) for v in selected]
        if len(set(selected)) != len(selected):
            raise ValueError('Cohort IDs must be unique')
        rows = pd.Index(ids).get_indexer(selected)
        if (rows < 0).any():
            raise ValueError('Cohort contains unknown IDs')
        return rows
    a, b = rows_for(ids_a), rows_for(ids_b)
    specs = features if features is not None else cfg.get('features', default_features(df))
    specs = {k: {'kind': v} if isinstance(v, str) else dict(v) for k, v in specs.items()}
    specs, metadata = configure_cluster_metadata(specs, cfg)
    records = []
    for name, spec in specs.items():
        feature = row_feature(df[name], spec, metadata.get(name))
        pa, pb = cohort_profile(feature, a), cohort_profile(feature, b)
        base = dict(feature=name, kind=spec['kind'], n_a=len(a), n_b=len(b),
                    n_known_a=pa['n_known'], n_known_b=pb['n_known'],
                    coverage_a=pa['coverage'], coverage_b=pb['coverage'], overlap=len(set(a) & set(b)))
        if spec['kind'] == 'continuous':
            records.append(dict(base, label=None, mean_a=pa['mean'], mean_b=pb['mean'],
                                median_a=pa['median'], median_b=pb['median'],
                                mean_delta=pa['mean']-pb['mean'] if pa['mean'] is not None and pb['mean'] is not None else None))
        else:
            for label in sorted(set(pa['counts']) | set(pb['counts'])):
                ca, cb = pa['counts'].get(label, 0), pb['counts'].get(label, 0)
                fa = ca/pa['n_known'] if pa['n_known'] else None
                fb = cb/pb['n_known'] if pb['n_known'] else None
                records.append(dict(base, label=label, label_display=feature['label_display'][label],
                    count_a=ca, count_b=cb, prevalence_a=fa, prevalence_b=fb,
                    percentage_point_delta=100*(fa-fb) if fa is not None and fb is not None else None,
                    prevalence_ratio=fa/fb if fa is not None and fb is not None and fb > 0 else None))
    return pd.DataFrame(records)


def edge_distances(z, graph, metric, batch_size=32):
    """Distances for supplied graph edges; does not assert nearest-neighbor accuracy."""
    distances = np.empty(graph.shape, dtype=np.float32)
    for start in range(0, len(z), batch_size):
        end = min(len(z), start+batch_size)
        a, b = z[start:end, None, :], z[graph[start:end]]
        if metric == 'cosine':
            distances[start:end] = np.clip(1 - (a*b).sum(axis=2) / (np.linalg.norm(a, axis=2)*np.linalg.norm(b, axis=2)),0,2)
        else:
            distances[start:end] = np.linalg.norm(a-b, axis=2)
    return distances


def build_explorer_data(df, z, graph, distances, specs, results, cfg, coords, comparison=None):
    """Bounded row-level snapshot; aggregate reports remain available above limits."""
    limit = cfg.get('explorer_max_rows', 10000)
    if not specs:
        return {'available':False,'reason':'No configured features to explore.'}
    if not cfg.get('explorer', True) or len(df) > limit:
        return {'available': False, 'reason': 'Row exploration disabled by config.' if not cfg.get('explorer', True)
                else f'Row exploration is capped at {limit:,} rows; this audit has {len(df):,}. '
                     'Audit an explicit sample or raise explorer_max_rows for a deliberate in-memory run.'}
    features = {}
    for name, spec in specs.items():
        result = results[name]
        record = row_feature(df[name], spec, result.get('metadata'))
        record.update(source=result.get('source', {}), geometry=result.get('geometry', pd.DataFrame()).to_dict('records'),
                      per_label=result.get('per_label', pd.DataFrame()).to_dict('records'),
                      explanation=result['explanation'])
        mapping = cfg.get('predictions', {}).get(name)
        if mapping:
            mapping = {'column': mapping} if isinstance(mapping, str) else mapping
            series = result.get('prediction_series')
            if series is not None:
                pred_spec = {**spec, 'empty_is_missing': False,
                             'missing_values': mapping.get('missing_values', spec.get('missing_values', []))}
                record['predictions'] = row_feature(series, pred_spec, result.get('metadata'))['values']
        features[name] = record
    ids = df[cfg['id_column']].astype(str).tolist()
    if distances is None:
        distances = edge_distances(z, graph, cfg['metric'])
    cohorts = cfg.get('cohorts', {})
    for name, selected in cohorts.items():
        if name in ['All evaluated rows','Current selection','__rest__'] or not isinstance(selected, list) or len(set(map(str, selected))) != len(selected) or set(map(str, selected))-set(ids):
            raise ValueError(f'Cohort {name!r} must contain unique IDs in the evaluated dataset')
    out = dict(available=True,run_id=cfg['_run_id'],ids=ids, features=features, ks=sorted(set(cfg['_evaluated_ks'])),
               input_rows=cfg['_input_rows'],excluded_rows=cfg['_excluded_rows'],
               seed=cfg['seed'],
               graph=graph.tolist(), distances=np.round(distances.astype(float), 6).tolist(),
               metric=cfg['metric'], embedding_name=cfg.get('embedding_name', 'Embedding A'),
               coordinates=None if coords is None else np.round(coords, 6).tolist(), cohorts=cohorts,
               item_urls=None, preview_urls=None, comparison=None)
    def safe_urls(column):
        if not column or column not in df:
            return None
        return [value if isinstance(value, str) and value.startswith(('https://', 'http://')) else None for value in df[column]]
    out['item_urls'] = safe_urls(cfg.get('item_url_column'))
    out['preview_urls'] = safe_urls(cfg.get('preview_url_column'))
    if comparison is not None:
        out['comparison'] = dict(name=cfg.get('comparison_name', 'Embedding B'),
            graph=comparison['graph'].tolist(), distances=np.round(comparison['distances'].astype(float), 6).tolist(),
            paired=comparison['paired'].to_dict('records'),
            scope='Identical rows, candidate pool, annotations, metric and k; paired evaluable anchors.',
            interval='95% percentile bootstrap conditional on fixed graphs; shared neighbors can induce dependence. '
                     'Group resampling uses group_column when provided. No search-error or retraining uncertainty is estimated.')
    return out


def copy_explorer_assets(out):
    directory = Path(out)/'assets'
    directory.mkdir(exist_ok=True)
    for filename in ['metric_options.js', 'explorer.js', 'explorer.css', 'enhancements.js']:
        shutil.copyfile(Path(__file__).parent/'assets'/filename, directory/filename)
