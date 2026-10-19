"""Plain DataFrame input, feature discovery, and explicit tag parsing."""
from pathlib import Path
import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    'score', 'fav_count', 'up_score', 'down_score', 'file_size',
    'image_width', 'image_height', 'tag_count', 'tag_count_general',
    'tag_count_artist', 'tag_count_character', 'tag_count_copyright', 'tag_count_meta',
]
TAG_COLUMNS = ['tag_string_artist', 'tag_string_character', 'tag_string_copyright',
               'tag_string_general', 'tag_string_meta']


def default_features(df):
    """Return {column: {kind: ...}}; IDs, URLs, dictionaries are not targets."""
    features = {c: {'kind': 'categorical'} for c in df if c.endswith('_cluster')}
    for c in ['rating', 'file_ext']:
        if c in df:
            features[c] = {'kind': 'categorical'}
    for c in df:
        if pd.api.types.is_bool_dtype(df[c]):
            features[c] = {'kind': 'categorical'}
    for c in NUMERIC_COLUMNS:
        if c in df:
            features[c] = {'kind': 'continuous'}
    for c in TAG_COLUMNS:
        if c in df:
            features[c] = {'kind': 'multilabel', 'empty_is_missing': True,
                           'min_count': 20, 'max_labels': 100}
    return features


def add_derived_features(df):
    """Create readable numerical targets on a copy, without altering sample_df."""
    df = df.copy(deep=False)
    if 'image_width' in df and 'image_height' in df:
        df = df.assign(
            megapixels=df.image_width * df.image_height / 1e6,
            aspect_ratio=df.image_width / df.image_height.replace(0, np.nan))
    for col in ['created_at', 'updated_at', 'last_commented_at',
                'last_noted_at', 'last_comment_bumped_at']:
        if col in df:
            dates = pd.to_datetime(df[col], utc=True, errors='coerce')
            df[col + '_year'] = dates.dt.year.astype(float)
    return df


def embedding_matrix(df, column='embedding', embeddings=None, *, allow_invalid=False):
    """Accept one vector per cell or an already row-aligned N×D numeric array."""
    if embeddings is not None:
        z = np.asarray(embeddings)
    else:
        if column not in df:
            raise ValueError(f'Missing embedding column: {column}')
        vectors = []
        dim = None
        for row, value in enumerate(df[column]):
            if isinstance(value, str):
                raise ValueError(f'Embedding row {row} is text. Use arrays or Parquet; printed ... cannot be restored.')
            a = np.asarray(value, dtype=np.float32)
            if a.ndim != 1 or a.size == 0:
                raise ValueError(f'Embedding row {row} must be a nonempty 1D vector')
            if dim is None:
                dim = a.size
            if a.size != dim:
                raise ValueError(f'Embedding row {row} has {a.size} dimensions; expected {dim}')
            vectors.append(a)
        z = np.stack(vectors) if vectors else np.empty((0, 0))
    if z.ndim != 2 or len(z) != len(df) or z.shape[1] == 0:
        raise ValueError('embeddings must be N×D, in exactly the DataFrame row order')
    z = np.asarray(z, dtype=np.float64)
    if len(z) < 2 or (not allow_invalid and not np.isfinite(z).all()):
        raise ValueError('Need at least two embeddings, all finite; fix alignment/missing embeddings first')
    return z


def align_predictions(df, predictions, id_column='id'):
    """Left-align a separate prediction table by unique post ID; never by index."""
    if id_column not in predictions or predictions[id_column].isna().any() or predictions[id_column].duplicated().any():
        raise ValueError('Prediction IDs must be present, nonmissing, and unique')
    canonical = predictions[id_column].astype(str)
    if canonical.duplicated().any():
        raise ValueError('Prediction IDs collide after string normalization')
    aligned = predictions.drop(columns=id_column).set_axis(canonical, axis=0).reindex(df[id_column].astype(str))
    aligned.index = df.index
    return aligned


def tag_sets(series, empty_is_missing=True):
    sets, valid = [], []
    for value in series:
        if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
            sets.append(set()); valid.append(False)
        elif isinstance(value, str):
            tags = set(value.split())
            sets.append(tags); valid.append(bool(tags) or not empty_is_missing)
        elif isinstance(value, (list, tuple, set, np.ndarray)):
            tags = {str(v) for v in value}
            sets.append(tags); valid.append(bool(tags) or not empty_is_missing)
        else:
            raise ValueError('Tags must be space-separated strings or lists of strings')
    return sets, np.asarray(valid, dtype=bool)


def encode_feature(series, spec, vocabulary=None):
    """Return values, known-row mask, and label names. Unknown stays separate."""
    kind = spec['kind']
    if kind == 'continuous':
        values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(values)
        for sentinel in spec.get('missing_values', []):
            valid &= values != sentinel
        return values, valid, []
    if kind == 'categorical':
        if spec.get('cluster_ids'):
            from .cluster_metadata import cluster_id
            values = np.array([cluster_id(v) for v in series], dtype=object)
            missing = {cluster_id(v) for v in spec.get('missing_values', [])}
            valid = series.notna().to_numpy() & ~np.isin(values, list(missing))
        else:
            valid = series.notna().to_numpy() & ~series.isin(spec.get('missing_values', [])).to_numpy()
            values = np.array([str(v) for v in series], dtype=object)
        names = sorted(set(values[valid])) if vocabulary is None else list(vocabulary)
        return values, valid, names
    if kind != 'multilabel':
        raise ValueError(f'Unknown feature kind: {kind}')
    sets, valid = tag_sets(series, spec.get('empty_is_missing', True))
    counts = {}
    for tags, known in zip(sets, valid):
        if known:
            for tag in tags:
                counts[tag] = counts.get(tag, 0) + 1
    if vocabulary is None:
        selected = sorted(counts, key=lambda tag: (-counts[tag], tag))
        selected = [tag for tag in selected if counts[tag] >= spec.get('min_count', 20)]
        limit = spec.get('max_labels', 100)
        names = list(spec['vocabulary']) if 'vocabulary' in spec else selected if limit is None else selected[:limit]
    else:
        names = list(vocabulary)
    lookup = {name: i for i, name in enumerate(names)}
    values = np.zeros((len(series), len(names)), dtype=np.uint8)
    for row, tags in enumerate(sets):
        for tag in tags:
            if tag in lookup:
                values[row, lookup[tag]] = 1
    return values, valid, names


def export_dataframe(df, directory, id_column='id', embedding_column='embedding'):
    """Save embeddings separately; omit irrelevant nested media_asset dictionaries."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    z = embedding_matrix(df, embedding_column)
    np.save(directory / 'embeddings.npy', z, allow_pickle=False)
    np.save(directory / 'embedding_ids.npy', df[id_column].astype(str).to_numpy(dtype=str), allow_pickle=False)
    df.drop(columns=[embedding_column, 'media_asset'], errors='ignore').to_parquet(directory / 'features.parquet', index=False)
    return directory


def load_dataframe(path):
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    raise ValueError('Use Parquet or pass a DataFrame in Python. A text preview is not a dataset.')
