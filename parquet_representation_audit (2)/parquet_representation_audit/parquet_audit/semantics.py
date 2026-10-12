"""Explicit scalar and set semantics. No dataset-specific names or code evaluation."""
import json
import math
from numbers import Integral, Real
from typing import Any
import numpy as np
import pandas as pd

KINDS = {'cluster_label':'categorical','categorical':'categorical','boolean':'categorical',
         'ordinal':'categorical','count':'continuous','numeric':'continuous',
         'multilabel':'multilabel','datetime':None,'text':None}


def is_null(value: Any) -> bool:
    """Containers are observations; False, zero and empty sets are not nulls."""
    return value is None or value is pd.NA or value is pd.NaT or isinstance(value, (float,np.floating)) and math.isnan(value)


def typed_key(value: Any, id_type: str) -> str:
    """Validate an ID's declared type without float conversion or string coercion."""
    if isinstance(value,np.generic): value=value.item()
    if id_type=='integer':
        if isinstance(value,bool) or not isinstance(value,Integral):
            raise ValueError(f'Expected an integer ID, got {value!r}; declare conversion explicitly')
        return str(int(value))
    if id_type=='boolean':
        if not isinstance(value,bool): raise ValueError(f'Expected boolean, got {value!r}')
        return 'true' if value else 'false'
    if id_type=='string':
        if not isinstance(value,str): raise ValueError(f'Expected string ID, got {value!r}')
        return value
    raise ValueError(f'Unsupported id_type {id_type!r}')


def equal_typed(a: Any,b: Any) -> bool:
    if isinstance(a,np.generic): a=a.item()
    if isinstance(b,np.generic): b=b.item()
    if isinstance(a,bool)!=isinstance(b,bool): return False
    if isinstance(a,str)!=isinstance(b,str): return False
    try: return bool(a==b)
    except (ValueError,TypeError): return False


def missing(value: Any,spec: dict) -> bool:
    if is_null(value): return True
    if isinstance(value,str) and value=='' and spec.get('empty_is_unknown',False): return True
    return any(equal_typed(value,s) for s in spec.get('missing_values',[]))


def convert(value: Any,spec: dict) -> Any:
    """Only declared, bounded conversions; nested values need an explicit path."""
    for key in spec.get('struct_path',[]):
        if is_null(value): return None
        if not isinstance(value,dict) or key not in value: raise ValueError(f'Cannot extract struct field {key!r}')
        value=value[key]
    if missing(value,spec): return None
    rule=spec.get('conversion','none')
    if rule=='integer':
        if isinstance(value,bool): raise ValueError('Boolean cannot be converted to a count or integer ID')
        if isinstance(value,str):
            text=value.strip()
            if not text or not text.lstrip('+-').isdigit(): raise ValueError(f'Not an exact integer: {value!r}')
            return int(text)
        if isinstance(value,Integral): return int(value)
        if isinstance(value,Real) and math.isfinite(value) and value==int(value) and abs(value)<=2**53: return int(value)
        raise ValueError(f'Not a lossless integer: {value!r}')
    if rule=='numeric':
        if isinstance(value,bool): raise ValueError('Boolean is not a numeric measurement')
        return float(value)
    if rule=='string':
        if isinstance(value,(str,Integral,bool)): return str(value)
        raise ValueError('String conversion accepts only exact primitive values')
    if rule=='boolean':
        mapping=spec.get('boolean_values',{})
        if isinstance(value,bool): return value
        if str(value) not in mapping: raise ValueError(f'Undeclared boolean value {value!r}')
        return mapping[str(value)]
    return value


def parse_tags(value: Any,spec: dict) -> list[str] | None:
    if missing(value,spec): return None
    encoding=spec['encoding']
    if encoding=='list':
        if not isinstance(value,(list,tuple,np.ndarray,set)): raise ValueError('Multilabel encoding=list requires a list')
        items=list(value)
    elif encoding=='delimited':
        if not isinstance(value,str): raise ValueError('Delimited tags require a string')
        items=[] if value=='' else value.split(spec['delimiter'])
    elif encoding=='json':
        if not isinstance(value,str): raise ValueError('JSON tags require a JSON string')
        items=json.loads(value)
        if not isinstance(items,list): raise ValueError('JSON multilabel value must be an array')
    else: raise ValueError(f'Unsupported tag encoding {encoding}')
    tags=[]
    for item in items:
        if not isinstance(item,str): raise ValueError('Each tag must be a string; nested/null/numeric tags are unsupported')
        if spec.get('strip',False): item=item.strip()
        if spec.get('case','preserve')=='lower': item=item.lower()
        if item=='':
            if spec.get('empty_token','error')=='drop': continue
            raise ValueError('Empty tag token; declare empty_token=drop if intended')
        tags.append(item)
    tags=sorted(set(tags))
    return None if not tags and spec.get('empty_is_unknown',False) else tags


def normalize_value(value: Any,spec: dict) -> Any:
    """Missing sentinels are handled before domain/range validation."""
    try:
        value=convert(value,spec)
        if value is None: return None
        kind=spec.get('kind')
        if kind=='multilabel': return parse_tags(value,spec)
        if kind in ['numeric','count']:
            if isinstance(value,(str,bool)) or not isinstance(value,Real): raise ValueError('Expected numeric scalar; strings need conversion=numeric')
            if not math.isfinite(value): raise ValueError('Nonfinite measurement')
            if kind=='count' and (value<0 or value!=int(value)): raise ValueError('Counts must be nonnegative integers; use numeric for net measurements')
            for limit,op in [('minimum',lambda x,y:x<y),('maximum',lambda x,y:x>y)]:
                if limit in spec and op(value,spec[limit]): raise ValueError(f'Value violates {limit}={spec[limit]}')
            return value
        if kind in ['cluster_label','categorical','ordinal','boolean']:
            typ='boolean' if kind=='boolean' else spec.get('id_type','string')
            key=typed_key(value,typ)
            if kind=='ordinal' and key not in [typed_key(x,typ) for x in spec['order']]: raise ValueError('Value outside declared ordinal order')
            return key
        if kind=='text':
            if not isinstance(value,str): raise ValueError('Text requires a scalar string')
            return value
        if kind=='datetime':
            time=pd.Timestamp(value)
            if time.tzinfo is None: time=time.tz_localize(spec['timezone'],ambiguous='raise',nonexistent='raise')
            return time.tz_convert(spec['timezone'])
        return value
    except (ValueError,TypeError,OverflowError,KeyError) as exc:
        if spec.get('invalid_values','error')=='unknown': return None
        raise ValueError(str(exc)) from exc


def normalize_series(series: pd.Series,spec: dict) -> pd.Series:
    values=[]
    for position,value in enumerate(series):
        try: values.append(normalize_value(value,spec))
        except ValueError as exc: raise ValueError(f'{series.name}, batch row {position}: {exc}') from exc
    # Object dtype retains exact int IDs and unknown versus known-empty lists.
    return pd.Series(values,index=series.index,name=series.name,dtype=object)


def computation_spec(spec: dict) -> dict:
    """Adapter to the preserved plain-array measurement/report implementation."""
    return {'kind':KINDS[spec['kind']], 'semantic_kind':spec['kind'], 'description':spec.get('description',''),
            'units':spec.get('units',''), 'family':spec.get('family',''), 'missing_values':[],
            'empty_is_missing':False, 'min_count':spec.get('min_count',1), 'max_labels':spec.get('max_labels',100),
            **({'vocabulary':spec['vocabulary']} if 'vocabulary' in spec else {}),
            **({'metadata_file':spec['metadata_path'],'normalized_ids':True} if spec.get('metadata_path') else {})}
