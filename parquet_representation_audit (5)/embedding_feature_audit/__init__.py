"""Readable DataFrame-first embedding feature auditing."""
from .dataframe import export_dataframe, default_features
from .dataframe_report import audit_dataframe
from .normalization import l2_normalize
from .topology import component_filtration, rips_h1
from .exploration import query_dataframe, compare_cohorts
__version__='0.6.0'
__all__=['audit_dataframe','export_dataframe','default_features','query_dataframe','compare_cohorts','l2_normalize','component_filtration','rips_h1']
