"""Schema-driven Parquet representation auditing through plain Python functions."""
__version__='1.0.0'
from .config import validate_bundle
from .ingest import inspect_parquet
from .runner import run_audit
__all__=['validate_bundle','inspect_parquet','run_audit']
