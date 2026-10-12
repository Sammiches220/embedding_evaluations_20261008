#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
parquet-representation-audit validate --schema dataset_schema.json --config audit_config.json
parquet-representation-audit run --schema dataset_schema.json --config audit_config.json --out "${1:-audit_output}"
