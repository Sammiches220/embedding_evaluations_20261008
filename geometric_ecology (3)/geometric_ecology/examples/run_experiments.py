"""Reproduce scale timings and cross-seed synthetic diagnostics.

Run from the project root after installation:
    python examples/run_experiments.py
Outputs are written beside this script, not to the installed package.
Each scale benchmark runs in a fresh process to make peak RSS comparable.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from geometric_ecology.core import build_map
from geometric_ecology.demo import synthetic_corpus
from geometric_ecology.io import dump_json
from geometric_ecology.validation import evaluate

ROOT = Path(__file__).resolve().parent


def main():
    rows = []
    program = """
import json,resource,time
from geometric_ecology.demo import synthetic_corpus
from geometric_ecology.core import build_map
x,ids,_=synthetic_corpus({n},48,42)
a=build_map(x,ids,repeats=4,seed=42)
print(json.dumps(dict(n={n},dimensions=48,build_s=a['manifest']['timings']['build_total_s'],peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)))
"""
    for n in [500, 1500, 5000]:
        env = os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        process = subprocess.run([sys.executable, "-c", program.format(n=n)], env=env, capture_output=True, text=True, check=True)
        rows.append(json.loads(process.stdout))
    with (ROOT / "scaling.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    diagnostics = []
    for seed in [7, 42, 101]:
        x, ids, metadata = synthetic_corpus(1100, 48, seed)
        model = build_map(x, ids, repeats=4, seed=seed)
        e = evaluate(model, metadata, include_nulls=False)
        diagnostics.append({"seed": seed, "mechanism_checks": e["mechanism_checks"], "heldout": e["heldout"], "observed": e["observed"]})
    dump_json(ROOT / "cross_seed.json", diagnostics)
    lines = ["# Replication and scale measurements", "", "Measured on the execution host, Linux, Python 3.12; 48 dimensions, four resamples, one BLAS thread. Peak RSS is Linux ru_maxrss of each fresh process (includes Python and imported libraries). Timings cover map construction, not the validation suite or HTML export. These are descriptive timings, not scaling guarantees.", "", "| Rows | Build seconds | Peak RSS MiB |", "|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r['n']} | {r['build_s']:.3f} | {r['peak_rss_mib']:.1f} |")
    lines.extend(["", "## Cross-seed planted mechanism diagnostics", "", "No training labels are supplied to the map. The same predeclared contrasts are evaluated for three independently generated corpora. These are synthetic rank-separation checks, not semantic accuracy or human task success.", "", "| Contrast | Seed 7 AUC | Seed 42 AUC | Seed 101 AUC |", "|---|---:|---:|---:|"])
    for i, check in enumerate(diagnostics[0]["mechanism_checks"]):
        values = [d["mechanism_checks"][i]["auc"] for d in diagnostics]
        lines.append(f"| {check['question']} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines.extend(["", "The exact neighbor backend still requires quadratic corpus work. A small-dimensional timing cannot establish feasibility at 8 million rows or 1024 dimensions. Reference-subset maps change the corpus-relative object being measured; an ANN replacement must report recall, topology changes, and profile stability.", ""])
    (ROOT / "REPLICATION.md").write_text("\n".join(lines))
    print("Wrote examples/scaling.csv, cross_seed.json and REPLICATION.md")

if __name__ == "__main__": main()
