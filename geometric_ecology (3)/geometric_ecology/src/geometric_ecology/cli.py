"""Run with `geometric-ecology` or `python -m geometric_ecology`."""
import argparse
import functools
import http.server
import json
from pathlib import Path
import sys
import webbrowser
import numpy as np
from .core import build_map, score_queries, route
from .demo import synthetic_corpus
from .io import save_model, load_model, export_explorer, dump_json, clean
from .validation import evaluate


def parser():
    p = argparse.ArgumentParser(description="Corpus-relative geometry and human navigation prototype")
    sub = p.add_subparsers(dest="command", required=True)
    for command in ["demo", "build"]:
        c = sub.add_parser(command, help="Map a synthetic corpus" if command == "demo" else "Map an embedding matrix")
        c.add_argument("--out", type=Path, required=True)
        c.add_argument("--scales", type=int, nargs="+", default=[12, 24, 48] if command == "demo" else [6, 12, 24])
        c.add_argument("--support", type=float, default=.5)
        c.add_argument("--resamples", type=int, default=3 if command == "demo" else 4)
        c.add_argument("--landmarks", type=int, default=96 if command == "demo" else 32)
        c.add_argument("--regions", type=int, default=72 if command == "demo" else 8)
        c.add_argument("--seed", type=int, default=42)
        c.add_argument("--skip-validation", action="store_true")
        c.add_argument("--skip-nulls", action="store_true")
        c.add_argument("--overwrite", action="store_true", help="Replace generated outputs in existing directory")
        if command == "demo":
            c.add_argument("--n", type=int, default=20000)
            c.add_argument("--dimensions", type=int, default=64)
        else:
            c.add_argument("--embeddings", type=Path, required=True)
            c.add_argument("--ids", type=Path, help="Aligned unique string array .npy or JSON list")
            c.add_argument("--metadata", type=Path, help="Aligned JSON list or JSONL objects; display/evaluation only")
            c.add_argument("--encoder", required=True, help="Frozen encoder name and version")
            c.add_argument("--allow-slow", action="store_true", help="Permit exact analysis above 20,000 rows")
    c = sub.add_parser("score", help="Attach new embeddings to a frozen map")
    c.add_argument("--map", type=Path, required=True)
    c.add_argument("--queries", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("route", help="Shortest support-graph route between corpus IDs")
    c.add_argument("--map", type=Path, required=True)
    c.add_argument("--start", required=True)
    c.add_argument("--end", required=True)
    c = sub.add_parser("serve", help="Serve the generated explorer locally")
    c.add_argument("--map", type=Path, required=True)
    c.add_argument("--port", type=int, default=8765)
    c.add_argument("--open", action="store_true", help="Open a browser")
    return p


def load_metadata(path):
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("Metadata must be a JSON list or JSONL stream of objects aligned to rows.")
    return data


def run(args):
    if args.command in ("demo", "build"):
        if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
            raise ValueError("Output directory is not empty. Choose a new --out or pass --overwrite.")
        if args.command == "demo":
            x, ids, metadata = synthetic_corpus(args.n, args.dimensions, args.seed)
            encoder = f"synthetic-mechanisms-v1/seed-{args.seed}"
        else:
            x = np.load(args.embeddings, mmap_mode="r", allow_pickle=False)
            if len(x) > 20000 and not args.allow_slow:
                raise ValueError("Exact prototype above 20,000 rows: sample first, or explicitly use --allow-slow. No ANN backend is provided.")
            ids = None
            if args.ids:
                ids = np.load(args.ids, allow_pickle=False).tolist() if args.ids.suffix == ".npy" else json.loads(args.ids.read_text(encoding="utf-8"))
                if not isinstance(ids, list) or any(isinstance(i, (list, dict)) for i in ids):
                    raise ValueError("IDs must be a flat array/list.")
            metadata = load_metadata(args.metadata) if args.metadata else None
            if metadata is not None and len(metadata) != len(x):
                raise ValueError("Metadata count differs from embedding row count.")
            encoder = args.encoder
        print(f"Mapping {len(x)} rows; exact backend; no labels used for geometry.", flush=True)
        model = build_map(x, ids, scales=args.scales, support=args.support, repeats=args.resamples,
                          landmarks=args.landmarks, regions=args.regions, seed=args.seed, encoder=encoder)
        print(f"Geometry built in {model['manifest']['timings']['build_total_s']:.2f}s.", flush=True)
        if not args.skip_validation: print("Running independent geometric controls...", flush=True)
        e = None if args.skip_validation else evaluate(model, metadata, include_nulls=not args.skip_nulls)
        save_model(model, args.out)
        if metadata is not None: dump_json(args.out / "metadata.json", metadata)
        else: (args.out / "metadata.json").unlink(missing_ok=True)
        print("Drawing automatic terrain, roads, contours and shorelines...", flush=True)
        export_explorer(model, args.out, e, metadata)
        print(f"Wrote {args.out.resolve() / 'explorer.html'}", flush=True)
    elif args.command == "score":
        model = load_model(args.map)
        result = score_queries(model, np.load(args.queries, allow_pickle=False))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        dump_json(args.out, {"corpus_sha256": model["manifest"]["corpus_sha256"], "queries": result})
        print(f"Wrote {args.out.resolve()}")
    elif args.command == "route":
        model = load_model(args.map)
        lookup = {item: i for i, item in enumerate(model["ids"])}
        if args.start not in lookup or args.end not in lookup: raise ValueError("Route endpoints must be known corpus IDs.")
        result = route(model["graph"], lookup[args.start], lookup[args.end])
        result["item_ids"] = [model["ids"][i] for i in result["path"]]
        print(json.dumps(clean(result), indent=2))
    elif args.command == "serve":
        directory = args.map.resolve()
        if not (directory / "explorer.html").is_file(): raise ValueError("Map directory has no explorer.html.")
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
        with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            url = f"http://127.0.0.1:{server.server_port}/explorer.html"
            print(f"Serving {url} — Ctrl+C to stop.", flush=True)
            if args.open: webbrowser.open(url)
            try: server.serve_forever()
            except KeyboardInterrupt: pass


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try: run(args)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        p.exit(2, f"Error: {exc}\n")

if __name__ == "__main__": main()
