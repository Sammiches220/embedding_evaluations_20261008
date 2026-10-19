from __future__ import annotations

import argparse

from .io import load_array, load_labels
from .report import run_categorical_audit, run_continuous_audit


def _ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Audit feature representation in an embedding")
    p.add_argument("--embeddings", required=True, help=".npy/.npz/.csv matrix with shape [N, D]")
    p.add_argument("--labels", required=True, help="Observed/source labels")
    p.add_argument("--label-column", default=None)
    p.add_argument("--predicted-labels", default=None, help="Optional extrinsic predictions aligned to the same rows")
    p.add_argument("--predicted-label-column", default=None)
    p.add_argument("--feature-type", choices=["categorical", "continuous"], default="categorical")
    p.add_argument("--ks", type=_ints, default=[5, 10, 25, 50, 100])
    p.add_argument("--permutations", type=int, default=30)
    p.add_argument("--missingness", type=_floats, default=[0.1, 0.25, 0.5, 0.75])
    p.add_argument("--out", default="audit_output")
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    z = load_array(args.embeddings)
    y = load_labels(args.labels, args.label_column)
    if len(z) != len(y):
        raise ValueError(f"Row mismatch: embeddings={len(z)} labels={len(y)}")

    if args.feature_type == "continuous":
        run_continuous_audit(z, y, args.out, args.ks, args.random_state)
        return

    y_pred = None
    if args.predicted_labels:
        y_pred = load_labels(args.predicted_labels, args.predicted_label_column)
        if len(y_pred) != len(y):
            raise ValueError("Predicted labels must align row-for-row with observed labels")

    run_categorical_audit(
        z, y, args.out, args.ks,
        y_pred=y_pred,
        n_permutations=args.permutations,
        missingness_fractions=args.missingness,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
