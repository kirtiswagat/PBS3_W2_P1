"""Sl 3 / T0c - frozen trunk verification.

Answers, in order:
  1. Does Adam_best_model.h5 load at all in this environment?
  2. What are the candidate embedding layers, and what are their dimensions?
  3. Are inference-time predictions deterministic and batch-independent?
  4. Does the checkpoint still reproduce the published macro AUC of 0.7892?

A NO-GO on (1) or (4) invalidates the 13-week compute plan, so run this first.

Usage:
    python t0c_verify_checkpoint.py
    python t0c_verify_checkpoint.py --limit 64      # fast smoke test
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (  # noqa: E402
    CLASSES_5,
    apply_work1_label_policy,
    load_batch,
)

PHD = os.environ.get("PHD_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CKPT = f"{PHD}/Result_Analysis/DenseNet_Adam/Adam_best_model.h5"
DATA_DIR = f"{PHD}/Dataset/CheXpert-v1.0-small"
VALID_CSV = f"{DATA_DIR}/valid.csv"
TARGET_AUC = 0.7892
TOLERANCE = 0.01

results = {}


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def check_environment():
    section("1. ENVIRONMENT")
    import PIL
    import tensorflow as tf

    print(f"Python     : {sys.version.split()[0]}")
    print(f"TensorFlow : {tf.__version__}")
    print(f"Keras      : {tf.keras.__version__}")
    print(f"Pillow     : {PIL.__version__}   <- resize parity depends on this")
    print(f"NumPy      : {np.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs       : {gpus or 'none (CPU only)'}")
    results["tensorflow"] = tf.__version__
    results["pillow"] = PIL.__version__
    results["gpu"] = bool(gpus)


def load_checkpoint():
    section("2. CHECKPOINT LOAD")
    import tensorflow as tf

    if not os.path.exists(CKPT):
        print(f"FAIL - checkpoint not found at {CKPT}")
        results["load"] = "FAIL - missing"
        return None

    size_mb = os.path.getsize(CKPT) / 1e6
    print(f"Path : {CKPT}")
    print(f"Size : {size_mb:.1f} MB")

    try:
        model = tf.keras.models.load_model(CKPT, compile=False)
    except Exception as exc:
        print(f"FAIL - could not load: {type(exc).__name__}: {exc}")
        print("\nTry: pip install 'tensorflow==2.15.*' in a dedicated venv,")
        print("or load weights into a freshly built graph from DenseNet121_Adam.py.")
        results["load"] = f"FAIL - {type(exc).__name__}"
        return None

    print("OK - loaded")
    print(f"Input  : {model.input_shape}")
    print(f"Output : {model.output_shape}")
    print(f"Params : {model.count_params():,}")
    results["load"] = "OK"
    results["output_shape"] = str(model.output_shape)
    return model


def _shape_of(layer):
    for attr in ("output_shape", None):
        try:
            return str(layer.output_shape) if attr else str(layer.output.shape)
        except Exception:
            continue
    return "?"


def inspect_embedding_layers(model):
    section("3. EMBEDDING LAYER CANDIDATES")
    print(f"{'idx':>4}  {'layer':<34}{'type':<22}{'output':<18}")
    print("-" * 72)
    for i, layer in enumerate(model.layers):
        print(
            f"{i:>4}  {layer.name:<34}{type(layer).__name__:<22}{_shape_of(layer):<18}"
        )

    nested = [
        l for l in model.layers if isinstance(l, __import__("tensorflow").keras.Model)
    ]
    if nested:
        base = nested[0]
        print(f"\nNested backbone : {base.name}  ->  {_shape_of(base)}")
        print("  candidate A : backbone pooled output (generic, 1024-d)")
        results["embedding_A"] = _shape_of(base)

    dense = [l for l in model.layers if type(l).__name__ == "Dense"]
    for d in dense[:-1]:
        print(f"  candidate B : {d.name} -> {_shape_of(d)}  (task-adapted)")
        results["embedding_B"] = _shape_of(d)

    print("\nNOTE: the plan assumed a single 1024-d embedding. There are two")
    print("      candidates. Extract BOTH in Phase 1 - same forward pass, and")
    print("      let Gate 2 decide which probes better.")


def check_determinism(model, x):
    section("4. DETERMINISM AND BATCH INDEPENDENCE")
    p_a = model.predict(x, batch_size=32, verbose=0)
    p_b = model.predict(x, batch_size=32, verbose=0)
    repeat_delta = float(np.abs(p_a - p_b).max())

    p_single = model.predict(x, batch_size=1, verbose=0)
    batch_delta = float(np.abs(p_a - p_single).max())

    print(f"Max |repeat run  - first run| : {repeat_delta:.3e}")
    print(f"Max |batch size 1 - size 32|  : {batch_delta:.3e}")

    results["repeat_delta"] = repeat_delta
    results["batch_delta"] = batch_delta

    if batch_delta > 1e-4:
        print("\n  *** WARNING - predictions depend on batch composition. ***")
        print("  Work-1 built the model with `base_model(inputs, training=True)`,")
        print("  which hard-codes BatchNorm into training mode and overrides the")
        print("  training=False that predict() normally passes.")
        print("  Consequence: embeddings would be non-deterministic.")
        print("  Fix before Phase 1: rebuild the graph with training=False and")
        print("  load weights by name, then re-run this check.")
        results["batchnorm_issue"] = True
    else:
        print("\n  OK - inference is batch-independent.")
        results["batchnorm_issue"] = False


def check_auc(model, limit=None):
    section("5. BASELINE REPRODUCTION")
    from sklearn.metrics import roc_auc_score

    df = pd.read_csv(VALID_CSV)
    df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
    if limit:
        df = df.head(limit)
    df = apply_work1_label_policy(df)
    print(f"Frontal validation images : {len(df)}")

    x, y = load_batch(df, DATA_DIR, CLASSES_5)
    preds = model.predict(x, batch_size=32, verbose=0)

    aucs = {}
    for i, c in enumerate(CLASSES_5):
        if len(np.unique(y[:, i])) < 2:
            continue
        aucs[c] = roc_auc_score(y[:, i], preds[:, i])

    print(f"\n{'Pathology':<20}{'AUC':>8}{'Work-1':>10}{'delta':>9}")
    print("-" * 47)
    work1 = {
        "Atelectasis": 0.7495,
        "Cardiomegaly": 0.7609,
        "Consolidation": 0.7093,
        "Edema": 0.8948,
        "Pleural Effusion": 0.8314,
    }
    for c, v in aucs.items():
        print(f"{c:<20}{v:>8.4f}{work1[c]:>10.4f}{v - work1[c]:>+9.4f}")

    macro = float(np.mean(list(aucs.values())))
    print("-" * 47)
    print(f"{'MACRO':<20}{macro:>8.4f}{TARGET_AUC:>10.4f}{macro - TARGET_AUC:>+9.4f}")

    results["macro_auc"] = macro
    results["per_class_auc"] = {k: float(v) for k, v in aucs.items()}
    return macro


def verdict(macro, limit):
    section("VERDICT")
    if results.get("load") != "OK":
        print("NO-GO - checkpoint does not load.")
        print("Impact: the 3 weeks saved by reusing the trunk are lost.")
        print("Action: resolve the TF version, or re-scope the plan today.")
        return 1

    if limit:
        print(f"PARTIAL - ran on a {limit}-image subset only; rerun without --limit.")
        return 0

    if macro >= TARGET_AUC - TOLERANCE:
        print(f"GO - macro AUC {macro:.4f} meets the {TARGET_AUC} baseline.")
        print("Sl 3 / T0c can be marked complete. Proceed to Phase 1 extraction.")
        code = 0
    else:
        print(f"NO-GO - macro AUC {macro:.4f} is below {TARGET_AUC - TOLERANCE:.4f}.")
        print("Check in order: (a) preprocessing parity, (b) the BatchNorm issue")
        print("above, (c) label policy. Do not extract embeddings until resolved.")
        code = 1

    if results.get("batchnorm_issue"):
        print("\nOutstanding: BatchNorm training-mode issue must be fixed before")
        print("embedding extraction, even though the AUC gate passed.")
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="use only N images")
    ap.add_argument("--out", default="t0c_results.json")
    args = ap.parse_args()

    print("Sl 3 / T0c - FROZEN TRUNK VERIFICATION")

    check_environment()
    model = load_checkpoint()
    if model is None:
        json.dump(results, open(args.out, "w"), indent=2)
        sys.exit(verdict(0.0, args.limit))

    inspect_embedding_layers(model)

    df = pd.read_csv(VALID_CSV)
    df = df[df["Frontal/Lateral"] == "Frontal"].head(32).reset_index(drop=True)
    df = apply_work1_label_policy(df)
    x_small, _ = load_batch(df, DATA_DIR, CLASSES_5)
    check_determinism(model, x_small)

    macro = check_auc(model, args.limit)

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nResults written to {args.out}")
    sys.exit(verdict(macro, args.limit))


if __name__ == "__main__":
    main()
