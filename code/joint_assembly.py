"""T23 - joint metrics and the analytic combination assembly.

WHAT THIS MEASURES AND WHY IT IS NOT JUST ANOTHER METRIC
    Binary Relevance scores each disease independently, so it can post good
    per-class AUC while getting whole studies wrong - predicting the right
    labels, but never the right *set*. Hamming loss and exact-match ratio are
    where that shows up, and the size of the gap between them is the only
    quantitative justification this study has for deferring Tier 3. H4 is
    currently deferred on an argument; this turns it into a number.

THE ASSEMBLY IS ANALYTIC, NOT 3,125 TRAINING RUNS
    Under plain-concatenation BR the heads are independent by construction:
    disease i's policy cannot change disease j's probabilities. So the 5^5
    grid over the competition labels factorises - fit 5 policies x N labels
    once, cache the probability columns, and every combination is a column
    selection afterwards. 70 fits per seed, then 3,125 evaluations for free.

EVALUATION POPULATION
    Exact-match needs a complete label vector, so rows carrying -1 in any
    evaluated label are dropped rather than imputed (invariant 6). That leaves
    13,448 of 18,692 Train-C rows at 5 labels and 11,719 at 14. The count is
    reported with every result because it differs between the two label sets.

THRESHOLDS
    Fitted on Train-B only (invariant 2), per (label, policy, seed), then
    frozen. Joint metrics are threshold-dependent in a way AUC is not, so
    tuning them on the evaluation split would manufacture the result.
"""

import argparse
import itertools
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import export_results
import heads
import trunks
from calibration import apply_temperature, fit_temperature, fit_threshold
from heads import (
    HEAD_CLASSES,
    POLICIES,
    Standardizer,
    apply_policy,
    convergence_report,
    fit_head,
    load_split,
    reset_convergence,
    set_trunk,
    subsample_patients,
)
from metrics import joint_metrics

BASE = str(Path(__file__).resolve().parent.parent)
heads.BASE = BASE  # heads.py hardcodes a stale machine path; repoint it here
OUT = f"{BASE}/results"


def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def build_probability_cache(args, labels, cache_path):
    """Fit every (label, policy, seed) head once and cache Train-C probabilities."""
    train, trainb, evalset = (
        load_split("train_a"),
        load_split("train_b"),
        load_split("train_c"),
    )
    names = train["label_names"]
    seeds = list(range(42, 42 + args.seeds))

    scaler = Standardizer().fit(train["emb_backbone"])
    x_tr_all = scaler.transform(train["emb_backbone"])
    x_b_all = scaler.transform(trainb["emb_backbone"])
    x_e_all = scaler.transform(evalset["emb_backbone"])
    row_sets = {s: subsample_patients(train["paths"], s, args.fraction) for s in seeds}

    n_eval = x_e_all.shape[0]
    probs = np.full(
        (len(labels), len(POLICIES), len(seeds), n_eval), np.nan, dtype=np.float32
    )
    thresholds = np.full(
        (len(labels), len(POLICIES), len(seeds)), np.nan, dtype=np.float32
    )

    total = len(labels) * len(POLICIES) * len(seeds)
    done = 0
    for li_out, lab in enumerate(labels):
        li = names.index(lab)
        y_tr_raw = train["labels"][:, li]
        y_b_raw = trainb["labels"][:, li]
        b_mask = y_b_raw != -1
        y_b = np.nan_to_num(y_b_raw[b_mask], nan=0.0)

        for pi, policy in enumerate(POLICIES):
            for si, s in enumerate(seeds):
                rows = row_sets[s]
                idx, y, w = apply_policy(y_tr_raw[rows], policy)
                clf = fit_head(
                    x_tr_all[rows][idx], y, w, C=args.C, max_iter=args.max_iter
                )
                done += 1
                if clf is None:
                    continue
                p_b = clf.predict_proba(x_b_all[b_mask])[:, 1]
                t = fit_temperature(y_b, p_b)
                thresholds[li_out, pi, si] = fit_threshold(
                    y_b, apply_temperature(p_b, t)
                )
                probs[li_out, pi, si] = apply_temperature(
                    clf.predict_proba(x_e_all)[:, 1], t
                )
            print(f"  {lab:<28} {policy:<9} [{done:>3}/{total}]")

    np.savez_compressed(
        cache_path,
        probs=probs,
        thresholds=thresholds,
        labels=np.array(labels),
        policies=np.array(POLICIES),
        seeds=np.array(seeds),
        eval_labels=evalset["labels"],
        label_names=np.array(names),
    )
    print(f"\nCached {cache_path}")
    return cache_path


def _population(eval_labels, names, labels):
    """Rows with no -1 in any evaluated label, plus the 0/1 target matrix."""
    cols = [names.index(l) for l in labels]
    sub = eval_labels[:, cols]
    complete = (sub != -1).all(axis=1)
    return complete, np.nan_to_num(sub[complete], nan=0.0)


def _joint_for(probs, thresholds, y_mat, complete, policy_per_label):
    """Joint metrics for one policy assignment, averaged over seeds."""
    n_seeds = probs.shape[2]
    out = []
    for si in range(n_seeds):
        cols, thr = [], []
        for li, pi in enumerate(policy_per_label):
            col = probs[li, pi, si][complete]
            if np.isnan(col).any():
                return None
            cols.append(col)
            thr.append(thresholds[li, pi, si])
        p_mat = np.column_stack(cols)
        pred = (p_mat >= np.asarray(thr)[None, :]).astype(int)
        out.append(
            {
                "hamming_loss": float((pred != y_mat).mean()),
                "exact_match": float((pred == y_mat).all(axis=1).mean()),
                "mean_per_label_accuracy": float((pred == y_mat).mean()),
            }
        )
    return {
        k: {
            "mean": float(np.mean([o[k] for o in out])),
            "sd": float(np.std([o[k] for o in out], ddof=1)) if len(out) > 1 else 0.0,
        }
        for k in out[0]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", default="densenet121", choices=list(trunks.TRUNKS))
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fraction", type=float, default=0.8)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument(
        "--sweep-tag", default=None, help="source of the per-label best policy"
    )
    ap.add_argument(
        "--reuse-cache", action="store_true", help="skip refitting if the cache exists"
    )
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    set_trunk(args.trunk)
    reset_convergence()
    suffix = "" if args.trunk == "densenet121" else f"_{args.trunk}"
    sweep_tag = args.sweep_tag or f"sweep_train_c{suffix}_C0.1"
    tag = args.tag or f"joint_assembly_{args.trunk}"
    cache_path = f"{OUT}/joint_probs_{args.trunk}_C{args.C}.npz"

    print(f"T23 - JOINT METRICS  [trunk: {args.trunk}, C: {args.C}]")
    t_start = time.time()

    all_labels = list(
        np.load(f"{trunks.emb_dir(BASE, args.trunk)}/train_c.npz", allow_pickle=True)[
            "label_names"
        ]
    )
    all_labels = [str(x) for x in all_labels]

    section("PROBABILITY CACHE")
    if args.reuse_cache and os.path.exists(cache_path):
        print(f"Reusing {cache_path}")
    else:
        build_probability_cache(args, all_labels, cache_path)

    cache = np.load(cache_path, allow_pickle=True)
    probs, thresholds = cache["probs"], cache["thresholds"]
    labels = [str(x) for x in cache["labels"]]
    policies = [str(x) for x in cache["policies"]]
    names = [str(x) for x in cache["label_names"]]
    eval_labels = cache["eval_labels"]

    with open(f"{OUT}/{sweep_tag}.json") as f:
        sweep = json.load(f)
    best_policy = {l: r.get("_best") for l, r in sweep.get("results", {}).items()}

    results = {}
    for set_name, label_set in (
        ("all_14", labels),
        ("competition_5", list(HEAD_CLASSES)),
    ):
        label_set = [l for l in label_set if l in labels]
        complete, y_mat = _population(eval_labels, names, label_set)
        rows = [labels.index(l) for l in label_set]
        sub_probs, sub_thr = probs[rows], thresholds[rows]

        section(
            f"JOINT METRICS - {set_name} ({len(label_set)} labels, n={complete.sum()})"
        )
        block = {
            "n_labels": len(label_set),
            "n_rows": int(complete.sum()),
            "global": {},
        }

        for pi, policy in enumerate(policies):
            m = _joint_for(sub_probs, sub_thr, y_mat, complete, [pi] * len(label_set))
            if m:
                block["global"][policy] = m
                print(
                    f"  global {policy:<10} hamming={m['hamming_loss']['mean']:.4f} "
                    f"exact_match={m['exact_match']['mean']:.4f}"
                )

        # Tier 2: each disease gets the policy the sweep found best for it
        assign = [policies.index(best_policy.get(l) or policies[0]) for l in label_set]
        m = _joint_for(sub_probs, sub_thr, y_mat, complete, assign)
        if m:
            block["per_label_best"] = {
                "assignment": {l: policies[a] for l, a in zip(label_set, assign)},
                **m,
            }
            print(
                f"  per-label best   hamming={m['hamming_loss']['mean']:.4f} "
                f"exact_match={m['exact_match']['mean']:.4f}"
            )

        best_global = min(
            block["global"].items(), key=lambda kv: kv[1]["hamming_loss"]["mean"]
        )
        if m:
            block["tier2_vs_best_global"] = {
                "best_global_policy": best_global[0],
                "hamming_delta": m["hamming_loss"]["mean"]
                - best_global[1]["hamming_loss"]["mean"],
                "exact_match_delta": m["exact_match"]["mean"]
                - best_global[1]["exact_match"]["mean"],
            }

        # the BR weakness, quantified: per-label accuracy is high, exact-match is not
        if m:
            block["br_gap"] = {
                "mean_per_label_accuracy": m["mean_per_label_accuracy"]["mean"],
                "exact_match": m["exact_match"]["mean"],
                "gap": m["mean_per_label_accuracy"]["mean"] - m["exact_match"]["mean"],
            }
            print(
                f"  BR gap: per-label acc {block['br_gap']['mean_per_label_accuracy']:.4f} "
                f"vs exact-match {block['br_gap']['exact_match']:.4f} "
                f"-> {block['br_gap']['gap']:.4f}"
            )

        results[set_name] = block

    # 3,125 analytic assembly over the competition labels
    section("ANALYTIC ASSEMBLY - 5^5 over the competition labels")
    label_set = [l for l in HEAD_CLASSES if l in labels]
    complete, y_mat = _population(eval_labels, names, label_set)
    rows = [labels.index(l) for l in label_set]
    sub_probs, sub_thr = probs[rows], thresholds[rows]

    surface = []
    for combo in itertools.product(range(len(policies)), repeat=len(label_set)):
        m = _joint_for(sub_probs, sub_thr, y_mat, complete, list(combo))
        if m:
            surface.append((combo, m["hamming_loss"]["mean"], m["exact_match"]["mean"]))
    print(
        f"  evaluated {len(surface)} combinations on n={complete.sum()} rows (no refitting)"
    )

    if surface:
        by_h = sorted(surface, key=lambda x: x[1])
        by_e = sorted(surface, key=lambda x: -x[2])
        results["assembly"] = {
            "n_combinations": len(surface),
            "n_rows": int(complete.sum()),
            "labels": label_set,
            "best_hamming": {
                "assignment": {l: policies[c] for l, c in zip(label_set, by_h[0][0])},
                "hamming_loss": by_h[0][1],
                "exact_match": by_h[0][2],
            },
            "worst_hamming": {
                "assignment": {l: policies[c] for l, c in zip(label_set, by_h[-1][0])},
                "hamming_loss": by_h[-1][1],
                "exact_match": by_h[-1][2],
            },
            "best_exact_match": {
                "assignment": {l: policies[c] for l, c in zip(label_set, by_e[0][0])},
                "hamming_loss": by_e[0][1],
                "exact_match": by_e[0][2],
            },
            "hamming_spread": by_h[-1][1] - by_h[0][1],
            "exact_match_spread": by_e[0][2] - by_e[-1][2],
        }
        a = results["assembly"]
        print(
            f"  best  hamming={a['best_hamming']['hamming_loss']:.4f}  {a['best_hamming']['assignment']}"
        )
        print(f"  worst hamming={a['worst_hamming']['hamming_loss']:.4f}")
        print(
            f"  spread: hamming {a['hamming_spread']:.4f}, exact-match {a['exact_match_spread']:.4f}"
        )

    os.makedirs(OUT, exist_ok=True)
    payload = {
        "phase": "joint",
        "trunk": args.trunk,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "C": args.C,
        "seeds": [int(s) for s in cache["seeds"]],
        "sweep_tag": sweep_tag,
        "eval_split": "train_c",
        "probability_cache": os.path.basename(cache_path),
        "results": results,
    }
    path = f"{OUT}/{tag}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    section("DONE")
    print(f"Written {path}")
    print(convergence_report())
    print(f"Total  : {time.time() - t_start:.1f}s")
    export_results.export()


if __name__ == "__main__":
    main()
