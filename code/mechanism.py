"""Sl 26 / T22 - the mechanism test.

H1 states that the optimal uncertainty policy is disease-specific and driven by
how much uncertainty a disease actually carries. Operationally:

    predictor : unc_fraction = uncertain labels / total training rows
                i.e. the fraction of rows whose label CHANGES when the policy
                flips. If almost no rows change, no policy can matter much.
    outcome   : spread = max(AUC across policies) - min(AUC across policies)
                i.e. how much the policy choice is worth for that disease.

H1 predicts a positive monotone relationship between the two.

CONTAMINATION CONTROL
    The frozen trunk was fine-tuned on five labels under one fixed uncertainty
    policy, so those five may favour it. The other nine were never in its
    objective. If the relationship holds on both groups, the trunk is not
    manufacturing the result. This is the control that converts the frozen-trunk
    design from a limitation into evidence.

Underpowered labels are excluded upstream by sweep.py --min-pos; on the 202-row
expert set, Pleural Other had one positive and produced the largest spread in
the table purely by chance.

Usage:
    python mechanism.py                      # uses results/sweep_train_c.json
    python mechanism.py --tag sweep_expert
"""

import argparse
import csv
import json
import os

import numpy as np
from scipy import stats

RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
BASELINE_POLICY = "u_ignore"  # the "do nothing with uncertain labels" reference


def section(t):
    print(f"\n{'=' * 82}\n{t}\n{'=' * 82}")


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg. Returns (adjusted p, rejected).

    Delegates to multiplicity.bh_fdr but keeps this module's argument order and
    its inclusive alpha comparison, so previously reported mechanism results are
    reproduced exactly.
    """
    from multiplicity import bh_fdr as _bh

    _, adj = _bh(pvals, alpha=alpha)
    return adj, adj <= alpha


def load(tag):
    with open(f"{RESULTS}/{tag}.json") as f:
        return json.load(f)


def build_rows(data):
    rows = []
    for lab, r in data["results"].items():
        policies = [p for p in data["policies"] if r.get(p)]
        if not policies:
            continue
        aucs = {p: r[p]["auc_mean"] for p in policies}
        st = data["label_stats"][lab]
        if st["uncertain"] == 0:
            continue  # no uncertain labels means no policy can do anything
        rows.append(
            {
                "label": lab,
                "trunk_seen": lab in data["trunk_seen"],
                "unc_fraction": st["unc_fraction"],
                "unc_over_mentioned": st["unc_over_mentioned"],
                "uncertain": st["uncertain"],
                "eval_n_pos": st.get("eval_n_pos"),
                "best": r["_best"],
                "spread": r["_spread"],
                **{f"auc_{p}": aucs[p] for p in policies},
                "_per_seed": {p: r[p]["auc_per_seed"] for p in policies},
            }
        )
    return sorted(rows, key=lambda x: -x["unc_fraction"])


def correlate(rows, label):
    if len(rows) < 3:
        print(f"{label:<26} too few labels ({len(rows)}) to correlate")
        return None
    x = [r["unc_fraction"] for r in rows]
    y = [r["spread"] for r in rows]
    rho, p = stats.spearmanr(x, y)
    print(
        f"{label:<26} n={len(rows):<3} Spearman rho={rho:+.3f}  p={p:.4f}"
        f"   {'SUPPORTS H1' if (rho > 0 and p < 0.05) else 'not significant'}"
    )
    return {"n": len(rows), "rho": float(rho), "p": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sweep_train_c")
    args = ap.parse_args()

    data = load(args.tag)
    rows = build_rows(data)

    print("MECHANISM TEST - does uncertainty burden predict policy sensitivity?")
    print(f"Source: {args.tag}.json   eval on: {data.get('eval_on', 'expert')}")

    section("PER-LABEL RESULTS")
    pol = data["policies"]
    head = f"{'Label':<28}{'unc frac':>10}{'n_pos':>8}" + "".join(
        f"{p:>10}" for p in pol
    )
    print(head + f"{'best':>10}{'spread':>9}{'seen':>6}")
    print("-" * len(head + " " * 25))
    for r in rows:
        line = f"{r['label']:<28}{r['unc_fraction']:>10.4f}{r['eval_n_pos'] or 0:>8,}"
        for p in pol:
            v = r.get(f"auc_{p}")
            line += f"{v:>10.4f}" if v is not None else f"{'-':>10}"
        line += (
            f"{r['best']:>10}{r['spread']:>9.4f}{'yes' if r['trunk_seen'] else 'no':>6}"
        )
        print(line)

    section("H1 - CORRELATION BETWEEN UNCERTAINTY BURDEN AND POLICY SENSITIVITY")
    overall = correlate(rows, "All labels")
    seen = correlate([r for r in rows if r["trunk_seen"]], "Trunk-seen (5 labels)")
    unseen = correlate(
        [r for r in rows if not r["trunk_seen"]], "Trunk-unseen (control)"
    )

    if seen and unseen:
        print("\nContamination control: if the unseen group behaves like the seen")
        print("group, the frozen trunk is not driving the relationship.")

    section("PER-LABEL SIGNIFICANCE - best policy vs u_ignore baseline")
    tests = []
    for r in rows:
        ps = r["_per_seed"]
        if r["best"] == BASELINE_POLICY or BASELINE_POLICY not in ps:
            continue
        a, b = np.array(ps[r["best"]]), np.array(ps[BASELINE_POLICY])
        if len(a) < 2 or np.allclose(a, b):
            continue
        t, p = stats.ttest_rel(a, b)
        tests.append(
            {
                "label": r["label"],
                "best": r["best"],
                "delta": float(a.mean() - b.mean()),
                "p": float(p),
            }
        )

    if tests:
        adj, rej = bh_fdr([t["p"] for t in tests])
        print(
            f"{'Label':<28}{'best policy':>12}{'delta AUC':>12}{'p':>10}{'BH p':>10}{'sig':>6}"
        )
        print("-" * 78)
        for t, a, r_ in zip(tests, adj, rej):
            t["p_bh"] = float(a)
            t["significant"] = bool(r_)
            print(
                f"{t['label']:<28}{t['best']:>12}{t['delta']:>+12.4f}"
                f"{t['p']:>10.4f}{a:>10.4f}{'yes' if r_ else 'no':>6}"
            )
        n_sig = int(sum(rej))
        print(
            f"\n{n_sig}/{len(tests)} labels show a significant gain over "
            f"{BASELINE_POLICY} after BH-FDR correction."
        )

    section("VERDICT")
    if overall and overall["rho"] > 0 and overall["p"] < 0.05:
        print("H1 SUPPORTED - uncertainty burden predicts policy sensitivity.")
    elif overall:
        print("H1 NOT SUPPORTED on this evidence.")
        print(f"  Spearman rho = {overall['rho']:+.3f}, p = {overall['p']:.4f}")
        print("\n  This is a publishable null if reported honestly. The pre-registered")
        print("  fallback is H2, the efficiency hypothesis, which is unaffected.")
        print("  Before concluding, check: (a) is the spread dominated by one or two")
        print("  labels, (b) does unc_over_mentioned work better as the predictor,")
        print("  (c) are any labels still underpowered.")

    out = f"{RESULTS}/mechanism_{args.tag}.csv"
    cols = [c for c in rows[0] if not c.startswith("_")]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print(f"\nWritten {out}")

    with open(f"{RESULTS}/mechanism_{args.tag}.json", "w") as f:
        json.dump(
            {
                "correlations": {
                    "all": overall,
                    "trunk_seen": seen,
                    "trunk_unseen": unseen,
                },
                "per_label_tests": tests,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
