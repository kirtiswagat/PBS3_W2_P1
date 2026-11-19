"""Collect every number Paper 1 quotes, so the manuscript is transcribed not retyped."""

import collections
import csv
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
PRIMARY = "sweep_train_c_xception_C0.1"


def head(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


head("TIER 1")
for r in csv.DictReader(open(R / "runs.csv")):
    if r["run_id"] in ("tier1", "tier1_xception") and r["embedding"] in (
        "emb_1024",
        "emb_backbone",
    ):
        print(
            f"  {r['run_id']:<18}{r['embedding']:<14}macro={r['macro_auc']} sd={r['macro_sd']} gate={r['gate2_passed']}"
        )

head("MECHANISM (all runs)")
for r in csv.DictReader(open(R / "mechanism_all.csv")):
    print(
        f"  {r['run_id']:<32}{r['trunk']:<12}C={r['C']:<5}{r['group']:<14}"
        f"n={r['n_labels']:<4}rho={r['spearman_rho']:<9}p={r['p_value']:<9}H1={r['supports_h1']}"
    )

head("POLICY TABLE - primary run, all 14 labels")
rows = [
    r for r in csv.DictReader(open(R / "sweep_cells.csv")) if r["run_id"] == PRIMARY
]
by = collections.defaultdict(dict)
for r in rows:
    by[r["pathology"]][r["policy"]] = float(r["auc_mean"])
pols = ["u_ignore", "u_ones", "u_zeros", "u_soft", "u_both"]
print(
    f"{'pathology':<28}"
    + "".join(f"{p:>10}" for p in pols)
    + f"{'best':>10}{'spread':>9}"
)
for lab, d in by.items():
    best = max(d, key=d.get)
    spread = max(d.values()) - min(d.values())
    print(
        f"{lab:<28}"
        + "".join(f"{d.get(p, float('nan')):>10.4f}" for p in pols)
        + f"{best:>10}{spread:>9.4f}"
    )

head("U_ONES RANK per pathology (1 = best)")
for lab, d in by.items():
    order = sorted(d, key=d.get, reverse=True)
    print(f"  {lab:<28} u_ones rank {order.index('u_ones') + 1}/5   best={order[0]}")

head("TOST POLICY EQUIVALENCE")
rows = list(csv.DictReader(open(R / "tost_policy_cells.csv")))
print("verdicts:", collections.Counter(r["verdict"] for r in rows))
print(
    "by contrast:",
    collections.Counter((r["contrast_type"], r["verdict"]) for r in rows),
)
ne = [r for r in rows if r["verdict"] != "equivalent"]
print(f"\nnot equivalent ({len(ne)}):")
for r in ne:
    print(
        f"  {r['run_id']:<26}{r['pathology']:<26}{r['policy_a']} vs {r['policy_b']:<10}"
        f"diff={float(r['mean_diff']):+.4f} p_bh={r['p_bh']}"
    )

head("JOINT METRICS")
for r in csv.DictReader(open(R / "joint_cells.csv")):
    print(
        f"  {r['label_set']:<16}{r['configuration']:<22}n={r['n_rows']:<7}"
        f"ham={r['hamming_loss']} exact={r['exact_match']} acc={r['mean_per_label_accuracy']} gap={r['br_gap']}"
    )

head("ASSEMBLY")
a = json.loads((R / "joint_assembly_xception.json").read_text())["results"]["assembly"]
print(f"  combos={a['n_combinations']} n={a['n_rows']}")
print(
    f"  hamming spread={a['hamming_spread']:.4f} exact spread={a['exact_match_spread']:.4f}"
)
for k in ("best_hamming", "best_exact_match", "worst_hamming"):
    print(
        f"  {k}: ham={a[k]['hamming_loss']:.4f} exact={a[k]['exact_match']:.4f} {a[k]['assignment']}"
    )

head("POWER AUDIT")
d = json.loads((R / "power_audit.json").read_text())
print(" ", json.dumps(d["summary"]))
print(" method:", d["method"])
print(" data use:", d["data_use_note"])

head("LABEL STATS from the primary sweep")
s = json.loads((R / f"{PRIMARY}.json").read_text())
ls = s.get("label_stats", {})
print(json.dumps(ls, indent=1)[:1400])
