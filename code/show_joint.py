"""Print the T23 joint-metric results as a readable table."""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
tag = sys.argv[1] if len(sys.argv) > 1 else "joint_assembly_xception"

d = json.loads((RESULTS / f"{tag}.json").read_text())
r = d["results"]

print(f"{tag}  trunk={d['trunk']}  C={d['C']}  seeds={len(d['seeds'])}\n")

for s in ("all_14", "competition_5"):
    if s not in r:
        continue
    b = r[s]
    print(f"=== {s}: {b['n_labels']} labels, n={b['n_rows']} rows ===")
    print(f"{'configuration':<24}{'hamming':>10}{'exact':>10}{'perlabel':>10}")
    for p, m in b["global"].items():
        print(
            f"{'global ' + p:<24}{m['hamming_loss']['mean']:>10.4f}"
            f"{m['exact_match']['mean']:>10.4f}{m['mean_per_label_accuracy']['mean']:>10.4f}"
        )
    pb = b.get("per_label_best")
    if pb:
        print(
            f"{'per-label best':<24}{pb['hamming_loss']['mean']:>10.4f}"
            f"{pb['exact_match']['mean']:>10.4f}{pb['mean_per_label_accuracy']['mean']:>10.4f}"
        )
        print("  assignment:")
        for lab, pol in pb["assignment"].items():
            print(f"    {lab:<30}{pol}")
    t = b.get("tier2_vs_best_global")
    if t:
        print(
            f"  vs best global ({t['best_global_policy']}): "
            f"hamming {t['hamming_delta']:+.4f}, exact-match {t['exact_match_delta']:+.4f}"
        )
    g = b.get("br_gap")
    if g:
        print(
            f"  BR GAP: per-label acc {g['mean_per_label_accuracy']:.4f} - "
            f"exact-match {g['exact_match']:.4f} = {g['gap']:.4f}"
        )
    print()

a = r.get("assembly")
if a:
    print(
        f"=== analytic assembly: {a['n_combinations']} combinations, n={a['n_rows']} ==="
    )
    print(f"  hamming spread     {a['hamming_spread']:.4f}")
    print(f"  exact-match spread {a['exact_match_spread']:.4f}")
    for k in ("best_hamming", "worst_hamming", "best_exact_match"):
        v = a[k]
        print(f"  {k:<18} hamming={v['hamming_loss']:.4f} exact={v['exact_match']:.4f}")
        print(f"    {v['assignment']}")
