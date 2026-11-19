"""Figure for the power audit: why a 202-image endpoint cannot resolve policy effects.

Colourblind-safe, 600 dpi, RGB - Diagnostics requires colour figures at >=600 dpi.
"""

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
R = BASE / "results"
OUT = R / "figures"

BLUE, ORANGE, TEAL, RED = "#1f6fb4", "#e07b1a", "#0d8b8b", "#b3312c"


def load():
    rows = []
    for c in csv.DictReader(open(R / "power_audit_cells.csv")):
        try:
            spread = float(c["observed_spread"])
            exp_hw = float(c["expert_ci95_halfwidth"])
            tc_hw = float(c["train_c_ci95_halfwidth"])
        except (TypeError, ValueError):
            continue
        # Fracture has 0 expert positives, so its SE is undefined rather than large
        if spread <= 0 or not np.isfinite(exp_hw) or not np.isfinite(tc_hw):
            continue
        rows.append(
            {
                "lab": c["pathology"],
                "n_pos": int(c["expert_n_pos"]),
                "spread": spread,
                "exp_hw": exp_hw,
                "tc_hw": tc_hw,
                "ratio": exp_hw / spread,
            }
        )
    return sorted(rows, key=lambda r: r["ratio"])


def main():
    rows = load()
    labs = [r["lab"] for r in rows]
    y = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 7.2), gridspec_kw={"width_ratios": [1.25, 1]}
    )

    # left: effect size vs the uncertainty of each endpoint
    ax1.barh(
        y + 0.26,
        [r["exp_hw"] for r in rows],
        height=0.26,
        color=RED,
        label="Expert set CI half-width (n = 202)",
    )
    ax1.barh(
        y,
        [r["tc_hw"] for r in rows],
        height=0.26,
        color=TEAL,
        label="Train-C CI half-width (n = 18,692)",
    )
    ax1.barh(
        y - 0.26,
        [r["spread"] for r in rows],
        height=0.26,
        color=BLUE,
        label="Observed policy effect (best - worst AUC)",
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(labs, fontsize=9)
    ax1.set_xscale("log")
    ax1.set_xlabel("AUC units (log scale)", fontsize=10)
    ax1.set_title(
        "(a) Measurement noise against the effect being measured",
        fontsize=11,
        fontweight="bold",
    )
    ax1.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    ax1.grid(axis="x", alpha=0.3, linestyle=":")
    ax1.set_axisbelow(True)

    # right: how many times larger the noise is than the effect
    ratios = [r["ratio"] for r in rows]
    colors = [TEAL if v < 1 else (ORANGE if v < 10 else RED) for v in ratios]
    ax2.barh(y, ratios, color=colors, height=0.62)
    ax2.axvline(1.0, color="black", linestyle="--", linewidth=1.4)
    ax2.text(
        1.15,
        len(rows) - 0.6,
        "resolvable\nthreshold",
        fontsize=8.5,
        va="top",
        color="black",
    )
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xscale("log")
    ax2.set_xlabel("Expert-set CI half-width / observed effect", fontsize=10)
    ax2.set_title(
        "(b) No pathology is resolvable at n = 202", fontsize=11, fontweight="bold"
    )
    ax2.grid(axis="x", alpha=0.3, linestyle=":")
    ax2.set_axisbelow(True)
    for i, r in enumerate(rows):
        ax2.text(
            r["ratio"] * 1.12,
            i,
            f"{r['ratio']:.0f}x  (n+ = {r['n_pos']})",
            va="center",
            fontsize=8,
        )
    ax2.set_xlim(right=max(ratios) * 6)

    fig.suptitle(
        "Why a 202-image validation endpoint cannot support per-pathology policy selection",
        fontsize=12.5,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(OUT, exist_ok=True)
    path = OUT / "F0_power_audit.png"
    fig.savefig(path, dpi=600, facecolor="white")
    plt.close(fig)
    print(
        f"Wrote {path}  ({len(rows)} pathologies, median ratio "
        f"{np.median(ratios):.1f}x)"
    )


if __name__ == "__main__":
    main()
