"""Regenerate every figure in the manuscript from hard-coded values.

This script is deliberately self-contained: it reads nothing from results/ and
needs only matplotlib and numpy. Every number below was transcribed from the
analysis outputs at the time the manuscript was prepared, so the figures it
produces are byte-for-byte equivalent in content to the published versions.

Output names follow the manuscript numbering, not the internal analysis names:

    Figure_1_precision_audit.png     precision of the 202-image endpoint
    Figure_2_mechanism.png           uncertainty burden vs policy sensitivity
    Figure_3_policy_effects.png      mean-centred policy effects
    Figure_4_policy_surface.png      policy x pathology AUC surface
    Figure_5_forest.png              per-pathology AUC with bootstrap CIs

Usage:
    python generate_manuscript_figures.py                 # all five, 600 dpi
    python generate_manuscript_figures.py --figure 1 3    # a subset
    python generate_manuscript_figures.py --dpi 300 --outdir ./fig --format tiff
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

# --------------------------------------------------------------------------
# Palette - colourblind-safe, high contrast in both RGB and greyscale print
# --------------------------------------------------------------------------
BLUE = "#1f6fb4"
ORANGE = "#e07b1a"
TEAL = "#0a7d72"
RED = "#b3312c"
VIOLET = "#6a4c9c"
GREY = "#5a6a78"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "axes.edgecolor": "#39444d",
        "axes.labelcolor": "#17212b",
        "text.color": "#17212b",
        "xtick.color": "#39444d",
        "ytick.color": "#39444d",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

POLICIES = ["u_ignore", "u_ones", "u_zeros", "u_soft", "u_both"]

# --------------------------------------------------------------------------
# Figure 1 data - precision audit of the 202-image validation endpoint.
# spread = observed best-minus-worst AUC; hw = 95% CI half-width.
# Fracture has zero positive cases, so its expert-set SE is undefined.
# --------------------------------------------------------------------------
POWER_AUDIT = [
    # (pathology, expert positives, policy spread, expert CI hw, Train-C CI hw)
    ("Lung Opacity", 117, 0.0004, 0.0710, 0.0075),
    ("Enlarged Cardiomediastinum", 105, 0.0119, 0.0763, 0.0198),
    ("Support Devices", 99, 0.0002, 0.0644, 0.0065),
    ("Atelectasis", 75, 0.0077, 0.0779, 0.0116),
    ("Cardiomegaly", 66, 0.0013, 0.0630, 0.0104),
    ("Pleural Effusion", 64, 0.0008, 0.0615, 0.0058),
    ("Edema", 42, 0.0026, 0.0790, 0.0076),
    ("Consolidation", 32, 0.0317, 0.1048, 0.0160),
    ("Pneumonia", 8, 0.0229, 0.2016, 0.0259),
    ("Pneumothorax", 7, 0.0008, 0.2081, 0.0129),
    ("Lung Lesion", 1, 0.0015, 0.5648, 0.0210),
    ("Pleural Other", 1, 0.0068, 0.5473, 0.0347),
    ("Fracture", 0, 0.0017, None, 0.0215),  # undefined, excluded from panel (b)
]

# --------------------------------------------------------------------------
# Figure 2 data - uncertainty fraction vs policy spread, per frozen backbone.
# trunk_seen marks the five findings the backbones were trained on.
# --------------------------------------------------------------------------
MECHANISM = {
    "Xception": [
        ("Atelectasis", 0.156493, 0.007662, True),
        ("Consolidation", 0.128072, 0.031740, True),
        ("Pneumonia", 0.084266, 0.022913, False),
        ("Edema", 0.061980, 0.002614, True),
        ("Enlarged Cardiomediastinum", 0.053596, 0.011881, False),
        ("Pleural Effusion", 0.050099, 0.000819, True),
        ("Cardiomegaly", 0.035585, 0.001290, True),
        ("Lung Opacity", 0.023037, 0.000366, False),
        ("Pneumothorax", 0.014123, 0.000817, False),
        ("Pleural Other", 0.009385, 0.006836, False),
        ("Lung Lesion", 0.005888, 0.001469, False),
        ("Support Devices", 0.004627, 0.000218, False),
        ("Fracture", 0.002614, 0.001747, False),
    ],
    "DenseNet121": [
        ("Atelectasis", 0.156493, 0.011920, True),
        ("Consolidation", 0.128072, 0.024720, True),
        ("Pneumonia", 0.084266, 0.029626, False),
        ("Edema", 0.061980, 0.002045, True),
        ("Enlarged Cardiomediastinum", 0.053596, 0.015232, False),
        ("Pleural Effusion", 0.050099, 0.000803, True),
        ("Cardiomegaly", 0.035585, 0.001112, True),
        ("Lung Opacity", 0.023037, 0.000357, False),
        ("Pneumothorax", 0.014123, 0.000853, False),
        ("Pleural Other", 0.009385, 0.010340, False),
        ("Lung Lesion", 0.005888, 0.001973, False),
        ("Support Devices", 0.004627, 0.000186, False),
        ("Fracture", 0.002614, 0.002772, False),
    ],
}
MECHANISM_RHO = {"Xception": (0.643, 0.018), "DenseNet121": (0.571, 0.041)}

# --------------------------------------------------------------------------
# Figures 3 and 4 data - mean AUC per policy and pathology, primary run
# (Xception, C = 0.1, five patient-resampling seeds, evaluation partition).
# --------------------------------------------------------------------------
POLICY_SURFACE = {
    "No Finding": [0.841256, 0.841256, 0.841256, 0.841256, 0.841256],
    "Enlarged Cardiomediastinum": [0.630247, 0.618366, 0.628809, 0.629148, 0.629448],
    "Cardiomegaly": [0.846310, 0.845352, 0.846140, 0.846413, 0.846641],
    "Lung Opacity": [0.701614, 0.701263, 0.701422, 0.701449, 0.701629],
    "Lung Lesion": [0.734602, 0.733817, 0.734429, 0.735286, 0.734913],
    "Edema": [0.834623, 0.832008, 0.833582, 0.833655, 0.834173],
    "Consolidation": [0.729732, 0.697992, 0.729066, 0.716435, 0.720501],
    "Pneumonia": [0.718011, 0.731547, 0.711320, 0.734232, 0.734096],
    "Atelectasis": [0.686765, 0.679104, 0.682391, 0.684384, 0.686068],
    "Pneumothorax": [0.768152, 0.767514, 0.767799, 0.768331, 0.768234],
    "Pleural Effusion": [0.859609, 0.859663, 0.859103, 0.859879, 0.859922],
    "Pleural Other": [0.765792, 0.760680, 0.765425, 0.766251, 0.767516],
    "Fracture": [0.691009, 0.692618, 0.691005, 0.692756, 0.692312],
    "Support Devices": [0.776930, 0.776885, 0.776724, 0.776984, 0.776895],
}

# --------------------------------------------------------------------------
# Figure 5 data - per-pathology AUC with 95% bootstrap intervals
# (Xception baseline heads on the evaluation partition).
# --------------------------------------------------------------------------
FOREST = [
    ("Atelectasis", 0.774635, 0.708265, 0.837804),
    ("Cardiomegaly", 0.792135, 0.723312, 0.858339),
    ("Consolidation", 0.848640, 0.795763, 0.910130),
    ("Edema", 0.920923, 0.877138, 0.956125),
    ("Pleural Effusion", 0.872305, 0.822004, 0.922807),
]


def _save(fig, outdir, name, dpi, fmt):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.{fmt}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# --------------------------------------------------------------------------
def figure_1(outdir, dpi, fmt):
    """Precision of the historical 202-image endpoint against the effect size."""
    rows = sorted(
        [r for r in POWER_AUDIT if r[3] is not None and r[2] > 0],
        key=lambda r: r[3] / r[2],
    )
    labs = [r[0] for r in rows]
    y = np.arange(len(rows))
    ratios = [r[3] / r[2] for r in rows]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 7.0), gridspec_kw={"width_ratios": [1.25, 1]}
    )

    ax1.barh(
        y + 0.26,
        [r[3] for r in rows],
        height=0.26,
        color=RED,
        label="Expert-set 95% CI half-width (n = 202)",
    )
    ax1.barh(
        y,
        [r[4] for r in rows],
        height=0.26,
        color=TEAL,
        label="Evaluation-partition CI half-width (n = 18,692)",
    )
    ax1.barh(
        y - 0.26,
        [r[2] for r in rows],
        height=0.26,
        color=BLUE,
        label="Observed policy effect (best − worst AUC)",
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(labs, fontsize=9)
    ax1.set_xscale("log")
    ax1.set_xlabel("AUC units (logarithmic scale)", fontsize=10)
    ax1.set_title(
        "(a) Measurement noise against the effect being measured",
        fontsize=11,
        fontweight="bold",
    )
    ax1.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    ax1.grid(axis="x", alpha=0.3, linestyle=":")
    ax1.set_axisbelow(True)

    colors = [TEAL if v < 1 else (ORANGE if v < 10 else RED) for v in ratios]
    ax2.barh(y, ratios, color=colors, height=0.62)
    ax2.axvline(1.0, color="black", linestyle="--", linewidth=1.4)
    ax2.text(1.18, len(rows) - 0.55, "resolvable\nthreshold", fontsize=8.5, va="top")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xscale("log")
    ax2.set_xlabel("Endpoint CI half-width ÷ observed effect", fontsize=10)
    ax2.set_title(
        "(b) No pathology is resolvable at n = 202", fontsize=11, fontweight="bold"
    )
    ax2.grid(axis="x", alpha=0.3, linestyle=":")
    ax2.set_axisbelow(True)
    for i, (r, v) in enumerate(zip(rows, ratios)):
        ax2.text(v * 1.12, i, f"{v:.0f}×  (n+ = {r[1]})", va="center", fontsize=8)
    ax2.set_xlim(right=max(ratios) * 6)

    fig.suptitle(
        "Why a 202-image validation endpoint cannot support per-pathology policy selection",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return _save(fig, outdir, "Figure_1_precision_audit", dpi, fmt)


def figure_2(outdir, dpi, fmt):
    """Uncertainty burden against policy sensitivity, both backbones."""
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    colors = {"Xception": BLUE, "DenseNet121": ORANGE}

    for trunk, pts in MECHANISM.items():
        x = [p[1] for p in pts]
        yv = [p[2] for p in pts]
        seen = [p[3] for p in pts]
        rho, pval = MECHANISM_RHO[trunk]
        ax.scatter(
            [a for a, s in zip(x, seen) if s],
            [b for b, s in zip(yv, seen) if s],
            s=95,
            color=colors[trunk],
            marker="o",
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
            label=f"{trunk} — backbone-trained  (ρ = {rho}, p = {pval})",
        )
        ax.scatter(
            [a for a, s in zip(x, seen) if not s],
            [b for b, s in zip(yv, seen) if not s],
            s=95,
            facecolor="none",
            edgecolor=colors[trunk],
            marker="s",
            linewidth=1.8,
            zorder=3,
            label=f"{trunk} — contamination control",
        )
        z = np.polyfit(x, yv, 1)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(
            xs,
            np.poly1d(z)(xs),
            color=colors[trunk],
            linewidth=1.6,
            alpha=0.55,
            linestyle="--",
            zorder=2,
        )

    for lab, ux, sp, _ in MECHANISM["Xception"]:
        if sp > 0.010 or ux > 0.12:
            ax.annotate(
                lab,
                (ux, sp),
                textcoords="offset points",
                xytext=(9, 5),
                fontsize=8.5,
                color=GREY,
            )

    ax.set_xlabel("Uncertain fraction of training mentions", fontsize=10.5)
    ax.set_ylabel("Policy sensitivity  (best − worst AUC)", fontsize=10.5)
    ax.set_title(
        "Uncertainty burden predicts sensitivity to the uncertainty-handling rule",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.28, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.6, framealpha=0.95, loc="upper left")
    fig.tight_layout()
    return _save(fig, outdir, "Figure_2_mechanism", dpi, fmt)


def _surface_matrix():
    labs = list(POLICY_SURFACE)
    return labs, np.array([POLICY_SURFACE[l] for l in labs])


def figure_3(outdir, dpi, fmt):
    """Policy effects after removing each pathology's mean AUC."""
    labs, mat = _surface_matrix()
    centred = (mat - mat.mean(axis=1, keepdims=True)) * 100  # AUC points

    lim = float(np.abs(centred).max())
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    im = ax.imshow(
        centred,
        cmap="RdBu",
        aspect="auto",
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
    )

    ax.set_xticks(range(len(POLICIES)))
    ax.set_xticklabels(POLICIES, fontsize=9.5, rotation=20, ha="right")
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=9)

    for i in range(len(labs)):
        for j in range(len(POLICIES)):
            v = centred[i, j]
            ax.text(
                j,
                i,
                f"{v:+.2f}",
                ha="center",
                va="center",
                fontsize=7.4,
                color="white" if abs(v) > lim * 0.55 else "#17212b",
            )

    worst = int(np.argmin(centred[labs.index("Atelectasis")]))
    ax.add_patch(
        Rectangle(
            (worst - 0.5, labs.index("Atelectasis") - 0.5),
            1,
            1,
            fill=False,
            edgecolor=VIOLET,
            linewidth=2.6,
        )
    )

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("AUC points relative to the pathology mean", fontsize=9.5)
    ax.set_title(
        "Within-pathology policy effects (primary run)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    return _save(fig, outdir, "Figure_3_policy_effects", dpi, fmt)


def figure_4(outdir, dpi, fmt):
    """Absolute AUC surface across policies and pathologies."""
    labs, mat = _surface_matrix()
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(POLICIES)))
    ax.set_xticklabels(POLICIES, fontsize=9.5, rotation=20, ha="right")
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=9)

    for i in range(len(labs)):
        best = int(np.argmax(mat[i]))
        for j in range(len(POLICIES)):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if mat[i, j] < mat.mean() else "#101418",
            )
        ax.add_patch(
            Rectangle(
                (best - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#ffd43b",
                linewidth=2.0,
            )
        )

    ones = POLICIES.index("u_ones")
    ax.add_patch(
        Rectangle(
            (ones - 0.5, -0.5),
            1,
            len(labs),
            fill=False,
            edgecolor=RED,
            linewidth=2.6,
            linestyle="--",
        )
    )

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Mean AUC across five seeds", fontsize=9.5)
    ax.set_title(
        "Policy × pathology AUC surface\n"
        "gold = best policy per row; dashed red = positive-assignment rule",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    return _save(fig, outdir, "Figure_4_policy_surface", dpi, fmt)


def figure_5(outdir, dpi, fmt):
    """Per-pathology AUC with bootstrap intervals."""
    rows = sorted(FOREST, key=lambda r: r[1])
    labs = [r[0] for r in rows]
    mean = np.array([r[1] for r in rows])
    lo = np.array([r[2] for r in rows])
    hi = np.array([r[3] for r in rows])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    cmap = plt.get_cmap("plasma")
    colors = [cmap(0.15 + 0.6 * i / max(len(rows) - 1, 1)) for i in range(len(rows))]

    for i in range(len(rows)):
        ax.plot(
            [lo[i], hi[i]],
            [i, i],
            color=colors[i],
            linewidth=3.0,
            solid_capstyle="round",
            alpha=0.85,
            zorder=2,
        )
        ax.plot(
            mean[i],
            i,
            "o",
            markersize=9,
            color=colors[i],
            markeredgecolor="white",
            markeredgewidth=1.3,
            zorder=3,
        )
        ax.text(
            hi[i] + 0.006,
            i,
            f"{mean[i]:.3f}  [{lo[i]:.3f}, {hi[i]:.3f}]",
            va="center",
            fontsize=8.4,
            color=GREY,
        )

    ax.axvline(0.5, color=GREY, linestyle=":", linewidth=1.2)
    ax.text(0.503, len(rows) - 0.4, "chance", fontsize=8, color=GREY)

    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=10)
    ax.set_xlabel("AUC with 95% bootstrap confidence interval", fontsize=10.5)
    ax.set_xlim(0.48, 1.06)
    ax.set_title(
        "Per-pathology discrimination on the evaluation partition",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.28, linestyle=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, outdir, "Figure_5_forest", dpi, fmt)


FIGURES = {1: figure_1, 2: figure_2, 3: figure_3, 4: figure_4, 5: figure_5}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--outdir", default="manuscript_figures")
    ap.add_argument(
        "--dpi", type=int, default=600, help="Diagnostics asks for at least 600 dpi"
    )
    ap.add_argument("--format", default="png", choices=["png", "tiff", "jpg", "pdf"])
    ap.add_argument(
        "--figure",
        type=int,
        nargs="*",
        choices=sorted(FIGURES),
        help="subset to regenerate; default is all",
    )
    args = ap.parse_args()

    wanted = args.figure or sorted(FIGURES)
    print(f"Generating {len(wanted)} figure(s) at {args.dpi} dpi -> {args.outdir}/")
    for n in wanted:
        FIGURES[n](args.outdir, args.dpi, args.format)
    print("done.")


if __name__ == "__main__":
    main()
