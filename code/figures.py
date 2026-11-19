"""Publication figures (F1-F5). Reads only the exported CSVs in results/ - never
the source JSONs - so a figure can never show a number the CSVs don't also show.

Style: 300 dpi, serif (matches the journal), Matplotlib's 'tab10' palette which
is colourblind-distinguishable for up to 10 categories (the 5 policies and a
handful of methods both fit comfortably under that).

F4 needs conformal/coverage numbers that are not yet exported to CSV anywhere
(Tasks 11-13/17 only write JSON) - it is skipped with a clear warning rather
than silently reading nothing, until that exporter exists.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.family": "serif", "figure.dpi": 300, "savefig.dpi": 300})
PALETTE = plt.get_cmap("tab10").colors

# The primary converged runs. sweep_cells.csv also holds the n=202 expert sweep,
# whose spreads are noise artefacts (invariant 7) and must never reach a figure.
PRIMARY_RUNS = ["sweep_train_c_C0.1", "sweep_train_c_xception_C0.1"]
PRIMARY_TIER1 = "tier1_C0.1"


def _primary(df, runs=PRIMARY_RUNS):
    out = df[df["run_id"].isin(runs)]
    if out.empty:
        print(f"figures: WARNING - none of {runs} present, falling back to all runs")
        return df
    return out


def _out_dir(results_dir):
    d = os.path.join(results_dir, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def _read(results_dir, name):
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        print(
            f"figures: WARNING - {name} not found in {results_dir}, skipping figures that need it"
        )
        return None
    return pd.read_csv(path)


def fig1_mechanism(results_dir):
    """Uncertainty fraction vs policy spread, all labels, both trunks."""
    df = _read(results_dir, "sweep_cells.csv")
    if df is None:
        return
    df = _primary(df)
    spread = df.drop_duplicates(["trunk", "pathology"])[
        ["trunk", "pathology", "unc_fraction", "spread"]
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (trunk, g) in enumerate(spread.groupby("trunk")):
        ax.scatter(g["unc_fraction"], g["spread"], label=trunk, color=PALETTE[i], s=40)
    ax.set_xlabel("Uncertainty fraction")
    ax.set_ylabel("Policy spread (max AUC - min AUC)")
    ax.set_title("F1: uncertainty burden predicts policy sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(_out_dir(results_dir), "F1_mechanism.png"))
    plt.close(fig)


def fig2_heatmap(results_dir, highlight_policy="u_ones"):
    """5 policies x 14 pathologies main-effects heatmap, one per trunk."""
    df = _read(results_dir, "sweep_cells.csv")
    if df is None:
        return
    df = _primary(df)
    for trunk, g in df.groupby("trunk"):
        pivot = g.pivot_table(index="pathology", columns="policy", values="auc_mean")
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        if highlight_policy in pivot.columns:
            j = list(pivot.columns).index(highlight_policy)
            for i in range(len(pivot.index)):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2
                    )
                )
        fig.colorbar(im, ax=ax, label="AUC")
        ax.set_title(f"F2: main effects ({trunk}) - {highlight_policy} outlined")
        fig.tight_layout()
        fig.savefig(os.path.join(_out_dir(results_dir), f"F2_heatmap_{trunk}.png"))
        plt.close(fig)


def fig2b_centred(results_dir, highlight_policy="u_ones"):
    """Row-centred companion to F2: AUC minus per-pathology mean.

    F2's absolute scale is dominated by between-disease AUC range, which hides
    the within-disease policy effect entirely. Centring each row exposes where
    a policy sits relative to its alternatives - the only view in which
    u_ones being worst for Atelectasis and Consolidation is legible.
    """
    df = _read(results_dir, "sweep_cells.csv")
    if df is None:
        return
    df = _primary(df)
    for trunk, g in df.groupby("trunk"):
        pivot = g.pivot_table(index="pathology", columns="policy", values="auc_mean")
        centred = pivot.sub(pivot.mean(axis=1), axis=0) * 100  # AUC points
        order = centred.abs().max(axis=1).sort_values(ascending=False).index
        centred = centred.loc[order]

        lim = float(centred.abs().to_numpy().max())
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(
            centred.values, cmap="RdBu_r", aspect="auto", vmin=-lim, vmax=lim
        )
        ax.set_xticks(range(len(centred.columns)))
        ax.set_xticklabels(centred.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(centred.index)))
        ax.set_yticklabels(centred.index)
        for i in range(len(centred.index)):
            for j in range(len(centred.columns)):
                ax.text(
                    j,
                    i,
                    f"{centred.values[i, j]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                )
        if highlight_policy in centred.columns:
            j = list(centred.columns).index(highlight_policy)
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, -0.5),
                    1,
                    len(centred.index),
                    fill=False,
                    edgecolor="black",
                    lw=2.5,
                )
            )
        fig.colorbar(im, ax=ax, label="AUC points vs pathology mean")
        ax.set_title(f"F2b: within-pathology policy effect ({trunk})")
        fig.tight_layout()
        fig.savefig(os.path.join(_out_dir(results_dir), f"F2b_centred_{trunk}.png"))
        plt.close(fig)


def fig3_equivalence(results_dir, delta=0.02):
    """AUC vs subset size, with a +/- delta TOST equivalence band around the full-feature baseline."""
    df = _read(results_dir, "selection_cells.csv")
    if df is None:
        return
    for (trunk, lab), g in df.groupby(["trunk", "pathology"]):
        fig, ax = plt.subplots(figsize=(6, 5))
        baseline = g["baseline_full_auc_mean"].dropna()
        baseline = float(baseline.iloc[0]) if len(baseline) else None
        for i, (method, gm) in enumerate(g.groupby("method")):
            gm = gm.sort_values("n_features")
            ax.plot(
                gm["n_features"],
                gm["auc_mean"],
                marker="o",
                label=method,
                color=PALETTE[i % 10],
            )
        if baseline is not None:
            ax.axhline(
                baseline, color="black", linestyle="--", label="full-feature baseline"
            )
            ax.axhspan(
                baseline - delta,
                baseline + delta,
                color="grey",
                alpha=0.2,
                label=f"TOST band (Δ={delta})",
            )
        ax.set_xlabel("Number of features")
        ax.set_ylabel("AUC")
        ax.set_title(f"F3: {lab} ({trunk})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe_lab = lab.replace(" ", "_")
        fig.savefig(
            os.path.join(
                _out_dir(results_dir), f"F3_equivalence_{trunk}_{safe_lab}.png"
            )
        )
        plt.close(fig)


def fig4_conformal_coverage(results_dir):
    """Internal vs external conformal coverage, one panel per trunk."""
    df = _read(results_dir, "coverage_shift_cells.csv")
    if df is None:
        print("figures: F4 skipped - run export_coverage.py first")
        return
    if "trunk" not in df.columns:
        df = df.assign(trunk="densenet121")
    for trunk, g in df.groupby("trunk"):
        g = g.reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        x = range(len(g))
        ax.bar(
            x,
            g["internal_coverage"],
            width=0.4,
            label="internal (Train-C)",
            color=PALETTE[0],
        )
        ax.bar(
            [i + 0.4 for i in x],
            g["external_coverage"],
            width=0.4,
            label="external (NIH)",
            color=PALETTE[1],
        )
        ax.axhline(
            1 - g["alpha"].iloc[0],
            color="black",
            linestyle="--",
            label="nominal 1-alpha",
        )
        ax.set_xticks([i + 0.2 for i in x])
        ax.set_xticklabels(g["pathology"], rotation=45, ha="right")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(f"F4: conformal coverage under shift ({trunk})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(_out_dir(results_dir), f"F4_coverage_shift_{trunk}.png")
        )
        plt.close(fig)


def fig5_forest(results_dir):
    """Per-class forest plot with bootstrap CIs."""
    df = _read(results_dir, "per_class.csv")
    if df is None:
        return
    df = _primary(df, [PRIMARY_TIER1])
    if "embedding" in df.columns:
        df = df[df["embedding"].isin(["emb_1024", "emb_backbone"])]
    df = df.dropna(subset=["ci95_lo", "ci95_hi"])
    fig, ax = plt.subplots(figsize=(6, max(4, 0.3 * len(df))))
    y = range(len(df))
    ax.errorbar(
        df["auc_mean"],
        y,
        xerr=[df["auc_mean"] - df["ci95_lo"], df["ci95_hi"] - df["auc_mean"]],
        fmt="o",
        color=PALETTE[0],
        ecolor="grey",
        capsize=3,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["pathology"])
    ax.set_xlabel("AUC (95% bootstrap CI)")
    ax.set_title("F5: per-class forest plot")
    fig.tight_layout()
    fig.savefig(os.path.join(_out_dir(results_dir), "F5_forest.png"))
    plt.close(fig)


def make_all(results_dir):
    fig1_mechanism(results_dir)
    fig2_heatmap(results_dir)
    fig2b_centred(results_dir)
    fig3_equivalence(results_dir)
    fig4_conformal_coverage(results_dir)
    fig5_forest(results_dir)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir", default=str(Path(__file__).resolve().parent.parent / "results")
    )
    args = ap.parse_args()
    make_all(args.results_dir)
