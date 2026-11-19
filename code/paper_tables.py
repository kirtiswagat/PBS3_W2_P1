"""LaTeX (booktabs) tables for the paper, pivoted from the exported CSVs.

Effect sizes and CIs go in the main columns; p-values are secondary (last
column, small print in the caption) - per invariant 8, a p-value alone is not
the reportable quantity here.
"""

import os
from pathlib import Path

import pandas as pd

RESULTS_DIR = str(Path(__file__).resolve().parent.parent / "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")


def _write(name, latex):
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, name)
    with open(path, "w") as f:
        f.write(latex)
    print(f"  {name}")


def table_main_effects(results_dir=RESULTS_DIR):
    """Pivot of sweep_cells.csv: pathology x policy AUC, effect size = spread."""
    path = os.path.join(results_dir, "sweep_cells.csv")
    if not os.path.exists(path):
        print("paper_tables: sweep_cells.csv not found, skipping main-effects table")
        return
    df = pd.read_csv(path)
    for trunk, g in df.groupby("trunk"):
        pivot = g.pivot_table(index="pathology", columns="policy", values="auc_mean")
        spread = g.drop_duplicates("pathology").set_index("pathology")["spread"]
        pivot["spread"] = spread
        latex = pivot.round(4).to_latex(
            escape=True, bold_rows=False, caption=f"Main-effects AUC by policy ({trunk}).",
            label=f"tab:main_effects_{trunk}",
        )
        _write(f"main_effects_{trunk}.tex", latex)


def table_mechanism(results_dir=RESULTS_DIR):
    """mechanism_all.csv: one row per (run, correlation group) - the replication table."""
    path = os.path.join(results_dir, "mechanism_all.csv")
    if not os.path.exists(path):
        print("paper_tables: mechanism_all.csv not found, skipping mechanism table")
        return
    df = pd.read_csv(path)
    cols = ["run_id", "trunk", "C", "group", "n_labels", "spearman_rho", "p_value"]
    latex = df[cols].round(4).to_latex(
        index=False, escape=True, caption="Mechanism test (H1): uncertainty fraction vs policy spread.",
        label="tab:mechanism",
    )
    _write("mechanism.tex", latex)


def table_selection(results_dir=RESULTS_DIR):
    """selection_cells.csv: AUC by (pathology, fraction, method), with TOST verdicts as the effect-size column."""
    path = os.path.join(results_dir, "selection_cells.csv")
    if not os.path.exists(path):
        print("paper_tables: selection_cells.csv not found, skipping selection table")
        return
    df = pd.read_csv(path)
    cols = [
        "pathology", "fraction", "method", "n_features",
        "auc_mean", "auc_sd", "baseline_full_auc_mean", "tost_verdict", "tost_diff", "tost_p",
    ]
    latex = df[cols].round(4).to_latex(
        index=False, escape=True, caption="Feature selection (H2/H3): AUC by method and subset size.",
        label="tab:selection",
    )
    _write("selection.tex", latex)


def make_all():
    print(f"Writing LaTeX tables to {TABLES_DIR}/")
    table_main_effects()
    table_mechanism()
    table_selection()


if __name__ == "__main__":
    make_all()
