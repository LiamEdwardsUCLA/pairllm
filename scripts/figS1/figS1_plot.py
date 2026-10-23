"""
NOTE — data requirements
    This script regenerates the published figure from the FULL per-run result
    tree (one metrics.json per training run), which is NOT included in this
    repository. What ships here under data/ is the aggregated, seed-level CSV
    behind the panel — data/figS1_gemma4/scaling_results.csv and
    zero_shot_baseline.csv, which contain every number plotted below.

    To re-run this script as-is, fetch the full result tree from the Zenodo
    archive (see README) and point DATA_ROOT at it.

Figure S1 (Gemma-4 reproducibility): LD50 N-sweep, pairwise vs pointwise,
against the base Gemma-4 zero-shot floor. Same style convention (rcParams,
palette, line styles) as fig2_plot.py's panel A.

Data sources:
  - data/figS1_gemma4/ld50/{n}/seed*/{method}.json
      (test set — same held-out test.csv used everywhere else in this
      project, so directly comparable to Figure 2's results)
  - data/figS1_gemma4/ld50/zeroshot/{method}/temp0/metrics.json
      (base Gemma-4, no fine-tuning, same test set)

Usage:
    python scripts/figS1_plot.py
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style (matches fig2_plot.py)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

C = {"pairwise": "#C44E52", "pointwise": "#4C72B0"}
LS = {"pairwise": "-", "pointwise": "--"}

DATA_ROOT = Path("data/figS1_gemma4/ld50")
N_ORDER = ["n50", "n100", "n200", "n400", "n800", "n1600", "n3200"]


def n_to_int(n_label):
    return int(n_label[1:])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_eval() -> pd.DataFrame | None:
    """{n}/seed{s}/{method}.json -> one row per (n, seed, method)."""
    rows = []
    for p in DATA_ROOT.glob("*/seed*/*.json"):
        n_label, seed_dir = p.parts[-3], p.parts[-2]
        if n_label not in N_ORDER:
            continue
        try:
            seed = int(seed_dir.replace("seed", ""))
        except ValueError:
            continue
        method = p.stem  # "pairwise" or "pointwise"
        m = json.loads(p.read_text())
        rho = m.get("spearman_lstsq") or m.get("spearman")
        if rho is None or (isinstance(rho, float) and np.isnan(rho)):
            continue
        rows.append({"n": n_label, "seed": seed, "method": method, "spearman_lstsq": float(rho)})
    return pd.DataFrame(rows) if rows else None


def load_zeroshot() -> dict:
    """Base Gemma-4 (no fine-tuning), same test set."""
    result = {}
    for method in ["pairwise", "pointwise"]:
        p = DATA_ROOT / "zeroshot" / method / "temp0" / "metrics.json"
        if p.exists():
            d = json.loads(p.read_text())
            rho = d.get("spearman_lstsq") or d.get("spearman")
            if rho is not None:
                result[method] = float(rho)
    return result


def test_scaling_stats(test_df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        test_df.groupby(["n", "method"])["spearman_lstsq"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    stats["N"] = stats["n"].apply(n_to_int)
    return stats.sort_values("N")


# ---------------------------------------------------------------------------
# Panel A
# ---------------------------------------------------------------------------

def plot_scaling(ax, test_df: pd.DataFrame, zs: dict):
    test_stats = test_scaling_stats(test_df)
    present_Ns = set()
    y_lo, y_hi = [], []

    for method in ["pairwise", "pointwise"]:
        sub = test_stats[test_stats["method"] == method]
        if sub.empty:
            continue
        Ns = sub["N"].values
        mu = sub["mean"].values
        sd = sub["std"].fillna(0).values
        ax.plot(Ns, mu, f"{LS[method]}o", ms=4, color=C[method], label=method, lw=1.5)
        ax.fill_between(Ns, mu - sd, mu + sd, color=C[method], alpha=0.15)
        present_Ns.update(Ns.tolist())
        y_lo.append((mu - sd).min())
        y_hi.append((mu + sd).max())

        if method in zs:
            ax.axhline(zs[method], ls=":", color=C[method], lw=1.2, alpha=0.7)
            y_lo.append(zs[method])

    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Spearman ρ (test)")
    ax.set_title("LD50 — Gemma-4 12B, scaffold-disjoint val")
    present_Ns = sorted(present_Ns)
    ax.set_xticks(present_Ns)
    ax.set_xticklabels([str(n) for n in present_Ns], rotation=40, ha="right")
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    span = max(y_hi) - min(y_lo)
    pad = 0.08 * span if span > 0 else 0.02
    ax.set_ylim(min(y_lo) - pad, max(y_hi) + pad)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def make_figS1():
    test_df = load_test_eval()
    if test_df is None:
        print("No Figure S1 results found yet.")
        sys.exit(1)
    zs = load_zeroshot()

    fig, ax_a = plt.subplots(figsize=(3.8, 3.6))
    plot_scaling(ax_a, test_df, zs)
    ax_a.legend(loc="lower right")

    fig.suptitle(
        "Figure S1 (progress)  Gemma-4 12B LD50 — N-sweep",
        fontweight="bold", y=1.02, fontsize=8,
    )

    out_dir = Path("figures/figure_S1")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        p = out_dir / f"figS1_real.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    make_figS1()
