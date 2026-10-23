"""
NOTE — data requirements
    This script regenerates the published figure from the FULL per-run result
    tree (one metrics.json per training run, plus per-run training logs), which
    is NOT included in this repository. What ships here under data/ is the
    aggregated, seed-level CSV behind the test-set panels —
    data/fig2_scaling/scaling_results.csv and zero_shot_baseline.csv — which
    carry every per-(dataset, method, N, seed) test correlation plotted in the
    scaling and bar panels. The validation training curves are NOT in that CSV:
    they are parsed per step from each run's own training log, which only the
    full result tree contains.

    To re-run this script as-is, fetch the full result tree from the Zenodo
    archive (see README) and point DATA_ROOT / CKPT_ROOT_TMPL at it.

    NOTE — the bar panel's N is chosen automatically, as the N maximising the
    pairwise-minus-pointwise gap (pick_bar_n_from_curves / pick_bar_n_from_test
    below). It is NOT fixed at the prespecified N=50 benchmark budget, and in
    multi-dataset mode the choice is made on test-set means. Set the N
    explicitly if you need a fixed-budget comparison.

Figure 2, scaffold-disjoint validation split — one dataset or all three side
by side, via --datasets. Works for any of ld50/caco2/solubility.

  python scripts/fig2_plot.py --datasets ld50
      -> single-dataset 3-panel figure (A: test scaling curve vs N, B: grouped
         bars at best N, C: val training curves) -> fig2_{dataset}_scaffold.{png,svg}

  python scripts/fig2_plot.py --datasets ld50 caco2 solubility --bar-n n50
      -> the published Figure 2 layout: N datasets side by side, method
         comparison at the fixed budget on top (panels A..C) and learning
         curves below (panels D..F). No validation-curve panel here, since
         training logs for 3 datasets side by side would be too dense to read.
         -> fig2_scaffold_all3.{png,svg}
         This mode reads only the shipped aggregate CSVs when the per-run tree
         is absent, so it reproduces from a clean checkout.

Data sources:
  - data/fig2_scaling/splits/{dataset}/scaffold_val_eval/{n}/seed*/{method}/metrics.json
      (test set, panels A/B — evaluated against the SAME held-out test.csv
      used everywhere else, so it's directly comparable to fig2_real)
  - $CKPT_ROOT/{dataset}_scaffold_val/{n}/seed*/{method}_train.log
      (val Spearman per step, parsed directly from each run's own training
      log — there's no consolidated training_curves.csv for this experiment;
      only read in single-dataset mode, for panel C)
  - data/fig2_scaling/splits/{dataset}/sota_scaffold/results/{name}/sota_{n}.json
  - data/fig2_scaling/splits/{dataset}/zeroshot/{method}/temp0/metrics.json

SoTA overlays: MiniMol + Morgan-only XGBoost (ADMET-trix omitted from the
plot for now — computed too, just not shown, to keep the overlay readable).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Style (matches make_fig2.py)
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

SOTA_STYLE = {
    "minimol":     {"color": "#55A868", "marker": "^", "label": "MiniMol"},
    "morgan_only": {"color": "#8C8C8C", "marker": "s", "label": "XGBoost (Morgan)"},
}

# Train_pool_scaffold.csv sizes — only matters if an "n=full" point ever
# existed for that dataset (none currently do).
POOL_N_BY_DATASET = {"ld50": 5306, "caco2": 590, "solubility": 7183}
DS_LABEL = {"ld50": "LD50", "caco2": "Caco-2", "solubility": "Solubility"}
N_ORDER = ["n50", "n100", "n200", "n400", "n800", "n1600", "n3200", "full"]

# Fine-tuning budget shown in the bar panel. Default is the prespecified n50
# benchmark budget; "auto" restores the legacy gap-maximising selection.
BAR_N = "n50"

DATA_ROOT = Path("data/fig2_scaling/splits")
# Root of the per-run checkpoint/log tree (panel C only). Not shipped — see the
# note at the top of this file. Override with the CKPT_ROOT environment variable.
CKPT_ROOT_TMPL = os.environ.get("CKPT_ROOT", "checkpoints") + "/{ds}_scaffold_val"

PW_LOG_PAT = re.compile(r"\[pairwise_eval\] step=(\d+)\s+spearman=([\-0-9.naninf]+)")
PT_LOG_PAT = re.compile(r"\[pointwise_eval\] step=(\d+)\s+spearman=([\-0-9.naninf]+)")


def n_to_int(n_label, ds):
    return POOL_N_BY_DATASET[ds] if n_label == "full" else int(n_label[1:])


# ---------------------------------------------------------------------------
# Data loading (parameterized by dataset, so both modes share it)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Aggregated-CSV source
#
# The per-run metrics.json tree is not shipped with this repository, but every
# test-set number it holds is in data/fig2_scaling/scaling_results.csv. These
# loaders read that CSV so the test-set panels reproduce from the repo alone.
# Panel C (validation training curves) still needs the per-run logs.
# ---------------------------------------------------------------------------

SCALING_CSV  = Path("data/fig2_scaling/scaling_results.csv")
ZEROSHOT_CSV = Path("data/fig2_scaling/zero_shot_baseline.csv")

# scaling_results.csv method name -> name used inside this script
CSV_METHOD_MAP = {
    "pairwise":            "pairwise",
    "pointwise":           "pointwise",
    "minimol":             "minimol",
    "xgboost_morgan_only": "morgan_only",
}


def load_test_eval_csv(ds: str) -> pd.DataFrame | None:
    """scaling_results.csv -> same shape as load_test_eval (pairwise/pointwise only)."""
    if not SCALING_CSV.exists():
        return None
    df = pd.read_csv(SCALING_CSV)
    df = df[(df["dataset"] == ds) & (df["method"].isin(["pairwise", "pointwise"]))]
    if df.empty:
        return None
    return pd.DataFrame({
        "n":               "n" + df["N"].astype(int).astype(str),
        "seed":            df["seed"].astype(int),
        "method":          df["method"],
        "spearman_lstsq":  df["spearman_rho"].astype(float),
    }).reset_index(drop=True)


def load_sota_csv(ds: str, name: str) -> pd.DataFrame | None:
    """scaling_results.csv -> same shape as load_sota, for the baseline overlays."""
    if not SCALING_CSV.exists():
        return None
    csv_name = {v: k for k, v in CSV_METHOD_MAP.items()}.get(name)
    df = pd.read_csv(SCALING_CSV)
    df = df[(df["dataset"] == ds) & (df["method"] == csv_name)]
    if df.empty:
        return None
    g = df.groupby("N")["spearman_rho"].agg(["mean", "std", "count"]).reset_index()
    g["n"] = "n" + g["N"].astype(int).astype(str)
    return g[["n", "N", "mean", "std", "count"]].sort_values("N")


def load_zeroshot_csv(ds: str) -> dict:
    if not ZEROSHOT_CSV.exists():
        return {}
    df = pd.read_csv(ZEROSHOT_CSV)
    df = df[df["dataset"] == ds]
    return {r["method"]: float(r["spearman_rho"]) for _, r in df.iterrows()}


def load_test_eval(ds: str) -> pd.DataFrame | None:
    """scaffold_val_eval/{n}/seed{s}/{method}/metrics.json -> one row per (n, seed, method)."""
    test_eval_dir = DATA_ROOT / ds / "scaffold_val_eval"
    rows = []
    for p in test_eval_dir.glob("*/seed*/*/metrics.json"):
        n_label, seed_str, method = p.relative_to(test_eval_dir).parts[:3]
        try:
            seed = int(seed_str.replace("seed", ""))
        except ValueError:
            continue
        m = json.loads(p.read_text())
        rho = m.get("spearman_lstsq") or m.get("spearman")
        if rho is None or (isinstance(rho, float) and np.isnan(rho)):
            continue
        rows.append({"n": n_label, "seed": seed, "method": method, "spearman_lstsq": float(rho)})
    if rows:
        return pd.DataFrame(rows)
    return load_test_eval_csv(ds)       # fall back to the shipped aggregate


def load_curves(ds: str) -> pd.DataFrame | None:
    """Parse every {n}/seed{s}/{method}_train.log for [*_eval] step/spearman lines.
    Only used in single-dataset mode (panel C)."""
    ckpt_root = Path(CKPT_ROOT_TMPL.format(ds=ds))
    rows = []
    for log_path in ckpt_root.glob("*/seed*/*_train.log"):
        n_label, seed_dir = log_path.parts[-3], log_path.parts[-2]
        try:
            seed = int(seed_dir.replace("seed", ""))
        except ValueError:
            continue
        method = "pairwise" if log_path.name.startswith("pairwise") else "pointwise"
        pat = PW_LOG_PAT if method == "pairwise" else PT_LOG_PAT
        text = log_path.read_text()
        for m in pat.finditer(text):
            step, rho_str = m.groups()
            try:
                rho = float(rho_str)
            except ValueError:
                continue
            if np.isnan(rho):
                continue
            rows.append({"n": n_label, "seed": seed, "method": method,
                         "step": int(step), "val_spearman": rho})
    return pd.DataFrame(rows) if rows else None


def load_sota(ds: str, name: str) -> pd.DataFrame | None:
    """sota_scaffold/results/{name}/sota_{n_label}.json -> DataFrame(n, N, mean, std, count)."""
    d = DATA_ROOT / ds / "sota_scaffold" / "results" / name
    if not d.exists():
        return load_sota_csv(ds, name)  # fall back to the shipped aggregate
    rows = []
    for p in d.glob("sota_n*.json"):
        m = json.loads(p.read_text())
        n_label = m["n_label"]
        rows.append({
            "n": n_label, "N": n_to_int(n_label, ds),
            "mean": m["spearman_mean"], "std": m["spearman_std"],
            "count": m["n_seeds"],
        })
    return pd.DataFrame(rows).sort_values("N") if rows else None


def load_zeroshot(ds: str) -> dict:
    """Same zero-shot baseline as fig2_real (same base model, same test set)."""
    result = {}
    for method in ["pairwise", "pointwise"]:
        p = DATA_ROOT / ds / "zeroshot" / method / "temp0" / "metrics.json"
        if p.exists():
            d = json.loads(p.read_text())
            rho = d.get("spearman_lstsq") or d.get("spearman")
            if rho is not None:
                result[method] = float(rho)
    return result or load_zeroshot_csv(ds)  # fall back to the shipped aggregate


def peak_spearman(df_curves: pd.DataFrame) -> pd.DataFrame:
    return (
        df_curves.groupby(["n", "seed", "method"])["val_spearman"]
        .max().reset_index().rename(columns={"val_spearman": "peak_rho"})
    )


def test_scaling_stats(test_df: pd.DataFrame, ds: str) -> pd.DataFrame:
    stats = (
        test_df.groupby(["n", "method"])["spearman_lstsq"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    stats["N"] = stats["n"].apply(lambda n: n_to_int(n, ds))
    return stats.sort_values("N")


def pick_bar_n_from_curves(df_curves: pd.DataFrame) -> str:
    """Single-dataset mode: pick N from peak val-curve rho (avoids test-set peeking)."""
    peak = peak_spearman(df_curves)
    stats = peak.groupby(["n", "method"])["peak_rho"].mean().reset_index()
    pw = stats[stats["method"] == "pairwise"].set_index("n")["peak_rho"]
    po = stats[stats["method"] == "pointwise"].set_index("n")["peak_rho"]
    common = pw.index.intersection(po.index)
    if common.empty:
        return peak["n"].iloc[0]
    return (pw[common] - po[common]).idxmax()


def pick_bar_n_from_test(test_stats: pd.DataFrame) -> str | None:
    """Multi-dataset mode: no training logs loaded, so pick directly from test-set means."""
    pw = test_stats[test_stats["method"] == "pairwise"].set_index("n")["mean"]
    po = test_stats[test_stats["method"] == "pointwise"].set_index("n")["mean"]
    common = pw.index.intersection(po.index)
    if not common.empty:
        return (pw[common] - po[common]).idxmax()
    any_method = test_stats.sort_values("N")
    return any_method["n"].iloc[-1] if not any_method.empty else None


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_tag(ax, tag):
    ax.text(-0.18, 1.06, tag, transform=ax.transAxes,
            fontweight="bold", fontsize=11, va="top")


def plot_scaling(ax, ds: str, test_df: pd.DataFrame, zs: dict):
    test_stats = test_scaling_stats(test_df, ds)
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

    if "pointwise" in zs:
        ax.axhline(zs["pointwise"], ls=":", color=C["pointwise"], lw=1.2, alpha=0.7)
        y_lo.append(zs["pointwise"])

    for name, style in SOTA_STYLE.items():
        sota_df = load_sota(ds, name)
        if sota_df is None or sota_df.empty:
            continue
        Ns, mu, sd = sota_df["N"].values, sota_df["mean"].values, sota_df["std"].fillna(0).values
        ax.plot(Ns, mu, ":", marker=style["marker"], ms=4, color=style["color"],
                label=style["label"], lw=1.5)
        ax.fill_between(Ns, mu - sd, mu + sd, color=style["color"], alpha=0.12)
        present_Ns.update(Ns.tolist())
        y_lo.append((mu - sd).min())
        y_hi.append((mu + sd).max())

    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Spearman ρ (test)")
    ax.set_title(f"{DS_LABEL[ds]} — scaffold-disjoint val")
    present_Ns = sorted(present_Ns)
    ax.set_xticks(present_Ns)
    ax.set_xticklabels([str(n) for n in present_Ns], rotation=40, ha="right")
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    # Zoom the y-range to the data so the zero-shot line sits near the bottom
    # of the plot instead of floating in the middle of a 0-1 axis — makes the
    # lift over baseline the visually dominant feature.
    if y_lo and y_hi:
        span = max(y_hi) - min(y_lo)
        pad = 0.08 * span if span > 0 else 0.02
        ax.set_ylim(min(y_lo) - pad, max(y_hi) + pad)


def plot_bars(ax, ds: str, test_df: pd.DataFrame, bar_n: str | None):
    if bar_n is None:
        ax.set_axis_off()
        return
    bar_int = n_to_int(bar_n, ds)
    method_stats = {}
    for method in ["pairwise", "pointwise"]:
        sub = test_df[(test_df["n"] == bar_n) & (test_df["method"] == method)]
        if not sub.empty:
            mu = sub["spearman_lstsq"].mean()
            sd = sub["spearman_lstsq"].std(ddof=1) if len(sub) > 1 else 0.0
            method_stats[method] = (mu, sd, C[method])

    for name, style in SOTA_STYLE.items():
        sota_df = load_sota(ds, name)
        if sota_df is None:
            continue
        row = sota_df[sota_df["N"] == bar_int]
        if not row.empty:
            method_stats[name] = (row["mean"].iloc[0], row["std"].iloc[0], style["color"])

    ordered = [m for m in ["pairwise", "pointwise"] + list(SOTA_STYLE) if m in method_stats]
    labels = [SOTA_STYLE[m]["label"] if m in SOTA_STYLE else m for m in ordered]
    x = np.arange(len(ordered))
    for i, method in enumerate(ordered):
        mu, sd, color = method_stats[method]
        ax.bar(i, mu, 0.5, color=color, yerr=sd if not np.isnan(sd) else 0, capsize=4)

    ax.set_xticks(x[:len(ordered)])
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Spearman ρ (test)")
    ax.set_title(f"{DS_LABEL[ds]} scaffold-val  N={bar_int}")
    ax.set_ylim(bottom=0)


def plot_training_curves(ax, ds: str, df_curves: pd.DataFrame, curve_n: str):
    sub = df_curves[df_curves["n"] == curve_n]
    for method in ["pairwise", "pointwise"]:
        msub = sub[sub["method"] == method]
        if msub.empty:
            continue
        all_steps = sorted(msub["step"].unique())
        seed_curves = []
        for _, sdf in msub.groupby("seed"):
            sdf = sdf.sort_values("step")
            seed_curves.append(np.interp(all_steps, sdf["step"], sdf["val_spearman"]))
        arr = np.array(seed_curves)
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0)
        ax.plot(all_steps, mu, LS[method], color=C[method], label=method, lw=1.5)
        ax.fill_between(all_steps, mu - sd, mu + sd, color=C[method], alpha=0.15)

    ax.set_xlabel("training step")
    ax.set_ylabel("val Spearman ρ (scaffold-disjoint)")
    ax.set_title(f"{DS_LABEL[ds]} scaffold-val  N={n_to_int(curve_n, ds)}")
    ax.set_ylim(bottom=0)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def make_single_dataset_figure(ds: str):
    """3-panel (A/B/C) figure for one dataset — includes training-curve panel C."""
    test_df = load_test_eval(ds)
    df_curves = load_curves(ds)
    if test_df is None or df_curves is None:
        print(f"[{ds}] No scaffold-val results found yet.")
        sys.exit(1)
    zs = load_zeroshot(ds)

    fig = plt.figure(figsize=(3.8, 9))
    gs = GridSpec(3, 1, figure=fig, hspace=0.6)

    ax_a = fig.add_subplot(gs[0, 0])
    plot_scaling(ax_a, ds, test_df, zs)
    panel_tag(ax_a, "A")
    ax_a.legend(loc="lower right")

    bar_n = pick_bar_n_from_curves(df_curves) if BAR_N == "auto" else BAR_N

    ax_b = fig.add_subplot(gs[1, 0])
    plot_bars(ax_b, ds, test_df, bar_n)
    panel_tag(ax_b, "B")

    ax_c = fig.add_subplot(gs[2, 0])
    plot_training_curves(ax_c, ds, df_curves, bar_n)
    panel_tag(ax_c, "C")
    ax_c.legend(loc="lower right")

    ckpt_root = Path(CKPT_ROOT_TMPL.format(ds=ds))
    n_tested = test_df["n"].nunique()
    n_total = len([n for n in N_ORDER if (ckpt_root / n).exists()])
    eval_note = f"A/B: test Spearman — {n_tested}/{n_total} N values evaluated"

    legend_elems = [
        Line2D([0], [0], color=C["pairwise"],  ls="-",  lw=2, label="pairwise (test)"),
        Line2D([0], [0], color=C["pointwise"], ls="--", lw=2, label="pointwise (test)"),
        Line2D([0], [0], color=C["pointwise"], ls=":", lw=1.2, alpha=0.7, label="ZS pointwise"),
    ]
    for name, style in SOTA_STYLE.items():
        if load_sota(ds, name) is not None:
            legend_elems.append(Line2D([0], [0], color=style["color"], ls=":", marker=style["marker"],
                                        ms=4, lw=1.5, label=style["label"]))
    fig.legend(handles=legend_elems, loc="upper center", ncol=len(legend_elems),
               bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=8)

    fig.suptitle(
        f"Figure 2 (scaffold-val)  {DS_LABEL[ds]} — scaling (A), best-N comparison (B), training dynamics (C)\n"
        f"[{eval_note};  C: val Spearman, scaffold-disjoint from train]",
        fontweight="bold", y=1.02, fontsize=8,
    )

    out_dir = Path("figures/figure_2")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        p = out_dir / f"fig2_{ds}_scaffold.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"Wrote {p}")
    plt.close(fig)


def make_multi_dataset_figure(datasets):
    """2-panel-per-column (A/B only, no training curves) figure, N datasets side by side."""
    per_ds = {}
    for ds in datasets:
        test_df = load_test_eval(ds)
        if test_df is None:
            print(f"[{ds}] No scaffold-val test results found yet — skipping.")
            continue
        per_ds[ds] = test_df

    if not per_ds:
        print("No scaffold-val results found for any requested dataset.")
        return

    ds_list = [ds for ds in datasets if ds in per_ds]
    ncols = len(ds_list)
    fig = plt.figure(figsize=(3.8 * ncols, 6.2))
    gs = GridSpec(2, ncols, figure=fig, hspace=0.55, wspace=0.45)

    any_sota = {name: False for name in SOTA_STYLE}
    for col, ds in enumerate(ds_list):
        test_df = per_ds[ds]
        zs = load_zeroshot(ds)

        # Row 0 — fixed-budget method comparison (panels A..C), one per dataset
        if BAR_N == "auto":
            bar_n = pick_bar_n_from_test(test_scaling_stats(test_df, ds))
        else:
            bar_n = BAR_N

        ax_bar = fig.add_subplot(gs[0, col])
        plot_bars(ax_bar, ds, test_df, bar_n)
        panel_tag(ax_bar, chr(ord("A") + col))

        # Row 1 — learning curves (panels D..F), same column order
        ax_sc = fig.add_subplot(gs[1, col])
        plot_scaling(ax_sc, ds, test_df, zs)
        panel_tag(ax_sc, chr(ord("A") + ncols + col))
        if col == 0:
            ax_sc.legend(loc="lower right")

        for name in SOTA_STYLE:
            if load_sota(ds, name) is not None:
                any_sota[name] = True

    legend_elems = [
        Line2D([0], [0], color=C["pairwise"],  ls="-",  lw=2, label="pairwise (test)"),
        Line2D([0], [0], color=C["pointwise"], ls="--", lw=2, label="pointwise (test)"),
        Line2D([0], [0], color=C["pointwise"], ls=":", lw=1.2, alpha=0.7, label="ZS pointwise"),
    ]
    for name, style in SOTA_STYLE.items():
        if any_sota[name]:
            legend_elems.append(Line2D([0], [0], color=style["color"], ls=":", marker=style["marker"],
                                        ms=4, lw=1.5, label=style["label"]))
    fig.legend(handles=legend_elems, loc="upper center", ncol=len(legend_elems),
               bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)

    fig.suptitle(
        f"Figure 2 (scaffold-val)  {BAR_N} method comparison (top) and learning curves (bottom)",
        fontweight="bold", y=1.08, fontsize=9,
    )

    out_dir = Path("figures/figure_2")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        p = out_dir / f"fig2_scaffold_all3.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ld50"],
                         choices=list(POOL_N_BY_DATASET))
    parser.add_argument("--bar-n", default="n50",
                        help="Fine-tuning budget for the bar panel, e.g. n50 (default) — the "
                             "prespecified benchmark budget. 'auto' restores the legacy "
                             "behaviour of picking the N that maximises the pairwise-minus-"
                             "pointwise gap, which selects the budget on the outcome.")
    args = parser.parse_args()

    BAR_N = args.bar_n
    if BAR_N != "auto" and BAR_N not in N_ORDER:
        parser.error(f"--bar-n must be 'auto' or one of {N_ORDER}")
    print(f"Bar panel budget: {BAR_N}")

    if len(args.datasets) == 1:
        make_single_dataset_figure(args.datasets[0])
    else:
        make_multi_dataset_figure(args.datasets)
