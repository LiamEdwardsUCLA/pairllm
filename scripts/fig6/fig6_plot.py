"""
NOTE — data requirements
    This script regenerates the published figure from the FULL per-run result
    tree (one metrics.json per training run), which is NOT included in this
    repository. What ships here under data/ is the aggregated, seed-level CSV
    behind each panel — for this figure, data/fig6_subset_selection/bars/
    bars_raw.csv and mmd_sweep/mmd_sweep_raw.csv, which contain every number
    plotted below. The compound selections themselves ARE shipped, under
    data/fig6_subset_selection/selections/.

    To re-run this script as-is, fetch the full result tree from the Zenodo
    archive (see README) and point EVAL_ROOT at it.

Figure 6: compound selection results.
  Panel A — N=200 bar chart (4 selectors × pairwise/pointwise)
  Panel B — N=50  bar chart
  Panel C — MMD sweep: subset distribution distance vs pairwise Spearman ρ (N=50)

Data: data/fig6_subset_selection/scaffold_eval/bars/
Output: data/fig6_subset_selection/fig6_scaffold_bars.png / .svg
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# NOTE: umap, rdkit and tqdm are imported lazily inside the functions that need
# them (the UMAP panel and the fingerprint/MMD recomputation). The manuscript
# Figure 6 layout reads everything from the shipped CSVs and needs none of them.

EVAL_ROOT = Path("data/fig6_subset_selection/scaffold_eval/bars")
DATA_ROOT = Path("data/fig6_subset_selection/selections/bars/n50")
POOL_CSV  = Path("data/fig2_scaling/splits/ld50/train_pool_scaffold.csv")
OUT_DIR   = Path("figures/figure_6")

# Aggregated CSVs — the per-run metrics.json tree is not shipped, but these
# carry every seed-level number the panels plot.
BARS_CSV  = Path("data/fig6_subset_selection/bars/bars_raw.csv")
SWEEP_CSV = Path("data/fig6_subset_selection/mmd_sweep/mmd_sweep_raw.csv")

# Degenerate runs excluded by default.
#
# (n50, kmedoids, pairwise, seed 3) collapsed to rho = -0.023 while the SAME
# compound subset trained pointwise reached 0.394, so the failure is in that
# training run rather than in the selected subset. Including it moves the
# k-medoids pairwise mean from 0.431 to 0.340 and inflates its SD from 0.015
# to 0.204, so it dominates the panel. Figure 2 excludes a comparably
# degenerate Caco-2 XGBoost fit, and this keeps the two figures consistent.
# Pass --include-degenerate to restore the published behaviour.
DEGENERATE_RUNS = {("n50", "kmedoids", "pairwise", 3)}
EXCLUDE_DEGENERATE = True

mpl.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── Bar chart config ──────────────────────────────────────────────────────────
SELECTORS = ["random", "maxmin", "kmedoids", "kernel_herding"]
SEL_LABEL = {"random": "Random", "maxmin": "MaxMin",
             "kmedoids": "k-medoids", "kernel_herding": "Kernel herding"}
SEL_COLOR = {"random": "#B0B0B0", "maxmin": "#55A868",
             "kmedoids": "#8172B3", "kernel_herding": "#2196F3"}
METHODS   = ["pairwise", "pointwise"]

# ── Sweep config ──────────────────────────────────────────────────────────────
# Each entry: (condition_name, color, list_of_seeds)
SWEEP = [
    ("random",            "#9E9E9E", list(range(1, 21))),
    ("kernel_herding",    "#2196F3", list(range(1, 21))),
    ("partial_maxmin_25", "#66BB6A", [1, 2, 3]),
    ("partial_maxmin_50", "#FFA726", [1, 2, 3]),
    ("partial_maxmin_75", "#EF5350", [1, 2, 3]),
    ("partial_maxmin_88", "#AB47BC", [1, 2, 3]),
    ("maxmin",            "#795548", list(range(1, 6))),
]
SWEEP_LABEL = {
    "random":            "Random",
    "kernel_herding":    "Kernel herding",
    "partial_maxmin_25": "Partial MaxMin 25%",
    "partial_maxmin_50": "Partial MaxMin 50%",
    "partial_maxmin_75": "Partial MaxMin 75%",
    "partial_maxmin_88": "Partial MaxMin 88%",
    "maxmin":            "MaxMin",
}


def panel_tag(ax, tag):
    ax.text(-0.14, 1.06, tag, transform=ax.transAxes,
            fontweight="bold", fontsize=11, va="top")


# ── Bar chart helpers ─────────────────────────────────────────────────────────
def _keep(n_label, selector, method, seed):
    """False for runs excluded as degenerate (see DEGENERATE_RUNS)."""
    if not EXCLUDE_DEGENERATE:
        return True
    return (n_label, selector, method, int(seed)) not in DEGENERATE_RUNS


def load_bars_csv(n_label, selector, method):
    """bars_raw.csv -> array of seed-level rho, honouring the exclusion list."""
    if not BARS_CSV.exists():
        return np.array([])
    df = pd.read_csv(BARS_CSV)
    df = df[(df["n_label"] == n_label)
            & (df["selector"] == SEL_LABEL[selector])
            & (df["method"] == method)]
    keep = [float(r["spearman_rho"]) for _, r in df.iterrows()
            if _keep(n_label, selector, method, r["seed"])]
    return np.array(keep)


def load_bars(n_label, selector, method):
    rhos = []
    for seed in range(1, 21):
        f = EVAL_ROOT / n_label / selector / f"seed{seed}" / method / "metrics.json"
        if f.exists() and _keep(n_label, selector, method, seed):
            d = json.loads(f.read_text())
            v = d.get("spearman_lstsq") or d.get("spearman")
            if v is not None:
                rhos.append(float(v))
    if rhos:
        return np.array(rhos)
    return load_bars_csv(n_label, selector, method)  # shipped aggregate


def plot_bars(ax, n_label, tag, title):
    def pairwise_mean(sel):
        rhos = load_bars(n_label, sel, "pairwise")
        return rhos.mean() if len(rhos) else -1.0

    ordered = sorted(SELECTORS, key=pairwise_mean, reverse=True)
    x = np.arange(len(METHODS))
    w = 0.12
    offsets = np.linspace(-(len(ordered) - 1) / 2,
                           (len(ordered) - 1) / 2,
                           len(ordered)) * (w + 0.01)

    for k, sel in enumerate(ordered):
        means, errs = [], []
        for m in METHODS:
            rhos = load_bars(n_label, sel, m)
            means.append(rhos.mean() if len(rhos) else np.nan)
            errs.append(rhos.std()   if len(rhos) else 0.0)
        bars = ax.bar(x + offsets[k], means, w, yerr=errs, capsize=2,
                      color=SEL_COLOR[sel], label=SEL_LABEL[sel], alpha=0.9)
        for m_val, bar in zip(means, bars):
            if np.isnan(m_val):
                bar.set_alpha(0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel("Spearman ρ (test)")
    ax.set_title(title, fontsize=8.5)
    ax.set_ylim(0, 0.75)
    panel_tag(ax, tag)


# ── Sweep helpers ─────────────────────────────────────────────────────────────
def compute_pool_fps():
    """Return (pool_fp_matrix, pool_mean, pool_df) — computed once, shared by sweep + UMAP."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from tqdm import tqdm
    print("Computing pool fingerprints...", flush=True)
    df = pd.read_csv(POOL_CSV)
    df = df[df["Drug"].notna() & (df["Drug"] != "Not Available")].reset_index(drop=True)
    fps = []
    for smi in tqdm(df["Drug"], desc="Pool FPs", leave=False):
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
                                dtype=np.float32))
    mat = np.stack(fps)
    return mat, mat.mean(axis=0), df


def compute_mmd(csv_path, pool_mean):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    sub = pd.read_csv(csv_path)
    fps = []
    for smi in sub["Drug"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
                                dtype=np.float32))
    if not fps:
        return np.nan
    diff = np.mean(fps, axis=0) - pool_mean
    return float(np.dot(diff, diff))


def collect_sweep_csv():
    """mmd_sweep_raw.csv -> same {sel: {mmds, rhos, color}} shape, no RDKit needed."""
    if not SWEEP_CSV.exists():
        return {}
    df = pd.read_csv(SWEEP_CSV)
    cond_data = {}
    for sel, color, _ in SWEEP:
        sub = df[df["condition"] == SWEEP_LABEL[sel]]
        if sub.empty:
            continue
        cond_data[sel] = {
            "mmds":  [float(v) for v in sub["mmd_squared"]],
            "rhos":  [float(v) for v in sub["spearman_rho"]],
            "color": color,
        }
    return cond_data


def collect_sweep(pool_mean):  # pool_mean kept for backward compat signature
    """Returns dict: sel -> {mmds, rhos, color} for all available seeds."""
    cond_data = {}
    for sel, color, seeds in SWEEP:
        mmds, rhos = [], []
        for seed in seeds:
            csv_path  = DATA_ROOT / sel / f"seed{seed}.csv"
            eval_path = EVAL_ROOT / "n50" / sel / f"seed{seed}" / "pairwise" / "metrics.json"
            if not csv_path.exists() or not eval_path.exists():
                continue
            mmd = compute_mmd(csv_path, pool_mean)
            d = json.loads(eval_path.read_text())
            rho = d.get("spearman_lstsq") or d.get("spearman")
            if rho is None or np.isnan(mmd):
                continue
            mmds.append(mmd)
            rhos.append(float(rho))
        if mmds:
            cond_data[sel] = {"mmds": mmds, "rhos": rhos, "color": color}
    return cond_data or collect_sweep_csv()  # shipped aggregate


def plot_sweep(ax, cond_data, tag):
    BOX_WIDTH = 0.032

    all_mean_x, all_mean_y, all_mmds = [], [], []
    for sel, color, _ in SWEEP:
        if sel not in cond_data:
            continue
        d = cond_data[sel]
        x_pos = float(np.mean(d["mmds"]))
        rhos  = d["rhos"]
        all_mean_x.append(x_pos)
        all_mean_y.append(float(np.mean(rhos)))
        all_mmds.extend(d["mmds"])

        ax.boxplot(
            rhos,
            positions=[x_pos],
            widths=BOX_WIDTH,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, markerfacecolor=color,
                            markeredgecolor=color, alpha=0.6),
            medianprops=dict(color="white", linewidth=1.5),
            boxprops=dict(facecolor=color, alpha=0.75, linewidth=0),
            whiskerprops=dict(color=color, linewidth=1.2),
            capprops=dict(color=color, linewidth=1.2),
        )
        ax.scatter([], [], color=color, s=35,
                   label=f"{SWEEP_LABEL[sel]} (n={len(rhos)})")

    # Trend through condition means; also compute all-points stats
    if len(all_mean_x) >= 3:
        # Condition-means regression (used for trend line)
        slope, intercept, r_means, p_means, _ = stats.linregress(all_mean_x, all_mean_y)
        # All-individual-points regression
        all_pts_x = sum([cond_data[s]["mmds"] for s in cond_data], [])
        all_pts_y = sum([cond_data[s]["rhos"] for s in cond_data], [])
        _, _, r_pts, p_pts, _ = stats.linregress(all_pts_x, all_pts_y)

        def fmt_p(p):
            return "p<0.001" if p < 0.001 else f"p={p:.3f}"

        print(f"[sweep] condition means (n={len(all_mean_x)}): r={r_means:.3f}  {fmt_p(p_means)}  slope={slope:.4f}")
        print(f"[sweep] all points     (n={len(all_pts_x)}):  r={r_pts:.3f}  {fmt_p(p_pts)}")

        xfit = np.linspace(min(all_mmds) - 0.05, max(all_mmds) + 0.05, 300)
        ax.plot(xfit, slope * xfit + intercept, "k--", lw=1, alpha=0.35, zorder=2)

        # Annotation bottom-left, away from upper-right legend
        ann = (
            f"condition means: r={r_means:.2f}, {fmt_p(p_means)}\n"
            f"all points:          r={r_pts:.2f}, {fmt_p(p_pts)}"
        )
        ax.text(0.03, 0.05, ann, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=6.5, color="#555",
                linespacing=1.6)

    ax.set_xlabel("MMD² (subset ↔ pool)")
    ax.set_ylabel("Spearman ρ (test, pairwise)")
    ax.set_title("N=50: distribution distance vs performance", fontsize=8.5)
    ax.legend(fontsize=5.5, loc="upper right", framealpha=0.8, edgecolor="none")
    panel_tag(ax, tag)


# ── UMAP panel ───────────────────────────────────────────────────────────────

UMAP_METHODS = [
    ("kernel_herding", "#2196F3", "D", "Kernel herding"),
    ("random",         "#555555", "o", "Random"),
    ("maxmin",         "#795548", "s", "MaxMin"),
]
UMAP_SEED = 1   # representative seed shown for each method


def subset_fps(sel, seed):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    csv = DATA_ROOT / sel / f"seed{seed}.csv"
    if not csv.exists():
        return None
    sub = pd.read_csv(csv)
    fps = []
    for smi in sub["Drug"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
                                dtype=np.float32))
    return np.stack(fps) if fps else None


def plot_umap(ax, pool_mat, tag):
    import umap
    print("Fitting UMAP on pool...", flush=True)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="jaccard", random_state=42, low_memory=False)
    pool_emb = reducer.fit_transform(pool_mat)

    # Pool background
    ax.scatter(pool_emb[:, 0], pool_emb[:, 1],
               c="#D0D0D0", s=2, alpha=0.35, linewidths=0, zorder=1, rasterized=True)

    handles = [mpatches.Patch(color="#D0D0D0", label=f"Pool (n={len(pool_mat)})")]

    for sel, color, marker, label in UMAP_METHODS:
        fps = subset_fps(sel, UMAP_SEED)
        if fps is None:
            continue
        emb = reducer.transform(fps)
        ax.scatter(emb[:, 0], emb[:, 1],
                   c=color, marker=marker, s=28, alpha=0.85,
                   edgecolors="white", linewidths=0.4, zorder=3)
        handles.append(mpatches.Patch(color=color, label=f"{label} (seed {UMAP_SEED})"))

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Chemical space coverage — N=50 selections", fontsize=8.5)
    ax.legend(handles=handles, fontsize=6, loc="lower right",
              framealpha=0.8, edgecolor="none")
    ax.set_xticks([]); ax.set_yticks([])
    panel_tag(ax, tag)


# ── Build figure ──────────────────────────────────────────────────────────────
def main():
    global EXCLUDE_DEGENERATE

    ap = argparse.ArgumentParser(description="Figure 6 / Figure S5 compound-selection panels.")
    ap.add_argument("--figure", choices=["6", "all"], default="6",
                    help="'6' (default): the manuscript Figure 6 layout — N=50 selector "
                         "comparison (A) and the MMD-squared sweep (B). Reads only the shipped "
                         "CSVs, so it needs no RDKit/UMAP and no per-run result tree. "
                         "'all': the full 2x2 working figure, adding the N=200 panel and the "
                         "UMAP projection (needs rdkit, umap-learn, tqdm).")
    ap.add_argument("--include-degenerate", action="store_true",
                    help="Keep runs listed in DEGENERATE_RUNS. Default is to exclude them; "
                         "see the comment on that constant for the rationale.")
    args = ap.parse_args()

    EXCLUDE_DEGENERATE = not args.include_degenerate
    print(f"Degenerate runs: {'INCLUDED' if args.include_degenerate else 'excluded'}"
          f" ({sorted(DEGENERATE_RUNS)})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.figure == "6":
        cond_data = collect_sweep_csv()
        print(f"  {sum(len(v['rhos']) for v in cond_data.values())} sweep points "
              f"across {len(cond_data)} conditions", flush=True)

        fig, (ax_A, ax_B) = plt.subplots(1, 2, figsize=(11, 4.2))
        fig.subplots_adjust(wspace=0.3)

        plot_bars(ax_A, "n50", "A", "N=50: selector comparison")
        plot_sweep(ax_B, cond_data, "B")

        handles, labels = ax_A.get_legend_handles_labels()
        ax_A.legend(handles, labels, fontsize=6, ncol=2, loc="upper left",
                    framealpha=0.7, edgecolor="none")

        stem = "fig6_selection"
    else:
        pool_mat, pool_mean, _ = compute_pool_fps()
        cond_data = collect_sweep(pool_mean)
        print(f"  {sum(len(v['rhos']) for v in cond_data.values())} sweep points "
              f"across {len(cond_data)} conditions", flush=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.subplots_adjust(hspace=0.45, wspace=0.42)
        fig.suptitle("Compound selection — scaffold val split",
                     fontweight="bold", fontsize=10, y=1.01)

        ax_A, ax_B = axes[0, 0], axes[0, 1]
        ax_C, ax_D = axes[1, 0], axes[1, 1]

        plot_bars(ax_A, "n200", "A", "N=200: selector comparison")
        plot_bars(ax_B, "n50",  "B", "N=50: selector comparison")
        plot_sweep(ax_C, cond_data, "C")
        plot_umap(ax_D, pool_mat, "D")

        handles, labels = ax_A.get_legend_handles_labels()
        ax_A.legend(handles, labels, fontsize=6, ncol=2, loc="upper left",
                    framealpha=0.7, edgecolor="none")

        stem = "fig6_scaffold_bars"

    for fmt in ("png", "svg"):
        out = OUT_DIR / f"{stem}.{fmt}"
        fig.savefig(out, bbox_inches="tight")
        print(f"[wrote] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
