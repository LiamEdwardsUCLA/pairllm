"""
NOTE — data requirements
    This script regenerates the published figure from the FULL per-run result
    tree (one metrics.json per training run), which is NOT included in this
    repository. What ships here under data/ is the aggregated, seed-level CSV
    behind each panel — see data/fig3_ablations/results/, which contains every
    number plotted below. The compound selections and the per-compound
    translation scores ARE shipped, under data/fig3_ablations/selections/ and
    data/fig3_ablations/translation/.

    To re-run this script as-is, fetch the full result tree from the Zenodo
    archive (see README) and point EVAL_ROOT / SOTA_ROOT at it.

Figure 3 — Ablations (LD50, scaffold split). Reads from
data/fig3_ablations/updated/. Missing results render as grey
pending bars, so this is safe to re-run at any point during training.

Layout:
  Row 0: A — representation at N=50  (grouped: pairwise / pointwise / SoTA)
          B — representation at N=100 (same)
  Row 1: C — random init  (pairwise + pointwise, N=50/100)
          D — sign flip / label inversion (pairwise + pointwise, N=50/100)
  Row 2: E — name→SMILES translation, base model (no fine-tuning)
          F — name→SMILES translation, fine-tuned models (N=50/100)

Usage:
    python scripts/fig3_plot.py
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

mpl.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

C = {"pointwise": "#4C72B0", "pairwise": "#C44E52", "sota": "#8C8C8C"}
C_PEND = "#D8D8D8"

BASE              = Path(__file__).resolve().parent.parent.parent
EVAL_ROOT         = BASE / "data/fig3_ablations/updated/eval"
SOTA_ROOT         = BASE / "data/fig3_ablations/updated/sota"
ABLATIONS_DIR     = BASE / "data/fig3_ablations"
OUT_DIR           = BASE / "data/fig3_ablations/updated"
ZEROSHOT_DIR      = BASE / "data/fig3_ablations/updated/eval/zeroshot"
FIG2_ZEROSHOT     = BASE / "data/fig2_scaling/splits/ld50/zeroshot"
TRANSLATION_TRAINED = BASE / "data/fig3_ablations/updated/translation/trained"

SEEDS  = [1, 2, 3]
N_VALS = ["n50", "n100"]
N_LABEL = {"n50": "N=50", "n100": "N=100"}


# ── helpers (from make_fig3_real.py) ─────────────────────────────────────────

def panel_tag(ax, tag):
    ax.text(-0.16, 1.08, tag, transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")


def pending(ax, msg="pending"):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color="#999", style="italic")


def read_rho(metrics_path: Path):
    if not Path(metrics_path).exists():
        return None
    d = json.loads(Path(metrics_path).read_text())
    rho = d.get("spearman_lstsq") or d.get("spearman")
    return float(rho) if rho is not None else None


def load_group(cond: str, n_label: str, method: str, seeds=SEEDS):
    """Mean/std/count of Spearman ρ across seeds for one (cond, n_label, method)."""
    vals = []
    for s in seeds:
        p = EVAL_ROOT / cond / n_label / f"seed{s}" / method / "metrics.json"
        rho = read_rho(p)
        if rho is not None:
            vals.append(rho)
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


def load_zeroshot_rho(repr_key: str):
    """Zero-shot pointwise Spearman for a given representation."""
    if repr_key == "standard":
        p = FIG2_ZEROSHOT / "pointwise/temp0/metrics.json"
    else:
        p = ZEROSHOT_DIR / repr_key / "pointwise/metrics.json"
    return read_rho(p)


def load_trans_tanimoto(method: str, n_label: str, seeds=SEEDS):
    """Mean/std Tanimoto across seeds for trained translation models."""
    vals = []
    for s in seeds:
        p = TRANSLATION_TRAINED / n_label / f"seed{s}" / method / "ld50_translation_scores.csv"
        if p.exists():
            df = pd.read_csv(p)
            vals.append(float(df["tanimoto_name2smiles"].mean()))
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


def load_sota_json(feature_set: str, n_label: str):
    """Try sota_{feature_set}_{n_label}.json then sota_{n_label}.json."""
    for fname in (f"sota_{feature_set}_{n_label}.json", f"sota_{n_label}.json"):
        p = SOTA_ROOT / feature_set / fname
        if p.exists():
            d = json.loads(p.read_text())
            return {"mean": d["spearman_mean"], "std": d["spearman_std"], "n": d["n_seeds"]}
    return None


def auto_ylim(stats_list, pad_lo=0.05, pad_hi=0.08):
    """Y limits tight around data — never forced to start at 0."""
    vals = []
    for s in stats_list:
        if s:
            vals += [s["mean"] - s["std"], s["mean"] + s["std"]]
    if not vals:
        return (0.0, 0.85)
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else 0.2
    return (lo - pad_lo * span, hi + pad_hi * span)


# ── A/B — representation ablation ────────────────────────────────────────────

def repr_panel(ax, n_label: str, tag: str):
    """
    3 repr conditions × 3 sub-bars per condition.
    Sub-bar order (per user): pairwise LLM, pointwise LLM, SoTA.
    """
    repr_conds = [
        # (eval_cond,    sota_feature_set,  x_label)
        ("standard",    "smiles_name",     "SMILES+name"),
        ("name_only",   "name_only",       "name only"),
        ("smiles_only", "admet_trix",      "SMILES only"),
    ]
    n_repr  = len(repr_conds)
    width   = 0.22
    offsets = [-width, 0.0, width]
    colors  = [C["pairwise"], C["pointwise"], C["sota"]]
    sub_labels = ["pairwise", "pointwise", "SoTA"]

    all_stats = []
    for ri, (llm_cond, sota_fs, _) in enumerate(repr_conds):
        pw   = load_group(llm_cond, n_label, "pairwise")
        pt   = load_group(llm_cond, n_label, "pointwise")
        sota = load_sota_json(sota_fs, n_label)
        sub_stats = [pw, pt, sota]
        all_stats.extend(sub_stats)
        for stat, color, off in zip(sub_stats, colors, offsets):
            x = ri + off
            if stat is None:
                ax.bar(x, 0.02, width * 0.9, color=C_PEND, bottom=0)
                ax.text(x, 0.03, "…", ha="center", fontsize=6, color="#aaa")
            else:
                ax.bar(x, stat["mean"], width * 0.9, color=color,
                       yerr=stat["std"], capsize=2.5, error_kw=dict(lw=0.9))
                top = stat["mean"] + stat["std"]
                ax.text(x, top + 0.004, f"n={stat['n']}",
                        ha="center", fontsize=5, color="#555")

    # Zero-shot dotted lines — one per representation group
    zs_rhos = [load_zeroshot_rho(cond) for cond, _, _ in repr_conds]
    for ri, zs_rho in enumerate(zs_rhos):
        if zs_rho is not None:
            ax.hlines(zs_rho, ri - 0.40, ri + 0.40,
                      colors=C["pointwise"], linestyles=":", lw=1.5, zorder=5)
    all_stats += [{"mean": r, "std": 0} for r in zs_rhos if r is not None]

    ylim = auto_ylim(all_stats)
    if ylim[0] < 0:
        ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(ylim)
    ax.set_xticks(range(n_repr))
    ax.set_xticklabels([c[2] for c in repr_conds])
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"Representation ablation — {N_LABEL[n_label]}", pad=5)

    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, sub_labels)]
    handles.append(plt.Line2D([], [], color=C["pointwise"], ls=":", lw=1.5, label="zero-shot"))
    ax.legend(handles=handles, fontsize=6, loc="lower right")
    panel_tag(ax, tag)


# ── C — random initialization (combined, both methods) ───────────────────────

def random_init_panel(ax, tag: str = None):
    groups = [("pointwise", "n50"), ("pointwise", "n100"),
              ("pairwise",  "n50"), ("pairwise",  "n100")]
    x = np.arange(len(groups))
    w = 0.35

    pretrained = [load_group("standard",    n, m) for m, n in groups]
    rand_init  = [load_group("random_init", n, m) for m, n in groups]
    colors = [C[m] for m, _ in groups]

    means_p = [s["mean"] if s else 0 for s in pretrained]
    errs_p  = [s["std"]  if s else 0 for s in pretrained]
    means_r = [s["mean"] if s else 0 for s in rand_init]
    errs_r  = [s["std"]  if s else 0 for s in rand_init]

    ax.bar(x - w / 2, means_p, w, yerr=errs_p, color=colors, capsize=2)
    ax.bar(x + w / 2, means_r, w, yerr=errs_r, color=colors, capsize=2,
           alpha=0.4, hatch="//")

    for i, (sp, sr) in enumerate(zip(pretrained, rand_init)):
        if sp is None:
            ax.text(x[i] - w / 2, 0.02, "pending", ha="center",
                    fontsize=5, rotation=90, color="#999")
        if sr is None:
            ax.text(x[i] + w / 2, 0.02, "pending", ha="center",
                    fontsize=5, rotation=90, color="#999")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{n}" for m, n in groups], fontsize=6.5)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Random initialization")
    ylim = auto_ylim(pretrained + rand_init)
    if ylim[0] < 0:
        ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(ylim)
    legend_handles = [
        mpatches.Patch(facecolor="#888", label="pretrained"),
        mpatches.Patch(facecolor="#888", alpha=0.4, hatch="//", label="rand-init"),
    ]
    ax.legend(handles=legend_handles, fontsize=6, loc="upper right")
    if tag:
        panel_tag(ax, tag)


# ── D — model scale ───────────────────────────────────────────────────────────

def scale_panel(ax, method: str, tag: str = None):
    conds  = ["model_2b", "model_4b", "standard"]
    labels = ["2B", "4B", "9B"]
    width  = 0.22
    color  = C[method]
    all_stats = []
    for ni, n in enumerate(N_VALS):
        off = (ni - 0.5) * width * 2.0
        for xi, cond in enumerate(conds):
            stat = load_group(cond, n, method)
            all_stats.append(stat)
            x = xi + off
            if stat is None:
                ax.bar(x, 0.02, width * 0.9, color=C_PEND)
                ax.text(x, 0.03, "…", ha="center", fontsize=6, color="#aaa")
            else:
                shade = 0.65 + 0.35 * ni
                c = (*mpl.colors.to_rgb(color), shade)
                ax.bar(x, stat["mean"], width * 0.9, color=c,
                       yerr=stat["std"], capsize=2.5, error_kw=dict(lw=0.9),
                       label=N_LABEL[n] if xi == 0 else "_")
                top = stat["mean"] + stat["std"]
                ax.text(x, top + 0.004, f"n={stat['n']}",
                        ha="center", fontsize=5, color="#555")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman ρ")
    ax.set_title(f"Model scale — {method}", pad=5)
    ax.set_ylim(auto_ylim(all_stats))
    c = C[method]
    legend_handles = [
        mpatches.Patch(facecolor=(*mpl.colors.to_rgb(c), 0.65), label="N=50"),
        mpatches.Patch(facecolor=(*mpl.colors.to_rgb(c), 1.0),  label="N=100"),
    ]
    ax.legend(handles=legend_handles, fontsize=6, loc="lower right")
    if tag:
        panel_tag(ax, tag)


# ── E — label inversion (combined, both methods) ──────────────────────────────

def sign_flip_panel(ax, tag: str = None):
    llm_groups = [("pointwise", "n50"), ("pointwise", "n100"),
                  ("pairwise",  "n50"), ("pairwise",  "n100")]
    sota_ns    = ["n50", "n100"]

    x_llm  = np.arange(len(llm_groups))
    x_sota = np.arange(len(sota_ns)) + len(llm_groups) + 0.7
    w = 0.32

    normal        = [load_group("standard",  n, m) for m, n in llm_groups]
    flipped       = [load_group("sign_flip", n, m) for m, n in llm_groups]
    sota_std      = [load_sota_json("morgan_only",       n) for n in sota_ns]
    sota_flipped  = [load_sota_json("sign_flip_morgan_only", n) for n in sota_ns]
    colors        = [C[m] for m, _ in llm_groups]

    # LLM bars: standard (solid) + sign-flipped (hatched)
    ax.bar(x_llm - w / 2, [s["mean"] if s else 0 for s in normal],  w,
           yerr=[s["std"] if s else 0 for s in normal],  color=colors, capsize=2)
    ax.bar(x_llm + w / 2, [s["mean"] if s else 0 for s in flipped], w,
           yerr=[s["std"] if s else 0 for s in flipped], color=colors, capsize=2,
           alpha=0.4, hatch="//")

    for i, (sn, sf) in enumerate(zip(normal, flipped)):
        if sn is None:
            ax.text(x_llm[i] - w / 2, 0.02, "pending", ha="center",
                    fontsize=5, rotation=90, color="#999")
        if sf is None:
            ax.text(x_llm[i] + w / 2, 0.02, "pending", ha="center",
                    fontsize=5, rotation=90, color="#999")

    # XGBoost bars: standard (solid) + sign-flipped (hatched)
    ax.bar(x_sota - w / 2, [s["mean"] if s else 0 for s in sota_std],     w,
           yerr=[s["std"] if s else 0 for s in sota_std],     color=C["sota"], capsize=2)
    ax.bar(x_sota + w / 2, [s["mean"] if s else 0 for s in sota_flipped], w,
           yerr=[s["std"] if s else 0 for s in sota_flipped], color=C["sota"], capsize=2,
           alpha=0.4, hatch="//")

    ax.axvline(len(llm_groups) + 0.2, color="#ccc", lw=0.8, ls="--")
    ax.axhline(0, color="k", lw=0.8)

    all_x  = list(x_llm) + list(x_sota)
    labels = [f"{m}\n{n}" for m, n in llm_groups] + [f"XGBoost\n{n}" for n in sota_ns]
    ax.set_xticks(all_x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Spearman ρ (vs true labels)")
    ax.set_title("Label inversion")
    ylim = auto_ylim(normal + flipped + sota_std + sota_flipped, pad_lo=0.08, pad_hi=0.08)
    ax.set_ylim(ylim)
    legend_handles = [
        mpatches.Patch(facecolor="#888", label="standard"),
        mpatches.Patch(facecolor="#888", alpha=0.4, hatch="//", label="sign-flipped"),
        mpatches.Patch(facecolor=C["sota"], label="XGBoost (Morgan)"),
    ]
    ax.legend(handles=legend_handles, fontsize=6, loc="upper right")
    if tag:
        panel_tag(ax, tag)


# ── E — name→SMILES: base model Tanimoto histogram ───────────────────────────

def panel_translation_base(ax, tag: str):
    csv_path = ABLATIONS_DIR / "translation" / "ld50_translation_scores.csv"
    if not csv_path.exists():
        pending(ax)
        ax.set_title("name→SMILES (base model)")
        panel_tag(ax, tag)
        return
    df = pd.read_csv(csv_path)

    mask_valid = df["tanimoto_name2smiles"] > 0
    valid   = df.loc[mask_valid,  "tanimoto_name2smiles"].values
    invalid = df.loc[~mask_valid, "tanimoto_name2smiles"].values

    bins  = np.linspace(0, 1, 26)
    bin_w = bins[1] - bins[0]

    ax.bar(bins[0] + bin_w / 2, len(invalid), width=bin_w * 0.9,
           color="#E07040", alpha=0.85)
    ax.hist(valid, bins=bins, color=C["pointwise"], alpha=0.8)

    mean_valid = float(valid.mean()) if len(valid) else 0.0
    ax.axvline(mean_valid, ls="--", color="k", lw=1)
    exact_rate = df["exact_match_name2smiles"].mean()

    handles = [
        mpatches.Patch(facecolor=C["pointwise"], alpha=0.8,
                       label=f"valid ({exact_rate:.0%} exact)"),
        mpatches.Patch(facecolor="#E07040", alpha=0.85,
                       label=f"invalid ({len(invalid)/len(df):.0%})"),
    ]
    ax.legend(handles=handles, fontsize=6, loc="upper right")
    ax.set_xlabel("Tanimoto (pred vs true)")
    ax.set_ylabel("count")
    ax.set_title("name→SMILES (base model)")
    panel_tag(ax, tag)


# ── F — name→SMILES translation: trained models vs base ──────────────────────

def panel_translation(ax, tag: str):
    groups = [("pointwise", "n50"), ("pointwise", "n100"),
              ("pairwise",  "n50"), ("pairwise",  "n100")]
    stats  = [load_trans_tanimoto(m, n) for m, n in groups]
    colors = [C[m] for m, _ in groups]
    x      = np.arange(len(groups))
    w      = 0.55

    for xi, (stat, color) in enumerate(zip(stats, colors)):
        if stat is None:
            ax.bar(x[xi], 0.01, w, color=C_PEND)
            ax.text(x[xi], 0.015, "pending", ha="center",
                    fontsize=5, rotation=90, color="#999")
        else:
            ax.bar(x[xi], stat["mean"], w, color=color,
                   yerr=stat["std"], capsize=2.5, error_kw=dict(lw=0.9))
            ax.text(x[xi], stat["mean"] + stat["std"] + 0.003, f"n={stat['n']}",
                    ha="center", fontsize=5, color="#555")

    # Base model reference line
    base_csv = ABLATIONS_DIR / "translation/ld50_translation_scores.csv"
    if base_csv.exists():
        df_base  = pd.read_csv(base_csv)
        base_tan = float(df_base["tanimoto_name2smiles"].mean())
        ax.axhline(base_tan, ls="--", color="#555", lw=1.2, zorder=5)
        ax.text(len(groups) - 0.08, base_tan + 0.004, "base model",
                ha="right", fontsize=5.5, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{n}" for m, n in groups], fontsize=6.5)
    ax.set_ylabel("Tanimoto (name→SMILES)")
    ax.set_title("name→SMILES translation")

    present = [s for s in stats if s]
    if present:
        ylim = auto_ylim(present, pad_lo=0.05, pad_hi=0.15)
        ax.set_ylim(max(0, ylim[0]), ylim[1])
    else:
        ax.set_ylim(0, 0.5)

    handles = [
        mpatches.Patch(color=C["pointwise"], label="pointwise"),
        mpatches.Patch(color=C["pairwise"],  label="pairwise"),
        plt.Line2D([], [], color="#555", ls="--", lw=1.2, label="base model"),
    ]
    ax.legend(handles=handles, fontsize=6, loc="lower right")
    panel_tag(ax, tag)


# ── build ─────────────────────────────────────────────────────────────────────

def build(out_dir: Path):
    fig = plt.figure(figsize=(13, 12))
    # Row 0: A (repr N=50, 2-wide), B (repr N=100, 2-wide)
    # Row 1: C (rand-init, 2-wide), D (label inversion, 2-wide)
    # Row 2: E (base model Tanimoto histogram, 2-wide), F (trained translation summary, 2-wide)
    gs  = GridSpec(3, 4, figure=fig, hspace=0.65, wspace=0.42,
                   height_ratios=[1, 1, 1])

    ax_A = fig.add_subplot(gs[0, :2])
    ax_B = fig.add_subplot(gs[0, 2:])
    ax_C = fig.add_subplot(gs[1, :2])
    ax_D = fig.add_subplot(gs[1, 2:])
    ax_E = fig.add_subplot(gs[2, :2])
    ax_F = fig.add_subplot(gs[2, 2:])

    repr_panel(ax_A, "n50",  "A")
    repr_panel(ax_B, "n100", "B")

    random_init_panel(ax_C, tag="C")
    sign_flip_panel(ax_D, tag="D")

    panel_translation_base(ax_E, "E")
    panel_translation(ax_F, "F")

    fig.suptitle("Figure 3 — Ablations (LD50, scaffold split)",
                 fontweight="bold", y=1.005, fontsize=11)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        p = out_dir / f"fig3_updated.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"[wrote] {p}")

    plt.close(fig)


if __name__ == "__main__":
    build(OUT_DIR)
