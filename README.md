# Comparative Adaptation Enables Language Models to Rank Molecules with Limited Fine-Tuning Data

Code and frozen data for reproducing the experiments in the paper: the scaling /
frugal-regime comparison across LD50, Caco-2, and Solubility (Figure 2), the
ablation studies (Figure 3), the compound-selection and MMD² analysis
(Figure 6), and the Gemma-4 reproducibility check (Figure S1).

This repository contains everything needed to **re-run the pipeline**: the
frozen scaffold splits, the exact compound subsets, the pair-sampling
procedure, the prompts, the training and merging scripts, the evaluation and
aggregation code, and the seed-level results behind every figure.

**Terminology** — the paper's *comparative adaptation* is called `pairwise`
throughout the code, CLI flags and result CSVs; the single-molecule baseline it
is compared against is `pointwise`.

---

## Contents

```
.
├── README.md
├── LICENSE                     ← Apache-2.0, covers all code
├── NOTICE                      ← attribution for datasets and base models
├── requirements-train.txt      ← LoRA fine-tuning + merging
├── requirements-eval.txt       ← vLLM inference (separate environment)
├── requirements-analysis.txt   ← selection, baselines, stats, figures (CPU)
├── data/
│   ├── LICENSE                 ← per-path terms for the derived data
│   ├── fig2_scaling/
│   │   ├── scaling_results.csv      ← Spearman ρ (+ MAE) per dataset × method × N × seed
│   │   ├── zero_shot_baseline.csv   ← zero-shot (no fine-tuning) Spearman ρ
│   │   └── splits/{ld50,caco2,solubility}/
│   │       ├── train_pool_scaffold.csv   ← scaffold-disjoint training pool
│   │       ├── val_scaffold.csv          ← scaffold-disjoint validation set (early stopping only)
│   │       ├── test.csv                  ← held-out test set (TDC scaffold split; never re-split)
│   │       └── normalization.json        ← y_min/y_max for the 0–1000 label rescaling
│   ├── fig6_subset_selection/
│   │   ├── bars/{bars_raw,bars_summary}.csv
│   │   ├── mmd_sweep/{mmd_sweep_raw,mmd_sweep_regression}.csv
│   │   └── selections/           ← the exact compound lists chosen by each selector
│   ├── fig3_ablations/
│   │   ├── results/              ← seed-level CSVs for each ablation panel
│   │   ├── selections/{n50,n100}/seed{1,2,3}.csv
│   │   └── translation/          ← per-compound name→SMILES scores
│   └── figS1_gemma4/
│       ├── scaling_results.csv
│       └── zero_shot_baseline.csv
└── scripts/
    ├── prompts.py                  ← all prompt templates
    ├── train.py                    ← LoRA fine-tuning (pairwise or pointwise)
    ├── merge.py                    ← merge LoRA adapter into base weights (Qwen3.5)
    ├── eval_pairwise_ranking.py    ← pairwise eval: margin → Spearman ρ via least-squares aggregation
    ├── eval_pointwise_ranking.py   ← pointwise eval: direct prediction → Spearman ρ
    ├── fig2/
    │   ├── build_scaffold_split.py     ← scaffold split + N-compound subsets + training JSONL
    │   ├── run_sota_scaffold.py        ← XGBoost baselines (Optuna-tuned)
    │   ├── minimol_scaffold_worker.py  ← MiniMol embedding + MLP/Ridge head baseline
    │   └── fig2_plot.py
    ├── fig3/fig3_plot.py               ← ablation panels
    ├── fig6/
    │   ├── fig6_select_compounds.py    ← random / MaxMin / k-medoids / kernel herding / partial MaxMin
    │   └── fig6_plot.py
    └── figS1/
        ├── merge_gemma.py              ← Gemma-4 merging (merge.py does not apply)
        └── figS1_plot.py
```

### A note on the `*_plot.py` scripts

The figure scripts regenerate the published figures from the **full per-run
result tree** (one `metrics.json` per training run), which is too large to
include here and is archived separately (see
[Data availability](#data-availability)). What ships in this repository is the
**aggregated, seed-level CSV behind every panel** — these contain every number
plotted, and every statistic reported in the paper can be recomputed from them
directly. Each plot script carries a header saying which CSV corresponds to it.

---

## Requirements

**Base models**

| Figure | Model |
|---|---|
| 2, 3, 4 | `Qwen/Qwen3.5-9B` |
| 5 | `google/gemma-4-12B-it` |

**Environments** — training and vLLM inference must be separate (conflicting
`transformers` pins):

```bash
python -m venv .venv-train    && . .venv-train/bin/activate    && pip install -r requirements-train.txt
python -m venv .venv-eval     && . .venv-eval/bin/activate     && pip install -r requirements-eval.txt
python -m venv .venv-analysis && . .venv-analysis/bin/activate && pip install -r requirements-analysis.txt
```

Python 3.10+. The Qwen3.5-9B figures (2, 3 and 6) need a single GPU with
≥48 GB for training; Figure S1 needs more headroom for the 12B base model.

**Qwen3.5 workaround** — training runs for Figures 2, 3 and 6 need

```bash
export LD_PRELOAD=<conda-env>/lib/libstdc++.so.6
export FLA_DISABLE_BACKEND_DISPATCH=1
```

to avoid a linear-attention (gated-delta-net) backend-dispatch bug specific to
this model family's fast path. Not needed for Figure S1.

---

## Key hyperparameters

### LoRA adapter and optimizer (all conditions)

| Parameter | Value |
|---|---|
| Rank (r) / Alpha (α) / Dropout | 64 / 128 / 0 |
| Target modules | all linear layers (`all-linear`) |
| Precision | bfloat16 |
| Optimizer | AdamW (8-bit) |
| Learning rate | 2 × 10⁻⁴ (see Figure S1 for its one exception) |
| LR schedule / warmup | cosine / 0.1 |
| Effective batch size | 64 (per-device 8 × grad-accum 8, single GPU) |
| Loss | cross-entropy, masked to the response span only |
| Early stopping | patience 5 evals, min delta 0.005, burn-in val ρ > 0.2 |

### Training schedule by N

Epoch-aligned: the eval interval equals the number of steps in one epoch, so
every N is checked at the same fractional-epoch granularity.

| N | 50 | 100 | 200 | 400 | 800 | 1600 | 3200 |
|---|---|---|---|---|---|---|---|
| max steps | 40 | 156 | 622 | 2,494 | 3,126 | 3,126 | 3,126 |
| eval / save interval | 1 | 2 | 4 | 7 | 13 | 25 | 50 |

Pairwise and pointwise use an identical schedule at each N, so the two are
directly comparable at the same training depth despite pairwise having far more
nominal training examples (C(N,2) pairs vs. N compounds) per epoch.

Figure 6 instead uses a **fixed-step** schedule, to decouple training depth
from the number of pairs different selectors generate:

| N | mode | max steps | eval interval |
|---|---|---|---|
| 50 | pairwise / pointwise | 40 | 5 / 1 |
| 200 | pairwise / pointwise | 622 | 50 / 4 |

### Evaluation

- **Pairwise**: 60,000 test pairs per dataset, sampled uniformly at random
  (seed 42) from all C(n,2) pairs of the held-out test set.
- Three aggregations are computed from the predicted margins: margin-sum,
  Bradley–Terry (sign only, iterative MM fit), and **least-squares recovery**
  (sparse solve of `s_j − s_i ≈ predicted margin` per pair, plus one row
  anchoring `mean(s)` to the training mean, via `scipy.sparse.linalg.lsqr`).
  **Least-squares recovery is the value reported as "pairwise" Spearman ρ**
  throughout the paper (`spearman_lstsq` in the output JSON).
- MAE is computed on the least-squares score, denormalized to native units.

---

## Dataset splits

Scaffold-disjoint (Bemis–Murcko, RDKit) throughout. The **test set** is TDC's
native scaffold split and is never touched during training, validation, or
hyperparameter selection. A second, independent scaffold-disjoint split is
carved out of TDC's training portion to produce the **validation set** used for
early stopping — this prevents near-identical analogs from leaking between a
training subsample and its own early-stopping signal.

| Dataset | Property (Y units) | Train pool | Val | Test |
|---|---|---|---|---|
| LD50 | Acute oral toxicity, pLD50 = log10(1/LD50), LD50 in mol/kg | 5,306 | 601 | 1,478 |
| Caco-2 | Intestinal permeability, log Papp (cm/s) | 655 | 73 | 182 |
| Solubility | Aqueous solubility, log S (mol/L) | 7,183 | 802 | 1,997 |

For each (dataset, N), 8 independent seeds each sample N compounds from the
train pool via `random.Random(seed).sample()`. N-grid: 50, 100, 200, 400, 800,
1600, 3200 for LD50 and Solubility; 50, 100, 200, 400 for Caco-2 (pool-limited).

Raw labels are rescaled to an integer 0–1000 scale before training:
`label = round((Y − y_min) / (y_max − y_min) × 1000)`, fit once per dataset on
the full training CSV (`y_min`/`y_max` in each `normalization.json`).

The splits in `data/fig2_scaling/splits/` are **frozen** — they are the exact
splits used for every result in the paper. `build_scaffold_split.py` reuses
them by default and regenerates the training JSONL files around them. Passing
`--force` re-derives the split from scratch and requires the raw TDC
`train.csv`; doing so will not reproduce the paper.

---

## Reproducing a run

```bash
# 1. Build the training subset for one (dataset, N, seed). Reuses the frozen
#    split and writes compounds.csv, pairwise_train.jsonl, pointwise_train.jsonl.
python scripts/fig2/build_scaffold_split.py --dataset ld50 --n-list 50 --seeds 1

# 2. Train (pairwise). Use --finetune-name pointwise + pointwise_train.jsonl
#    for the pointwise arm; see the schedule table for max-steps/eval-steps.
torchrun --nproc_per_node=1 scripts/train.py \
    --model Qwen/Qwen3.5-9B \
    --dataset <seed_dir>/pairwise_train.jsonl \
    --finetune-name pairwise \
    --output-dir <checkpoint_dir> \
    --batch-size 8 --grad-accum 8 \
    --max-steps 40 --eval-steps 1 --save-steps 1 \
    --lora-targets all-linear \
    --val-file <dataset_dir>/val_scaffold_pairwise.jsonl \
    --patience 5 --early-stopping-burnin 0.2 --early-stopping-min-delta 0.005

# 3. Merge the adapter into the base weights
python scripts/merge.py \
    --adapter <checkpoint_dir>/pairwise-<port>/best_checkpoint \
    --base-model Qwen/Qwen3.5-9B \
    --output-dir <merged_dir>

# 4. Evaluate (in the vLLM environment)
python scripts/eval_pairwise_ranking.py \
    --dataset ld50 \
    --model <merged_dir> \
    --test-csv data/fig2_scaling/splits/ld50/test.csv \
    --normalization data/fig2_scaling/splits/ld50/normalization.json \
    --train-csv data/fig2_scaling/splits/ld50/train_pool_scaffold.csv \
    --n-pairs 60000 \
    --output-json <results>/metrics.json
```

Steps 1–4 repeat across the full N × seed × method grid. For the pointwise arm,
step 4 uses `eval_pointwise_ranking.py` with `--dataset`, `--model`,
`--test-csv`, `--normalization` and `--output-json` only — there are no pairs to
sample, so `--n-pairs` and `--train-csv` do not apply.

### Figure 3 — ablations

LD50 only, N ∈ {50, 100}, seeds 1–3, against a standard condition of
SMILES+name / pretrained backbone / correct labels.

| Ablation | How to run | Manuscript panel |
|---|---|---|
| Pretraining control (random init) | `train.py --random-init` and `merge.py --random-init --seed <s>` | Fig. 3A |
| Target inversion (label sign flip) | Sign-flipped training labels | Fig. 3B |
| Molecular input (name-only / SMILES-only) | `--representation names-only` or `smiles-only` on both train and eval | Fig. 3C (N=100); Fig. S2A (N=50) |
| Name→SMILES translation | Prompt the trained model for each test compound's SMILES; score by Tanimoto (Morgan, radius 2) vs. truth | Fig. S2B; Fig. S2C |

The compound subsets are the same ones used in Figure 2 and ship under
`data/fig3_ablations/selections/`.

### Figure 6 — compound selection

`fig6_select_compounds.py` reproduces the selections in
`data/fig6_subset_selection/selections/`. All selectors operate on Morgan
fingerprints (ECFP4, 2,048-bit, radius 2); compounds whose SMILES cannot be
parsed are excluded first.

| Algorithm | Description |
|---|---|
| Random | Uniform sample without replacement (`numpy.random.default_rng(seed).choice`) |
| MaxMin | Greedy max-min Tanimoto diversity picker (`rdSimDivPickers.MaxMinPicker`); high MMD² |
| k-Medoids | KMeans proxy (N clusters); nearest-to-centroid pool compound per cluster; empty clusters filled randomly |
| Kernel herding | Greedy MMD²-minimizing under a linear kernel on Morgan FPs; first compound random; low MMD² |
| Partial MaxMin | First ⌊f × N⌋ picked by MaxMin, remainder random, for f ∈ {25, 50, 75, 88}% — the MMD² sweep |

Seed counts differ by condition (random and kernel herding were replicated more
heavily to tighten the mean): 20 seeds at N=50 and 10 at N=200 for those two,
5 seeds elsewhere, pointwise always 5. The MMD² sweep uses 3 seeds per
condition, pairwise only. Exact counts are in `bars_summary.csv`.

### Figure S1 — Gemma-4

Reuses Figure 2's LD50 split and per-(N, seed) subsets exactly — no new
sampling — so any difference is attributable to the model, not the data.
Three changes are required, for architecture compatibility only:

1. **Attention**: `--attn-implementation sdpa`. Gemma-4's full-attention layers
   use head_dim=512, above FlashAttention-2's 256 cap.
2. **Merging**: `scripts/figS1/merge_gemma.py`, not `merge.py`. Gemma-4's
   text-only class expects a key structure that does not match the
   checkpoint's on-disk keys; loading via `AutoModelForCausalLM` resolves to
   the correct multimodal wrapper and needs no key remapping.
3. **Inference**: vLLM ≥ 0.26, in its own environment.

Plus one genuine hyperparameter change: **pairwise uses lr 5×10⁻⁵ and
early-stopping burn-in 0.35** (vs. 2×10⁻⁴ / 0.2). Gemma-4's zero-shot pairwise
val ρ ≈ 0.32 already exceeds the 0.2 burn-in threshold at step 0 — a threshold
calibrated against Qwen's much lower zero-shot floor — so early stopping arms
immediately and locks in a checkpoint before training has progressed. Pointwise
is unchanged. Everything else (patience, min-delta, eval schedule, LoRA config,
N-grid, seeds) matches Figure 2.

### SoTA baselines

- **XGBoost (Morgan)** — `run_sota_scaffold.py --features morgan`. Self-contained.
- **XGBoost (ADMET-trix)** — needs the third-party ADMET-trix feature generator
  (`fingerprint_gen.py`), which is **not redistributed here**. Obtain the
  checkout and set `ADMET_TRIX_SRC` to its path.
- **MiniMol** — needs its own environment
  ([graphcore-research/minimol](https://github.com/graphcore-research/minimol));
  set `MINIMOL_PYTHON` to that interpreter. `run_sota_scaffold.py` shells out to
  `minimol_scaffold_worker.py` through it.

---

## Statistical testing

Every comparison uses a **paired t-test** (`scipy.stats.ttest_rel`) across
matched seeds — valid because, within any single comparison, every condition
trains on the exact same seeded N-compound subset, so seed-level outcomes are
paired. Figure S1 additionally reports Wilcoxon signed-rank as a nonparametric
complement.

Unpaired **Welch's t-test** (`ttest_ind(..., equal_var=False)`) is used only
where a paired test is invalid: against methods with a different, non-seed-matched
protocol (MiniMol/XGBoost at seed counts that do not line up 1:1), and
throughout Figure 6, where selectors have unequal seed counts. Figure 6 reports
Mann–Whitney U alongside Welch's.

No multiple-comparison correction is applied, across N values or across the
several comparisons made at a given N.

These are one-line calls over the seed-level columns already present in each
figure's raw CSV, and are reconstructable from the shipped data without running
any model.

---

## Data files

**`data/fig2_scaling/scaling_results.csv`** — one row per (dataset, method, N,
seed): `dataset, method, N, seed, spearman_rho, mae, n_compounds_train`.
`method` ∈ {`pairwise`, `pointwise`, `minimol`, `xgboost_morgan_only`,
`xgboost_admet_trix`}. `zero_shot_baseline.csv` has the same base models with no
fine-tuning, one row per (dataset, method).

**`data/fig6_subset_selection/`** — `bars/bars_raw.csv` is one row per
(selector, N, mode, seed): `n_label, selector, method, seed, spearman_rho`.
`bars_summary.csv` adds `n_seeds, mean, std, ci95, min, max` and Welch
`vs_random_t`/`vs_random_p`. `mmd_sweep/mmd_sweep_raw.csv` is one row per
(condition, seed) with `mmd_squared` and `spearman_rho`;
`mmd_sweep_regression.csv` holds the regression of ρ on MMD² for condition
means and for all individual points. `selections/` holds the chosen compounds
as `Drug_ID, Drug, Y` at `bars/{n50,n200}/{selector}/seed{N}.csv` and
`mmd_sweep/n50/{condition}/seed{1..3}.csv`; the random, MaxMin and kernel-herding
files in the sweep are the same objects as the corresponding `bars/n50` files.

**`data/fig3_ablations/results/`** — `repr_raw.csv`, `random_init_raw.csv`,
`label_inversion_raw.csv` are one row per (condition, N, method, seed) with
`spearman_rho`; each has a `*_summary.csv` adding the paired t-statistic and
p-value against the standard condition. `zeroshot.csv` holds the three
zero-shot representation baselines. `translation_base_model.csv` and
`translation/{n50,n100}/seed{1,2,3}/{pairwise,pointwise}/ld50_translation_scores.csv`
hold per-compound translation scores; `translation_raw.csv` and
`translation_summary.csv` aggregate them.

**`data/figS1_gemma4/scaling_results.csv`** — one row per (method, N, seed),
same columns and conventions as Figure 2. `zero_shot_baseline.csv` is base
Gemma-4, pairwise only.

---

## Data availability

The source datasets are publicly available through the
[Therapeutics Data Commons](https://tdcommons.ai/) and their originating
repositories; see the paper for citations. The frozen splits, subset indices,
and seed-level results are included in this repository.

The full per-run result tree, training logs, and trained LoRA adapters are
archived at **[TODO: Zenodo DOI]**.

## Licensing

This repository mixes original code, derived data, and model derivatives.
Terms differ by path.

| Path | License |
|---|---|
| Everything except `data/` | Apache-2.0 (see `LICENSE`) |
| `data/fig2_scaling/splits/solubility/` | CC0 1.0 |
| `data/fig2_scaling/splits/{ld50,caco2}/`, `data/fig6_subset_selection/selections/`, `data/fig3_ablations/selections/` | CC BY 4.0 |
| All other files under `data/` (seed-level results) | CC0 1.0 |
| LoRA adapters, where released | Apache-2.0, as modified derivatives of the base models |

`data/LICENSE` gives the per-path detail. Attribution for all third-party
sources is in `NOTICE` — please preserve it if you redistribute any part of
this work.

**On the dataset terms.** The solubility splits derive from AqSolDB, whose
Harvard Dataverse deposit (doi:10.7910/DVN/OVHAW8) is CC0 1.0. The LD50 and
Caco-2 splits derive from TDC datasets for which TDC records the upstream
license as "Not Specified" and which TDC redistributes as CC BY 4.0; we follow
that position and note the uncertainty rather than asserting it is settled.

**On the model weights.** Both base models — `Qwen/Qwen3.5-9B` and
`google/gemma-4-12B-it` — are Apache-2.0, with no use restrictions and no
flow-down obligations. Any adapters released with this work are LoRA deltas,
not full weights, and are modified derivative works of those models; the
Apache-2.0 text ships alongside them. Qwen and Gemma are trademarks of their
respective owners, and nothing here is endorsed by or affiliated with Alibaba
Cloud or Google.

**Intended use.** This is a research artifact for ADMET property prediction on
public benchmarks. It is not validated or approved for clinical, regulatory, or
any other decision-making use, and is not a substitute for experimental
measurement.

## Citation

```bibtex
@article{[TODO: key],
  title  = {Comparative Adaptation Enables Language Models to Rank Molecules
            with Limited Fine-Tuning Data},
  author = {[TODO]},
  year   = {[TODO]},
  journal = {[TODO]},
  doi    = {[TODO]},
}
```
