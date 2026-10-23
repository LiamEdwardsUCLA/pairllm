"""
Figure 6 compound selection.

Generates all compound subsets used in Figure 6:

  Bar charts (Panels A/B):
    Selectors:   random, MaxMin, k-medoids, kernel herding
    N=50 :  random / kernel_herding  seeds 1-20;  maxmin / kmedoids  seeds 1-5
    N=200:  random / kernel_herding  seeds 1-10;  maxmin / kmedoids  seeds 1-5

  MMD² sweep (Panel C): partial MaxMin at fractions {25%, 50%, 75%, 88%}
    N=50, seeds 1-3 per fraction

Input pool: data/fig2_scaling/splits/ld50/train_pool_scaffold.csv
            (scaffold-disjoint training pool, 5,306 compounds after SMILES filtering)

Output:
  data/fig6_subset_selection/selections/bars/n{N}/{method}/seed{s}.csv
  data/fig6_subset_selection/selections/mmd_sweep/n50/{method}/seed{s}.csv

Run from the project root directory.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.SimDivFilters import rdSimDivPickers
from sklearn.cluster import KMeans

# ── Config ────────────────────────────────────────────────────────────────────
POOL_CSV = Path("data/fig2_scaling/splits/ld50/train_pool_scaffold.csv")
OUT_DIR  = Path("data/fig6_subset_selection/selections")

BAR_SEEDS = {
    "n50": {
        "random":         list(range(1, 21)),   # 20 seeds
        "kernel_herding": list(range(1, 21)),   # 20 seeds
        "maxmin":         list(range(1, 6)),    #  5 seeds
        "kmedoids":       list(range(1, 6)),    #  5 seeds
    },
    "n200": {
        "random":         list(range(1, 11)),   # 10 seeds
        "kernel_herding": list(range(1, 11)),   # 10 seeds
        "maxmin":         list(range(1, 6)),    #  5 seeds
        "kmedoids":       list(range(1, 6)),    #  5 seeds
    },
}

SWEEP_FRACTIONS = [0.25, 0.50, 0.75, 0.88]
SWEEP_SEEDS     = [1, 2, 3]

# ── Load pool ─────────────────────────────────────────────────────────────────
print("Loading train pool...", flush=True)
df = pd.read_csv(POOL_CSV)
df = df[df["Drug"].notna() & (df["Drug"] != "Not Available")].reset_index(drop=True)

mols, valid_idx = [], []
for i, smi in enumerate(df["Drug"]):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is not None:
        mols.append(mol)
        valid_idx.append(i)
df = df.iloc[valid_idx].reset_index(drop=True)
n_pool = len(df)
print(f"  {n_pool} valid compounds", flush=True)

# ── Fingerprints ──────────────────────────────────────────────────────────────
print("Computing Morgan fingerprints (ECFP4, 2048 bits)...", flush=True)
fps_rdkit = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]
fp_matrix = np.zeros((n_pool, 2048), dtype=np.float32)
for i, fp in enumerate(fps_rdkit):
    DataStructs.ConvertToNumpyArray(fp, fp_matrix[i])
pool_mean = fp_matrix.mean(axis=0)

# ── Save helper ───────────────────────────────────────────────────────────────
def save(indices, *parts):
    path = OUT_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"    skip (exists): {path.name}", flush=True)
        return
    df.iloc[sorted(set(indices))][["Drug_ID", "Drug", "Y"]].to_csv(path, index=False)
    print(f"    {path.parent.name}/{path.name}: {len(indices)} compounds", flush=True)

# ── Selectors ─────────────────────────────────────────────────────────────────
def select_random(N, seed):
    # Draw 200 first so that n50 is always a prefix of n200 for the same seed.
    full = np.random.default_rng(seed).choice(n_pool, 200, replace=False).tolist()
    return full[:N]


def select_maxmin(N, seed):
    picker = rdSimDivPickers.MaxMinPicker()
    return list(picker.LazyBitVectorPick(fps_rdkit, n_pool, N, seed=seed))


def select_kmedoids(N, seed):
    km = KMeans(n_clusters=N, random_state=seed, n_init=5, max_iter=300)
    labels = km.fit_predict(fp_matrix)
    centroids = km.cluster_centers_
    selected = []
    for c in range(N):
        mask = np.where(labels == c)[0]
        if len(mask) == 0:
            continue
        dists = np.linalg.norm(fp_matrix[mask] - centroids[c], axis=1)
        selected.append(int(mask[dists.argmin()]))
    if len(selected) < N:
        remaining = list(set(range(n_pool)) - set(selected))
        extra = np.random.default_rng(seed).choice(remaining, N - len(selected), replace=False).tolist()
        selected += extra
    return selected


def select_kernel_herding(N, seed):
    rng = np.random.default_rng(seed)
    excluded = np.zeros(n_pool, dtype=bool)
    first = int(rng.choice(n_pool))
    selected = [first]
    excluded[first] = True
    # float64 accumulator for numerical stability
    subset_mean = fp_matrix[first].astype(np.float64)
    for t in range(1, N):
        target = pool_mean - subset_mean
        scores = fp_matrix @ target.astype(np.float32)
        scores[excluded] = -np.inf
        pick = int(np.argmax(scores))
        selected.append(pick)
        excluded[pick] = True
        subset_mean = (t * subset_mean + fp_matrix[pick]) / (t + 1)
    return selected


def select_partial_maxmin(N, fraction, seed):
    n_maxmin = int(fraction * N)
    n_random = N - n_maxmin
    picker = rdSimDivPickers.MaxMinPicker()
    if n_maxmin > 0:
        maxmin_idx = list(picker.LazyBitVectorPick(fps_rdkit, n_pool, n_maxmin, seed=seed))
    else:
        maxmin_idx = []
    if n_random > 0:
        excluded = set(maxmin_idx)
        remaining = [i for i in range(n_pool) if i not in excluded]
        random_picks = np.random.default_rng(seed).choice(remaining, n_random, replace=False).tolist()
    else:
        random_picks = []
    return maxmin_idx + random_picks


# ═════════════════════════════════════════════════════════════════════════════
# BAR SELECTIONS
# ═════════════════════════════════════════════════════════════════════════════
SELECTOR_FNS = {
    "random":         select_random,
    "maxmin":         select_maxmin,
    "kmedoids":       select_kmedoids,
    "kernel_herding": select_kernel_herding,
}

for n_key, selector_seeds in BAR_SEEDS.items():
    N = int(n_key[1:])
    print(f"\n{'='*60}\nBAR SELECTIONS  N={N}\n{'='*60}", flush=True)
    for method, seeds in selector_seeds.items():
        print(f"\n── {method} ──", flush=True)
        fn = SELECTOR_FNS[method]
        for seed in seeds:
            idx = fn(N, seed)
            save(idx, "bars", n_key, method, f"seed{seed}.csv")

# ═════════════════════════════════════════════════════════════════════════════
# MMD² SWEEP SELECTIONS
# ═════════════════════════════════════════════════════════════════════════════
N_SWEEP = 50
print(f"\n{'='*60}\nMMD² SWEEP  N={N_SWEEP}\n{'='*60}", flush=True)
for fraction in SWEEP_FRACTIONS:
    pct = int(fraction * 100)
    method = f"partial_maxmin_{pct}"
    print(f"\n── {method} ──", flush=True)
    for seed in SWEEP_SEEDS:
        idx = select_partial_maxmin(N_SWEEP, fraction, seed)
        save(idx, "mmd_sweep", f"n{N_SWEEP}", method, f"seed{seed}.csv")

print("\nDone.", flush=True)
