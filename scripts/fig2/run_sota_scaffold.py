"""
Run all 3 SoTA baselines, in sequence, against any {n_label}/seed{s}/
compounds.csv tree. Works for any of ld50/caco2/solubility via --dataset.
Trains on train.csv, evaluates on test.csv — the same held-out test set used
by the LLM pipeline, so results are directly comparable.

Baselines (in order):
  1. Morgan-only XGBoost   — bare ECFP fingerprint from SMILES
  2. ADMET-trix XGBoost    — Morgan + Avalon + ErG + RDKit descriptors
  3. MiniMol               — pretrained MiniMol embeddings + MLP/Ridge head
                              (needs minimol_env — a separate torch/graphium
                              stack incompatible with the training/eval
                              environment, so this stage subprocesses into
                              that env via minimol_scaffold_worker.py)

Usage:
    python scripts/run_sota_scaffold.py --dataset ld50 \
        --seed-root data/fig2_scaling/splits/ld50/scaffold_val \
        --n-list 50 100 200 400 800 1600 3200 --seeds 1 2 3
    python scripts/run_sota_scaffold.py --dataset caco2 \
        --seed-root data/fig2_scaling/splits/caco2/scaffold_val \
        --n-list 50 100 200 400 --seeds 1 2 3
    python scripts/run_sota_scaffold.py --skip minimol   # if its env isn't ready
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

BASE = Path(__file__).resolve().parent.parent.parent

# The ADMET-trix feature generator (Morgan+Avalon+ErG+RDKit descriptors) is a
# third-party checkout, NOT redistributed here — see "SoTA baselines" in the
# README for where to obtain it. It only operates on SMILES, so it's
# dataset-agnostic and shared regardless of --dataset. Override the location
# with the ADMET_TRIX_SRC environment variable.
SOTA_SRC = Path(os.environ.get(
    "ADMET_TRIX_SRC",
    BASE / "third_party" / "ics-admet-prediction-tdc-versiontwo",
))

# MiniMol needs its own environment (incompatible pins with the training env).
# Point MINIMOL_PYTHON at that environment's interpreter.
MINIMOL_ENV_PYTHON = Path(os.environ.get(
    "MINIMOL_PYTHON",
    Path.home() / "miniconda3" / "envs" / "minimol_env" / "bin" / "python3",
))


# ---------------------------------------------------------------------------
# Shared XGBoost/Optuna machinery (identical tuning recipe for both feature sets)
# ---------------------------------------------------------------------------

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "tree_method": "hist", "device": "cpu", "nthread": 4, "random_state": 42,
    }
    from xgboost import XGBRegressor
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)


def run_trial(seed, X_train, y_train, X_test, y_test, n_optuna_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_train))
    cut = int(0.8 * len(idx))
    X_tr, X_val = X_train[idx[:cut]], X_train[idx[cut:]]
    y_tr, y_val = y_train[idx[:cut]], y_train[idx[cut:]]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: objective(t, X_tr, y_tr, X_val, y_val), n_trials=n_optuna_trials, show_progress_bar=False)

    best = study.best_params
    best.update({"tree_method": "hist", "device": "cpu", "nthread": 4, "random_state": seed})

    from xgboost import XGBRegressor
    model = XGBRegressor(**best)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "seed": seed,
        "mae": float(mean_absolute_error(y_test, preds)),
        "spearman": float(spearmanr(preds, y_test).statistic),
        "best_params": best,
        "preds": preds.tolist(),
    }


def sanitize(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6)


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

def generate_morgan_only(smiles_series, n_bits=1024, radius=2):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import DataStructs, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    mols = smiles_series.apply(Chem.MolFromSmiles)
    valid_mols = mols.dropna()
    fps = valid_mols.apply(lambda m: rdMolDescriptors.GetHashedMorganFingerprint(m, nBits=n_bits, radius=radius))
    rows = []
    for fp in fps:
        arr = np.zeros((n_bits,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return np.stack(rows)


def generate_admet_trix(smiles_series):
    if not SOTA_SRC.exists():
        sys.exit(
            f"ERROR: ADMET-trix feature generator not found at {SOTA_SRC}.\n"
            f"It is third-party code and is not redistributed in this repository — see "
            f"'SoTA baselines' in the README, then set ADMET_TRIX_SRC to your checkout.\n"
            f"The Morgan-only XGBoost baseline (--features morgan) needs no extra checkout."
        )
    sys.path.insert(0, str(SOTA_SRC))
    from fingerprint_gen import generate_morgan_fps, generate_avalon_fps, generate_erg_fps, compute_rdkit_features
    from rdkit import Chem, RDLogger
    from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
    from rdkit.Chem import DataStructs
    RDLogger.DisableLog("rdApp.*")

    def generate_avalon_safe(mols, n_bits=1024):
        rows, n_fail = [], 0
        for mol in mols:
            try:
                fp = GetAvalonCountFP(mol, nBits=n_bits)
                arr = np.zeros(n_bits, dtype=np.int32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                rows.append(arr)
            except (Exception, SystemError):
                rows.append(np.zeros(n_bits, dtype=np.int32))
                n_fail += 1
        if n_fail:
            print(f"  Avalon: {n_fail}/{len(rows)} molecules zero-filled due to errors")
        return np.stack(rows)

    mols = smiles_series.apply(Chem.MolFromSmiles)
    valid_mols = mols.dropna()
    mol_list = list(valid_mols)
    return np.concatenate([
        generate_morgan_fps(valid_mols),
        generate_avalon_safe(mol_list),
        generate_erg_fps(valid_mols),
        compute_rdkit_features(valid_mols),
    ], axis=1)


FEATURE_SETS = {
    "morgan_only": generate_morgan_only,
    "admet_trix": generate_admet_trix,
}


# ---------------------------------------------------------------------------
# Driver for one feature set, across every (n_label, seed)
# ---------------------------------------------------------------------------

def run_xgboost_baseline(feature_set, ds_dir, seed_root, n_list, seeds, out_root, n_optuna_trials,
                         flip_labels=False):
    out_key = f"sign_flip_{feature_set}" if flip_labels else feature_set
    print(f"\n{'='*70}\nXGBoost baseline: {out_key}\n{'='*70}", flush=True)
    build_features = FEATURE_SETS[feature_set]

    train_df = pd.read_csv(ds_dir / "train.csv")
    test_df = pd.read_csv(ds_dir / "test.csv")

    from rdkit import Chem
    train_df = train_df[train_df["Drug"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)].reset_index(drop=True)
    test_df = test_df[test_df["Drug"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)].reset_index(drop=True)
    print(f"Train pool: {len(train_df)}, Test: {len(test_df)}", flush=True)

    flip_y_min = flip_y_max = None
    if flip_labels:
        norm = json.load(open(ds_dir / "normalization.json"))
        flip_y_min, flip_y_max = norm["y_min"], norm["y_max"]
        print(f"[flip-labels] y_min={flip_y_min:.4f} y_max={flip_y_max:.4f}; "
              f"training on mirrored labels, evaluating against true test Y", flush=True)

    cache_dir = ds_dir / "sota" / f"feature_cache_{feature_set}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cached_features(df, tag):
        p = cache_dir / f"{tag}_features.npy"
        if p.exists():
            return sanitize(np.load(p))
        X = sanitize(build_features(df["Drug"]))
        np.save(p, X)
        return X

    print("Building test features...", flush=True)
    X_test = cached_features(test_df, "test")
    y_test = test_df["Y"].values

    print("Building train-pool features (cached once, subset per seed)...", flush=True)
    X_train_full = cached_features(train_df, "train")
    y_train_full = train_df["Y"].values
    if flip_labels:
        y_train_full = flip_y_min + flip_y_max - y_train_full
    train_id_to_idx = {did: i for i, did in enumerate(train_df["Drug_ID"])}

    out_dir = out_root / out_key
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in n_list:
        n_label = f"n{n}"
        out_path = out_dir / f"sota_{n_label}.json"
        results = []
        for seed in seeds:
            comp_path = seed_root / n_label / f"seed{seed}" / "compounds.csv"
            if not comp_path.exists():
                print(f"  {n_label}/seed{seed}: MISSING {comp_path}, skipping", flush=True)
                continue
            comp_df = pd.read_csv(comp_path)
            idxs = [train_id_to_idx[did] for did in comp_df["Drug_ID"] if did in train_id_to_idx]
            X_tr, y_tr = X_train_full[idxs], y_train_full[idxs]

            print(f"  {n_label}/seed{seed}: train_size={len(X_tr)}", flush=True)
            r = run_trial(seed, X_tr, y_tr, X_test, y_test, n_optuna_trials)
            print(f"  {n_label}/seed{seed}: MAE={r['mae']:.4f}, Spearman={r['spearman']:.4f}", flush=True)
            results.append(r)

        if not results:
            print(f"  no seeds available for {n_label}, skipping aggregation", flush=True)
            continue

        maes = [r["mae"] for r in results]
        rhos = [r["spearman"] for r in results]
        n_nan = int(np.sum(np.isnan(rhos)))
        if n_nan:
            print(f"  [{feature_set}] {n_label}: {n_nan}/{len(rhos)} seed(s) had NaN Spearman "
                  f"(degenerate fit) — excluded from spearman_mean/std", flush=True)
        summary = {
            "n_label": n_label, "feature_set": out_key,
            "n_train": int(len(X_tr)), "n_test": int(len(X_test)), "n_seeds": len(results),
            "n_seeds_valid_spearman": len(results) - n_nan,
            "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "spearman_mean": float(np.nanmean(rhos)), "spearman_std": float(np.nanstd(rhos)),
            "seeds": [{k: v for k, v in r.items() if k != "preds"} for r in results],
        }
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        preds_path = out_dir / f"per_seed_{n_label}.csv"
        with open(preds_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "Drug_ID", "Drug", "true_Y", "pred_Y"])
            for r in results:
                for did, smi, ty, py in zip(test_df["Drug_ID"], test_df["Drug"], y_test, r["preds"]):
                    writer.writerow([r["seed"], did, smi, ty, py])

        print(f"  [{feature_set}] {n_label}: Spearman={np.nanmean(rhos):.4f}±{np.nanstd(rhos):.4f}  -> {out_path}", flush=True)


def run_minimol(dataset, seed_root, n_list, seeds, out_root):
    print(f"\n{'='*70}\nMiniMol baseline (subprocess -> minimol_env)\n{'='*70}", flush=True)
    if not MINIMOL_ENV_PYTHON.exists():
        print(f"  SKIP: {MINIMOL_ENV_PYTHON} not found — minimol_env not set up.", flush=True)
        return
    import subprocess
    cmd = [
        str(MINIMOL_ENV_PYTHON), str(BASE / "scripts" / "fig2" / "minimol_scaffold_worker.py"),
        "--dataset", dataset,
        "--seed-root", str(seed_root),
        "--n-list", *[str(n) for n in n_list],
        "--seeds", *[str(s) for s in seeds],
        "--out-root", str(out_root / "minimol"),
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(BASE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ld50", choices=["ld50", "caco2", "solubility"])
    parser.add_argument("--seed-root", default=None,
                        help="Directory containing {n_label}/seed{s}/compounds.csv "
                             "(default: data/fig2_scaling/splits/{dataset}/scaffold_val)")
    parser.add_argument("--n-list", nargs="+", type=int, default=[50, 100, 200, 400, 800, 1600, 3200])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--out-root", default=None,
                        help="default: data/fig2_scaling/splits/{dataset}/sota_scaffold/results")
    parser.add_argument("--n-optuna-trials", type=int, default=20)
    parser.add_argument("--skip", nargs="+", default=[],
                        choices=["morgan_only", "admet_trix", "minimol"],
                        help="Baselines to skip this run.")
    parser.add_argument("--flip-labels", action="store_true",
                        help="Train on mirrored labels (y_min+y_max-Y), evaluate against true test Y. "
                             "Outputs to sign_flip_{feature_set}/ subdirectory.")
    args = parser.parse_args()

    ds_dir = BASE / "data" / "fig2_scaling" / "splits" / args.dataset
    seed_root = Path(args.seed_root) if args.seed_root else ds_dir / "scaffold_val"
    out_root = Path(args.out_root) if args.out_root else ds_dir / "sota_scaffold" / "results"
    out_root.mkdir(parents=True, exist_ok=True)

    if "morgan_only" not in args.skip:
        run_xgboost_baseline("morgan_only", ds_dir, seed_root, args.n_list, args.seeds, out_root, args.n_optuna_trials,
                             flip_labels=args.flip_labels)
    if "admet_trix" not in args.skip:
        run_xgboost_baseline("admet_trix", ds_dir, seed_root, args.n_list, args.seeds, out_root, args.n_optuna_trials,
                             flip_labels=args.flip_labels)
    if "minimol" not in args.skip:
        run_minimol(args.dataset, seed_root, args.n_list, args.seeds, out_root)

    print("\nALL DONE")


if __name__ == "__main__":
    main()
