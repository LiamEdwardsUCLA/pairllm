"""
MiniMol worker for an arbitrary seed-root (e.g. scaffold_val) — meant to
be run inside minimol_env (graphium/torch-geometric stack; incompatible with
the training/eval environment's torch build). Invoked as a subprocess by
run_sota_scaffold.py so the orchestrator itself can run in any environment
with xgboost/optuna. Works for any of ld50/caco2/solubility via --dataset
(each gets its own minimol_cache/ — embeddings aren't transferable across
datasets).

Two phases:
  1. Embed (skipped if cache already covers every compound needed): embeds
     the FULL train.csv (superset of every split's pool) once, so the same
     cache serves both the standard random split and any scaffold-disjoint
     split without re-embedding.
  2. Train a 3-layer MLP (matching the MiniMol paper) + Ridge head per
     (n_label, seed), reading compounds.csv from --seed-root/{n_label}/seed{s}/.

Usage (inside minimol_env):
    python scripts/minimol_scaffold_worker.py --dataset ld50 \
        --seed-root data/fig2_scaling/splits/ld50/scaffold_val \
        --n-list 50 100 200 400 800 1600 3200 --seeds 1 2 3 \
        --out-root data/fig2_scaling/splits/ld50/sota_scaffold/results/minimol
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

BATCH_SIZE = 256
DS_DIR = None    # set in main() once --dataset is known
CACHE_DIR = None


def load_valid_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            smi = r["Drug"]
            if not smi or smi == "Not Available":
                continue
            from rdkit import Chem
            if Chem.MolFromSmiles(smi) is None:
                continue
            rows.append(r)
    return rows


def embed_rows(model, rows, tag):
    emb_path = CACHE_DIR / f"{tag}_embeddings.npy"
    meta_path = CACHE_DIR / f"{tag}_meta.json"
    if emb_path.exists() and meta_path.exists():
        print(f"  [skip] {emb_path} exists", flush=True)
        return

    smiles = [r["Drug"] for r in rows]
    embeddings = np.zeros((len(smiles), 512), dtype=np.float32)
    failed = []

    for i in range(0, len(smiles), BATCH_SIZE):
        batch = smiles[i : i + BATCH_SIZE]
        try:
            out = model(batch)
            for j, e in enumerate(out):
                embeddings[i + j] = e.detach().cpu().numpy()
        except Exception as e:
            print(f"    batch {i}-{i+len(batch)} failed as a group ({e}); retrying one-by-one", flush=True)
            for j, smi in enumerate(batch):
                try:
                    e = model([smi])[0]
                    embeddings[i + j] = e.detach().cpu().numpy()
                except Exception as e2:
                    failed.append(i + j)
                    print(f"    compound {i+j} ({smi}) failed: {e2}", flush=True)
        print(f"  {tag}: {min(i+BATCH_SIZE, len(smiles))}/{len(smiles)}", flush=True)

    np.save(emb_path, embeddings)
    meta = {
        "drug_ids": [r["Drug_ID"] for r in rows],
        "smiles": smiles,
        "y": [float(r["Y"]) for r in rows],
        "failed_indices": failed,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"  wrote {emb_path} shape={embeddings.shape}  ({len(failed)} failed compounds)", flush=True)


def ensure_embeddings():
    """Embed the FULL train.csv (superset of every split's pool) + test.csv,
    cached under the 'train_pool' tag for compatibility with minimol_ld50.py."""
    need_train = not ((CACHE_DIR / "train_pool_embeddings.npy").exists() and
                       (CACHE_DIR / "train_pool_meta.json").exists())
    need_test = not ((CACHE_DIR / "test_embeddings.npy").exists() and
                      (CACHE_DIR / "test_meta.json").exists())
    if not need_train and not need_test:
        print("Embedding cache already complete, skipping embed phase.", flush=True)
        return

    print("Loading MiniMol...", flush=True)
    from minimol import Minimol
    model = Minimol()

    if need_train:
        print("\n== train (full train.csv — superset of every split) ==", flush=True)
        rows = load_valid_rows(DS_DIR / "train.csv")
        print(f"  {len(rows)} valid compounds", flush=True)
        embed_rows(model, rows, "train_pool")

    if need_test:
        print("\n== test ==", flush=True)
        rows = load_valid_rows(DS_DIR / "test.csv")
        print(f"  {len(rows)} valid compounds", flush=True)
        embed_rows(model, rows, "test")


def load_embedding_lookup(tag):
    embeddings = np.load(CACHE_DIR / f"{tag}_embeddings.npy")
    meta = json.load(open(CACHE_DIR / f"{tag}_meta.json"))
    return {smi: embeddings[i] for i, smi in enumerate(meta["smiles"])}


def load_compounds(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def build_xy(rows, emb_lookup, y_min, y_max):
    X, y, missing = [], [], 0
    for r in rows:
        smi = r["Drug"]
        if smi not in emb_lookup:
            missing += 1
            continue
        X.append(emb_lookup[smi])
        y.append((float(r["Y"]) - y_min) / (y_max - y_min) * 1000)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float64), missing


def fit_mlp(X_tr, y_tr, X_te, seed):
    """3-layer MLP (hidden=1024x3) with a small LR sweep, matching the MiniMol paper's head."""
    best_preds, best_val = None, float("-inf")
    for lr in [1e-3, 3e-4, 1e-4]:
        mlp = MLPRegressor(
            hidden_layer_sizes=(1024, 1024, 1024),
            activation="relu",
            max_iter=200,
            learning_rate_init=lr,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=seed,
        )
        mlp.fit(X_tr, y_tr)
        val_score = mlp.best_validation_score_
        if val_score > best_val:
            best_val = val_score
            best_preds = mlp.predict(X_te)
    return np.clip(best_preds, 0, 1000)


def train_and_eval(train_rows, test_X, test_y, train_lookup, seed, y_min, y_max):
    X_tr, y_tr, n_missing = build_xy(train_rows, train_lookup, y_min, y_max)
    if n_missing:
        print(f"    WARNING: {n_missing} train compounds missing embeddings", flush=True)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(test_X)

    mlp_preds = fit_mlp(X_tr_s, y_tr, X_te_s, seed)
    mlp_mae = float(np.abs(mlp_preds - test_y).mean())
    mlp_spear = float(stats.spearmanr(mlp_preds, test_y)[0])

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_s, y_tr)
    ridge_preds = np.clip(ridge.predict(X_te_s), 0, 1000)
    ridge_mae = float(np.abs(ridge_preds - test_y).mean())
    ridge_spear = float(stats.spearmanr(ridge_preds, test_y)[0])

    return {
        "n_train": len(y_tr),
        "mlp_spearman": mlp_spear, "mlp_mae": mlp_mae,
        "ridge_spearman": ridge_spear, "ridge_mae": ridge_mae,
    }


def main():
    global DS_DIR, CACHE_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ld50", choices=["ld50", "caco2", "solubility"])
    parser.add_argument("--seed-root", required=True,
                        help="Directory containing {n_label}/seed{s}/compounds.csv")
    parser.add_argument("--n-list", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    DS_DIR = Path("data/fig2_scaling/splits") / args.dataset
    CACHE_DIR = DS_DIR / "minimol_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ensure_embeddings()

    norm = json.loads((DS_DIR / "normalization.json").read_text())
    y_min, y_max = norm["y_min"], norm["y_max"]

    print("Loading embedding caches...", flush=True)
    train_lookup = load_embedding_lookup("train_pool")
    test_lookup = load_embedding_lookup("test")

    test_rows = load_compounds(DS_DIR / "test.csv")
    test_X, test_y, n_missing_test = build_xy(test_rows, test_lookup, y_min, y_max)
    print(f"  {len(train_lookup)} train embeddings, {len(test_lookup)} test embeddings, "
          f"{len(test_y)} test rows ({n_missing_test} missing)", flush=True)

    seed_root = Path(args.seed_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for n in args.n_list:
        n_label = f"n{n}"
        sota_json = out_root / f"sota_{n_label}.json"
        per_seed_json = out_root / f"per_seed_{n_label}.json"

        print(f"\n== {n_label} ==", flush=True)
        per_seed_results = {}
        for seed in args.seeds:
            comp_path = seed_root / n_label / f"seed{seed}" / "compounds.csv"
            if not comp_path.exists():
                print(f"  seed{seed}: MISSING {comp_path}, skipping", flush=True)
                continue
            train_rows = load_compounds(comp_path)
            result = train_and_eval(train_rows, test_X, test_y, train_lookup, seed, y_min, y_max)
            per_seed_results[f"seed{seed}"] = result
            print(f"  seed{seed}: n_train={result['n_train']}  "
                  f"mlp_spearman={result['mlp_spearman']:.4f}  ridge_spearman={result['ridge_spearman']:.4f}",
                  flush=True)

        if not per_seed_results:
            print(f"  no seeds available for {n_label}, skipping aggregation", flush=True)
            continue

        rhos = [v["mlp_spearman"] for v in per_seed_results.values()]
        sota_summary = {
            "n_label": n_label,
            "spearman_mean": float(np.mean(rhos)),
            "spearman_std": float(np.std(rhos)),
            "n_seeds": len(rhos),
            "method": "minimol+mlp",
        }
        with open(sota_json, "w") as f:
            json.dump(sota_summary, f, indent=2)
        with open(per_seed_json, "w") as f:
            json.dump(per_seed_results, f, indent=2)
        print(f"  wrote {sota_json}: mean={sota_summary['spearman_mean']:.4f} "
              f"std={sota_summary['spearman_std']:.4f} (n={sota_summary['n_seeds']})", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
