"""
Build a scaffold-disjoint train/val split. Works for any of
ld50/caco2/solubility via --dataset.

Compounds are grouped by Bemis-Murcko scaffold and whole scaffold groups are
assigned to train or val, so val compounds share no core scaffold with
train — a random split could put near-identical analogs on both sides of
train/val, making validation look better-correlated with generalization than
it really is.

Steps:
  1. Load the full compound pool CSV for the dataset.
  2. Compute each compound's Bemis-Murcko scaffold (RDKit).
  3. Group by scaffold; greedily assign whole groups to val or pool, largest
     groups first, always adding to whichever side is currently furthest
     below its target size (standard scaffold-split bin-packing).
  4. Write train_pool_scaffold.csv / val_scaffold.csv and
     val_scaffold_{pairwise,pointwise}.jsonl.
  5. Sample N/seed compound subsets from train_pool_scaffold into
     data/fig2_scaling/splits/{dataset}/scaffold_val/{n_label}/seed{s}/,
     with pairwise_train.jsonl + pointwise_train.jsonl.

Usage:
    python scripts/build_scaffold_split.py --dataset ld50
    python scripts/build_scaffold_split.py --dataset caco2 --n-list 50 100 200 400 --seeds 1 2 3
    python scripts/build_scaffold_split.py --dataset solubility --n-list 50 100 200 400 800 1600 3200 --seeds 1 2 3
"""

import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts import (
    LD50_PAIRWISE_MARGIN_PROMPT, LD50_PROMPT_DIGIT_ONLY,
    CACO2_PAIRWISE_MARGIN_PROMPT, CACO2_PROMPT_DIGIT_ONLY,
    SOLUBILITY_PAIRWISE_MARGIN_PROMPT, SOLUBILITY_PROMPT_DIGIT_ONLY,
)

DATASET_CFG = {
    "ld50":       {"prompt_pairwise": LD50_PAIRWISE_MARGIN_PROMPT,       "prompt_pointwise": LD50_PROMPT_DIGIT_ONLY},
    "caco2":      {"prompt_pairwise": CACO2_PAIRWISE_MARGIN_PROMPT,      "prompt_pointwise": CACO2_PROMPT_DIGIT_ONLY},
    "solubility": {"prompt_pairwise": SOLUBILITY_PAIRWISE_MARGIN_PROMPT, "prompt_pointwise": SOLUBILITY_PROMPT_DIGIT_ONLY},
}

BASE = Path(__file__).resolve().parent.parent.parent
MAX_PAIRS = 100_000

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="ld50", choices=list(DATASET_CFG))
parser.add_argument("--n-list", nargs="+", type=int, default=[100, 200, 400, 800])
parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
parser.add_argument("--val-frac", type=float, default=0.10)
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

D_DIR = BASE / "data" / "fig2_scaling" / "splits" / args.dataset
PAIRWISE_PROMPT = DATASET_CFG[args.dataset]["prompt_pairwise"]
POINTWISE_PROMPT = DATASET_CFG[args.dataset]["prompt_pointwise"]


def load_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(rows, path, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_normalization(path):
    with open(path) as f:
        norm = json.load(f)
    return norm["y_min"], norm["y_max"]


def normalize(y, y_min, y_max):
    return round((float(y) - y_min) / (y_max - y_min) * 1000)


def row_to_compound(row, y_min, y_max):
    return {"Drug_ID": row["Drug_ID"], "smiles": row["Drug"],
            "label": normalize(row["Y"], y_min, y_max)}


def murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold) if scaffold is not None else ""


def scaffold_split(rows, val_frac, seed):
    """Greedy scaffold-disjoint split: group by scaffold, sort groups largest
    first, assign each whole group to whichever side is furthest below its
    target count. Rows with no valid scaffold (parse failure) fall back to
    their own singleton group, so they never get silently dropped."""
    groups = {}
    n_unparseable = 0
    for row in rows:
        scaf = murcko_scaffold(row["Drug"])
        if scaf is None:
            n_unparseable += 1
            scaf = f"__unparseable__{row['Drug_ID']}"
        groups.setdefault(scaf, []).append(row)
    if n_unparseable:
        print(f"  [warn] {n_unparseable} compounds failed to parse with RDKit; "
              f"kept as their own singleton scaffold group")

    group_list = sorted(groups.values(), key=len, reverse=True)
    rng = random.Random(seed)
    rng.shuffle(group_list)  # break ties among equal-size groups deterministically per seed

    n = len(rows)
    target_val = round(n * val_frac)
    val_rows, pool_rows = [], []
    for group in group_list:
        # Add to val until it reaches target; everything else goes to pool.
        if len(val_rows) < target_val:
            val_rows.extend(group)
        else:
            pool_rows.extend(group)
    print(f"  Scaffold split: {len(groups)} unique scaffolds, "
          f"{len(val_rows)} val / {len(pool_rows)} pool "
          f"(target val={target_val}, {val_frac:.0%})")
    return val_rows, pool_rows


def make_pair_record(a, b, prompt_template, rng):
    if rng.random() < 0.5:
        a, b = b, a
    diff = b["label"] - a["label"]
    response = f"+{diff}" if diff > 0 else str(diff)
    return {
        "messages": [
            {"role": "user", "content": prompt_template.format(
                drug_name_a=a["Drug_ID"], smiles_a=a["smiles"],
                drug_name_b=b["Drug_ID"], smiles_b=b["smiles"])},
            {"role": "assistant", "content": response},
        ],
        "label_a": a["label"], "label_b": b["label"],
        "margin": abs(a["label"] - b["label"]),
    }


def make_pointwise_record(c, prompt_template):
    return {
        "messages": [
            {"role": "user", "content": prompt_template.format(
                drug_name=c["Drug_ID"], smiles=c["smiles"])},
            {"role": "assistant", "content": str(c["label"])},
        ],
        "true_label": c["label"],
    }


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def build_pairwise_jsonl(compounds, prompt, seed, max_pairs, out_path):
    rng = random.Random(seed)
    all_idx_pairs = list(itertools.combinations(range(len(compounds)), 2))
    if len(all_idx_pairs) > max_pairs:
        all_idx_pairs = rng.sample(all_idx_pairs, max_pairs)
    records = [make_pair_record(compounds[i], compounds[j], prompt, rng)
               for i, j in all_idx_pairs]
    write_jsonl(records, out_path)
    return len(records)


def build_val_pairwise_jsonl(compounds, prompt, seed, max_pairs, out_path):
    rng = random.Random(seed)
    all_idx_pairs = list(itertools.combinations(range(len(compounds)), 2))
    if len(all_idx_pairs) > max_pairs:
        all_idx_pairs = rng.sample(all_idx_pairs, max_pairs)
    records = []
    for i, j in all_idx_pairs:
        a, b = compounds[i], compounds[j]
        ia, ib = i, j
        if rng.random() < 0.5:
            a, b = b, a
            ia, ib = j, i
        diff = b["label"] - a["label"]
        response = f"+{diff}" if diff > 0 else str(diff)
        records.append({
            "messages": [
                {"role": "user", "content": prompt.format(
                    drug_name_a=a["Drug_ID"], smiles_a=a["smiles"],
                    drug_name_b=b["Drug_ID"], smiles_b=b["smiles"])},
                {"role": "assistant", "content": response},
            ],
            "label_a": a["label"], "label_b": b["label"],
            "margin": abs(a["label"] - b["label"]), "idx_a": ia, "idx_b": ib,
        })
    write_jsonl(records, out_path)
    return len(records)


def build_pointwise_jsonl(compounds, prompt, out_path):
    records = [make_pointwise_record(c, prompt) for c in compounds]
    write_jsonl(records, out_path)
    return len(records)


def main():
    norm_path = D_DIR / "normalization.json"
    y_min, y_max = load_normalization(norm_path)
    print(f"{args.dataset}  y_min={y_min:.4f}  y_max={y_max:.4f}")

    train_csv = D_DIR / "train.csv"
    val_csv = D_DIR / "val_scaffold.csv"
    pool_csv = D_DIR / "train_pool_scaffold.csv"

    if val_csv.exists() and pool_csv.exists() and not args.force:
        # Frozen split shipped with this repo — train.csv (raw TDC download) not needed.
        val_rows = load_csv_rows(val_csv)
        pool_rows = load_csv_rows(pool_csv)
        print(f"  Reusing existing split: {len(pool_rows)} pool + {len(val_rows)} val")
    else:
        if not train_csv.exists():
            sys.exit(
                f"ERROR: {train_csv} not found. Rebuilding the split from scratch needs the raw "
                f"TDC training CSV (columns Drug_ID,Drug,Y). To reproduce the paper, use the "
                f"frozen split already in {D_DIR} and omit --force."
            )
        all_rows = load_csv_rows(train_csv)
        val_rows, pool_rows = scaffold_split(all_rows, args.val_frac, seed=0)
        write_csv(val_rows, val_csv, fieldnames=["Drug_ID", "Drug", "Y"])
        write_csv(pool_rows, pool_csv, fieldnames=["Drug_ID", "Drug", "Y"])

    val_compounds = [row_to_compound(r, y_min, y_max) for r in val_rows]
    val_pw_path = D_DIR / "val_scaffold_pairwise.jsonl"
    val_pt_path = D_DIR / "val_scaffold_pointwise.jsonl"

    if not val_pw_path.exists() or args.force:
        n_vp = build_val_pairwise_jsonl(val_compounds, PAIRWISE_PROMPT,
                                         seed=0, max_pairs=5000, out_path=val_pw_path)
        print(f"  val_scaffold_pairwise.jsonl: {n_vp:,} pairs from {len(val_compounds)} compounds")
    else:
        print(f"  val_scaffold_pairwise.jsonl: exists ({sum(1 for _ in open(val_pw_path)):,} pairs)")

    if not val_pt_path.exists() or args.force:
        n_vt = build_pointwise_jsonl(val_compounds, POINTWISE_PROMPT, val_pt_path)
        print(f"  val_scaffold_pointwise.jsonl: {n_vt} records")
    else:
        print(f"  val_scaffold_pointwise.jsonl: exists ({sum(1 for _ in open(val_pt_path))} records)")

    scaffold_val_dir = D_DIR / "scaffold_val"
    for n in args.n_list:
        if n > len(pool_rows):
            print(f"  n{n}: SKIP — exceeds pool size ({len(pool_rows)} compounds available)")
            continue
        n_label = f"n{n}"
        for seed in args.seeds:
            seed_dir = scaffold_val_dir / n_label / f"seed{seed}"
            comp_path = seed_dir / "compounds.csv"
            pw_path = seed_dir / "pairwise_train.jsonl"
            pt_path = seed_dir / "pointwise_train.jsonl"

            if comp_path.exists() and not args.force:
                compounds = [row_to_compound(r, y_min, y_max) for r in load_csv_rows(comp_path)]
            else:
                rng = random.Random(seed)
                sampled_rows = rng.sample(pool_rows, n)
                compounds = [row_to_compound(r, y_min, y_max) for r in sampled_rows]
                write_csv(sampled_rows, comp_path, fieldnames=["Drug_ID", "Drug", "Y"])

            if pw_path.exists() and not args.force:
                status_pw = f"exists ({sum(1 for _ in open(pw_path)):,})"
            else:
                n_written = build_pairwise_jsonl(compounds, PAIRWISE_PROMPT,
                                                  seed, MAX_PAIRS, pw_path)
                status_pw = f"wrote {n_written:,}"

            if pt_path.exists() and not args.force:
                status_pt = f"exists ({len(compounds):,})"
            else:
                n_pt = build_pointwise_jsonl(compounds, POINTWISE_PROMPT, pt_path)
                status_pt = f"wrote {n_pt:,}"

            print(f"  scaffold_val/{n_label}/seed{seed}: {len(compounds):,} compounds | "
                  f"pairwise={status_pw} | pointwise={status_pt}")

    print("\nDone.")


if __name__ == "__main__":
    main()
