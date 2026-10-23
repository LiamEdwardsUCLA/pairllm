"""
Evaluate a pairwise (margin) model: predict a signed integer (label_B -
label_A) for sampled test-set pairs, aggregate into per-compound scores, and
compute Spearman rho against the true labels.

Three aggregations are computed from the predicted margins: margin-sum
("Thurstone"), Bradley-Terry on the predicted sign, and least-squares
recovery. Least-squares recovery (`spearman_lstsq`) is the number reported as
"pairwise" Spearman rho throughout the paper.

Usage:
    python scripts/eval_pairwise_ranking.py \
        --dataset ld50 \
        --model <merged_dir> \
        --test-csv data/fig2_scaling/splits/ld50/test.csv \
        --normalization data/fig2_scaling/splits/ld50/normalization.json \
        --train-csv data/fig2_scaling/splits/ld50/train_pool_scaffold.csv \
        --n-pairs 60000 \
        --output-json <results>/metrics.json
"""

import argparse
import csv
import json
import os
import re
import random
import sys
from collections import defaultdict

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr as sparse_lsqr
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import (
    LD50_PAIRWISE_MARGIN_PROMPT,
    LD50_PAIRWISE_MARGIN_PROMPT_NAME_ONLY,
    LD50_PAIRWISE_MARGIN_PROMPT_SMILES_ONLY,
    CACO2_PAIRWISE_MARGIN_PROMPT,
    SOLUBILITY_PAIRWISE_MARGIN_PROMPT,
)

MARGIN_PROMPTS = {
    "ld50":       LD50_PAIRWISE_MARGIN_PROMPT,
    "caco2":      CACO2_PAIRWISE_MARGIN_PROMPT,
    "solubility": SOLUBILITY_PAIRWISE_MARGIN_PROMPT,
}


def normalize(y, y_min, y_max):
    return round((float(y) - y_min) / (y_max - y_min) * 1000)


def denormalize(s, y_min, y_max):
    return s / 1000.0 * (y_max - y_min) + y_min


def fit_bradley_terry(match_list, n, max_iter=500):
    active = set()
    opponents = defaultdict(list)
    wins_count = defaultdict(int)
    for w, l in match_list:
        active.add(w); active.add(l)
        opponents[w].append(l)
        opponents[l].append(w)
        wins_count[w] += 1

    strengths = np.ones(n)
    active_list = list(active)
    for iteration in range(max_iter):
        new_strengths = strengths.copy()
        for i in active:
            exp_i = sum(strengths[i] / (strengths[i] + strengths[j]) for j in opponents[i])
            if exp_i > 0:
                new_strengths[i] = wins_count[i] / exp_i if wins_count[i] > 0 else 1e-6
        new_strengths /= new_strengths[active_list].mean()
        delta = np.abs(new_strengths[active_list] - strengths[active_list]).max()
        strengths = new_strengths
        if delta < 1e-6:
            print(f"BT converged at iteration {iteration + 1}")
            break
    return strengths, active


def thurstone_aggregate(pair_records, predictions, n):
    """Sum predicted margins per compound. Higher score = predicted more toxic."""
    scores = np.zeros(n)
    for (a, b), pred in zip(pair_records, predictions):
        if pred is None:
            continue
        # pred = label_b - label_a (positive when A more toxic)
        scores[a] += pred
        scores[b] -= pred
    return scores


def least_squares_recover(pair_records, predictions, n, anchor_mean):
    """
    Recover cardinal 0-1000 scores from predicted margins via least squares.

    Solves min ||As - b||  where:
      A[k, j] = +1, A[k, i] = -1  for pair (i, j) with prediction p_k ~ s_j - s_i
      last row enforces mean(s) = anchor_mean

    Returns a length-n score array on approximately the same 0-1000 scale.
    """
    valid = [(i, j, p) for (i, j), p in zip(pair_records, predictions) if p is not None]
    m = len(valid)
    A = lil_matrix((m + 1, n))
    b = np.zeros(m + 1)
    for k, (i, j, pred) in enumerate(valid):
        A[k, j] =  1.0
        A[k, i] = -1.0
        b[k]    = float(pred)
    # Mean constraint row
    A[m, :] = 1.0 / n
    b[m]    = anchor_mean
    result = sparse_lsqr(A.tocsr(), b, atol=1e-6, btol=1e-6)
    return result[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      default="ld50", choices=list(MARGIN_PROMPTS),
                        help="Dataset/task — selects prompt template.")
    parser.add_argument("--model",        default=None, help="Merged model path")
    parser.add_argument("--base-model",   default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter",      default=None)
    parser.add_argument("--test-csv",     required=True)
    parser.add_argument("--normalization",required=True)
    parser.add_argument("--n-pairs",      type=int, default=60_000)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--tp-size",      type=int, default=2)
    parser.add_argument("--max-lora-rank",type=int, default=64)
    parser.add_argument("--output-json",  default=None)
    parser.add_argument("--train-csv",    default=None,
                        help="Train CSV for anchor mean in LS recovery. "
                             "If omitted, falls back to 500.")
    parser.add_argument("--n-compounds-train", type=int, default=None,
                        help="Number of training compounds the model was trained on (bookkeeping).")
    parser.add_argument("--representation", default=None,
                        choices=["both", "names-only", "smiles-only"],
                        help="Input representation ablation, Figure 3 (LD50 only). "
                             "'both'=name+SMILES (default), 'names-only'=name only, "
                             "'smiles-only'=SMILES only.")
    args = parser.parse_args()

    if args.representation in ("names-only", "smiles-only") and args.dataset != "ld50":
        sys.exit("ERROR: --representation ablations are defined for --dataset ld50 only.")

    random.seed(args.seed)

    with open(args.normalization) as f:
        norm = json.load(f)
    y_min, y_max = norm["y_min"], norm["y_max"]

    compounds = []
    with open(args.test_csv) as f:
        for row in csv.DictReader(f):
            compounds.append({
                "Drug_ID": row["Drug_ID"],
                "smiles":  row["Drug"],
                "label":   normalize(row["Y"], y_min, y_max),
                "raw_Y":   float(row["Y"]),
            })

    n = len(compounds)
    print(f"Loaded {n} test compounds")
    print(f"Sampling {args.n_pairs} pairs (avg {args.n_pairs * 2 / n:.1f} comparisons per compound)")

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sampled_indices = random.sample(all_pairs, min(args.n_pairs, len(all_pairs)))
    pair_records = []
    for i, j in sampled_indices:
        if random.random() < 0.5:
            i, j = j, i
        pair_records.append((i, j))

    true_margins = [compounds[b]["label"] - compounds[a]["label"] for a, b in pair_records]

    # ── Inference ─────────────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    model_path = args.model or args.base_model
    tokenizer  = AutoTokenizer.from_pretrained(model_path)

    prompt_template = MARGIN_PROMPTS[args.dataset]

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=args.tp_size,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enable_lora=args.adapter is not None,
        max_lora_rank=args.max_lora_rank,
    )
    lora_request    = LoRARequest("ranking_adapter", 1, args.adapter) if args.adapter else None
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8)

    def _build_single_prompt(ca, cb):
        rep = args.representation
        if rep == "names-only":
            tmpl = LD50_PAIRWISE_MARGIN_PROMPT_NAME_ONLY
            fmt_kwargs = dict(drug_name_a=ca["Drug_ID"], drug_name_b=cb["Drug_ID"])
        elif rep == "smiles-only":
            tmpl = LD50_PAIRWISE_MARGIN_PROMPT_SMILES_ONLY
            fmt_kwargs = dict(smiles_a=ca["smiles"], smiles_b=cb["smiles"])
        else:  # "both" or None
            tmpl = prompt_template
            fmt_kwargs = dict(
                drug_name_a=ca["Drug_ID"], smiles_a=ca["smiles"],
                drug_name_b=cb["Drug_ID"], smiles_b=cb["smiles"],
            )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": tmpl.format(**fmt_kwargs)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    print(f"\n── Prompt construction ──")
    print(f"  representation : {args.representation or 'both (name+SMILES)'}")

    prompts = [_build_single_prompt(compounds[a], compounds[b]) for a, b in pair_records]

    # Print a sample prompt (first pair) so the user can verify construction
    _sep = "─" * 60
    print(f"\n{_sep}")
    print(f"SAMPLE PROMPT [pair 0]  A={compounds[pair_records[0][0]]['Drug_ID']}  B={compounds[pair_records[0][1]]['Drug_ID']}")
    print(_sep)
    print(prompts[0])
    print(_sep)

    print(f"Running inference on {len(prompts)} pairs...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    predictions = []
    n_invalid = 0
    for out in outputs:
        raw = out.outputs[0].text.strip()
        m = re.search(r"[+-]?\d+", raw)
        if m is None:
            n_invalid += 1
            predictions.append(None)
        else:
            predictions.append(max(-1000, min(1000, int(m.group()))))

    n_valid = len(predictions) - n_invalid

    # ── Accuracy ──────────────────────────────────────────────────────────────
    valid_pairs = [(p, g) for p, g in zip(predictions, true_margins) if p is not None]
    n_correct   = sum(1 for p, g in valid_pairs if (p > 0) == (g > 0) or (p == 0 and g == 0))
    sign_acc    = n_correct / n_valid if n_valid > 0 else 0.0
    mae         = sum(abs(p - g) for p, g in valid_pairs) / n_valid if n_valid > 0 else 0.0
    print(f"\nSign accuracy: {sign_acc:.4f} ({n_correct}/{n_valid} correct direction, {n_invalid} invalid)")
    print(f"MAE (predicted vs true margin): {mae:.1f}")

    # ── Aggregation → Spearman ────────────────────────────────────────────────
    true_labels = np.array([c["label"] for c in compounds])

    thurstone = thurstone_aggregate(pair_records, predictions, n)
    rho, pval = spearmanr(thurstone, -true_labels)
    print(f"\nSpearman rho (Thurstone): {rho:.4f}  (p={pval:.2e})")

    # BT on sign for comparison
    match_list = []
    for (a, b), pred in zip(pair_records, predictions):
        if pred is None:
            continue
        match_list.append((a, b) if pred > 0 else (b, a))
    bt_scores, active = fit_bradley_terry(match_list, n)
    rho_bt, _ = spearmanr(bt_scores, -true_labels)
    print(f"Spearman rho (BT on sign): {rho_bt:.4f}")

    # Least-squares recovery of cardinal 0-1000 scores — the reported metric
    if args.train_csv:
        train_labels = []
        with open(args.train_csv) as _f:
            for row in csv.DictReader(_f):
                train_labels.append(normalize(row["Y"], y_min, y_max))
        anchor_mean = float(np.mean(train_labels))
        anchor_source = "train"
    else:
        anchor_mean = 500.0
        anchor_source = "prior (500)"
    print(f"\nLS anchor mean: {anchor_mean:.1f} ({anchor_source})")
    ls_scores = least_squares_recover(pair_records, predictions, n, anchor_mean)
    rho_ls, _ = spearmanr(ls_scores, true_labels)
    pearson_ls, _ = pearsonr(ls_scores, true_labels)
    mae_ls = float(np.mean(np.abs(ls_scores - true_labels)))
    spread_ratio_ls = float(np.std(ls_scores) / np.std(true_labels)) if np.std(true_labels) > 0 else float("nan")
    true_native = denormalize(true_labels, y_min, y_max)
    pred_native = denormalize(ls_scores, y_min, y_max)
    mae_ls_native = float(np.mean(np.abs(pred_native - true_native)))
    print(f"Spearman rho (least squares): {rho_ls:.4f}")
    print(f"Pearson r  (least squares): {pearson_ls:.4f}")
    print(f"MAE (least squares, 0-1000 scale): {mae_ls:.1f}")
    print(f"Spread ratio (lstsq): {spread_ratio_ls:.3f}")
    print(f"MAE (least squares, native units): {mae_ls_native:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

        results = {
            "method": "pairwise",
            "task": "margin",
            "dataset": args.dataset,
            "n_compounds_train": args.n_compounds_train,
            "sign_accuracy": sign_acc,
            "margin_mae": mae,
            "n_correct": n_correct,
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "spearman_thurstone": float(rho),
            "spearman_bt_sign": float(rho_bt),
            "spearman_lstsq": float(rho_ls),
            "pearson_lstsq": float(pearson_ls),
            "lstsq_mae": mae_ls,
            "spread_ratio_lstsq": spread_ratio_ls,
            "lstsq_mae_native": mae_ls_native,
            "lstsq_anchor_mean": anchor_mean,
            "lstsq_anchor_source": anchor_source,
            "n_pairs": len(pair_records),
            "n_compounds": n,
            "adapter": args.adapter,
            "model": model_path,
            "representation": args.representation,
            "seed": args.seed,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {args.output_json}")

        # Per-compound JSONL
        agg_ranks  = n - (-thurstone).argsort().argsort()
        true_ranks = n - true_labels.argsort().argsort()
        ls_ranks   = n - (-ls_scores).argsort().argsort()
        compounds_path = args.output_json.replace(".json", "_compounds.jsonl")
        with open(compounds_path, "w") as f:
            for i, c in enumerate(compounds):
                row = {
                    "Drug_ID":     c["Drug_ID"],
                    "smiles":      c["smiles"],
                    "true_label":  int(c["label"]),
                    "true_Y":      c["raw_Y"],
                    "agg_score":   float(thurstone[i]),
                    "pred_rank":   int(agg_ranks[i]),
                    "true_rank":   int(true_ranks[i]),
                    "lstsq_score": float(ls_scores[i]),
                    "lstsq_rank":  int(ls_ranks[i]),
                }
                f.write(json.dumps(row) + "\n")
        print(f"Saved: {compounds_path}")


if __name__ == "__main__":
    main()
