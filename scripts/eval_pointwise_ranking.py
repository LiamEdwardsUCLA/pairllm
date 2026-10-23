"""
Evaluate a pointwise (digit-prediction) model — works for any of
ld50/caco2/solubility via --dataset. Runs inference via vLLM, parses digit
predictions, and computes Spearman ρ — the same metric used by
eval_pairwise_ranking.py so results are directly comparable.

Usage:
    python scripts/eval_pointwise_ranking.py \
        --dataset ld50 \
        --model <merged_dir> \
        --test-csv data/fig2_scaling/splits/ld50/test.csv \
        --normalization data/fig2_scaling/splits/ld50/normalization.json \
        --output-json <results>/metrics.json
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import (
    LD50_PROMPT_DIGIT_ONLY,
    LD50_PROMPT_DIGIT_ONLY_NAME_ONLY, LD50_PROMPT_DIGIT_ONLY_SMILES_ONLY,
    SOLUBILITY_PROMPT_DIGIT_ONLY,
    CACO2_PROMPT_DIGIT_ONLY,
)

POINTWISE_PROMPTS = {
    "ld50":       LD50_PROMPT_DIGIT_ONLY,
    "solubility": SOLUBILITY_PROMPT_DIGIT_ONLY,
    "caco2":      CACO2_PROMPT_DIGIT_ONLY,
}


def extract_predicted_label(text):
    """Extract a predicted integer label from model output.

    Tries strict formats first (JSON, bare integer), then falls back to the
    first digit sequence in the output — matching the PointwiseEvalCallback's
    re.search(r'\\d+', text) so that train-val and test-eval are consistent.
    """
    try:
        parsed = json.loads(text)
        if "predicted_label" in parsed:
            return int(parsed["predicted_label"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = re.search(r'"predicted_label"\s*:\s*"?(\d+)"?', text)
    if m:
        return int(m.group(1))
    # Bare integer (whole output is just a number)
    m = re.search(r'^\s*(\d+)\s*$', text.strip())
    if m:
        return int(m.group(1))
    # Fallback: first number anywhere in the output (matches val-eval behavior)
    m = re.search(r'\d+', text)
    if m:
        return int(m.group())
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",              required=True)
    parser.add_argument("--test-csv",           required=True)
    parser.add_argument("--normalization",      required=True)
    parser.add_argument("--tp-size",            type=int, default=2)
    parser.add_argument("--output-json",        default=None)
    parser.add_argument("--dataset",            default="ld50",
                        choices=["ld50", "caco2", "solubility"])
    parser.add_argument("--seed",               type=int, default=None,
                        help="Random seed used for compound sampling (for bookkeeping only).")
    parser.add_argument("--n-compounds-train",  type=int, default=None,
                        help="Number of training compounds the model was trained on.")
    parser.add_argument("--representation", default=None, choices=["both", "names-only", "smiles-only"],
                        help="Input representation ablation, Figure 3 (LD50 only). "
                             "'both'=name+SMILES (default), 'names-only'=name only, "
                             "'smiles-only'=SMILES only.")
    args = parser.parse_args()

    if args.representation in ("names-only", "smiles-only") and args.dataset != "ld50":
        sys.exit("ERROR: --representation ablations are defined for --dataset ld50 only.")

    with open(args.normalization) as f:
        norm = json.load(f)
    y_min, y_max = norm["y_min"], norm["y_max"]

    def normalize(y):
        return round((float(y) - y_min) / (y_max - y_min) * 1000)

    def denormalize(label):
        return label / 1000 * (y_max - y_min) + y_min

    prompt_base = POINTWISE_PROMPTS[args.dataset]

    compounds = []
    with open(args.test_csv) as f:
        for row in csv.DictReader(f):
            compounds.append({
                "Drug_ID": row["Drug_ID"],
                "smiles":  row["Drug"],
                "label":   normalize(row["Y"]),
                "raw_Y":   float(row["Y"]),
            })

    print(f"Loaded {len(compounds)} test compounds", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=args.tp_size,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=6)

    prompts = []
    for c in compounds:
        if args.representation == "names-only":
            content = LD50_PROMPT_DIGIT_ONLY_NAME_ONLY.format(drug_name=c["Drug_ID"])
        elif args.representation == "smiles-only":
            content = LD50_PROMPT_DIGIT_ONLY_SMILES_ONLY.format(smiles=c["smiles"])
        else:
            content = prompt_base.format(
                drug_name=c["Drug_ID"], smiles=c["smiles"],
            )
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))

    print(f"Running inference on {len(prompts)} compounds...", flush=True)
    outputs = llm.generate(prompts, sampling_params)

    records = []
    n_invalid = 0
    for c, out in zip(compounds, outputs):
        raw = out.outputs[0].text
        pred = extract_predicted_label(raw)
        if pred is None:
            n_invalid += 1
            pred = 500
        records.append({
            "Drug_ID":   c["Drug_ID"],
            "smiles":    c["smiles"],
            "true":      c["label"],
            "true_Y":    c["raw_Y"],
            "predicted": max(0, min(1000, pred)),
            "pred_Y":    denormalize(max(0, min(1000, pred))),
            "invalid":   pred is None,
        })

    preds_arr = np.array([r["predicted"] for r in records], dtype=float)
    trues_arr = np.array([r["true"]      for r in records], dtype=float)

    print(f"Invalid outputs: {n_invalid}/{len(compounds)}", flush=True)

    spearman     = float(spearmanr(preds_arr, trues_arr).statistic)
    pearson      = float(pearsonr(preds_arr, trues_arr).statistic)
    mae          = float(np.mean(np.abs(preds_arr - trues_arr)))
    spread_ratio = float(np.std(preds_arr) / np.std(trues_arr)) if np.std(trues_arr) > 0 else float("nan")

    rng = np.random.default_rng(42)
    n = len(preds_arr)
    n_sample = min(10_000, n * (n - 1) // 2)
    idx_i = rng.integers(0, n, n_sample * 2)
    idx_j = rng.integers(0, n, n_sample * 2)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask][:n_sample], idx_j[mask][:n_sample]
    sign_acc = float(np.mean(
        np.sign(preds_arr[idx_i] - preds_arr[idx_j]) ==
        np.sign(trues_arr[idx_i] - trues_arr[idx_j])
    ))

    print(f"Spearman ρ: {spearman:.4f}  Pearson: {pearson:.4f}  MAE: {mae:.1f}  spread: {spread_ratio:.3f}  sign_acc: {sign_acc:.4f}", flush=True)

    result = {
        "method":            "pointwise",
        "dataset":           args.dataset,
        "seed":              args.seed,
        "n_compounds_train": args.n_compounds_train,
        "model":             args.model,
        "representation":    args.representation,
        "spearman_lstsq":    spearman,
        "pearson":           pearson,
        "mae":               mae,
        "spread_ratio":      spread_ratio,
        "sign_accuracy":     sign_acc,
        "n_compounds":       len(records),
        "n_invalid":         n_invalid,
        "predictions":       records,
    }

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2))
        print(f"Saved to {args.output_json}", flush=True)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
