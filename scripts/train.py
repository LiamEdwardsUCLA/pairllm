"""
LoRA SFT for pairwise (margin) or pointwise (digit) molecular property
prediction, used to train every model in every figure in this package —
only the CLI flags (model, dataset, schedule, LR/early-stopping) differ
between runs; see each figure's section in the README for the exact flags.

Labels are trained as literal digit tokens (e.g. "455" tokenizes digit-by-
digit) with a standard masked cross-entropy loss over the response span.
"""

import argparse
import json
import math
import os
import re
from datetime import datetime

# Per-run HF caches, keyed by a job id if running under a scheduler that sets
# one — avoids cache-lock contention when multiple runs start simultaneously
# on shared storage; harmless default of "local" otherwise.
_job_id = os.environ.get('SLURM_JOB_ID', 'local')
os.environ.setdefault("HF_HOME",          f"/tmp/hf_home_{_job_id}")
os.environ.setdefault("HF_DATASETS_CACHE", f"/tmp/hf_datasets_{_job_id}")

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainerCallback


# ── Collator ──────────────────────────────────────────────────────────────────

class PaddingCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            ids  = f["input_ids"]
            mask = f.get("attention_mask", [1] * len(ids))
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids":      torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels":         torch.tensor(labels),
        }


# ── Pairwise eval callback ────────────────────────────────────────────────────

class PairwiseEvalCallback(TrainerCallback):
    """
    Pairwise (margin task) val eval every eval_steps steps. Ground truth is a
    signed integer (e.g. "+312", "-47"); reports sign accuracy (direction) +
    mean absolute error, plus Spearman ρ via least-squares aggregation of the
    predicted margins into per-compound scores (requires idx_a/idx_b/label_a/
    label_b metadata in the val JSONL).
    """

    def __init__(self, val_file, tokenizer, eval_steps, batch_size=16,
                 patience=None, burnin_threshold=0.0, min_delta=0.005,
                 output_dir=None):
        self.tokenizer  = tokenizer
        self.eval_steps = eval_steps
        self.batch_size = batch_size
        self._patience          = patience
        self._burnin_threshold  = burnin_threshold
        self._min_delta         = min_delta
        self._best_rho          = float("-inf")
        self._n_no_improve      = 0
        self._burnin_reached    = False
        self._should_stop       = False
        self._output_dir        = output_dir

        records = [json.loads(l) for l in open(val_file)]
        self.ground_truth = [int(r["messages"][1]["content"].strip()) for r in records]

        # Build compound index for Spearman ρ via least-squares aggregation.
        compound_labels: dict[int, float] = {}
        for r in records:
            compound_labels[r["idx_a"]] = r["label_a"]
            compound_labels[r["idx_b"]] = r["label_b"]
        n_comp = max(compound_labels) + 1
        self._true_labels = np.array([compound_labels[i] for i in range(n_comp)], dtype=np.float64)
        self._pair_a_idx = np.array([r["idx_a"] for r in records], dtype=np.int32)
        self._pair_b_idx = np.array([r["idx_b"] for r in records], dtype=np.int32)

        self.prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": r["messages"][0]["content"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for r in records
        ]
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[PairwiseEvalCallback] {len(self.prompts)} val pairs from {val_file}", flush=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._run_eval(state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps == 0:
            self._run_eval(state, model)
            if self._patience is not None:
                base = model.module if hasattr(model, "module") else model
                device = next(base.parameters()).device
                import torch.distributed as dist
                if dist.is_initialized():
                    stop = torch.tensor(int(self._should_stop), dtype=torch.int32, device=device)
                    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                    if stop.item():
                        control.should_training_stop = True
                elif self._should_stop:
                    control.should_training_stop = True

    def _run_eval(self, state, model):
        if not state.is_world_process_zero:
            return

        base   = model.module if hasattr(model, "module") else model
        device = next(base.parameters()).device

        prev_padding_side           = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        base.eval()

        preds = []  # None = invalid
        with torch.no_grad():
            for i in range(0, len(self.prompts), self.batch_size):
                batch_prompts = self.prompts[i : i + self.batch_size]
                enc = self.tokenizer(
                    batch_prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=2048,
                ).to(device)
                out = base.generate(
                    **enc, max_new_tokens=8, do_sample=False,  # "+1000" is at most 5 tokens
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                prompt_len = enc["input_ids"].shape[1]
                for j in range(len(batch_prompts)):
                    text = self.tokenizer.decode(
                        out[j][prompt_len:], skip_special_tokens=True
                    ).strip()
                    m = re.search(r"[+-]?\d+", text)
                    preds.append(int(m.group()) if m else None)

        base.train()
        self.tokenizer.padding_side = prev_padding_side

        n_invalid = sum(1 for p in preds if p is None)
        valid_pairs = [(p, g) for p, g in zip(preds, self.ground_truth) if p is not None]
        n_valid = len(valid_pairs)

        # Sign accuracy: fraction where predicted direction matches true direction
        n_correct = sum(
            1 for p, g in valid_pairs
            if (p > 0) == (g > 0) or (p == 0 and g == 0)
        )
        sign_acc = n_correct / n_valid if n_valid > 0 else 0.0
        mae = sum(abs(p - g) for p, g in valid_pairs) / n_valid if n_valid > 0 else 0.0

        # Spearman ρ via least-squares aggregation
        spearman = float("nan")
        if n_valid >= 2:
            valid_mask = [p is not None for p in preds]
            pred_arr = np.array([p for p in preds if p is not None], dtype=np.float64)
            a_idx = self._pair_a_idx[valid_mask]
            b_idx = self._pair_b_idx[valid_mask]
            n_comp = len(self._true_labels)
            # Design matrix: row k => s[b_idx[k]] - s[a_idx[k]] ≈ pred_arr[k]
            A = np.zeros((len(pred_arr), n_comp), dtype=np.float64)
            A[np.arange(len(pred_arr)), b_idx] = +1.0
            A[np.arange(len(pred_arr)), a_idx] = -1.0
            scores, _, _, _ = np.linalg.lstsq(A, pred_arr, rcond=None)
            from scipy.stats import spearmanr
            rho, _ = spearmanr(scores, self._true_labels)
            spearman = float(rho)

        print(
            f"\n[pairwise_eval] step={state.global_step}"
            f"  spearman={spearman:.4f}  sign_acc={sign_acc:.3f}  mae={mae:.1f}"
            f"  ({n_correct}/{n_valid} correct direction, {n_invalid} invalid)\n",
            flush=True,
        )
        state.log_history.append({
            "step": state.global_step,
            "pairwise_val_spearman": spearman,
            "pairwise_val_sign_accuracy": sign_acc,
            "pairwise_val_mae": mae,
            "pairwise_val_n_correct": n_correct,
            "pairwise_val_n_invalid": n_invalid,
        })
        # Early stopping
        if self._patience is not None and not math.isnan(spearman):
            if spearman > self._burnin_threshold:
                self._burnin_reached = True
            if self._burnin_reached:
                if spearman > self._best_rho + self._min_delta:
                    self._best_rho = spearman
                    self._n_no_improve = 0
                    if self._output_dir is not None:
                        import shutil
                        best_path = os.path.join(self._output_dir, "best_checkpoint")
                        if os.path.exists(best_path):
                            shutil.rmtree(best_path)
                        base.save_pretrained(best_path)
                        self.tokenizer.save_pretrained(best_path)
                        print(f"[pairwise_eval] Best checkpoint saved → {best_path}", flush=True)
                else:
                    self._n_no_improve += 1
                    if self._n_no_improve >= self._patience:
                        print(
                            f"\n[pairwise_eval] EARLY STOP at step={state.global_step}"
                            f"  best_rho={self._best_rho:.4f}"
                            f"  no_improve={self._n_no_improve}\n",
                            flush=True,
                        )
                        self._should_stop = True


class PointwiseEvalCallback(TrainerCallback):
    """
    Pointwise val eval every eval_steps steps.
    Val JSONL format: {"messages": [user, assistant], "true_label": int}
    Reports Spearman ρ and MAE directly from per-compound predictions.
    """

    def __init__(self, val_file, tokenizer, eval_steps, batch_size=16,
                 patience=None, burnin_threshold=0.0, min_delta=0.005,
                 output_dir=None):
        self.tokenizer  = tokenizer
        self.eval_steps = eval_steps
        self.batch_size = batch_size
        self._patience          = patience
        self._burnin_threshold  = burnin_threshold
        self._min_delta         = min_delta
        self._best_rho          = float("-inf")
        self._n_no_improve      = 0
        self._burnin_reached    = False
        self._should_stop       = False
        self._output_dir        = output_dir

        records = [json.loads(l) for l in open(val_file)]
        self.true_labels = np.array([r["true_label"] for r in records], dtype=np.float64)
        self.prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": r["messages"][0]["content"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for r in records
        ]
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[PointwiseEvalCallback] {len(self.prompts)} val compounds from {val_file}", flush=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._run_eval(state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps == 0:
            self._run_eval(state, model)
            if self._patience is not None:
                base = model.module if hasattr(model, "module") else model
                device = next(base.parameters()).device
                import torch.distributed as dist
                if dist.is_initialized():
                    stop = torch.tensor(int(self._should_stop), dtype=torch.int32, device=device)
                    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                    if stop.item():
                        control.should_training_stop = True
                elif self._should_stop:
                    control.should_training_stop = True

    def _run_eval(self, state, model):
        if not state.is_world_process_zero:
            return

        base   = model.module if hasattr(model, "module") else model
        device = next(base.parameters()).device

        prev_padding_side           = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        base.eval()

        preds = []
        with torch.no_grad():
            for i in range(0, len(self.prompts), self.batch_size):
                batch_prompts = self.prompts[i : i + self.batch_size]
                enc = self.tokenizer(
                    batch_prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=2048,
                ).to(device)
                out = base.generate(
                    **enc, max_new_tokens=4, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                prompt_len = enc["input_ids"].shape[1]
                for j in range(len(batch_prompts)):
                    text = self.tokenizer.decode(
                        out[j][prompt_len:], skip_special_tokens=True
                    ).strip()
                    m = re.search(r"\d+", text)
                    preds.append(int(m.group()) if m else None)

        base.train()
        self.tokenizer.padding_side = prev_padding_side

        n_invalid = sum(1 for p in preds if p is None)
        valid_preds = [p for p in preds if p is not None]
        valid_true  = [t for p, t in zip(preds, self.true_labels) if p is not None]
        n_valid = len(valid_preds)

        spearman = float("nan")
        mae      = float("nan")
        if n_valid >= 2:
            from scipy.stats import spearmanr
            pred_arr = np.array(valid_preds, dtype=np.float64)
            true_arr = np.array(valid_true,  dtype=np.float64)
            rho, _   = spearmanr(pred_arr, true_arr)
            spearman = float(rho)
            mae      = float(np.mean(np.abs(pred_arr - true_arr)))

        print(
            f"\n[pointwise_eval] step={state.global_step}"
            f"  spearman={spearman:.4f}  mae={mae:.1f}"
            f"  ({n_valid} valid, {n_invalid} invalid)\n",
            flush=True,
        )
        state.log_history.append({
            "step":                    state.global_step,
            "pointwise_val_spearman":  spearman,
            "pointwise_val_mae":       mae,
            "pointwise_val_n_invalid": n_invalid,
        })
        # Early stopping
        if self._patience is not None and not math.isnan(spearman):
            if spearman > self._burnin_threshold:
                self._burnin_reached = True
            if self._burnin_reached:
                if spearman > self._best_rho + self._min_delta:
                    self._best_rho = spearman
                    self._n_no_improve = 0
                    if self._output_dir is not None:
                        import shutil
                        best_path = os.path.join(self._output_dir, "best_checkpoint")
                        if os.path.exists(best_path):
                            shutil.rmtree(best_path)
                        base.save_pretrained(best_path)
                        self.tokenizer.save_pretrained(best_path)
                        print(f"[pointwise_eval] Best checkpoint saved → {best_path}", flush=True)
                else:
                    self._n_no_improve += 1
                    if self._n_no_improve >= self._patience:
                        print(
                            f"\n[pointwise_eval] EARLY STOP at step={state.global_step}"
                            f"  best_rho={self._best_rho:.4f}"
                            f"  no_improve={self._n_no_improve}\n",
                            flush=True,
                        )
                        self._should_stop = True


# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="LoRA SFT for pairwise/pointwise molecular property prediction")
parser.add_argument("--model",         default="Qwen/Qwen3.5-9B")
parser.add_argument("--attn-implementation", default="flash_attention_2",
                    help="e.g. 'sdpa' for models FA2 doesn't support (Gemma-4's global "
                         "attention layers use head_dim=512, above FA2's 256 cap).")
parser.add_argument("--dataset",       required=True, help="Path to a pairwise or pointwise training JSONL")
parser.add_argument("--finetune-name", default="pairwise")
parser.add_argument("--output-dir",    default="checkpoints")
parser.add_argument("--max-seq-length", type=int,   default=4096)
parser.add_argument("--lora-r",         type=int,   default=64)
parser.add_argument("--lora-alpha",     type=int,   default=128)
parser.add_argument("--batch-size",     type=int,   default=2)
parser.add_argument("--grad-accum",     type=int,   default=8)
parser.add_argument("--epochs",         type=int,   default=4)
parser.add_argument("--max-steps",      type=int,   default=-1,
                    help="Fixed step budget, overrides --epochs (-1 = use epochs).")
parser.add_argument("--lr",             type=float, default=2e-4)
parser.add_argument("--warmup-ratio",   type=float, default=0.1)
parser.add_argument("--weight-decay",   type=float, default=0.01)
parser.add_argument("--save-total-limit", type=int,  default=4)
parser.add_argument("--logging-steps",  type=int,   default=10)
parser.add_argument("--seed",           type=int,   default=3407)
parser.add_argument("--lora-targets",   default="default",
                    help="LoRA target modules: 'default' (7 safe modules) or 'all-linear'.")
parser.add_argument("--save-steps",     type=int,   default=None,
                    help="Save a checkpoint every N steps. If set, switches to step-based saving (use for 1-epoch runs).")
parser.add_argument("--val-file",          default=None,
                    help="Pairwise val JSONL (with idx_a/idx_b); Spearman ρ evaluated every --eval-steps.")
parser.add_argument("--pointwise-val-file", default=None,
                    help="Pointwise val JSONL (with true_label); Spearman ρ evaluated every --eval-steps.")
parser.add_argument("--eval-steps",        type=int, default=None,
                    help="Run val eval every N steps (default: same as --save-steps). Applies to both val files.")
parser.add_argument("--patience",                 type=int,   default=None,
                    help="Early stopping patience: stop after N eval steps with no improvement. None = disabled.")
parser.add_argument("--early-stopping-burnin",    type=float, default=0.0,
                    help="Don't activate early stopping until val_spearman first exceeds this threshold.")
parser.add_argument("--early-stopping-min-delta", type=float, default=0.005,
                    help="Minimum improvement over best to count as a new best (default 0.005).")
parser.add_argument("--random-init",       action="store_true", default=False,
                    help="Initialize model with random weights (same config/architecture as "
                         "--model) instead of loading pretrained weights. For the Figure 3 "
                         "pretraining-collapse ablation.")
args = parser.parse_args()

_DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_lora_target_modules = (
    "all-linear"       if args.lora_targets == "all-linear"
    else _DEFAULT_TARGETS if args.lora_targets == "default"
    else args.lora_targets.split(",")
)

local_rank = int(os.environ.get("LOCAL_RANK", 0))

tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

if args.random_init:
    print(f"[random-init] Building model from {args.model}'s config with random weights "
          f"(no pretrained weights loaded), seeded with --seed={args.seed} for reproducibility ...")
    # Seed immediately before from_config so the random weights are deterministic
    # from args.seed alone — merge.py --random-init reconstructs the identical
    # backbone at eval time by seeding the same way before its own from_config call.
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(f"cuda:{local_rank}")
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": local_rank},
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
model.enable_input_require_grads()

model = get_peft_model(model, LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=_lora_target_modules,
    lora_dropout=0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
))
model.print_trainable_parameters()

_probe = tokenizer.apply_chat_template(
    [{"role": "user", "content": "U"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
if "<|im_start|>assistant\n" in _probe:
    # Qwen's chat template: the assistant-turn marker is directly present in
    # a single-message, add_generation_prompt=True render.
    _user_suffix = _probe.split("U", 1)[1].split("<|im_start|>assistant\n")[0]
    _response_template = f"{_user_suffix}<|im_start|>assistant\n"
else:
    # Generic fallback for any other chat format (e.g. Gemma's turn tokens).
    # NOT derived from the add_generation_prompt=True probe above — some chat
    # templates (Gemma-4) insert generation-prompt-only scaffolding (an empty
    # thinking-channel placeholder) that does not appear in a real multi-turn
    # render, which would make the derived template unfindable in actual
    # training examples. Instead, render a real 2-turn conversation with a
    # unique marker standing in for the assistant's content, and take
    # everything between the end of the user's message and the marker — this
    # is exactly the scaffold text that precedes the answer in every real
    # training example, regardless of chat format.
    _marker = "XYZ_RESPONSE_TEMPLATE_PROBE_MARKER"
    _full = tokenizer.apply_chat_template(
        [{"role": "user", "content": "U"}, {"role": "assistant", "content": _marker}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    _before_marker = _full.split(_marker, 1)[0]
    _response_template = _before_marker[_before_marker.index("U") + 1:]
_response_template_ids = tokenizer.encode(_response_template, add_special_tokens=False)
_template_len = len(_response_template_ids)
if local_rank == 0:
    print(f"[train] response template: {repr(_response_template)} → ids={_response_template_ids}", flush=True)

collator = PaddingCollator(tokenizer.pad_token_id)


def tokenize_and_mask(examples):
    """Mask everything up to and including the response template, so the loss
    is computed only over the assistant's answer span."""
    result = {"input_ids": [], "attention_mask": [], "labels": []}
    n_found, n_missing = 0, 0
    for convo in examples["messages"]:
        text = tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        enc = tokenizer(text, truncation=True, max_length=args.max_seq_length,
                        add_special_tokens=False)
        ids = enc["input_ids"]
        lbl = list(ids)

        found = False
        for pos in range(len(ids) - _template_len + 1):
            if ids[pos : pos + _template_len] == _response_template_ids:
                for i in range(pos + _template_len):
                    lbl[i] = -100
                found = True
                break
        if not found:
            lbl = [-100] * len(ids)
            n_missing += 1
        else:
            n_found += 1

        result["input_ids"].append(ids)
        result["attention_mask"].append(enc["attention_mask"])
        result["labels"].append(lbl)

    if n_missing > 0:
        print(f"[train] WARNING: response template not found in {n_missing} samples "
              f"(found in {n_found})", flush=True)
    return result


raw_records = [json.loads(l) for l in open(args.dataset)]
dataset = load_dataset("json", data_files={"train": args.dataset}, split="train")
dataset = dataset.map(tokenize_and_mask, batched=True, remove_columns=["messages"])
if local_rank == 0:
    print(f"[train] dataset tokenized: {len(dataset)} samples", flush=True)

job_id     = os.environ.get("SLURM_JOB_ID", os.environ.get("MASTER_PORT", datetime.now().strftime("%Y%m%d-%H%M%S")))
output_dir = os.path.join(args.output_dir, f"{args.finetune_name}-{job_id}")
os.makedirs(output_dir, exist_ok=True)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    data_collator=collator,
    args=SFTConfig(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        ddp_find_unused_parameters=False,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps or 1,
        save_total_limit=args.save_total_limit,
        output_dir=output_dir,
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        packing=False,
        max_length=args.max_seq_length,
    ),
)

_eval_steps = args.eval_steps or args.save_steps or 1
if args.val_file:
    trainer.add_callback(PairwiseEvalCallback(
        args.val_file, tokenizer, eval_steps=_eval_steps,
        patience=args.patience,
        burnin_threshold=args.early_stopping_burnin,
        min_delta=args.early_stopping_min_delta,
        output_dir=output_dir,
    ))
if args.pointwise_val_file:
    trainer.add_callback(PointwiseEvalCallback(
        args.pointwise_val_file, tokenizer, eval_steps=_eval_steps,
        patience=args.patience,
        burnin_threshold=args.early_stopping_burnin,
        min_delta=args.early_stopping_min_delta,
        output_dir=output_dir,
    ))

trainer.train()

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[train] adapter saved to {output_dir}", flush=True)

    # Save training metadata for reproducibility and result bookkeeping
    meta = {
        "finetune_name":    args.finetune_name,
        "base_model":       args.model,
        "dataset":          args.dataset,
        "output_dir":       output_dir,
        "lora_r":           args.lora_r,
        "lora_alpha":       args.lora_alpha,
        "lora_targets":     args.lora_targets,
        "batch_size":       args.batch_size,
        "grad_accum":       args.grad_accum,
        "effective_batch":  args.batch_size * args.grad_accum,
        "epochs":           args.epochs,
        "max_steps":        args.max_steps,
        "lr":               args.lr,
        "warmup_ratio":     args.warmup_ratio,
        "seed":             args.seed,
        "random_init":      args.random_init,
        "n_train_records":  len(raw_records),
        "val_file":         args.val_file,
        "job_id":           _job_id,
    }
    meta_path = os.path.join(output_dir, "training_metadata.json")
    with open(meta_path, "w") as _f:
        json.dump(meta, _f, indent=2)
    print(f"[train] metadata saved to {meta_path}", flush=True)
