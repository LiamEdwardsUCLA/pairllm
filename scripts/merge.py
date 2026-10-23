"""
Merge a LoRA adapter into the Qwen3.5 base model and save for vLLM inference.

Loads the base model via the text-only Qwen3_5ForCausalLM class specifically
(not AutoModelForCausalLM, which resolves Qwen3_5Config to the composite VL
class Qwen3_5ForConditionalGeneration and saves weights under a
language_model.* prefix that doesn't match the adapter's key structure).

Usage:
    python scripts/merge.py \
        --adapter /path/to/checkpoint-800 \
        --base-model Qwen/Qwen3.5-9B \
        --output-dir merged/ \
        --output-name my-merged-model
"""

import argparse
import json
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", required=True, help="Path to LoRA adapter checkpoint directory")
parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
parser.add_argument("--output-name", default="qwen3.5-9b-merged")
parser.add_argument("--output-dir", default="merged_models")
parser.add_argument("--skip-sanity-check", action="store_true")
parser.add_argument("--random-init", action="store_true",
                    help="Reconstruct the same random (untrained) backbone the adapter was "
                         "trained against, seeded with --seed, instead of loading pretrained "
                         "weights from --base-model. --base-model is still used for config/"
                         "tokenizer/visual+mtp weights (architecture only, never the LM weights).")
parser.add_argument("--seed", type=int, default=None,
                    help="Required with --random-init — must match the --seed train.py used "
                         "(torch.manual_seed is called identically in both places).")
args = parser.parse_args()
if args.random_init and args.seed is None:
    parser.error("--random-init requires --seed (read it from the adapter's training_metadata.json)")

output_path = Path(args.output_dir) / args.output_name

try:
    from transformers import Qwen3_5ForCausalLM  # type: ignore
    print(f"Using Qwen3_5ForCausalLM (text-only class)", flush=True)
    loader = Qwen3_5ForCausalLM
except ImportError:
    from transformers import AutoModelForCausalLM
    print(f"WARNING: Qwen3_5ForCausalLM not available, falling back to AutoModelForCausalLM", flush=True)
    loader = AutoModelForCausalLM

if args.random_init:
    from transformers import AutoConfig, AutoModelForCausalLM
    print(f"[random-init] Reconstructing random backbone from {args.base_model}'s config, "
          f"seeded with --seed={args.seed} (no pretrained weights loaded) ...", flush=True)
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    # AutoModelForCausalLM.from_config() here, not the Qwen3_5ForCausalLM
    # class used directly in the pretrained-loading branch below:
    # Qwen3_5ForCausalLM's __init__ expects a flat text config and errors on
    # the composite VL config AutoConfig.from_pretrained returns for this
    # checkpoint. AutoModelForCausalLM.from_config() resolves the composite
    # config to the correct text-only architecture (8.95B params).
    base = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")
else:
    print(f"Loading base model from {args.base_model} ...", flush=True)
    base = loader.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

# Capture base logits before adapter injection for sanity check
base_logits = None
tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
if not args.skip_sanity_check:
    sample = tokenizer("The synergy score is", return_tensors="pt").to(base.device)
    with torch.no_grad():
        base_logits = base(**sample).logits[0, -1].detach().clone()

print(f"Applying adapter from {args.adapter} ...", flush=True)
peft_model = PeftModel.from_pretrained(base, args.adapter)

if not args.skip_sanity_check:
    with torch.no_grad():
        peft_logits = peft_model(**sample).logits[0, -1]
    diff = (peft_logits - base_logits).abs().max().item()
    print(f"Max |adapter_logit - base_logit| = {diff:.4e}", flush=True)
    if diff < 1e-6:
        raise RuntimeError(
            f"Sanity check failed: adapter produces identical logits to base "
            f"(max diff {diff:.2e}). Adapter appears untrained or not loaded."
        )

print("Upcasting to float32 for precision merge ...", flush=True)
peft_model = peft_model.float()
print("Merging and unloading LoRA weights ...", flush=True)
merged = peft_model.merge_and_unload()
print("Casting merged model back to bfloat16 ...", flush=True)
merged = merged.to(torch.bfloat16)

if not args.skip_sanity_check:
    with torch.no_grad():
        merged_logits = merged(**sample).logits[0, -1]
    # Merged model must agree with the pre-merge adapter logits, not the base logits
    merge_diff = (merged_logits - peft_logits).abs().max().item()
    print(f"Max |merged_logit - adapter_logit| = {merge_diff:.4e}", flush=True)
    if merge_diff > 0.5:
        raise RuntimeError(
            f"Merge sanity check failed: merged model logits diverge from adapter logits "
            f"(max diff {merge_diff:.2e}). Expected <0.5 for float32-merged bfloat16 LoRA; "
            f"small diffs (~0.1) are normal bfloat16 rounding from different computation paths."
        )
    print("Merge sanity check passed.", flush=True)

output_path.mkdir(parents=True, exist_ok=True)
print(f"Saving merged model to {output_path} ...", flush=True)
merged.save_pretrained(str(output_path), safe_serialization=True)
tokenizer.save_pretrained(str(output_path))

# ── vLLM compatibility: inject visual + mtp weights and remap text keys ───────
#
# Qwen3_5ForCausalLM (text-only) saves weights with flat key structure:
#   model.embed_tokens.weight, model.layers.0.*, lm_head.weight
#
# vLLM only registers Qwen3_5ForConditionalGeneration (the VL class), which
# expects the full checkpoint format:
#   model.language_model.embed_tokens.weight  (text backbone under language_model.*)
#   model.visual.*                            (visual encoder — 333 weights)
#   mtp.*                                     (multi-token prediction — 15 weights)
#
# Fix: remap flat text keys to language_model.* and add visual+mtp from base.
#
import safetensors.torch as safetensors_st
import json as _json

print("Remapping text keys to VL format and injecting visual/mtp weights ...", flush=True)

# Load all merged text shards
merged_tensors: dict = {}
for shard in sorted(output_path.glob("*.safetensors")):
    merged_tensors.update(safetensors_st.load_file(str(shard)))

# Remap flat text keys to VL format if needed.
# If keys are already model.language_model.* (merged from VL base) → no change.
# If keys are flat model.embed_tokens.* / model.layers.* → add language_model. prefix.
has_lm_prefix = any(k.startswith("model.language_model.") for k in merged_tensors)
remapped: dict = {}
for k, v in merged_tensors.items():
    if not has_lm_prefix and k.startswith("model."):
        remapped["model.language_model." + k[len("model."):]] = v
    else:
        remapped[k] = v
if not has_lm_prefix:
    print("  Remapped flat model.* keys to model.language_model.* format.", flush=True)

# Load visual + mtp weights from base model (never modified by LoRA)
base_index_path = Path(args.base_model) / "model.safetensors.index.json"
if base_index_path.exists():
    base_index = _json.loads(base_index_path.read_text())
    base_shard_files = set(base_index["weight_map"].values())
    for shard_name in sorted(base_shard_files):
        shard_path = Path(args.base_model) / shard_name
        for k, v in safetensors_st.load_file(str(shard_path)).items():
            if k.startswith("model.visual") or k.startswith("mtp."):
                remapped[k] = v
else:
    # Single-shard base model
    base_shard = Path(args.base_model) / "model.safetensors"
    if base_shard.exists():
        for k, v in safetensors_st.load_file(str(base_shard)).items():
            if k.startswith("model.visual") or k.startswith("mtp."):
                remapped[k] = v

visual_count = sum(1 for k in remapped if k.startswith("model.visual"))
mtp_count    = sum(1 for k in remapped if k.startswith("mtp."))
lm_count     = sum(1 for k in remapped if "language_model" in k)
print(f"  language_model keys: {lm_count}, visual keys: {visual_count}, mtp keys: {mtp_count}", flush=True)

# Delete old flat shards and save remapped tensors
for shard in list(output_path.glob("*.safetensors")):
    shard.unlink()
index_path = output_path / "model.safetensors.index.json"
if index_path.exists():
    index_path.unlink()

# Split into ~4 GB shards
MAX_SHARD_BYTES = 4 * 1024 ** 3
shards_out: list[dict] = [{}]
current_size = 0
for k, v in remapped.items():
    nbytes = v.numel() * v.element_size()
    if current_size + nbytes > MAX_SHARD_BYTES and current_size > 0:
        shards_out.append({})
        current_size = 0
    shards_out[-1][k] = v
    current_size += nbytes

n_shards = len(shards_out)
weight_map: dict[str, str] = {}
for i, shard_tensors in enumerate(shards_out, 1):
    fname = f"model.safetensors-{i:05d}-of-{n_shards:05d}.safetensors" if n_shards > 1 else "model.safetensors"
    safetensors_st.save_file(shard_tensors, str(output_path / fname))
    for k in shard_tensors:
        weight_map[k] = fname

if n_shards > 1:
    (output_path / "model.safetensors.index.json").write_text(
        _json.dumps({"metadata": {"format": "pt"}, "weight_map": weight_map}, indent=2)
    )
print(f"Saved {n_shards} shard(s) with {len(remapped)} total weights.", flush=True)

# Replace config with base model's VL config (Qwen3_5ForConditionalGeneration format)
import shutil
cfg_path = output_path / "config.json"
base_cfg_path = Path(args.base_model) / "config.json"
if base_cfg_path.exists():
    shutil.copy(base_cfg_path, cfg_path)
    print(f"Replaced config.json with base model config (vLLM-compatible format)", flush=True)

# Copy preprocessor configs so vLLM can init the image processor for VL model loading
for pname in ("preprocessor_config.json", "video_preprocessor_config.json"):
    src = Path(args.base_model) / pname
    if src.exists():
        shutil.copy(src, output_path / pname)
        print(f"Copied {pname} from base model", flush=True)

# vLLM fails if tokenizer_config.json references TokenizersBackend (an Unsloth
# class not present in the vllm env). Remove it — vLLM doesn't need it.
tok_cfg_path = output_path / "tokenizer_config.json"
if tok_cfg_path.exists():
    tok_cfg = json.loads(tok_cfg_path.read_text())
    if tok_cfg.get("tokenizer_class") == "TokenizersBackend":
        tok_cfg.pop("tokenizer_class")
        tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2))
        print(f"Removed TokenizersBackend from tokenizer_config.json", flush=True)


# Verify expected output files exist
required = ["config.json", "tokenizer_config.json", "tokenizer.json"]
missing = [f for f in required if not (output_path / f).exists()]
shards = list(output_path.glob("*.safetensors"))
if missing:
    raise RuntimeError(f"Post-save check failed: missing files: {missing}")
if not shards:
    raise RuntimeError("Post-save check failed: no .safetensors weight files found in output directory.")
total_gb = sum(s.stat().st_size for s in shards) / 1e9
expected_keys = lm_count + visual_count + mtp_count + 1  # +1 for lm_head
print(f"Post-save check passed: {len(shards)} shard(s), {total_gb:.1f} GB, {len(remapped)} weights.", flush=True)

print(f"Done. Merged model at: {output_path}", flush=True)
