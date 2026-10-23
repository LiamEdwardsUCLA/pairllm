"""
Merge a LoRA adapter into the Gemma-4 base model and save for vLLM inference.

Loads the base model via AutoModelForCausalLM (not a text-only model class,
unlike merge.py's Qwen3.5 handling): this resolves Gemma-4's config to
Gemma4UnifiedForConditionalGeneration, the class matching the checkpoint's
on-disk key structure, so no key remapping is needed after merging. LoRA's
target_modules="all-linear" finds every Linear layer regardless of the
surrounding multimodal wrapper. Requires vLLM >= 0.26 to load the resulting
merged model for inference (see Requirements in the README).

Usage:
    python scripts/merge_gemma.py \
        --adapter /path/to/checkpoint-N \
        --base-model google/gemma-4-12B-it \
        --output-dir merged_models \
        --output-name my-merged-model
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", required=True, help="Path to LoRA adapter checkpoint directory")
parser.add_argument("--base-model", default="google/gemma-4-12B-it")
parser.add_argument("--output-name", default="gemma4-merged")
parser.add_argument("--output-dir", default="merged_models")
parser.add_argument("--skip-sanity-check", action="store_true")
args = parser.parse_args()

output_path = Path(args.output_dir) / args.output_name

print(f"Loading base model from {args.base_model} ...", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"  resolved class: {type(base).__name__}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

base_logits = None
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
    merge_diff = (merged_logits - peft_logits).abs().max().item()
    print(f"Max |merged_logit - adapter_logit| = {merge_diff:.4e}", flush=True)
    if merge_diff > 0.5:
        raise RuntimeError(
            f"Merge sanity check failed: merged model logits diverge from adapter logits "
            f"(max diff {merge_diff:.2e})."
        )
    print("Merge sanity check passed.", flush=True)

output_path.mkdir(parents=True, exist_ok=True)
print(f"Saving merged model to {output_path} ...", flush=True)
merged.save_pretrained(str(output_path), safe_serialization=True)
tokenizer.save_pretrained(str(output_path))
processor.save_pretrained(str(output_path))

shards = list(output_path.glob("*.safetensors"))
total_gb = sum(s.stat().st_size for s in shards) / 1e9
print(f"Done. {len(shards)} shard(s), {total_gb:.1f} GB. Merged model at: {output_path}", flush=True)
