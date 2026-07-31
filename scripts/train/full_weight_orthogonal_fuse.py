#!/usr/bin/env python3
"""Full-weight orthogonal residual fusion (correct version).

Merge both LoRAs into base, compute full weight deltas, orthogonalize ΔB against ΔA,
fuse: W = W_A + λ × R_B. This operates on the EFFECTIVE delta (B@A), not the LoRA
A/B parameters — which is what the competitor's formula means.
"""
import json, argparse, os, hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
BASE = str(ROOT / "models/OneReason-0.8B-pretrain-competition")
TARGETS = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}

def merge_lora(adapter_path):
    """Load base + merge LoRA -> full model weights."""
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, adapter_path, adapter_name="lora", is_trainable=False)
    model = model.merge_and_unload()
    return model.state_dict()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.25)
    args = ap.parse_args()

    print(f"Loading base + merging A ({args.model_a})...")
    sd_a = merge_lora(args.model_a)
    print(f"Loading base + merging B ({args.model_b})...")
    sd_b = merge_lora(args.model_b)
    print(f"Loading base (no LoRA)...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, low_cpu_mem_usage=True)
    sd_base = base_model.state_dict()

    print(f"Fusing with λ={args.lam}...")
    fused_sd = {}
    total_modules = 0
    for key in sd_base:
        if not any(t in key for t in TARGETS):
            # non-target weights: use A (keep A's strengths)
            fused_sd[key] = sd_a[key]
            continue
        # target module: compute deltas
        w_base = sd_base[key]
        w_a = sd_a[key]
        w_b = sd_b[key]
        delta_a = w_a - w_base  # A's effective delta
        delta_b = w_b - w_base  # B's effective delta
        # orthogonalize: R_B = ΔB - (⟨ΔB,ΔA⟩/‖ΔA‖²) × ΔA
        dot = torch.dot(delta_b.flatten(), delta_a.flatten())
        norm_sq = torch.dot(delta_a.flatten(), delta_a.flatten())
        if norm_sq > 0:
            r_b = delta_b - (dot / norm_sq) * delta_a
        else:
            r_b = delta_b
        # fuse: W = W_A + λ × R_B
        fused_sd[key] = (w_a + args.lam * r_b).to(torch.bfloat16)
        total_modules += 1
        if total_modules % 28 == 0:
            cos = (dot / (delta_a.norm() * delta_b.norm() + 1e-8)).item()
            print(f"  {key}: cos(ΔA,ΔB)={cos:.4f}", flush=True)

    print(f"Fused {total_modules} target modules. Saving as BF16 full-param...")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # save as HF model (full-param)
    base_model.load_state_dict({k: v.to(torch.float32) if v.dtype != torch.bfloat16 else v for k, v in fused_sd.items()})
    # actually need to convert all to bfloat16
    for k in fused_sd:
        fused_sd[k] = fused_sd[k].to(torch.bfloat16) if fused_sd[k].dtype != torch.bfloat16 else fused_sd[k]
    base_model.load_state_dict(fused_sd)
    base_model.save_pretrained(str(out), safe_serialization=True)
    # copy config/tokenizer
    import shutil
    src = Path(BASE)
    for f in ["config.json","generation_config.json","tokenizer.json","tokenizer_config.json","chat_template.jinja","special_tokens_map.json","vocab.json","merges.txt"]:
        if (src/f).exists(): shutil.copy(src/f, out/f)
    total = sum((out/f).stat().st_size for f in os.listdir(out))
    sha = hashlib.sha256()
    with open(out/"model.safetensors","rb") as f:
        for chunk in iter(lambda: f.read(8192*1024), b''): sha.update(chunk)
    print(f"\nFused full-param model: λ={args.lam}")
    print(f"  path: {out}")
    print(f"  total bytes: {total}")
    print(f"  model_sha256: {sha.hexdigest()}")

if __name__ == "__main__":
    main()
