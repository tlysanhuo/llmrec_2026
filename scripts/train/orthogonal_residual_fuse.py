#!/usr/bin/env python3
"""Orthogonal residual fusion (llmrec-post-training-notes technique).

R_B = ΔB - (⟨ΔB,ΔA⟩/‖ΔA‖²) × ΔA   (remove B's component parallel to A)
W_fused = ΔA + λ × R_B               (add only B's orthogonal/novel direction)

This preserves A's strengths while adding B's non-conflicting directions.
Different from naive soup (washes to mean) and task-vector (has conflicts).
"""
import json, argparse, os, hashlib
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
BASE = str(ROOT / "models/OneReason-0.8B-pretrain-competition")
TARGETS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"

def load_adapter(path, max_rank):
    """Load LoRA adapter, padding to max_rank if needed."""
    t = load_file(str(Path(path)/"adapter_model.safetensors"), device="cpu")
    cfg = json.loads(Path(path/"adapter_config.json").read_text())
    rank = cfg["r"]
    padded = {}
    for k in sorted(t):
        if k.endswith(A_SUFFIX):
            # A: [rank, in_dim] -> pad rows to max_rank
            if t[k].shape[0] < max_rank:
                t[k] = torch.cat([t[k], torch.zeros(max_rank - t[k].shape[0], t[k].shape[1], dtype=t[k].dtype)], dim=0)
        elif k.endswith(B_SUFFIX):
            # B: [out_dim, rank] -> pad cols to max_rank
            if t[k].shape[1] < max_rank:
                t[k] = torch.cat([t[k], torch.zeros(t[k].shape[0], max_rank - t[k].shape[1], dtype=t[k].dtype)], dim=1)
        padded[k] = t[k]
    return padded, cfg

def orthogonal_fuse(adapter_a, adapter_b, lam=0.25):
    """Fuse: W_fused = ΔA + λ × (ΔB - proj(ΔB onto ΔA))."""
    fused = {}
    a_keys = sorted(k for k in adapter_a if k.endswith(A_SUFFIX))
    for ak in a_keys:
        bk = ak.replace("lora_A","lora_B")
        prefix = ak[:-len(A_SUFFIX)]
        # Get A and B for both models
        a_A, a_B = adapter_a[ak].float(), adapter_a[bk].float()
        b_A, b_B = adapter_b[ak].float(), adapter_b[bk].float()
        # Flatten A+B into a single vector per module
        da = torch.cat([a_A.flatten(), a_B.flatten()])  # ΔA
        db = torch.cat([b_A.flatten(), b_B.flatten()])  # ΔB
        # Orthogonalize: R_B = ΔB - (⟨ΔB,ΔA⟩/‖ΔA‖²) × ΔA
        dot = torch.dot(db, da)
        norm_sq = torch.dot(da, da)
        if norm_sq > 0:
            r_b = db - (dot / norm_sq) * da
        else:
            r_b = db  # A is zero, keep all of B
        # Fuse: ΔA + λ × R_B
        fused_vec = da + lam * r_b
        # Reshape back
        a_shape = a_A.shape
        b_shape = a_B.shape
        fused_A = fused_vec[:a_shape.numel()].reshape(a_shape)
        fused_B = fused_vec[a_shape.numel():].reshape(b_shape)
        fused[ak] = fused_A.to(adapter_a[ak].dtype)
        fused[bk] = fused_B.to(adapter_a[bk].dtype)
    return fused

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="Model A (base, preserves its strengths)")
    ap.add_argument("--model-b", required=True, help="Model B (adds orthogonal directions)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.25)
    args = ap.parse_args()

    # Determine max rank
    cfg_a = json.loads(Path(args.model_a+"/adapter_config.json").read_text())
    cfg_b = json.loads(Path(args.model_b+"/adapter_config.json").read_text())
    max_rank = max(cfg_a["r"], cfg_b["r"])
    print(f"Model A: {args.model_a} (r={cfg_a['r']})")
    print(f"Model B: {args.model_b} (r={cfg_b['r']})")
    print(f"Max rank: {max_rank}, lambda: {args.lam}")

    # Load + pad both to max_rank
    adapter_a, _ = load_adapter(args.model_a, max_rank)
    adapter_b, _ = load_adapter(args.model_b, max_rank)
    print(f"Loaded {len(adapter_a)} tensors each (padded to r={max_rank})")

    # Orthogonal fuse
    fused = orthogonal_fuse(adapter_a, adapter_b, args.lam)

    # Verify keys match
    assert set(fused.keys()) == set(adapter_a.keys()), "key mismatch"

    # Write output
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_file(fused, str(out/"adapter_model.safetensors"), metadata={"format":"pt"})
    # Config: use max_rank, A's config as base
    out_cfg = dict(cfg_a)
    out_cfg["r"] = max_rank
    out_cfg["lora_alpha"] = max_rank
    out_cfg["inference_mode"] = True
    out_cfg["rank_pattern"] = {}
    out_cfg["alpha_pattern"] = {}
    out_cfg["target_modules"] = sorted(TARGETS)
    (out/"adapter_config.json").write_text(json.dumps(out_cfg, indent=2, sort_keys=True)+"\n")

    # Stats
    total_bytes = (out/"adapter_model.safetensors").stat().st_size + (out/"adapter_config.json").stat().st_size
    sha = hashlib.sha256(open(out/"adapter_model.safetensors","rb").read(8192*1024)).hexdigest() if False else hashlib.sha256()
    with open(out/"adapter_model.safetensors","rb") as f:
        for chunk in iter(lambda: f.read(8192*1024), b''): sha.update(chunk)
    print(f"\nFused adapter: r={max_rank}, lambda={args.lam}")
    print(f"  path: {out}")
    print(f"  total bytes: {total_bytes} (<400MB: {total_bytes < 400000000})")
    print(f"  adapter_sha256: {sha.hexdigest()}")
    # Orthogonality check: how much of B was orthogonal to A?
    a_flat = torch.cat([adapter_a[k].float().flatten() for k in sorted(adapter_a) if k.endswith(A_SUFFIX)])
    b_flat = torch.cat([adapter_b[k].float().flatten() for k in sorted(adapter_b) if k.endswith(A_SUFFIX)])
    cos = torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm() + 1e-8)
    print(f"  cos(A,B) on A-matrices: {cos.item():.6f} (0=orthogonal, 1=parallel)")

if __name__ == "__main__":
    main()
