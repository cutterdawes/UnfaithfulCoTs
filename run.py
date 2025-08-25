#!/usr/bin/env python3
import argparse
from dataclasses import asdict
from typing import List
import sys
import os
import time

# Early check for PyTorch to provide a clearer error if missing
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.stderr.write(
        "PyTorch is not installed. Activate your conda env or install deps.\n"
        "Try: 'conda env create -f environment.yml && conda activate unfaithfulcots'\n"
    )
    raise

from faithvec.model import HFModel
from faithvec.pipeline import run_pair, difference_in_means


def main():
    ap = argparse.ArgumentParser(description="MVP: Learn a faithfulness vector via difference-in-means.")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="HF model name (reasoning model suggested)")
    ap.add_argument("--n", type=int, default=50, help="Number of training pairs to sample")
    ap.add_argument("--n-test", type=int, default=20, help="Number of test pairs (next after training)")
    ap.add_argument("--max-new", type=int, default=64, help="Max new tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for pairs")
    ap.add_argument("--layer", type=int, default=20, help="Feature layer (transformer block index, 0=embed->block0). Use -1 for last layer.")
    ap.add_argument("--device", default=None, help="Device override: cpu|cuda|mps (default prefers CUDA, else CPU)")
    ap.add_argument("--device-map", default=None, help="Hugging Face device_map, e.g., 'auto' or 'none' (default: none)")
    ap.add_argument(
        "--wm-path",
        default="data/wm_cache/wm-us-city-lat.yaml",
        help="Path to a World Model YAML cache (defaults to local copy if present)",
    )
    ap.add_argument(
        "--wm-prop",
        default="latitude",
        help="Natural-language name of the property (e.g., 'population', 'longitude')",
    )
    args = ap.parse_args()

    # Normalize device and device_map
    dev = None
    if args.device:
        dev = args.device.strip().lower()
        if dev not in {"cpu", "cuda", "mps"}:
            raise SystemExit(f"Invalid --device '{args.device}'. Use cpu|cuda|mps.")
    dm = args.device_map
    if isinstance(dm, str) and dm.strip().lower() in {"none", "", "null"}:
        dm = None

    model = HFModel(model_name=args.model, device=dev, device_map=dm, feature_layer=args.layer)
    print(f"Model: {args.model} | Device: {model.device} | device_map: {dm or 'none'} | layer: {args.layer}")
    if args.wm_path and os.path.exists(args.wm_path):
        from faithvec.wm import load_wm_pairs_list
        all_pairs = load_wm_pairs_list(args.wm_path, seed=args.seed, property_name=args.wm_prop)
        print(f"Using WM cache: {args.wm_path} [prop={args.wm_prop}] (total CLEAR pairs: {len(all_pairs)})")
        pairs = all_pairs[: args.n]
        test_pairs = all_pairs[args.n : args.n + args.n_test]
    else:
        from faithvec.dataset import generate_numeric_pairs
        print("WM cache not found; falling back to synthetic numeric pairs.")
        all_pairs = generate_numeric_pairs(args.n + args.n_test, seed=args.seed)
        pairs = all_pairs[: args.n]
        test_pairs = all_pairs[args.n : args.n + args.n_test]

    def fmt_pair(p):
        if hasattr(p, "x") and hasattr(p, "y"):
            return f"{p.x},{p.y}"
        if hasattr(p, "left") and hasattr(p, "right"):
            return f"{p.left} | {p.right}"
        return "?"

    print("\n=== Train ===")
    print(f"Running on train set (n={len(pairs)})...")
    results = []
    for i, p in enumerate(pairs, 1):
        r = run_pair(model, p, max_new_tokens=args.max_new, temperature=args.temperature)
        results.append(r)
        status = "faithful" if r.is_faithful else ("unfaithful" if r.is_faithful is not None else "unknown")
        print(f"[{i:03d}] ({fmt_pair(p)}) -> {status}")

    fvec = difference_in_means(results)
    if fvec is None:
        print("Not enough labeled pairs to compute difference-in-means.")
        return
    print("\nLearned faithfulness vector (unit-norm): dim=", fvec.v.numel())

    # Save learned vector with training data info in filename
    def _slug(s: str) -> str:
        return ''.join(ch if ch.isalnum() else '-' for ch in s).strip('-')

    dataset_tag = 'toy'
    prop_tag = None
    if args.wm_path and os.path.exists(args.wm_path):
        dataset_tag = os.path.splitext(os.path.basename(args.wm_path))[0]
        prop_tag = args.wm_prop
    model_tag = _slug(args.model)
    ts = time.strftime('%Y%m%d-%H%M%S')
    parts = [model_tag, dataset_tag]
    if prop_tag:
        parts.append(f"prop-{_slug(prop_tag)}")
    parts += [f"n-{args.n}", f"seed-{args.seed}", ts]
    save_dir = os.path.join('data', 'faithvec')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '__'.join(parts) + '.pt')
    torch.save({
        'v': fvec.v,
        'mu_f': fvec.mu_f,
        'mu_u': fvec.mu_u,
        'meta': {
            'model': args.model,
            'device': model.device,
            'device_map': dm or 'none',
            'feature_layer': args.layer,
            'wm_path': args.wm_path if args.wm_path and os.path.exists(args.wm_path) else None,
            'wm_prop': args.wm_prop if prop_tag else None,
            'n_train': args.n,
            'n_test': args.n_test,
            'seed': args.seed,
            'timestamp': ts,
        }
    }, save_path)
    print(f"Saved vector to: {save_path}")

    # Simple separation score across labeled pairs
    labeled = [(r, fvec.project(r.pair_features)) for r in results if r.is_faithful is not None]
    if labeled:
        proj_f = [s for (r, s) in labeled if r.is_faithful]
        proj_u = [s for (r, s) in labeled if not r.is_faithful]
        print(f"Train: Avg proj faithful: {sum(proj_f)/max(1,len(proj_f)):.4f} (n={len(proj_f)})")
        print(f"Train: Avg proj unfaithful: {sum(proj_u)/max(1,len(proj_u)):.4f} (n={len(proj_u)})")

    # Evaluate on test set if available
    if test_pairs:
        print(f"\n=== Test ===")
        print(f"Running on test set (n={len(test_pairs)})...")
        test_results = []
        for i, p in enumerate(test_pairs, 1):
            r = run_pair(model, p, max_new_tokens=args.max_new, temperature=args.temperature)
            test_results.append(r)
            status = "faithful" if r.is_faithful else ("unfaithful" if r.is_faithful is not None else "unknown")
            if hasattr(p, "x") and hasattr(p, "y"):
                pair_str = f"{p.x},{p.y}"
            elif hasattr(p, "left") and hasattr(p, "right"):
                pair_str = f"{p.left} | {p.right}"
            else:
                pair_str = "?"
            print(f"[{i:03d}] ({pair_str}) -> {status}")
        labeled_t = [(r, fvec.project(r.pair_features)) for r in test_results if r.is_faithful is not None]
        if labeled_t:
            proj_f_t = [s for (r, s) in labeled_t if r.is_faithful]
            proj_u_t = [s for (r, s) in labeled_t if not r.is_faithful]
            print(f"Test: Avg proj faithful: {sum(proj_f_t)/max(1,len(proj_f_t)):.4f} (n={len(proj_f_t)})")
            print(f"Test: Avg proj unfaithful: {sum(proj_u_t)/max(1,len(proj_u_t)):.4f} (n={len(proj_u_t)})")
        else:
            print("Test: No labeled test pairs (couldn't parse Yes/No).")


if __name__ == "__main__":
    main()
