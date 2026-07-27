#!/usr/bin/env python3
"""
Prepare a base MLX model for serving from the Anchor server.

This script:
1. Downloads a HuggingFace model (e.g., mlx-community/Qwen3-0.6B-bf16)
2. Converts it to MLX format if needed
3. Places it in the Anchor's models/base/<model_name>/ directory
4. Generates SHA256 checksums for all files

Usage:
  python tools/prepare_base_model.py --model mlx-community/Qwen3-0.6B-bf16 --out models/base/Qwen3-0.6B-bf16
  python tools/prepare_base_model.py --model mlx-community/Qwen3-0.6B-bf16 --out models/base/Qwen3-0.6B-bf16 --anchor-dir .
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    from mlx_lm.convert import convert
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    print("Warning: mlx_lm not available, will only copy existing MLX models")


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def prepare_model(hf_repo: str, output_dir: Path, anchor_dir: Path = None) -> dict:
    """
    Download and prepare a base model for the Anchor server.

    Returns manifest dict with file info.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if model is already in MLX format (local directory)
    if Path(hf_repo).exists() and Path(hf_repo).is_dir():
        # Local MLX model directory - copy files
        print(f"Copying local MLX model from {hf_repo}")
        source_dir = Path(hf_repo)
    else:
        # Download from HuggingFace
        print(f"Downloading {hf_repo} from HuggingFace...")
        local_path = snapshot_download(repo_id=hf_repo)
        source_dir = Path(local_path)

        # Convert to MLX format if needed
        if MLX_LM_AVAILABLE:
            print("Converting to MLX format...")
            convert(str(source_dir), str(output_dir))
            source_dir = output_dir
        else:
            # If mlx_lm not available, just copy what we have
            print("mlx_lm not available, copying raw files...")
            shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)
            source_dir = output_dir

    # Standard MLX model files
    expected_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]

    # Generate manifest
    file_info = []
    for fname in expected_files:
        fpath = source_dir / fname
        if fpath.exists():
            sha256 = compute_sha256(fpath)
            size = fpath.stat().st_size
            file_info.append({"name": fname, "sha256": sha256, "size": size})
            print(f"  {fname}: sha256={sha256[:16]}... size={size:,} bytes")
        else:
            print(f"  WARNING: {fname} not found in {source_dir}")

    if not file_info:
        raise RuntimeError(f"No model files found in {source_dir}")

    # If anchor_dir specified, copy to Anchor's base model directory
    if anchor_dir:
        anchor_model_dir = anchor_dir / "models" / "base" / output_dir.name
        anchor_model_dir.mkdir(parents=True, exist_ok=True)
        for fname in expected_files:
            src = source_dir / fname
            if src.exists():
                dst = anchor_model_dir / fname
                shutil.copy2(src, dst)
                print(f"  Copied {fname} to Anchor at {anchor_model_dir}")
        print(f"\nModel ready at: {anchor_model_dir}")

    manifest = {
        "model_name": output_dir.name,
        "files": file_info
    }

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest saved to {manifest_path}")

    return manifest


def main():
    ap = argparse.ArgumentParser(description="Prepare base MLX model for Anchor server")
    ap.add_argument("--model", required=True, help="HuggingFace repo ID (e.g., mlx-community/Qwen3-0.6B-bf16) or local path")
    ap.add_argument("--out", required=True, help="Output directory name (e.g., models/base/Qwen3-0.6B-bf16)")
    ap.add_argument("--anchor-dir", default=".", help="Anchor server root directory (default: current dir)")
    args = ap.parse_args()

    output_dir = Path(args.out)
    anchor_dir = Path(args.anchor_dir)

    prepare_model(args.model, output_dir, anchor_dir)
    print("\nDone! You can now start the Anchor server and download the model from iOS.")


if __name__ == "__main__":
    main()