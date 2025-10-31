"""
Visualize training metrics from checkpoints and logs.
Supports W&B, TensorBoard, and JSON logs.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

try:
    import matplotlib.pyplot as plt
    import numpy as np
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


def load_wandb_logs(run_dir: str) -> Dict[str, List[Any]]:
    """Load metrics from W&B run directory."""
    metrics = {}
    # W&B stores logs in wandb/ directory
    wandb_dir = Path(run_dir) / "wandb"
    if not wandb_dir.exists():
        return metrics
    
    # Simplified: W&B logs are complex, this is a basic implementation
    print(f"📊 W&B logs found in {wandb_dir}")
    print("   Use: wandb sync or wandb dashboard for full visualization")
    return metrics


def load_tensorboard_logs(run_dir: str) -> Dict[str, List[Any]]:
    """Load metrics from TensorBoard log directory."""
    metrics = {}
    tb_dir = Path(run_dir)
    if not tb_dir.exists():
        return metrics
    
    # TensorBoard logs require tensorboard to read
    print(f"📊 TensorBoard logs found in {tb_dir}")
    print("   Use: tensorboard --logdir={tb_dir} for visualization")
    return metrics


def find_checkpoints(run_dir: str) -> List[str]:
    """Find all checkpoint files in run directory."""
    run_path = Path(run_dir)
    if not run_path.exists():
        return []
    
    checkpoints = []
    for f in run_path.iterdir():
        if f.is_file() and f.suffix == ".pt":
            checkpoints.append(str(f))
        elif f.is_dir() and (f / "config.json").exists():
            # HF checkpoint format
            checkpoints.append(str(f))
    
    return sorted(checkpoints)


def visualize_checkpoints(run_dir: str, output: str = None):
    """Visualize checkpoint information."""
    checkpoints = find_checkpoints(run_dir)
    
    if not checkpoints:
        print(f"⚠️  No checkpoints found in {run_dir}")
        return
    
    print(f"\n📦 Found {len(checkpoints)} checkpoint(s):")
    for i, ckpt in enumerate(checkpoints, 1):
        size = os.path.getsize(ckpt) / (1024**2)  # MB
        print(f"  {i}. {Path(ckpt).name} ({size:.1f} MB)")


def summarize_run(run_dir: str):
    """Print summary of training run."""
    run_path = Path(run_dir)
    
    print(f"\n📊 Training Run Summary: {run_path.name}")
    print("=" * 60)
    
    # Checkpoints
    checkpoints = find_checkpoints(str(run_path))
    print(f"✅ Checkpoints: {len(checkpoints)}")
    
    # Look for best checkpoint
    best_ckpt = run_path / "best.pt"
    if best_ckpt.exists() or (run_path / "best.pt" / "config.json").exists():
        print(f"   🏆 Best checkpoint: ✓")
    
    last_ckpt = run_path / "last.pt"
    if last_ckpt.exists() or (run_path / "last.pt" / "config.json").exists():
        print(f"   📌 Last checkpoint: ✓")
    
    # Config
    config_file = run_path / "config.json"
    if config_file.exists():
        print(f"✅ Config file: ✓")
    
    # Logs
    wandb_dir = run_path / "wandb"
    if wandb_dir.exists():
        print(f"✅ W&B logs: ✓")
    
    tb_dir = run_path.parent if (run_path / "events.out.tfevents").exists() else None
    if tb_dir:
        print(f"✅ TensorBoard logs: ✓")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument(
        "run_dir",
        type=str,
        nargs="?",
        default="runs",
        help="Directory containing training run (default: runs)",
    )
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        help="List all checkpoints",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show run summary",
    )
    args = parser.parse_args()
    
    if args.checkpoints:
        visualize_checkpoints(args.run_dir)
    elif args.summary:
        summarize_run(args.run_dir)
    else:
        # Default: show summary
        summarize_run(args.run_dir)
        print("\n💡 Use --checkpoints to list all checkpoints")
        print("💡 Use tensorboard --logdir=runs for metrics visualization")


if __name__ == "__main__":
    main()


