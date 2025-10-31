import argparse
import os
from typing import List, Optional, Tuple
import logging

import torch
import yaml
from datasets import load_dataset

from trainers.trainer import TrainerConfig
from build_trainer import build_trainer
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def read_yaml(path: str) -> dict:
    """Read and parse YAML configuration file with error handling."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config is None:
            raise ValueError(f"Empty or invalid YAML file: {path}")
        logger.info(f"Successfully loaded config from {path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {path}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error reading config file {path}: {e}", exc_info=True)
        raise


def to_cfg(config: dict) -> TrainerConfig:
    model = config.get("model", {})
    training = config.get("training", {})
    hardware = config.get("hardware", {})
    optimizer_cfg = config.get("optimizer", {})
    eval_cfg = config.get("eval", {})

    return TrainerConfig(
        seed=config.get("seed", 42),
        run_name=config.get("run_name", "run"),
        output_dir=config.get("output_dir", "runs/run"),
        model_name=model.get("name_or_path", "gpt2"),
        gradient_checkpointing=bool(model.get("gradient_checkpointing", True)),
        lora_enabled=bool(model.get("lora", {}).get("enabled", False)),
        lora_r=int(model.get("lora", {}).get("r", 16)),
        lora_alpha=int(model.get("lora", {}).get("alpha", 32)),
        lora_dropout=float(model.get("lora", {}).get("dropout", 0.05)),
        epochs=int(training.get("epochs", 3)),
        train_batch_size=int(training.get("train_batch_size", 8)),
        eval_batch_size=int(training.get("eval_batch_size", 8)),
        grad_accum_steps=int(training.get("grad_accum_steps", 2)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        learning_rate=float(training.get("learning_rate", 5e-5)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        warmup_ratio=float(training.get("warmup_ratio", 0.06)),
        scheduler=str(training.get("scheduler", "cosine")),
        mixed_precision=str(training.get("mixed_precision", "bf16")),
        early_stopping_patience=int(training.get("early_stopping_patience", 2)),
        log_interval=int(training.get("log_interval", 50)),
        eval_interval=int(training.get("eval_interval", 500)),
        device=str(hardware.get("device", "auto")),
        allow_tf32=bool(training.get("allow_tf32", True)),
        torch_compile=bool(training.get("torch_compile", False)),
        compile_mode=str(training.get("compile_mode", "default")),
        fused_adamw=bool(training.get("fused_adamw", True)),
        detect_anomaly=bool(training.get("detect_anomaly", False)),
        use_profiler=bool(training.get("use_profiler", False)),
        save_safetensors=bool(training.get("save_safetensors", True)),
        optimizer_type=str(optimizer_cfg.get("type", "adamw")),
        # store selected metric in cfg for trainer usage
        # (trainer will still log both loss and ppl)
        
        num_workers=int(config.get("data", {}).get("num_workers", 4)),
        prefetch_factor=int(config.get("data", {}).get("prefetch_factor", 2)),
        persistent_workers=bool(config.get("data", {}).get("persistent_workers", True)),
    )


def load_text_splits(
    dataset: str,
    subset: Optional[str],
    text_field: str,
    limit_per_split: int = 5000
) -> Tuple[List[str], List[str]]:
    """
    Load and split dataset into training and validation sets.
    
    Args:
        dataset: Dataset name from HuggingFace
        subset: Optional subset name
        text_field: Field name containing text data
        limit_per_split: Maximum samples per split
    
    Returns:
        Tuple of (train_texts, val_texts)
    """
    try:
        logger.info(f"Loading dataset: {dataset} (subset: {subset})")
        if subset:
            ds = load_dataset(dataset, subset)
        else:
            ds = load_dataset(dataset)
        
        if "train" not in ds:
            raise ValueError(f"Dataset {dataset} does not contain 'train' split")
        
        train = ds["train"][text_field][:limit_per_split]
        val_limit = max(256, limit_per_split // 10)
        
        if "validation" in ds:
            val = ds["validation"][text_field][:val_limit]
        elif "val" in ds:
            val = ds["val"][text_field][:val_limit]
        else:
            logger.warning(f"No validation split found, using train split for validation")
            val = ds["train"][text_field][:val_limit]
        
        train_list = list(train) if not isinstance(train, list) else train
        val_list = list(val) if not isinstance(val, list) else val
        
        logger.info(f"Loaded {len(train_list)} training samples and {len(val_list)} validation samples")
        return train_list, val_list
    except Exception as e:
        logger.error(f"Error loading dataset {dataset}: {e}", exc_info=True)
        raise


def main() -> None:
    """Main training function with improved error handling and logging."""
    parser = argparse.ArgumentParser(
        description="Train LLM with TruthGPT optimization core",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="optimization_core/configs/llm_default.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--data_limit",
        type=int,
        default=5000,
        help="Limit number of samples per split"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional path to log file"
    )
    args = parser.parse_args()

    # Setup file logging if requested
    if args.log_file:
        setup_logger(__name__, log_file=args.log_file)

    try:
        logger.info("=" * 80)
        logger.info("Starting LLM Training")
        logger.info("=" * 80)

        # Load configuration
        cfg_dict = read_yaml(args.config)
        cfg = to_cfg(cfg_dict)

        # Extract data configuration
        data_cfg = cfg_dict.get("data", {})
        dataset = str(data_cfg.get("dataset", "wikitext"))
        subset = data_cfg.get("subset", "wikitext-2-raw-v1")
        text_field = str(data_cfg.get("text_field", "text"))

        # Load datasets
        train_texts, val_texts = load_text_splits(
            dataset, subset, text_field, args.data_limit
        )

        # Log configuration summary
        logger.info("Training Configuration:")
        logger.info(f"  Device: {cfg.device}")
        logger.info(f"  Model: {cfg.model_name}")
        logger.info(f"  Output Directory: {cfg.output_dir}")
        logger.info(f"  Epochs: {cfg.epochs}")
        logger.info(f"  Batch Size: {cfg.train_batch_size}")
        logger.info(f"  Learning Rate: {cfg.learning_rate}")
        logger.info(f"  Mixed Precision: {cfg.mixed_precision}")
        logger.info(f"  Gradient Accumulation Steps: {cfg.grad_accum_steps}")

        # Build and train
        trainer = build_trainer(
            cfg=cfg,
            raw_cfg=cfg_dict,
            train_texts=train_texts,
            val_texts=val_texts,
            max_seq_len=int(data_cfg.get("max_seq_len", args.max_seq_len)),
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully!")

        # Generate sample
        logger.info("Generating sample output...")
        sample = trainer.generate("The theory of transformers is", max_new_tokens=64)
        logger.info("Sample Output:")
        logger.info(sample)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
