from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import argparse
import sys
import asyncio
from pathlib import Path
from config_loader import (
from model_training import create_model_trainer, DeviceManager
import logging
        import json
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Experiment Runner Script
========================

This script runs experiments using the configuration system and training pipeline.
"""

    load_config_from_yaml, 
    create_experiment_config, 
    save_experiment_config,
    validate_config,
    quick_config_setup
)

def setup_logging(experiment_id: str, log_dir: str = "logs"):
    """Setup logging for the experiment."""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path / f"{experiment_id}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

async def run_experiment(config_path: str, experiment_id: str = None, description: str = ""):
    """Run a complete experiment."""
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        config = load_config_from_yaml(config_path)
        
        # Validate configuration
        logger.info("Validating configuration")
        if not validate_config(config):
            raise ValueError("Configuration validation failed")
        
        # Create experiment config with metadata
        if experiment_id is None:
            experiment_id = f"exp_{config.model_name}_{config.num_epochs}epochs"
        
        exp_config = create_experiment_config(experiment_id, description, config)
        
        # Save experiment configuration
        output_dir = Path(config.output_dir)
        output_dir.mkdir(exist_ok=True)
        save_experiment_config(exp_config, str(output_dir))
        
        # Setup logging
        logger = setup_logging(experiment_id)
        logger.info(f"Starting experiment: {experiment_id}")
        logger.info(f"Description: {description}")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Dataset: {config.dataset_path}")
        logger.info(f"Epochs: {config.num_epochs}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Learning rate: {config.learning_rate}")
        
        # Create trainer and run training
        device_manager = DeviceManager()
        trainer = await create_model_trainer(device_manager)
        
        logger.info("Starting training...")
        results = await trainer.train(config)
        
        # Log results
        logger.info("Training completed successfully!")
        logger.info(f"Best validation accuracy: {results.get('training_summary', {}).get('best_val_accuracy', 'N/A')}")
        logger.info(f"Final test accuracy: {results.get('evaluation_result', {}).get('test_accuracy', 'N/A')}")
        logger.info(f"Total training time: {results.get('total_training_time', 'N/A')} seconds")
        
        # Save results
        results_path = output_dir / f"{experiment_id}_results.json"
        with open(results_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise

async def run_quick_experiment(model_name: str, dataset_path: str, experiment_id: str, description: str = ""):
    """Run a quick experiment with default settings."""
    logger = logging.getLogger(__name__)
    
    try:
        # Create quick experiment config
        exp_config = quick_config_setup(model_name, dataset_path, experiment_id, description)
        
        # Setup logging
        logger = setup_logging(experiment_id)
        logger.info(f"Starting quick experiment: {experiment_id}")
        
        # Create trainer and run training
        device_manager = DeviceManager()
        trainer = await create_model_trainer(device_manager)
        
        logger.info("Starting training...")
        results = await trainer.train(exp_config.config)
        
        # Save results
        output_dir = Path(exp_config.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        save_experiment_config(exp_config, str(output_dir))
        
        logger.info("Quick experiment completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Quick experiment failed: {e}", exc_info=True)
        raise

def main():
    
    """main function."""
parser = argparse.ArgumentParser(description="Run NLP training experiments")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--experiment-id", type=str, help="Experiment ID")
    parser.add_argument("--description", type=str, default="", help="Experiment description")
    
    # Quick experiment options
    parser.add_argument("--quick", action="store_true", help="Run quick experiment")
    parser.add_argument("--model", type=str, help="Model name for quick experiment")
    parser.add_argument("--dataset", type=str, help="Dataset path for quick experiment")
    
    args = parser.parse_args()
    
    if args.quick:
        if not args.model or not args.dataset:
            print("Error: --model and --dataset are required for quick experiments")
            sys.exit(1)
        
        asyncio.run(run_quick_experiment(
            args.model, 
            args.dataset, 
            args.experiment_id or f"quick_{args.model}",
            args.description
        ))
    
    elif args.config:
        asyncio.run(run_experiment(
            args.config,
            args.experiment_id,
            args.description
        ))
    
    else:
        print("Error: Either --config or --quick must be specified")
        print("Examples:")
        print("  python run_experiment.py --config configs/baseline.yaml")
        print("  python run_experiment.py --quick --model distilbert-base-uncased --dataset data/sentiment.csv")
        sys.exit(1)

match __name__:
    case "__main__":
    main() 