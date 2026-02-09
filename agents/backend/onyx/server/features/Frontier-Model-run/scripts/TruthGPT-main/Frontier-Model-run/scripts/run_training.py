#!/usr/bin/env python3
"""Main entry point for running training with YAML configuration."""
import argparse
import sys
from pathlib import Path

from scripts.config_parser import ConfigParser
from scripts.kf_grpo_train import main


def main_with_config() -> None:
    """Main function to run training with YAML config.
    
    Loads configuration from YAML file and runs the training script.
    Exits with appropriate error codes on failure.
    """
    parser = argparse.ArgumentParser(description='Run KF-GRPO training with YAML config')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    
    config_path = Path(args.config)
    
    try:
        config = ConfigParser.load_config(str(config_path))
        script_args, training_args, model_args = ConfigParser.convert_to_all_args(config)
        main(script_args, training_args, model_args)
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid configuration: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to run training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main_with_config()  