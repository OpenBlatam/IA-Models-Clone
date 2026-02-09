"""Experiment tracking utilities."""
import logging
from typing import Any, Callable, Dict

import mlflow
import wandb

from .types import KFGRPOScriptArguments

DEFAULT_WANDB_PROJECT = "kf-grpo"
DEFAULT_WANDB_CODE_DIR = "."
DEFAULT_WANDB_START_METHOD = "thread"


def _safe_execute(func: Callable[[], None], error_message: str) -> None:
    """Safely execute a function with error handling."""
    try:
        func()
    except Exception as e:
        logging.warning(f"{error_message}: {e}")


def setup_experiment_tracking(args: KFGRPOScriptArguments, config: Dict[str, Any]) -> None:
    """Setup experiment tracking with wandb and MLflow."""
    if "wandb" in args.report_to:
        _safe_execute(
            lambda: wandb.init(
                project=DEFAULT_WANDB_PROJECT,
                config=config,
                settings=wandb.Settings(
                    code_dir=DEFAULT_WANDB_CODE_DIR,
                    disable_git=True,
                    start_method=DEFAULT_WANDB_START_METHOD
                )
            ),
            "Failed to initialize Wandb"
        )
        logging.info("Wandb initialized successfully")
    
    def _init_mlflow():
        mlflow.start_run()
        mlflow.log_params(config)
    
    _safe_execute(_init_mlflow, "Failed to initialize MLflow")
    logging.info("MLflow initialized successfully")


def finish_experiment_tracking(args: KFGRPOScriptArguments) -> None:
    """Finish experiment tracking."""
    _safe_execute(mlflow.end_run, "Failed to end MLflow run")
    
    if "wandb" in args.report_to:
        _safe_execute(wandb.finish, "Failed to finish Wandb")

