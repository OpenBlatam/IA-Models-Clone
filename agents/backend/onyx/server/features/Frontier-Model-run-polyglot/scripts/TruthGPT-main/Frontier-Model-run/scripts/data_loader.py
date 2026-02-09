"""Data loading utilities for training."""
from typing import Any, Optional

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer

DEFAULT_STREAMING = True
DEFAULT_TRUST_REMOTE_CODE = True


def load_training_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    cache_dir: Optional[str] = None,
    streaming: bool = DEFAULT_STREAMING
) -> DatasetDict:
    """Load dataset for training."""
    if not dataset_name:
        raise ValueError("dataset_name cannot be empty")
    
    try:
        return load_dataset(
            dataset_name,
            name=dataset_config,
            cache_dir=cache_dir,
            streaming=streaming
        )
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset_name}': {e}")


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = DEFAULT_TRUST_REMOTE_CODE
) -> PreTrainedTokenizerBase:
    """Load tokenizer for the model."""
    if not model_name:
        raise ValueError("model_name cannot be empty")
    
    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer for model '{model_name}': {e}")


