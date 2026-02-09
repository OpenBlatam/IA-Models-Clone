"""
Dataset Loader Module
=====================

Handles loading and validation of JSON datasets with prompt-response format.
Provides dataset splitting and validation utilities.

Author: BUL System
Date: 2024
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datasets import Dataset, DatasetDict

# Import format loaders
try:
    from .data.formats import DatasetFormatLoader
except ImportError:
    # Fallback if data module not available
    DatasetFormatLoader = None

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads and validates JSON datasets for LLM training.
    
    Supports datasets with "prompt" and "response" fields.
    Handles both list and nested dictionary formats.
    
    Attributes:
        dataset_path: Path to the dataset file
        raw_data: Loaded raw data from JSON
        validation_split: Fraction of data for validation (default: 0.2)
        min_size_for_split: Minimum dataset size to create validation split
        
    Example:
        >>> loader = DatasetLoader("data/training.json")
        >>> data = loader.load()
        >>> dataset = loader.prepare_dataset(data)
    """
    
    def __init__(
        self,
        dataset_path: Union[str, Path],
        validation_split: float = 0.2,
        min_size_for_split: int = 100
    ):
        """
        Initialize DatasetLoader.
        
        Args:
            dataset_path: Path to JSON dataset file
            validation_split: Fraction of data for validation (default: 0.2)
            min_size_for_split: Minimum size to create validation split (default: 100)
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
        """
        self.dataset_path = Path(dataset_path)
        self.validation_split = validation_split
        self.min_size_for_split = min_size_for_split
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        logger.info(f"DatasetLoader initialized for: {self.dataset_path}")
    
    def load(self) -> List[Dict[str, str]]:
        """
        Load and validate dataset from file.
        
        Supports multiple formats:
        - JSON (.json)
        - CSV (.csv)
        - Parquet (.parquet) - requires pandas
        
        Returns:
            List of dictionaries with "prompt" and "response" keys
            
        Raises:
            ValueError: If dataset format is invalid
            FileNotFoundError: If file doesn't exist
        """
        try:
            # Use format loader if available (for CSV/Parquet support)
            if DatasetFormatLoader:
                try:
                    data = DatasetFormatLoader.load(self.dataset_path)
                    # Validate format
                    self._validate_data(data)
                    logger.info(f"Loaded {len(data)} examples from {self.dataset_path.suffix} file")
                    return data
                except Exception as e:
                    # Fallback to JSON if format loader fails
                    logger.debug(f"Format loader failed, trying JSON: {e}")
            
            # Fallback to JSON loading
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both list and dict formats
            data = self._normalize_format(data)
            
            # Validate format
            self._validate_data(data)
            
            logger.info(f"Loaded {len(data)} examples from dataset")
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")
    
    def _normalize_format(self, data: Union[List, Dict]) -> List[Dict[str, str]]:
        """
        Normalize dataset format to list of dictionaries.
        
        Args:
            data: Raw loaded data (list or dict)
            
        Returns:
            List of dictionaries
            
        Raises:
            ValueError: If format is invalid
        """
        if isinstance(data, dict):
            # Try common keys
            for key in ["data", "examples", "dataset", "samples"]:
                if key in data:
                    data = data[key]
                    break
            else:
                raise ValueError(
                    "Dataset dict must contain one of: 'data', 'examples', 'dataset', 'samples'"
                )
        
        if not isinstance(data, list):
            raise ValueError("Dataset must be a list of examples or a dict with a list value")
        
        return data
    
    def _validate_data(self, data: List[Dict]) -> None:
        """
        Validate dataset format and content.
        
        Args:
            data: List of example dictionaries
            
        Raises:
            ValueError: If validation fails
        """
        if not data:
            raise ValueError("Dataset is empty")
        
        required_fields = ["prompt", "response"]
        
        for i, example in enumerate(data):
            if not isinstance(example, dict):
                raise ValueError(f"Example {i} must be a dictionary")
            
            for field in required_fields:
                if field not in example:
                    raise ValueError(f"Example {i} missing required field: '{field}'")
                
                if not isinstance(example[field], str):
                    raise ValueError(
                        f"Example {i} field '{field}' must be a string, "
                        f"got {type(example[field]).__name__}"
                    )
                
                if not example[field].strip():
                    raise ValueError(f"Example {i} field '{field}' is empty")
    
    def prepare_dataset(
        self,
        data: List[Dict[str, str]],
        split: bool = True
    ) -> Union[Dataset, DatasetDict]:
        """
        Prepare HuggingFace dataset from raw data.
        
        Args:
            data: List of example dictionaries
            split: Whether to split into train/validation
            
        Returns:
            Dataset or DatasetDict
        """
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(data)
        
        if split and len(dataset) >= self.min_size_for_split:
            # Split into train/validation
            dataset = dataset.train_test_split(
                test_size=self.validation_split,
                seed=42
            )
            dataset["validation"] = dataset.pop("test")
            logger.info(
                f"Dataset split: {len(dataset['train'])} train, "
                f"{len(dataset['validation'])} validation"
            )
        else:
            if split:
                logger.info(
                    f"Dataset too small for validation split "
                    f"(< {self.min_size_for_split} examples), "
                    f"using all data for training"
                )
            dataset = DatasetDict({"train": dataset})
        
        return dataset
    
    def get_statistics(self, data: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Args:
            data: List of example dictionaries
            
        Returns:
            Dictionary with dataset statistics
        """
        if not data:
            return {
                "total_examples": 0,
                "prompt_length": {"min": 0, "max": 0, "avg": 0},
                "response_length": {"min": 0, "max": 0, "avg": 0},
            }
        
        prompt_lengths = [len(ex["prompt"]) for ex in data]
        response_lengths = [len(ex["response"]) for ex in data]
        total_lengths = [p + r for p, r in zip(prompt_lengths, response_lengths)]
        
        def _calculate_percentiles(values: List[int]) -> Dict[str, float]:
            """Calculate percentile statistics."""
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            return {
                "p50": sorted_vals[n // 2] if n > 0 else 0,
                "p75": sorted_vals[int(n * 0.75)] if n > 0 else 0,
                "p90": sorted_vals[int(n * 0.90)] if n > 0 else 0,
                "p95": sorted_vals[int(n * 0.95)] if n > 0 else 0,
            }
        
        return {
            "total_examples": len(data),
            "prompt_length": {
                "min": min(prompt_lengths),
                "max": max(prompt_lengths),
                "avg": sum(prompt_lengths) / len(prompt_lengths),
                **_calculate_percentiles(prompt_lengths),
            },
            "response_length": {
                "min": min(response_lengths),
                "max": max(response_lengths),
                "avg": sum(response_lengths) / len(response_lengths),
                **_calculate_percentiles(response_lengths),
            },
            "total_length": {
                "min": min(total_lengths),
                "max": max(total_lengths),
                "avg": sum(total_lengths) / len(total_lengths),
                **_calculate_percentiles(total_lengths),
            },
        }
    
    def validate_dataset_quality(self, data: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Validate dataset quality and return quality metrics.
        
        Args:
            data: List of example dictionaries
            
        Returns:
            Dictionary with quality metrics and warnings
        """
        warnings_list = []
        quality_score = 100.0
        
        if not data:
            return {"quality_score": 0, "warnings": ["Dataset is empty"], "is_valid": False}
        
        # Check for duplicates
        unique_prompts = set(ex["prompt"].lower().strip() for ex in data)
        duplicate_ratio = 1 - (len(unique_prompts) / len(data))
        if duplicate_ratio > 0.1:
            warnings_list.append(f"High duplicate ratio: {duplicate_ratio:.2%}")
            quality_score -= duplicate_ratio * 20
        
        # Check for empty or very short responses
        short_responses = sum(1 for ex in data if len(ex["response"].strip()) < 10)
        if short_responses > len(data) * 0.1:
            warnings_list.append(f"Many short responses: {short_responses}/{len(data)}")
            quality_score -= (short_responses / len(data)) * 15
        
        # Check for very long responses (potential outliers)
        avg_length = sum(len(ex["response"]) for ex in data) / len(data)
        long_responses = sum(1 for ex in data if len(ex["response"]) > avg_length * 3)
        if long_responses > len(data) * 0.05:
            warnings_list.append(f"Some very long responses detected: {long_responses}")
            quality_score -= 5
        
        # Check dataset size
        if len(data) < 10:
            warnings_list.append("Dataset is very small (< 10 examples)")
            quality_score -= 10
        elif len(data) < 100:
            warnings_list.append("Dataset is small (< 100 examples), may not train well")
            quality_score -= 5
        
        return {
            "quality_score": max(0, quality_score),
            "warnings": warnings_list,
            "is_valid": quality_score >= 70,
            "duplicate_ratio": duplicate_ratio,
            "avg_response_length": sum(len(ex["response"]) for ex in data) / len(data),
        }

