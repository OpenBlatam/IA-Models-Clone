from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import math
import random
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import (
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ..models.sequence import EmailSequence, SequenceStep
from ..models.subscriber import Subscriber
from ..models.template import EmailTemplate
from typing import Any, List, Dict, Optional
"""
Data Splitting and Cross-Validation for Email Sequence System

Advanced data splitting strategies including train/validation/test splits,
cross-validation, stratified sampling, and time-series aware splitting
for email sequence datasets.
"""


    train_test_split,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    GroupKFold,
    LeaveOneOut,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)


logger = logging.getLogger(__name__)


@dataclass
class DataSplitConfig:
    """Configuration for data splitting and cross-validation"""
    # Split ratios
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Cross-validation parameters
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # stratified, kfold, timeseries, group
    
    # Stratification parameters
    stratify_by: str = "sequence_type"  # sequence_type, subscriber_type, template_category
    n_strata: int = 10
    
    # Time-series parameters
    time_column: str = "created_at"
    time_order: bool = True
    
    # Group parameters
    group_by: str = "subscriber_id"  # subscriber_id, sequence_id, template_id
    
    # Random state
    random_state: int = 42
    
    # Validation parameters
    ensure_distribution: bool = True
    min_samples_per_split: int = 10
    
    # Output parameters
    save_splits: bool = True
    splits_dir: str = "./data_splits"


class DataSplitter:
    """Advanced data splitter for email sequences"""
    
    def __init__(self, config: DataSplitConfig):
        
    """__init__ function."""
self.config = config
        self.splits = {}
        self.split_metadata = {}
        
        # Set random state
        random.seed(config.random_state)
        np.random.seed(config.random_state)
        torch.manual_seed(config.random_state)
        
        logger.info("Data Splitter initialized")
    
    def split_data(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        strategy: str = "random"
    ) -> Dict[str, List[int]]:
        """Split data using specified strategy"""
        
        if strategy == "random":
            return self._random_split(sequences, subscribers, templates)
        elif strategy == "stratified":
            return self._stratified_split(sequences, subscribers, templates)
        elif strategy == "time_series":
            return self._time_series_split(sequences, subscribers, templates)
        elif strategy == "group":
            return self._group_split(sequences, subscribers, templates)
        elif strategy == "sequence_aware":
            return self._sequence_aware_split(sequences, subscribers, templates)
        else:
            raise ValueError(f"Unknown splitting strategy: {strategy}")
    
    def _random_split(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate]
    ) -> Dict[str, List[int]]:
        """Random split of data"""
        
        # Create indices for all data points
        all_indices = list(range(len(sequences)))
        random.shuffle(all_indices)
        
        # Calculate split points
        n_total = len(all_indices)
        n_train = int(n_total * self.config.train_ratio)
        n_val = int(n_total * self.config.validation_ratio)
        
        # Split indices
        train_indices = all_indices[:n_train]
        val_indices = all_indices[n_train:n_train + n_val]
        test_indices = all_indices[n_train + n_val:]
        
        splits = {
            "train": train_indices,
            "validation": val_indices,
            "test": test_indices
        }
        
        # Store metadata
        self.split_metadata = {
            "strategy": "random",
            "total_samples": n_total,
            "train_samples": len(train_indices),
            "validation_samples": len(val_indices),
            "test_samples": len(test_indices)
        }
        
        return splits
    
    def _stratified_split(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate]
    ) -> Dict[str, List[int]]:
        """Stratified split based on specified criteria"""
        
        # Create stratification labels
        if self.config.stratify_by == "sequence_type":
            labels = self._get_sequence_type_labels(sequences)
        elif self.config.stratify_by == "subscriber_type":
            labels = self._get_subscriber_type_labels(sequences, subscribers)
        elif self.config.stratify_by == "template_category":
            labels = self._get_template_category_labels(sequences, templates)
        else:
            raise ValueError(f"Unknown stratification criteria: {self.config.stratify_by}")
        
        # Create indices
        indices = list(range(len(sequences)))
        
        # Perform stratified split
        train_indices, temp_indices, train_labels, temp_labels = train_test_split(
            indices, labels,
            test_size=(1 - self.config.train_ratio),
            stratify=labels,
            random_state=self.config.random_state
        )
        
        # Split remaining data into validation and test
        val_ratio = self.config.validation_ratio / (self.config.validation_ratio + self.config.test_ratio)
        val_indices, test_indices, _, _ = train_test_split(
            temp_indices, temp_labels,
            test_size=(1 - val_ratio),
            stratify=temp_labels,
            random_state=self.config.random_state
        )
        
        splits = {
            "train": train_indices,
            "validation": val_indices,
            "test": test_indices
        }
        
        # Store metadata
        self.split_metadata = {
            "strategy": "stratified",
            "stratify_by": self.config.stratify_by,
            "total_samples": len(indices),
            "train_samples": len(train_indices),
            "validation_samples": len(val_indices),
            "test_samples": len(test_indices),
            "strata_distribution": self._get_strata_distribution(labels, splits)
        }
        
        return splits
    
    def _get_sequence_type_labels(self, sequences: List[EmailSequence]) -> List[str]:
        """Get sequence type labels for stratification"""
        labels = []
        for sequence in sequences:
            # Categorize sequences based on length and content
            if len(sequence.steps) <= 2:
                labels.append("short")
            elif len(sequence.steps) <= 5:
                labels.append("medium")
            else:
                labels.append("long")
        return labels
    
    def _get_subscriber_type_labels(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber]
    ) -> List[str]:
        """Get subscriber type labels for stratification"""
        labels = []
        for sequence in sequences:
            # Find corresponding subscriber
            subscriber = next((s for s in subscribers if s.id == sequence.subscriber_id), None)
            if subscriber:
                if subscriber.company_size == "small":
                    labels.append("small_company")
                elif subscriber.company_size == "medium":
                    labels.append("medium_company")
                else:
                    labels.append("large_company")
            else:
                labels.append("unknown")
        return labels
    
    def _get_template_category_labels(
        self,
        sequences: List[EmailSequence],
        templates: List[EmailTemplate]
    ) -> List[str]:
        """Get template category labels for stratification"""
        labels = []
        for sequence in sequences:
            # Find corresponding template
            template = next((t for t in templates if t.id == sequence.template_id), None)
            if template:
                labels.append(template.category)
            else:
                labels.append("unknown")
        return labels
    
    def _time_series_split(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate]
    ) -> Dict[str, List[int]]:
        """Time-series aware split"""
        
        # Create DataFrame with timestamps
        data = []
        for i, sequence in enumerate(sequences):
            data.append({
                "index": i,
                "created_at": sequence.created_at if hasattr(sequence, 'created_at') else pd.Timestamp.now(),
                "sequence_id": sequence.id
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values("created_at")
        
        # Calculate split points
        n_total = len(df)
        n_train = int(n_total * self.config.train_ratio)
        n_val = int(n_total * self.config.validation_ratio)
        
        # Split based on time order
        train_indices = df.iloc[:n_train]["index"].tolist()
        val_indices = df.iloc[n_train:n_train + n_val]["index"].tolist()
        test_indices = df.iloc[n_train + n_val:]["index"].tolist()
        
        splits = {
            "train": train_indices,
            "validation": val_indices,
            "test": test_indices
        }
        
        # Store metadata
        self.split_metadata = {
            "strategy": "time_series",
            "total_samples": n_total,
            "train_samples": len(train_indices),
            "validation_samples": len(val_indices),
            "test_samples": len(test_indices),
            "time_range": {
                "train": (df.iloc[0]["created_at"], df.iloc[n_train-1]["created_at"]),
                "validation": (df.iloc[n_train]["created_at"], df.iloc[n_train+n_val-1]["created_at"]),
                "test": (df.iloc[n_train+n_val]["created_at"], df.iloc[-1]["created_at"])
            }
        }
        
        return splits
    
    def _group_split(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate]
    ) -> Dict[str, List[int]]:
        """Group-based split to avoid data leakage"""
        
        if self.config.group_by == "subscriber_id":
            groups = [seq.subscriber_id for seq in sequences]
        elif self.config.group_by == "sequence_id":
            groups = [seq.id for seq in sequences]
        elif self.config.group_by == "template_id":
            groups = [seq.template_id for seq in sequences]
        else:
            raise ValueError(f"Unknown group by criteria: {self.config.group_by}")
        
        # Get unique groups
        unique_groups = list(set(groups))
        random.shuffle(unique_groups)
        
        # Split groups
        n_groups = len(unique_groups)
        n_train_groups = int(n_groups * self.config.train_ratio)
        n_val_groups = int(n_groups * self.config.validation_ratio)
        
        train_groups = unique_groups[:n_train_groups]
        val_groups = unique_groups[n_train_groups:n_train_groups + n_val_groups]
        test_groups = unique_groups[n_train_groups + n_val_groups:]
        
        # Get indices for each split
        train_indices = [i for i, group in enumerate(groups) if group in train_groups]
        val_indices = [i for i, group in enumerate(groups) if group in val_groups]
        test_indices = [i for i, group in enumerate(groups) if group in test_groups]
        
        splits = {
            "train": train_indices,
            "validation": val_indices,
            "test": test_indices
        }
        
        # Store metadata
        self.split_metadata = {
            "strategy": "group",
            "group_by": self.config.group_by,
            "total_samples": len(sequences),
            "total_groups": n_groups,
            "train_samples": len(train_indices),
            "validation_samples": len(val_indices),
            "test_samples": len(test_indices),
            "train_groups": len(train_groups),
            "validation_groups": len(val_groups),
            "test_groups": len(test_groups)
        }
        
        return splits
    
    def _sequence_aware_split(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate]
    ) -> Dict[str, List[int]]:
        """Sequence-aware split considering sequence completeness"""
        
        # Categorize sequences by completeness
        complete_sequences = []
        incomplete_sequences = []
        
        for i, sequence in enumerate(sequences):
            if len(sequence.steps) >= 3 and all(step.content for step in sequence.steps):
                complete_sequences.append(i)
            else:
                incomplete_sequences.append(i)
        
        # Split complete sequences
        random.shuffle(complete_sequences)
        n_complete = len(complete_sequences)
        n_train_complete = int(n_complete * self.config.train_ratio)
        n_val_complete = int(n_complete * self.config.validation_ratio)
        
        train_complete = complete_sequences[:n_train_complete]
        val_complete = complete_sequences[n_train_complete:n_train_complete + n_val_complete]
        test_complete = complete_sequences[n_train_complete + n_val_complete:]
        
        # Split incomplete sequences
        random.shuffle(incomplete_sequences)
        n_incomplete = len(incomplete_sequences)
        n_train_incomplete = int(n_incomplete * self.config.train_ratio)
        n_val_incomplete = int(n_incomplete * self.config.validation_ratio)
        
        train_incomplete = incomplete_sequences[:n_train_incomplete]
        val_incomplete = incomplete_sequences[n_train_incomplete:n_train_incomplete + n_val_incomplete]
        test_incomplete = incomplete_sequences[n_train_incomplete + n_val_incomplete:]
        
        # Combine splits
        splits = {
            "train": train_complete + train_incomplete,
            "validation": val_complete + val_incomplete,
            "test": test_complete + test_incomplete
        }
        
        # Store metadata
        self.split_metadata = {
            "strategy": "sequence_aware",
            "total_samples": len(sequences),
            "complete_sequences": n_complete,
            "incomplete_sequences": n_incomplete,
            "train_samples": len(splits["train"]),
            "validation_samples": len(splits["validation"]),
            "test_samples": len(splits["test"]),
            "completeness_distribution": {
                "train": {"complete": len(train_complete), "incomplete": len(train_incomplete)},
                "validation": {"complete": len(val_complete), "incomplete": len(val_incomplete)},
                "test": {"complete": len(test_complete), "incomplete": len(test_incomplete)}
            }
        }
        
        return splits
    
    def _get_strata_distribution(
        self,
        labels: List[str],
        splits: Dict[str, List[int]]
    ) -> Dict[str, Dict[str, int]]:
        """Get distribution of strata across splits"""
        
        distribution = {}
        for split_name, indices in splits.items():
            split_labels = [labels[i] for i in indices]
            distribution[split_name] = {}
            for label in set(labels):
                distribution[split_name][label] = split_labels.count(label)
        
        return distribution
    
    def save_splits(self, filepath: str):
        """Save splits to file"""
        
        if self.config.save_splits:
            splits_dir = Path(self.config.splits_dir)
            splits_dir.mkdir(parents=True, exist_ok=True)
            
            filepath = splits_dir / filepath
            
            data = {
                "splits": self.splits,
                "metadata": self.split_metadata,
                "config": self.config.__dict__
            }
            
            with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Splits saved to {filepath}")
    
    def load_splits(self, filepath: str) -> Dict[str, List[int]]:
        """Load splits from file"""
        
        splits_dir = Path(self.config.splits_dir)
        filepath = splits_dir / filepath
        
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            data = json.load(f)
        
        self.splits = data["splits"]
        self.split_metadata = data["metadata"]
        
        logger.info(f"Splits loaded from {filepath}")
        return self.splits


class CrossValidator:
    """Advanced cross-validation for email sequences"""
    
    def __init__(self, config: DataSplitConfig):
        
    """__init__ function."""
self.config = config
        self.cv_results = {}
        
        logger.info("Cross Validator initialized")
    
    def cross_validate(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        model_func: Callable,
        cv_strategy: str = None
    ) -> Dict[str, Any]:
        """Perform cross-validation"""
        
        if cv_strategy is None:
            cv_strategy = self.config.cv_strategy
        
        if cv_strategy == "stratified":
            return self._stratified_cv(sequences, subscribers, templates, model_func)
        elif cv_strategy == "kfold":
            return self._kfold_cv(sequences, subscribers, templates, model_func)
        elif cv_strategy == "timeseries":
            return self._timeseries_cv(sequences, subscribers, templates, model_func)
        elif cv_strategy == "group":
            return self._group_cv(sequences, subscribers, templates, model_func)
        else:
            raise ValueError(f"Unknown CV strategy: {cv_strategy}")
    
    def _stratified_cv(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        model_func: Callable
    ) -> Dict[str, Any]:
        """Stratified cross-validation"""
        
        # Create labels for stratification
        labels = self._create_stratification_labels(sequences, subscribers, templates)
        
        # Create cross-validation splits
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        return self._perform_cv(cv, sequences, labels, model_func, "stratified")
    
    def _kfold_cv(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        model_func: Callable
    ) -> Dict[str, Any]:
        """K-fold cross-validation"""
        
        cv = KFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        return self._perform_cv(cv, sequences, None, model_func, "kfold")
    
    def _timeseries_cv(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        model_func: Callable
    ) -> Dict[str, Any]:
        """Time-series cross-validation"""
        
        cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        return self._perform_cv(cv, sequences, None, model_func, "timeseries")
    
    def _group_cv(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        model_func: Callable
    ) -> Dict[str, Any]:
        """Group cross-validation"""
        
        # Create groups
        if self.config.group_by == "subscriber_id":
            groups = [seq.subscriber_id for seq in sequences]
        elif self.config.group_by == "sequence_id":
            groups = [seq.id for seq in sequences]
        else:
            groups = [seq.template_id for seq in sequences]
        
        cv = GroupKFold(n_splits=self.config.cv_folds)
        
        return self._perform_cv(cv, sequences, groups, model_func, "group")
    
    def _perform_cv(
        self,
        cv,
        sequences: List[EmailSequence],
        labels: Optional[List] = None,
        model_func: Callable = None,
        cv_type: str = "unknown"
    ) -> Dict[str, Any]:
        """Perform cross-validation with given strategy"""
        
        indices = list(range(len(sequences)))
        scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(indices, labels)):
            logger.info(f"Training fold {fold + 1}/{self.config.cv_folds}")
            
            # Split data
            train_sequences = [sequences[i] for i in train_idx]
            val_sequences = [sequences[i] for i in val_idx]
            
            # Train and evaluate model
            if model_func:
                score = model_func(train_sequences, val_sequences)
                scores.append(score)
            
            fold_results.append({
                "fold": fold + 1,
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "score": score if model_func else None
            })
        
        # Calculate statistics
        cv_results = {
            "cv_type": cv_type,
            "n_folds": self.config.cv_folds,
            "scores": scores,
            "mean_score": np.mean(scores) if scores else None,
            "std_score": np.std(scores) if scores else None,
            "fold_results": fold_results
        }
        
        self.cv_results = cv_results
        return cv_results
    
    def _create_stratification_labels(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate]
    ) -> List[str]:
        """Create labels for stratification"""
        
        if self.config.stratify_by == "sequence_type":
            return self._get_sequence_type_labels(sequences)
        elif self.config.stratify_by == "subscriber_type":
            return self._get_subscriber_type_labels(sequences, subscribers)
        elif self.config.stratify_by == "template_category":
            return self._get_template_category_labels(sequences, templates)
        else:
            return [str(i % self.config.n_strata) for i in range(len(sequences))]
    
    def _get_sequence_type_labels(self, sequences: List[EmailSequence]) -> List[str]:
        """Get sequence type labels"""
        labels = []
        for sequence in sequences:
            if len(sequence.steps) <= 2:
                labels.append("short")
            elif len(sequence.steps) <= 5:
                labels.append("medium")
            else:
                labels.append("long")
        return labels
    
    def _get_subscriber_type_labels(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber]
    ) -> List[str]:
        """Get subscriber type labels"""
        labels = []
        for sequence in sequences:
            subscriber = next((s for s in subscribers if s.id == sequence.subscriber_id), None)
            if subscriber:
                labels.append(subscriber.company_size or "unknown")
            else:
                labels.append("unknown")
        return labels
    
    def _get_template_category_labels(
        self,
        sequences: List[EmailSequence],
        templates: List[EmailTemplate]
    ) -> List[str]:
        """Get template category labels"""
        labels = []
        for sequence in sequences:
            template = next((t for t in templates if t.id == sequence.template_id), None)
            if template:
                labels.append(template.category)
            else:
                labels.append("unknown")
        return labels
    
    def plot_cv_results(self, save_path: str = None):
        """Plot cross-validation results"""
        
        if not self.cv_results or not self.cv_results.get("scores"):
            logger.warning("No CV results to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot scores across folds
        axes[0].plot(range(1, len(self.cv_results["scores"]) + 1), self.cv_results["scores"], 'bo-')
        axes[0].axhline(y=self.cv_results["mean_score"], color='r', linestyle='--', label=f'Mean: {self.cv_results["mean_score"]:.4f}')
        axes[0].set_title(f"{self.cv_results['cv_type'].title()} Cross-Validation Scores")
        axes[0].set_xlabel("Fold")
        axes[0].set_ylabel("Score")
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot score distribution
        axes[1].hist(self.cv_results["scores"], bins=10, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=self.cv_results["mean_score"], color='r', linestyle='--', label=f'Mean: {self.cv_results["mean_score"]:.4f}')
        axes[1].set_title("Score Distribution")
        axes[1].set_xlabel("Score")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


class DataSplitManager:
    """Manager for data splitting and cross-validation"""
    
    def __init__(self, config: DataSplitConfig):
        
    """__init__ function."""
self.config = config
        self.splitter = DataSplitter(config)
        self.cross_validator = CrossValidator(config)
        
        # Performance tracking
        self.manager_stats = defaultdict(int)
        
        logger.info("Data Split Manager initialized")
    
    async def create_splits(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        strategy: str = "stratified"
    ) -> Dict[str, List[int]]:
        """Create data splits"""
        
        splits = self.splitter.split_data(sequences, subscribers, templates, strategy)
        
        # Validate splits
        self._validate_splits(splits)
        
        # Save splits
        if self.config.save_splits:
            self.splitter.save_splits(f"splits_{strategy}_{int(time.time())}.json")
        
        self.manager_stats[f"{strategy}_splits_created"] += 1
        
        return splits
    
    async def perform_cross_validation(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        model_func: Callable = None,
        cv_strategy: str = None
    ) -> Dict[str, Any]:
        """Perform cross-validation"""
        
        cv_results = self.cross_validator.cross_validate(
            sequences, subscribers, templates, model_func, cv_strategy
        )
        
        # Plot results
        if self.config.save_splits:
            plots_dir = Path(self.config.splits_dir) / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            self.cross_validator.plot_cv_results(
                plots_dir / f"cv_results_{cv_strategy or self.config.cv_strategy}.png"
            )
        
        self.manager_stats["cv_runs"] += 1
        
        return cv_results
    
    def _validate_splits(self, splits: Dict[str, List[int]]):
        """Validate data splits"""
        
        # Check for overlap
        train_set = set(splits["train"])
        val_set = set(splits["validation"])
        test_set = set(splits["test"])
        
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise ValueError("Data splits have overlapping indices")
        
        # Check minimum samples
        for split_name, indices in splits.items():
            if len(indices) < self.config.min_samples_per_split:
                raise ValueError(f"{split_name} split has fewer than {self.config.min_samples_per_split} samples")
        
        # Check distribution if required
        if self.config.ensure_distribution:
            total_samples = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
            expected_ratios = {
                "train": self.config.train_ratio,
                "validation": self.config.validation_ratio,
                "test": self.config.test_ratio
            }
            
            for split_name, indices in splits.items():
                actual_ratio = len(indices) / total_samples
                expected_ratio = expected_ratios[split_name]
                
                if abs(actual_ratio - expected_ratio) > 0.05:  # 5% tolerance
                    logger.warning(f"{split_name} split ratio ({actual_ratio:.3f}) differs significantly from expected ({expected_ratio:.3f})")
    
    async def get_split_report(self) -> Dict[str, Any]:
        """Generate comprehensive split report"""
        
        return {
            "manager_stats": dict(self.manager_stats),
            "splitter_metadata": self.splitter.split_metadata,
            "cv_results": self.cross_validator.cv_results,
            "config": {
                "train_ratio": self.config.train_ratio,
                "validation_ratio": self.config.validation_ratio,
                "test_ratio": self.config.test_ratio,
                "cv_folds": self.config.cv_folds,
                "cv_strategy": self.config.cv_strategy,
                "stratify_by": self.config.stratify_by
            }
        } 