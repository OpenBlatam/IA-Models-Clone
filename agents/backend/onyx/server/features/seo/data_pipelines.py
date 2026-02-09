from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Iterator, Generator
from dataclasses import dataclass, field
from functools import partial, reduce
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Functional Data Processing Pipelines for SEO Service
Pure functions and data transformations following functional programming principles
"""


logger = logging.getLogger(__name__)

@dataclass
class TextData:
    """Immutable data structure for text data"""
    text: str
    label: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> Any:
        """Validate data after initialization"""
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if len(self.text.strip()) == 0:
            raise ValueError("text cannot be empty")

@dataclass
class ProcessedData:
    """Immutable data structure for processed data"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Pure functions for text preprocessing
def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters"""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep basic punctuation
    cleaned = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', cleaned)
    
    return cleaned

def normalize_text(text: str) -> str:
    """Normalize text to lowercase and standardize formatting"""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Convert to lowercase
    normalized = text.lower()
    
    # Standardize whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized.strip()

def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text"""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Remove HTML tags
    cleaned = re.sub(r'<[^>]+>', '', text)
    
    # Remove HTML entities
    cleaned = re.sub(r'&[a-zA-Z]+;', '', cleaned)
    
    return cleaned.strip()

def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract potential keywords from text"""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Clean and normalize text
    cleaned = normalize_text(clean_text(text))
    
    # Split into words
    words = re.findall(r'\b\w+\b', cleaned)
    
    # Filter by length and common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    keywords = [word for word in words if len(word) >= min_length and word not in stop_words]
    
    return keywords

def calculate_text_metrics(text: str) -> Dict[str, Any]:
    """Calculate various text metrics"""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    cleaned = clean_text(text)
    words = cleaned.split()
    sentences = re.split(r'[.!?]+', cleaned)
    
    return {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'avg_sentence_length': np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0,
        'unique_words': len(set(words)),
        'text_length': len(cleaned)
    }

# Functional data transformation pipelines
def create_text_preprocessing_pipeline(*functions: Callable[[str], str]) -> Callable[[str], str]:
    """Create a pipeline of text preprocessing functions"""
    def pipeline(text: str) -> str:
        return reduce(lambda result, func: func(result), functions, text)
    return pipeline

def create_data_validation_pipeline(*validators: Callable[[TextData], bool]) -> Callable[[TextData], bool]:
    """Create a pipeline of data validation functions"""
    def pipeline(data: TextData) -> bool:
        return all(validator(data) for validator in validators)
    return pipeline

# Data validation functions
def validate_text_length(data: TextData, min_length: int = 10, max_length: int = 10000) -> bool:
    """Validate text length"""
    return min_length <= len(data.text) <= max_length

def validate_text_content(data: TextData) -> bool:
    """Validate text content quality"""
    # Check for meaningful content
    words = data.text.split()
    if len(words) < 3:
        return False
    
    # Check for excessive repetition
    unique_words = set(words)
    if len(unique_words) / len(words) < 0.3:
        return False
    
    return True

def validate_label(data: TextData, num_classes: int) -> bool:
    """Validate label if present"""
    if data.label is None:
        return True
    return 0 <= data.label < num_classes

# Data loading functions
async def load_text_data_from_file(file_path: str) -> List[TextData]:
    """Load text data from file asynchronously"""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = []
    
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        async for line in file:
            line = line.strip()
            if line:
                try:
                    # Parse JSON line format
                    parsed = json.loads(line)
                    text_data = TextData(
                        text=parsed.get('text', ''),
                        label=parsed.get('label'),
                        metadata=parsed.get('metadata', {})
                    )
                    data.append(text_data)
                except json.JSONDecodeError:
                    # Treat as plain text
                    data.append(TextData(text=line))
    
    return data

def load_text_data_from_dataframe(df: pd.DataFrame, text_column: str, label_column: Optional[str] = None) -> List[TextData]:
    """Load text data from pandas DataFrame"""
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame")
    
    data = []
    
    for _, row in df.iterrows():
        text = str(row[text_column])
        label = row.get(label_column) if label_column else None
        metadata = {col: row[col] for col in df.columns if col not in [text_column, label_column]}
        
        data.append(TextData(text=text, label=label, metadata=metadata))
    
    return data

# Data filtering functions
def filter_data_by_condition(data: List[TextData], condition: Callable[[TextData], bool]) -> List[TextData]:
    """Filter data based on condition"""
    return list(filter(condition, data))

def filter_by_text_length(data: List[TextData], min_length: int = 10, max_length: int = 10000) -> List[TextData]:
    """Filter data by text length"""
    return filter_data_by_condition(data, lambda x: validate_text_length(x, min_length, max_length))

def filter_by_content_quality(data: List[TextData]) -> List[TextData]:
    """Filter data by content quality"""
    return filter_data_by_condition(data, validate_text_content)

# Data transformation functions
def transform_text_data(data: List[TextData], transform_func: Callable[[str], str]) -> List[TextData]:
    """Transform text data using a function"""
    return [
        TextData(
            text=transform_func(item.text),
            label=item.label,
            metadata=item.metadata
        )
        for item in data
    ]

def add_text_metrics(data: List[TextData]) -> List[TextData]:
    """Add text metrics to metadata"""
    return [
        TextData(
            text=item.text,
            label=item.label,
            metadata={**item.metadata, 'metrics': calculate_text_metrics(item.text)}
        )
        for item in data
    ]

def add_keywords(data: List[TextData], min_length: int = 3) -> List[TextData]:
    """Add extracted keywords to metadata"""
    return [
        TextData(
            text=item.text,
            label=item.label,
            metadata={**item.metadata, 'keywords': extract_keywords(item.text, min_length)}
        )
        for item in data
    ]

# Tokenization functions
def create_tokenization_pipeline(tokenizer_name: str, max_length: int = 512) -> Callable[[List[TextData]], List[ProcessedData]]:
    """Create a tokenization pipeline"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def tokenize_batch(data: List[TextData]) -> List[ProcessedData]:
        texts = [item.text for item in data]
        labels = [item.label for item in data]
        
        # Tokenize
        encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Convert labels to tensor if present
        label_tensor = None
        if any(label is not None for label in labels):
            label_tensor = torch.tensor([label if label is not None else -100 for label in labels])
        
        return [
            ProcessedData(
                input_ids=encodings['input_ids'][i],
                attention_mask=encodings['attention_mask'][i],
                labels=label_tensor[i] if label_tensor is not None else None,
                metadata=data[i].metadata
            )
            for i in range(len(data))
        ]
    
    return tokenize_batch

# Data splitting functions
def split_data(data: List[TextData], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[TextData], List[TextData], List[TextData]]:
    """Split data into train, validation, and test sets"""
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("Ratios must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("Sum of train and validation ratios must be less than 1")
    
    # Shuffle data
    shuffled = data.copy()
    np.random.shuffle(shuffled)
    
    total_size = len(shuffled)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = shuffled[:train_size]
    val_data = shuffled[train_size:train_size + val_size]
    test_data = shuffled[train_size + val_size:]
    
    return train_data, val_data, test_data

def stratified_split_data(data: List[TextData], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[TextData], List[TextData], List[TextData]]:
    """Stratified split maintaining label distribution"""
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("Ratios must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("Sum of train and validation ratios must be less than 1")
    
    # Group by label
    label_groups = {}
    for item in data:
        label = item.label if item.label is not None else 'unlabeled'
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(item)
    
    train_data, val_data, test_data = [], [], []
    
    for label, group in label_groups.items():
        np.random.shuffle(group)
        total_size = len(group)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        train_data.extend(group[:train_size])
        val_data.extend(group[train_size:train_size + val_size])
        test_data.extend(group[train_size + val_size:])
    
    return train_data, val_data, test_data

# Data augmentation functions
def augment_text_data(data: List[TextData], augmentation_funcs: List[Callable[[str], str]], num_augmentations: int = 1) -> List[TextData]:
    """Augment text data using multiple functions"""
    augmented_data = []
    
    for item in data:
        augmented_data.append(item)  # Original data
        
        for _ in range(num_augmentations):
            for func in augmentation_funcs:
                try:
                    augmented_text = func(item.text)
                    if augmented_text != item.text:  # Only add if different
                        augmented_data.append(TextData(
                            text=augmented_text,
                            label=item.label,
                            metadata={**item.metadata, 'augmented': True}
                        ))
                except Exception as e:
                    logger.warning(f"Augmentation failed: {e}")
    
    return augmented_data

# Data sampling functions
def sample_data(data: List[TextData], sample_size: int, random_state: Optional[int] = None) -> List[TextData]:
    """Sample data randomly"""
    if random_state is not None:
        np.random.seed(random_state)
    
    if sample_size >= len(data):
        return data
    
    indices = np.random.choice(len(data), sample_size, replace=False)
    return [data[i] for i in indices]

def sample_data_by_label(data: List[TextData], samples_per_label: int, random_state: Optional[int] = None) -> List[TextData]:
    """Sample data maintaining label balance"""
    if random_state is not None:
        np.random.seed(random_state)
    
    # Group by label
    label_groups = {}
    for item in data:
        label = item.label if item.label is not None else 'unlabeled'
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(item)
    
    sampled_data = []
    for label, group in label_groups.items():
        if len(group) <= samples_per_label:
            sampled_data.extend(group)
        else:
            sampled_indices = np.random.choice(len(group), samples_per_label, replace=False)
            sampled_data.extend([group[i] for i in sampled_indices])
    
    return sampled_data

# Data export functions
async def export_data_to_jsonl(data: List[TextData], file_path: str) -> None:
    """Export data to JSONL format"""
    async with aiofiles.open(file_path, 'w', encoding='utf-8') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        for item in data:
            json_line = json.dumps({
                'text': item.text,
                'label': item.label,
                'metadata': item.metadata
            }, ensure_ascii=False)
            await file.write(json_line + '\n')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")

def export_data_to_dataframe(data: List[TextData]) -> pd.DataFrame:
    """Export data to pandas DataFrame"""
    records = []
    for item in data:
        record = {
            'text': item.text,
            'label': item.label,
            **item.metadata
        }
        records.append(record)
    
    return pd.DataFrame(records)

# Pipeline composition functions
def compose_pipelines(*pipelines: Callable) -> Callable:
    """Compose multiple pipelines into a single pipeline"""
    def composed_pipeline(data: Any) -> Any:
        result = data
        for pipeline in pipelines:
            result = pipeline(result)
        return result
    return composed_pipeline

# Example pipeline configurations
def create_seo_preprocessing_pipeline() -> Callable[[List[TextData]], List[TextData]]:
    """Create a complete SEO preprocessing pipeline"""
    text_pipeline = create_text_preprocessing_pipeline(
        remove_html_tags,
        clean_text,
        normalize_text
    )
    
    validation_pipeline = create_data_validation_pipeline(
        lambda x: validate_text_length(x, min_length=10, max_length=5000),
        validate_text_content
    )
    
    def full_pipeline(data: List[TextData]) -> List[TextData]:
        # Transform text
        transformed = transform_text_data(data, text_pipeline)
        
        # Filter valid data
        filtered = filter_data_by_condition(transformed, validation_pipeline)
        
        # Add metrics and keywords
        enhanced = add_text_metrics(filtered)
        enhanced = add_keywords(enhanced)
        
        return enhanced
    
    return full_pipeline

def create_training_pipeline(tokenizer_name: str, max_length: int = 512) -> Callable[[List[TextData]], List[ProcessedData]]:
    """Create a complete training data pipeline"""
    preprocessing = create_seo_preprocessing_pipeline()
    tokenization = create_tokenization_pipeline(tokenizer_name, max_length)
    
    return compose_pipelines(preprocessing, tokenization)

# Utility functions
def get_data_statistics(data: List[TextData]) -> Dict[str, Any]:
    """Get comprehensive statistics about the dataset"""
    if not data:
        return {}
    
    # Basic statistics
    total_samples = len(data)
    labeled_samples = sum(1 for item in data if item.label is not None)
    
    # Label distribution
    label_counts = {}
    for item in data:
        if item.label is not None:
            label_counts[item.label] = label_counts.get(item.label, 0) + 1
    
    # Text length statistics
    text_lengths = [len(item.text) for item in data]
    
    # Word count statistics
    word_counts = [len(item.text.split()) for item in data]
    
    return {
        'total_samples': total_samples,
        'labeled_samples': labeled_samples,
        'unlabeled_samples': total_samples - labeled_samples,
        'label_distribution': label_counts,
        'text_length_stats': {
            'mean': np.mean(text_lengths),
            'std': np.std(text_lengths),
            'min': np.min(text_lengths),
            'max': np.max(text_lengths)
        },
        'word_count_stats': {
            'mean': np.mean(word_counts),
            'std': np.std(word_counts),
            'min': np.min(word_counts),
            'max': np.max(word_counts)
        }
    }

# Example usage
if __name__ == "__main__":
    # Example data
    sample_data = [
        TextData("This is a great SEO article about machine learning.", label=1),
        TextData("Poor quality content with no value.", label=0),
        TextData("Excellent guide for beginners in SEO optimization.", label=1)
    ]
    
    # Create and run pipeline
    pipeline = create_seo_preprocessing_pipeline()
    processed_data = pipeline(sample_data)
    
    # Print results
    for item in processed_data:
        print(f"Text: {item.text[:50]}...")
        print(f"Label: {item.label}")
        print(f"Keywords: {item.metadata.get('keywords', [])}")
        print(f"Metrics: {item.metadata.get('metrics', {})}")
        print("---")
    
    # Get statistics
    stats = get_data_statistics(processed_data)
    print(f"Dataset Statistics: {stats}") 