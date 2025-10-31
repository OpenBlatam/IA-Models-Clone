from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, TypeVar, Iterator
from functools import reduce, partial, lru_cache
from dataclasses import dataclass
from enum import Enum
import warnings
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
        import json
            import json
        from sklearn.model_selection import StratifiedKFold
        import random
        import random
        import random
            from googletrans import Translator
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Functional Data Processing Pipeline
Pure functional programming approach for data processing pipelines
that complements object-oriented model architectures.
"""




# Type variables for functional programming
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

@dataclass(frozen=True)
class DataPoint:
    """Immutable data point for functional processing"""
    text: str
    label: Optional[Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> Any:
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})

@dataclass(frozen=True)
class ProcessingConfig:
    """Immutable configuration for data processing"""
    max_length: int = 512
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = False
    lemmatize: bool = False
    min_word_length: int = 2
    max_words: Optional[int] = None

class DataTransformation:
    """Pure functions for data transformations"""
    
    @staticmethod
    def identity(data: T) -> T:
        """Identity function - returns data unchanged"""
        return data
    
    @staticmethod
    def filter_empty(data: List[DataPoint]) -> List[DataPoint]:
        """Filter out empty data points"""
        return list(filter(lambda x: x.text.strip() != "", data))
    
    @staticmethod
    def filter_length(data: List[DataPoint], min_length: int = 10) -> List[DataPoint]:
        """Filter data points by minimum text length"""
        return list(filter(lambda x: len(x.text) >= min_length, data))
    
    @staticmethod
    def lowercase_text(data: List[DataPoint]) -> List[DataPoint]:
        """Convert text to lowercase"""
        return [
            DataPoint(
                text=point.text.lower(),
                label=point.label,
                metadata=point.metadata
            )
            for point in data
        ]
    
    @staticmethod
    def remove_punctuation(data: List[DataPoint]) -> List[DataPoint]:
        """Remove punctuation from text"""
        return [
            DataPoint(
                text=re.sub(r'[^\w\s]', '', point.text),
                label=point.label,
                metadata=point.metadata
            )
            for point in data
        ]
    
    @staticmethod
    def remove_stopwords(data: List[DataPoint]) -> List[DataPoint]:
        """Remove stopwords from text"""
        stop_words = set(stopwords.words('english'))
        return [
            DataPoint(
                text=' '.join([word for word in point.text.split() if word.lower() not in stop_words]),
                label=point.label,
                metadata=point.metadata
            )
            for point in data
        ]
    
    @staticmethod
    def lemmatize_text(data: List[DataPoint]) -> List[DataPoint]:
        """Lemmatize text"""
        lemmatizer = WordNetLemmatizer()
        return [
            DataPoint(
                text=' '.join([lemmatizer.lemmatize(word) for word in point.text.split()]),
                label=point.label,
                metadata=point.metadata
            )
            for point in data
        ]
    
    @staticmethod
    def filter_word_length(data: List[DataPoint], min_length: int = 2) -> List[DataPoint]:
        """Filter words by minimum length"""
        return [
            DataPoint(
                text=' '.join([word for word in point.text.split() if len(word) >= min_length]),
                label=point.label,
                metadata=point.metadata
            )
            for point in data
        ]
    
    @staticmethod
    def limit_words(data: List[DataPoint], max_words: int) -> List[DataPoint]:
        """Limit number of words per text"""
        return [
            DataPoint(
                text=' '.join(point.text.split()[:max_words]),
                label=point.label,
                metadata=point.metadata
            )
            for point in data
        ]
    
    @staticmethod
    def add_length_metadata(data: List[DataPoint]) -> List[DataPoint]:
        """Add text length to metadata"""
        return [
            DataPoint(
                text=point.text,
                label=point.label,
                metadata={**point.metadata, 'length': len(point.text.split())}
            )
            for point in data
        ]
    
    @staticmethod
    def add_sentiment_metadata(data: List[DataPoint]) -> List[DataPoint]:
        """Add sentiment analysis to metadata"""
        def get_sentiment(text: str) -> str:
            positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic'}
            negative_words = {'bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing'}
            
            words = set(text.lower().split())
            positive_count = len(words.intersection(positive_words))
            negative_count = len(words.intersection(negative_words))
            
            if positive_count > negative_count:
                return 'positive'
            elif negative_count > positive_count:
                return 'negative'
            else:
                return 'neutral'
        
        return [
            DataPoint(
                text=point.text,
                label=point.label,
                metadata={**point.metadata, 'sentiment': get_sentiment(point.text)}
            )
            for point in data
        ]

class DataPipeline:
    """Functional data processing pipeline"""
    
    def __init__(self, config: ProcessingConfig):
        
    """__init__ function."""
self.config = config
        self.transformations: List[Callable] = []
    
    def add_transformation(self, transformation: Callable[[List[DataPoint]], List[DataPoint]]) -> 'DataPipeline':
        """Add transformation to pipeline (immutable operation)"""
        new_pipeline = DataPipeline(self.config)
        new_pipeline.transformations = self.transformations + [transformation]
        return new_pipeline
    
    def compose(self, other: 'DataPipeline') -> 'DataPipeline':
        """Compose two pipelines"""
        new_pipeline = DataPipeline(self.config)
        new_pipeline.transformations = self.transformations + other.transformations
        return new_pipeline
    
    def process(self, data: List[DataPoint]) -> List[DataPoint]:
        """Apply all transformations in sequence"""
        return reduce(lambda acc, transform: transform(acc), self.transformations, data)
    
    @staticmethod
    def create_standard_pipeline(config: ProcessingConfig) -> 'DataPipeline':
        """Create standard processing pipeline"""
        pipeline = DataPipeline(config)
        
        # Add standard transformations
        if config.lowercase:
            pipeline = pipeline.add_transformation(DataTransformation.lowercase_text)
        
        if config.remove_punctuation:
            pipeline = pipeline.add_transformation(DataTransformation.remove_punctuation)
        
        if config.remove_stopwords:
            pipeline = pipeline.add_transformation(DataTransformation.remove_stopwords)
        
        if config.lemmatize:
            pipeline = pipeline.add_transformation(DataTransformation.lemmatize_text)
        
        if config.min_word_length > 1:
            pipeline = pipeline.add_transformation(
                partial(DataTransformation.filter_word_length, min_length=config.min_word_length)
            )
        
        if config.max_words:
            pipeline = pipeline.add_transformation(
                partial(DataTransformation.limit_words, max_words=config.max_words)
            )
        
        # Add metadata
        pipeline = pipeline.add_transformation(DataTransformation.add_length_metadata)
        pipeline = pipeline.add_transformation(DataTransformation.add_sentiment_metadata)
        
        return pipeline

class DataLoader:
    """Functional data loading utilities"""
    
    @staticmethod
    def load_csv(file_path: str, text_column: str, label_column: Optional[str] = None) -> List[DataPoint]:
        """Load data from CSV file"""
        df = pd.read_csv(file_path)
        
        data_points = []
        for _, row in df.iterrows():
            text = str(row[text_column])
            label = row[label_column] if label_column else None
            metadata = {col: row[col] for col in df.columns if col not in [text_column, label_column]}
            
            data_points.append(DataPoint(text=text, label=label, metadata=metadata))
        
        return data_points
    
    @staticmethod
    def load_json(file_path: str, text_key: str, label_key: Optional[str] = None) -> List[DataPoint]:
        """Load data from JSON file"""
        
        with open(file_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            data = json.load(f)
        
        data_points = []
        for item in data:
            text = str(item[text_key])
            label = item.get(label_key) if label_key else None
            metadata = {k: v for k, v in item.items() if k not in [text_key, label_key]}
            
            data_points.append(DataPoint(text=text, label=label, metadata=metadata))
        
        return data_points
    
    @staticmethod
    def save_data(data: List[DataPoint], file_path: str, format: str = 'csv') -> None:
        """Save data to file"""
        if format == 'csv':
            df_data = []
            for point in data:
                row = {'text': point.text}
                if point.label is not None:
                    row['label'] = point.label
                row.update(point.metadata)
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(file_path, index=False)
        
        elif format == 'json':
            
            data_dict = []
            for point in data:
                item = {'text': point.text}
                if point.label is not None:
                    item['label'] = point.label
                item.update(point.metadata)
                data_dict.append(item)
            
            with open(file_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(data_dict, f, indent=2)

class DataSplitting:
    """Functional data splitting utilities"""
    
    @staticmethod
    def split_train_val_test(data: List[DataPoint], 
                           train_ratio: float = 0.8,
                           val_ratio: float = 0.1,
                           test_ratio: float = 0.1,
                           random_state: int = 42) -> Tuple[List[DataPoint], List[DataPoint], List[DataPoint]]:
        """Split data into train/validation/test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Extract labels for stratification
        labels = [point.label for point in data]
        
        # Split into train and temp
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            data, labels, 
            train_size=train_ratio,
            random_state=random_state,
            stratify=labels if len(set(labels)) > 1 else None
        )
        
        # Split temp into validation and test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels,
            train_size=val_ratio_adjusted,
            random_state=random_state,
            stratify=temp_labels if len(set(temp_labels)) > 1 else None
        )
        
        return train_data, val_data, test_data
    
    @staticmethod
    def k_fold_split(data: List[DataPoint], k: int = 5, random_state: int = 42) -> List[Tuple[List[DataPoint], List[DataPoint]]]:
        """Create k-fold cross-validation splits"""
        
        labels = [point.label for point in data]
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        
        splits = []
        for train_idx, val_idx in skf.split(data, labels):
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx]
            splits.append((train_data, val_data))
        
        return splits

class DataAugmentation:
    """Functional data augmentation utilities"""
    
    @staticmethod
    def synonym_replacement(data: List[DataPoint], 
                          synonym_dict: Dict[str, List[str]],
                          replacement_prob: float = 0.3) -> List[DataPoint]:
        """Replace words with synonyms"""
        
        def replace_synonyms(text: str) -> str:
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() in synonym_dict and random.random() < replacement_prob:
                    synonyms = synonym_dict[word.lower()]
                    words[i] = random.choice(synonyms)
            return ' '.join(words)
        
        return [
            DataPoint(
                text=replace_synonyms(point.text),
                label=point.label,
                metadata=point.metadata
            )
            for point in data
        ]
    
    @staticmethod
    def random_insertion(data: List[DataPoint], 
                        insertion_words: List[str],
                        insertion_prob: float = 0.1) -> List[DataPoint]:
        """Randomly insert words"""
        
        def insert_random_words(text: str) -> str:
            words = text.split()
            for _ in range(len(words)):
                if random.random() < insertion_prob:
                    insert_word = random.choice(insertion_words)
                    insert_pos = random.randint(0, len(words))
                    words.insert(insert_pos, insert_word)
            return ' '.join(words)
        
        return [
            DataPoint(
                text=insert_random_words(point.text),
                label=point.label,
                metadata=point.metadata
            )
            for point in data
        ]
    
    @staticmethod
    def random_deletion(data: List[DataPoint], 
                       deletion_prob: float = 0.1) -> List[DataPoint]:
        """Randomly delete words"""
        
        def delete_random_words(text: str) -> str:
            words = text.split()
            words = [word for word in words if random.random() > deletion_prob]
            return ' '.join(words)
        
        return [
            DataPoint(
                text=delete_random_words(point.text),
                label=point.label,
                metadata=point.metadata
            )
            for point in data
        ]
    
    @staticmethod
    def back_translation(data: List[DataPoint], 
                       target_language: str = 'fr',
                       source_language: str = 'en') -> List[DataPoint]:
        """Back translation augmentation"""
        try:
            translator = Translator()
            
            def back_translate(text: str) -> str:
                # Translate to target language
                translated = translator.translate(text, dest=target_language)
                # Translate back to source language
                back_translated = translator.translate(translated.text, dest=source_language)
                return back_translated.text
            
            return [
                DataPoint(
                    text=back_translate(point.text),
                    label=point.label,
                    metadata=point.metadata
                )
                for point in data
            ]
        
        except ImportError:
            logger.warning("googletrans not installed. Skipping back translation.")
            return data

class DataAnalysis:
    """Functional data analysis utilities"""
    
    @staticmethod
    def analyze_text_lengths(data: List[DataPoint]) -> Dict[str, float]:
        """Analyze text length distribution"""
        lengths = [len(point.text.split()) for point in data]
        
        return {
            'mean_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'total_texts': len(data)
        }
    
    @staticmethod
    def analyze_labels(data: List[DataPoint]) -> Dict[str, Any]:
        """Analyze label distribution"""
        labels = [point.label for point in data if point.label is not None]
        
        if not labels:
            return {'error': 'No labels found'}
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        return {
            'unique_labels': unique_labels.tolist(),
            'label_counts': dict(zip(unique_labels, counts)),
            'total_samples': len(labels),
            'class_imbalance': np.max(counts) / np.min(counts) if len(counts) > 1 else 1.0
        }
    
    @staticmethod
    def analyze_vocabulary(data: List[DataPoint]) -> Dict[str, Any]:
        """Analyze vocabulary"""
        all_words = []
        for point in data:
            all_words.extend(point.text.lower().split())
        
        word_counts = pd.Series(all_words).value_counts()
        
        return {
            'vocabulary_size': len(word_counts),
            'total_words': len(all_words),
            'unique_words': len(word_counts),
            'most_common_words': word_counts.head(20).to_dict(),
            'avg_words_per_text': len(all_words) / len(data)
        }
    
    @staticmethod
    def generate_wordcloud_data(data: List[DataPoint]) -> Dict[str, int]:
        """Generate word frequency data for wordcloud"""
        all_words = []
        for point in data:
            all_words.extend(point.text.lower().split())
        
        word_counts = pd.Series(all_words).value_counts()
        return word_counts.to_dict()

class DataValidation:
    """Functional data validation utilities"""
    
    @staticmethod
    def validate_data_points(data: List[DataPoint]) -> List[str]:
        """Validate data points and return list of errors"""
        errors = []
        
        for i, point in enumerate(data):
            if not isinstance(point, DataPoint):
                errors.append(f"Item {i} is not a DataPoint")
                continue
            
            if not isinstance(point.text, str):
                errors.append(f"Item {i}: text is not a string")
            
            if point.text.strip() == "":
                errors.append(f"Item {i}: text is empty")
            
            if not isinstance(point.metadata, dict):
                errors.append(f"Item {i}: metadata is not a dictionary")
        
        return errors
    
    @staticmethod
    def check_data_quality(data: List[DataPoint]) -> Dict[str, Any]:
        """Check data quality metrics"""
        if not data:
            return {'error': 'Empty dataset'}
        
        # Basic statistics
        total_points = len(data)
        empty_texts = sum(1 for point in data if point.text.strip() == "")
        duplicate_texts = len(data) - len(set(point.text for point in data))
        
        # Text length statistics
        lengths = [len(point.text.split()) for point in data]
        
        return {
            'total_points': total_points,
            'empty_texts': empty_texts,
            'duplicate_texts': duplicate_texts,
            'avg_text_length': np.mean(lengths),
            'min_text_length': np.min(lengths),
            'max_text_length': np.max(lengths),
            'quality_score': (total_points - empty_texts - duplicate_texts) / total_points
        }

# Functional composition utilities
def compose(*functions: Callable) -> Callable:
    """Compose multiple functions"""
    def inner(arg) -> Any:
        return reduce(lambda acc, f: f(acc), reversed(functions), arg)
    return inner

def pipe(data: T, *functions: Callable) -> Any:
    """Pipe data through multiple functions"""
    return reduce(lambda acc, f: f(acc), functions, data)

def curry(func: Callable, *args, **kwargs) -> Callable:
    """Curry a function with partial arguments"""
    return partial(func, *args, **kwargs)

# Example usage functions
def create_standard_pipeline() -> DataPipeline:
    """Create a standard data processing pipeline"""
    config = ProcessingConfig(
        max_length=512,
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=False,
        lemmatize=False,
        min_word_length=2
    )
    
    return DataPipeline.create_standard_pipeline(config)

def process_text_data(data: List[DataPoint], pipeline: DataPipeline) -> List[DataPoint]:
    """Process text data using functional pipeline"""
    return pipeline.process(data)

def analyze_dataset(data: List[DataPoint]) -> Dict[str, Any]:
    """Analyze dataset using functional approach"""
    return {
        'text_analysis': DataAnalysis.analyze_text_lengths(data),
        'label_analysis': DataAnalysis.analyze_labels(data),
        'vocabulary_analysis': DataAnalysis.analyze_vocabulary(data),
        'quality_check': DataValidation.check_data_quality(data)
    }

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = [
        DataPoint(text="This is a great product!", label=1),
        DataPoint(text="Terrible service, would not recommend.", label=0),
        DataPoint(text="Amazing experience with excellent support.", label=1),
        DataPoint(text="Poor quality and disappointing results.", label=0),
        DataPoint(text="Good value for money.", label=1)
    ]
    
    # Create and apply pipeline
    pipeline = create_standard_pipeline()
    processed_data = process_text_data(sample_data, pipeline)
    
    # Analyze dataset
    analysis = analyze_dataset(processed_data)
    
    print("Data processing completed!")
    print(f"Original data points: {len(sample_data)}")
    print(f"Processed data points: {len(processed_data)}")
    print(f"Analysis: {analysis}") 