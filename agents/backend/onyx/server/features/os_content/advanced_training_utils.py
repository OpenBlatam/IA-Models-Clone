from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import logging
import time
import gc
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
            from lime.lime_text import LimeTextExplainer
        from sklearn.metrics import roc_curve
        from itertools import product
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Advanced Training Utilities
Additional utilities for deep learning workflows including data augmentation,
model interpretability, and advanced evaluation metrics.
"""



    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score
)



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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class DataAugmentation:
    """Advanced data augmentation techniques for text data"""
    
    def __init__(self, augmentation_config: Dict = None):
        
    """__init__ function."""
self.config = augmentation_config or {}
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Synonym replacement dictionary (simplified)
        self.synonyms = {
            'good': ['great', 'excellent', 'fine', 'nice'],
            'bad': ['terrible', 'awful', 'horrible', 'poor'],
            'happy': ['joyful', 'cheerful', 'pleased', 'glad'],
            'sad': ['unhappy', 'depressed', 'melancholy', 'gloomy'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'miniature', 'petite']
        }
    
    def augment_text(self, text: str, augmentation_type: str = 'random') -> List[str]:
        """Apply text augmentation techniques"""
        augmented_texts = []
        
        if augmentation_type == 'random':
            # Apply random augmentation
            techniques = ['synonym_replacement', 'random_insertion', 'random_deletion', 'random_swap']
            for technique in techniques:
                if np.random.random() < 0.5:  # 50% chance for each technique
                    augmented_texts.append(self._apply_technique(text, technique))
        
        elif augmentation_type == 'all':
            # Apply all augmentation techniques
            techniques = ['synonym_replacement', 'random_insertion', 'random_deletion', 'random_swap']
            for technique in techniques:
                augmented_texts.append(self._apply_technique(text, technique))
        
        else:
            # Apply specific technique
            augmented_texts.append(self._apply_technique(text, augmentation_type))
        
        return augmented_texts
    
    def _apply_technique(self, text: str, technique: str) -> str:
        """Apply specific augmentation technique"""
        if technique == 'synonym_replacement':
            return self._synonym_replacement(text)
        elif technique == 'random_insertion':
            return self._random_insertion(text)
        elif technique == 'random_deletion':
            return self._random_deletion(text)
        elif technique == 'random_swap':
            return self._random_swap(text)
        else:
            return text
    
    def _synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace words with synonyms"""
        words = word_tokenize(text.lower())
        n = min(n, len(words))
        
        for _ in range(n):
            if len(words) == 0:
                break
            
            random_word = np.random.choice(words)
            if random_word in self.synonyms:
                synonym = np.random.choice(self.synonyms[random_word])
                words = [synonym if word == random_word else word for word in words]
        
        return ' '.join(words)
    
    def _random_insertion(self, text: str, n: int = 1) -> str:
        """Insert random words"""
        words = word_tokenize(text.lower())
        n = min(n, len(words))
        
        for _ in range(n):
            if len(words) == 0:
                break
            
            random_word = np.random.choice(list(self.synonyms.keys()))
            random_idx = np.random.randint(0, len(words) + 1)
            words.insert(random_idx, random_word)
        
        return ' '.join(words)
    
    def _random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words"""
        words = word_tokenize(text.lower())
        
        # Keep words that are not stop words with probability p
        words = [word for word in words if word not in self.stop_words or np.random.random() > p]
        
        return ' '.join(words)
    
    def _random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swap adjacent words"""
        words = word_tokenize(text.lower())
        n = min(n, len(words) - 1)
        
        for _ in range(n):
            if len(words) < 2:
                break
            
            idx = np.random.randint(0, len(words) - 1)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        
        return ' '.join(words)
    
    def augment_dataset(self, texts: List[str], labels: List, 
                       augmentation_factor: int = 2) -> Tuple[List[str], List]:
        """Augment entire dataset"""
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        for i, (text, label) in enumerate(zip(texts, labels)):
            # Generate augmented versions
            augmented_versions = self.augment_text(text, 'all')
            
            # Add augmented versions to dataset
            for aug_text in augmented_versions[:augmentation_factor]:
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
        
        return augmented_texts, augmented_labels

class ModelInterpretability:
    """Model interpretability and explainability tools"""
    
    def __init__(self, model: nn.Module, tokenizer=None):
        
    """__init__ function."""
self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def get_attention_weights(self, text: str, layer_idx: int = -1) -> np.ndarray:
        """Extract attention weights from transformer model"""
        if not hasattr(self.model, 'bert') and not hasattr(self.model, 'transformer'):
            raise ValueError("Model must be a transformer model")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attention_weights = outputs.attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]
        
        return attention_weights.cpu().numpy()
    
    def visualize_attention(self, text: str, layer_idx: int = -1, head_idx: int = 0):
        """Visualize attention weights"""
        attention_weights = self.get_attention_weights(text, layer_idx)
        
        # Get tokens
        tokens = self.tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # Create attention heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_weights[head_idx],
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Blues',
            annot=True,
            fmt='.2f'
        )
        plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key')
        plt.ylabel('Query')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, text: str, class_idx: int = 1) -> Dict[str, float]:
        """Get feature importance using gradient-based attribution"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Enable gradient computation
        inputs['input_ids'].requires_grad_(True)
        
        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Backward pass
        logits[0, class_idx].backward()
        
        # Get gradients
        gradients = inputs['input_ids'].grad
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Calculate importance scores
        importance_scores = {}
        for i, (token, grad) in enumerate(zip(tokens, gradients[0])):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                importance_scores[token] = abs(grad.item())
        
        return importance_scores
    
    def visualize_feature_importance(self, text: str, class_idx: int = 1):
        """Visualize feature importance"""
        importance_scores = self.get_feature_importance(text, class_idx)
        
        # Create bar plot
        tokens = list(importance_scores.keys())
        scores = list(importance_scores.values())
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(tokens)), scores)
        plt.xlabel('Tokens')
        plt.ylabel('Importance Score')
        plt.title(f'Feature Importance for Class {class_idx}')
        plt.xticks(range(len(tokens)), tokens, rotation=45)
        plt.tight_layout()
        plt.show()
    
    def generate_lime_explanation(self, text: str, num_samples: int = 1000):
        """Generate LIME explanation for text classification"""
        try:
            
            # Create explainer
            explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
            
            # Define prediction function
            def predict_proba(texts) -> Any:
                inputs = self.tokenizer(
                    texts,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                
                return probabilities.cpu().numpy()
            
            # Generate explanation
            exp = explainer.explain_instance(text, predict_proba, num_features=10)
            
            # Show explanation
            exp.show_in_notebook()
            
            return exp
            
        except ImportError:
            logger.warning("LIME not installed. Install with: pip install lime")
            return None

class AdvancedEvaluation:
    """Advanced evaluation metrics and analysis"""
    
    def __init__(self) -> Any:
        self.metrics_history = []
    
    def calculate_comprehensive_metrics(self, y_true: List, y_pred: List, 
                                      y_prob: Optional[List] = None) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'], metrics['recall'], metrics['f1'], _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # ROC AUC for binary classification
        if y_prob is not None and len(np.unique(y_true)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        class_report = classification_report(y_true, y_pred, output_dict=True)
        metrics['per_class_metrics'] = class_report
        
        # Additional metrics
        metrics['total_samples'] = len(y_true)
        metrics['unique_classes'] = len(np.unique(y_true))
        
        return metrics
    
    def calculate_regression_metrics(self, y_true: List, y_pred: List) -> Dict:
        """Calculate regression metrics"""
        metrics = {}
        
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Additional metrics
        metrics['total_samples'] = len(y_true)
        metrics['mean_target'] = np.mean(y_true)
        metrics['std_target'] = np.std(y_true)
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: List, y_pred: List, 
                            class_names: Optional[List] = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_true: List, y_prob: List):
        """Plot ROC curve"""
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, train_metrics: List[Dict], val_metrics: List[Dict]):
        """Plot training history"""
        epochs = range(1, len(train_metrics) + 1)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(epochs, [m['loss'] for m in train_metrics], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, [m['loss'] for m in val_metrics], 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        if 'accuracy' in train_metrics[0]:
            axes[0, 1].plot(epochs, [m['accuracy'] for m in train_metrics], 'b-', label='Train Accuracy')
            axes[0, 1].plot(epochs, [m['accuracy'] for m in val_metrics], 'r-', label='Val Accuracy')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning Rate
        if 'learning_rate' in train_metrics[0]:
            axes[1, 0].plot(epochs, [m['learning_rate'] for m in train_metrics], 'g-')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True)
        
        # F1 Score
        if 'f1' in val_metrics[0]:
            axes[1, 1].plot(epochs, [m['f1'] for m in val_metrics], 'r-', label='Val F1')
            axes[1, 1].set_title('Validation F1 Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_wordcloud(self, texts: List[str], title: str = "Word Cloud"):
        """Generate word cloud from texts"""
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(combined_text)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.show()

class CrossValidation:
    """Cross-validation utilities"""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        
    """__init__ function."""
self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    async def cross_validate(self, texts: List[str], labels: List, 
                           model_creator: Callable, 
                           training_function: Callable) -> Dict:
        """Perform cross-validation"""
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(texts, labels)):
            logger.info(f"Training fold {fold + 1}/{self.n_splits}")
            
            # Split data
            train_texts = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            
            # Create and train model
            model = model_creator()
            metrics = await training_function(model, train_texts, train_labels, val_texts, val_labels)
            
            fold_metrics.append(metrics)
            
            logger.info(f"Fold {fold + 1} metrics: {metrics}")
        
        # Calculate average metrics
        avg_metrics = self._calculate_average_metrics(fold_metrics)
        
        return {
            'fold_metrics': fold_metrics,
            'average_metrics': avg_metrics,
            'std_metrics': self._calculate_std_metrics(fold_metrics)
        }
    
    def _calculate_average_metrics(self, fold_metrics: List[Dict]) -> Dict:
        """Calculate average metrics across folds"""
        avg_metrics = {}
        
        for key in fold_metrics[0].keys():
            if isinstance(fold_metrics[0][key], (int, float)):
                values = [m[key] for m in fold_metrics]
                avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def _calculate_std_metrics(self, fold_metrics: List[Dict]) -> Dict:
        """Calculate standard deviation of metrics across folds"""
        std_metrics = {}
        
        for key in fold_metrics[0].keys():
            if isinstance(fold_metrics[0][key], (int, float)):
                values = [m[key] for m in fold_metrics]
                std_metrics[key] = np.std(values)
        
        return std_metrics

class HyperparameterOptimization:
    """Hyperparameter optimization utilities"""
    
    def __init__(self, optimization_config: Dict = None):
        
    """__init__ function."""
self.config = optimization_config or {}
        self.best_params = None
        self.best_score = float('inf')
    
    async def optimize_hyperparameters(self, param_grid: Dict, 
                                     objective_function: Callable) -> Dict:
        """Optimize hyperparameters using grid search or random search"""
        optimization_method = self.config.get('method', 'grid_search')
        
        if optimization_method == 'grid_search':
            return await self._grid_search(param_grid, objective_function)
        elif optimization_method == 'random_search':
            return await self._random_search(param_grid, objective_function)
        else:
            raise ValueError(f"Unsupported optimization method: {optimization_method}")
    
    async def _grid_search(self, param_grid: Dict, objective_function: Callable) -> Dict:
        """Grid search optimization"""
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        results = []
        
        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_names, combination))
            logger.info(f"Testing parameters {i + 1}/{len(param_combinations)}: {params}")
            
            # Evaluate objective function
            score = await objective_function(params)
            results.append({'params': params, 'score': score})
            
            # Update best parameters
            if score < self.best_score:
                self.best_score = score
                self.best_params = params
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': results
        }
    
    async def _random_search(self, param_grid: Dict, objective_function: Callable) -> Dict:
        """Random search optimization"""
        n_trials = self.config.get('n_trials', 20)
        results = []
        
        for i in range(n_trials):
            # Sample random parameters
            params = {}
            for param_name, param_values in param_grid.items():
                params[param_name] = np.random.choice(param_values)
            
            logger.info(f"Trial {i + 1}/{n_trials}: {params}")
            
            # Evaluate objective function
            score = await objective_function(params)
            results.append({'params': params, 'score': score})
            
            # Update best parameters
            if score < self.best_score:
                self.best_score = score
                self.best_params = params
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': results
        }

# Example usage functions
async def run_advanced_evaluation(y_true: List, y_pred: List, y_prob: Optional[List] = None):
    """Run advanced evaluation with comprehensive metrics and visualizations"""
    evaluator = AdvancedEvaluation()
    
    # Calculate metrics
    metrics = evaluator.calculate_comprehensive_metrics(y_true, y_pred, y_prob)
    
    # Generate visualizations
    evaluator.plot_confusion_matrix(y_true, y_pred)
    
    if y_prob is not None:
        evaluator.plot_roc_curve(y_true, y_prob)
    
    return metrics

async def run_data_augmentation_example(texts: List[str], labels: List):
    """Run data augmentation example"""
    augmenter = DataAugmentation()
    
    # Augment dataset
    augmented_texts, augmented_labels = augmenter.augment_dataset(texts, labels, augmentation_factor=2)
    
    logger.info(f"Original dataset size: {len(texts)}")
    logger.info(f"Augmented dataset size: {len(augmented_texts)}")
    
    return augmented_texts, augmented_labels

async def run_model_interpretability_example(model: nn.Module, tokenizer, text: str):
    """Run model interpretability example"""
    interpreter = ModelInterpretability(model, tokenizer)
    
    # Get feature importance
    importance_scores = interpreter.get_feature_importance(text)
    
    # Visualize feature importance
    interpreter.visualize_feature_importance(text)
    
    # Visualize attention weights
    interpreter.visualize_attention(text)
    
    return importance_scores

# Example usage
if __name__ == "__main__":
    async def main():
        
    """main function."""
# Example: Data augmentation
        texts = [
            "This is a great product",
            "I love this service",
            "Terrible experience",
            "Amazing quality"
        ]
        labels = [1, 1, 0, 1]
        
        augmented_texts, augmented_labels = await run_data_augmentation_example(texts, labels)
        
        print("Data augmentation completed!")
        print(f"Augmented texts: {augmented_texts[:5]}")  # Show first 5
        
        # Example: Advanced evaluation
        y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 0, 1, 1, 1]
        y_prob = [0.1, 0.9, 0.2, 0.3, 0.1, 0.8, 0.7, 0.9]
        
        metrics = await run_advanced_evaluation(y_true, y_pred, y_prob)
        print(f"Evaluation metrics: {metrics}")
    
    asyncio.run(main()) 