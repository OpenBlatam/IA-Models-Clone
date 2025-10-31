from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from collections import defaultdict
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat
from textblob import TextBlob
from ..models.sequence import EmailSequence, SequenceStep
from ..models.subscriber import Subscriber
from ..models.template import EmailTemplate
from typing import Any, List, Dict, Optional
"""
Evaluation Metrics for Email Sequence System

Comprehensive evaluation metrics for email sequence models including
content quality, engagement prediction, personalization effectiveness,
and business impact metrics.
"""


    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
    brier_score_loss
)


logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')


@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics"""
    # Content quality metrics
    enable_content_quality: bool = True
    enable_readability: bool = True
    enable_sentiment: bool = True
    enable_grammar: bool = True
    
    # Engagement metrics
    enable_engagement: bool = True
    enable_cta_analysis: bool = True
    enable_urgency_detection: bool = True
    
    # Personalization metrics
    enable_personalization: bool = True
    enable_relevance: bool = True
    enable_customization: bool = True
    
    # Business metrics
    enable_business_impact: bool = True
    enable_conversion: bool = True
    enable_revenue: bool = True
    
    # Technical metrics
    enable_technical: bool = True
    enable_performance: bool = True
    
    # Thresholds
    min_content_length: int = 50
    max_content_length: int = 2000
    min_readability_score: float = 30.0
    max_readability_score: float = 80.0
    
    # Weights for composite scores
    content_weight: float = 0.3
    engagement_weight: float = 0.3
    personalization_weight: float = 0.2
    business_weight: float = 0.2


class ContentQualityMetrics:
    """Content quality evaluation metrics"""
    
    def __init__(self, config: MetricsConfig):
        
    """__init__ function."""
self.config = config
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        logger.info("Content Quality Metrics initialized")
    
    def evaluate_content_quality(
        self,
        content: str,
        target_audience: Optional[Subscriber] = None
    ) -> Dict[str, float]:
        """Evaluate content quality metrics"""
        
        if not content or len(content.strip()) == 0:
            return {"content_quality_score": 0.0}
        
        metrics = {}
        
        # Basic content metrics
        metrics.update(self._basic_content_metrics(content))
        
        # Readability metrics
        if self.config.enable_readability:
            metrics.update(self._readability_metrics(content))
        
        # Sentiment analysis
        if self.config.enable_sentiment:
            metrics.update(self._sentiment_metrics(content))
        
        # Grammar and style
        if self.config.enable_grammar:
            metrics.update(self._grammar_metrics(content))
        
        # Calculate composite score
        metrics["content_quality_score"] = self._calculate_content_quality_score(metrics)
        
        return metrics
    
    def _basic_content_metrics(self, content: str) -> Dict[str, float]:
        """Calculate basic content metrics"""
        
        words = word_tokenize(content.lower())
        sentences = sent_tokenize(content)
        
        metrics = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "unique_word_ratio": len(set(words)) / max(len(words), 1),
            "stop_word_ratio": len([w for w in words if w in self.stop_words]) / max(len(words), 1),
            "content_length_score": self._calculate_length_score(len(content))
        }
        
        return metrics
    
    def _readability_metrics(self, content: str) -> Dict[str, float]:
        """Calculate readability metrics"""
        
        try:
            metrics = {
                "flesch_reading_ease": textstat.flesch_reading_ease(content),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(content),
                "gunning_fog": textstat.gunning_fog(content),
                "smog_index": textstat.smog_index(content),
                "automated_readability_index": textstat.automated_readability_index(content),
                "coleman_liau_index": textstat.coleman_liau_index(content),
                "linsear_write_formula": textstat.linsear_write_formula(content),
                "dale_chall_readability": textstat.dale_chall_readability_score(content),
                "difficult_words": textstat.difficult_words(content),
                "syllable_count": textstat.syllable_count(content),
                "lexicon_count": textstat.lexicon_count(content),
                "sentence_count": textstat.sentence_count(content)
            }
            
            # Calculate readability score
            metrics["readability_score"] = self._calculate_readability_score(metrics)
            
        except Exception as e:
            logger.warning(f"Error calculating readability metrics: {e}")
            metrics = {"readability_score": 0.0}
        
        return metrics
    
    def _sentiment_metrics(self, content: str) -> Dict[str, float]:
        """Calculate sentiment metrics"""
        
        try:
            # VADER sentiment analysis
            vader_scores = self.sentiment_analyzer.polarity_scores(content)
            
            # TextBlob sentiment analysis
            blob = TextBlob(content)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            metrics = {
                "vader_positive": vader_scores['pos'],
                "vader_negative": vader_scores['neg'],
                "vader_neutral": vader_scores['neu'],
                "vader_compound": vader_scores['compound'],
                "textblob_polarity": textblob_polarity,
                "textblob_subjectivity": textblob_subjectivity,
                "sentiment_score": self._calculate_sentiment_score(vader_scores, textblob_polarity)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating sentiment metrics: {e}")
            metrics = {"sentiment_score": 0.0}
        
        return metrics
    
    def _grammar_metrics(self, content: str) -> Dict[str, float]:
        """Calculate grammar and style metrics"""
        
        try:
            blob = TextBlob(content)
            
            # Basic grammar checks
            word_count = len(blob.words)
            sentence_count = len(blob.sentences)
            
            # Calculate various ratios
            metrics = {
                "avg_words_per_sentence": word_count / max(sentence_count, 1),
                "sentence_variety": self._calculate_sentence_variety(content),
                "word_complexity": self._calculate_word_complexity(content),
                "grammar_score": self._calculate_grammar_score(content)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating grammar metrics: {e}")
            metrics = {"grammar_score": 0.0}
        
        return metrics
    
    def _calculate_content_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite content quality score"""
        
        scores = []
        
        # Length score
        if "content_length_score" in metrics:
            scores.append(metrics["content_length_score"])
        
        # Readability score
        if "readability_score" in metrics:
            scores.append(metrics["readability_score"])
        
        # Sentiment score
        if "sentiment_score" in metrics:
            scores.append(metrics["sentiment_score"])
        
        # Grammar score
        if "grammar_score" in metrics:
            scores.append(metrics["grammar_score"])
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_length_score(self, length: int) -> float:
        """Calculate content length score"""
        
        if length < self.config.min_content_length:
            return 0.0
        elif length > self.config.max_content_length:
            return 0.5
        else:
            # Optimal length range
            optimal_min = 200
            optimal_max = 800
            if optimal_min <= length <= optimal_max:
                return 1.0
            else:
                # Gradual decrease outside optimal range
                if length < optimal_min:
                    return length / optimal_min
                else:
                    return max(0.5, 1.0 - (length - optimal_max) / (self.config.max_content_length - optimal_max))
    
    def _calculate_readability_score(self, metrics: Dict[str, float]) -> float:
        """Calculate readability score"""
        
        try:
            flesch_ease = metrics.get("flesch_reading_ease", 0)
            
            # Normalize Flesch Reading Ease (0-100 scale)
            if flesch_ease >= 90:
                score = 1.0
            elif flesch_ease >= 80:
                score = 0.9
            elif flesch_ease >= 70:
                score = 0.8
            elif flesch_ease >= 60:
                score = 0.7
            elif flesch_ease >= 50:
                score = 0.6
            elif flesch_ease >= 30:
                score = 0.5
            else:
                score = 0.3
            
            return score
            
        except Exception:
            return 0.5
    
    def _calculate_sentiment_score(
        self,
        vader_scores: Dict[str, float],
        textblob_polarity: float
    ) -> float:
        """Calculate sentiment score"""
        
        # Combine VADER and TextBlob scores
        vader_score = vader_scores['compound']
        combined_score = (vader_score + textblob_polarity) / 2
        
        # Normalize to 0-1 range
        normalized_score = (combined_score + 1) / 2
        
        return normalized_score
    
    def _calculate_sentence_variety(self, content: str) -> float:
        """Calculate sentence variety score"""
        
        sentences = sent_tokenize(content)
        if len(sentences) < 2:
            return 0.5
        
        # Calculate sentence length variety
        lengths = [len(sent.split()) for sent in sentences]
        length_std = np.std(lengths)
        length_mean = np.mean(lengths)
        
        # Normalize variety score
        variety_score = min(1.0, length_std / max(length_mean, 1))
        
        return variety_score
    
    def _calculate_word_complexity(self, content: str) -> float:
        """Calculate word complexity score"""
        
        words = word_tokenize(content.lower())
        if not words:
            return 0.0
        
        # Count syllables per word
        syllable_counts = [textstat.syllable_count(word) for word in words]
        avg_syllables = np.mean(syllable_counts)
        
        # Normalize complexity (1-3 syllables is optimal)
        if avg_syllables <= 1:
            complexity_score = 0.3
        elif avg_syllables <= 2:
            complexity_score = 1.0
        elif avg_syllables <= 3:
            complexity_score = 0.8
        else:
            complexity_score = 0.4
        
        return complexity_score
    
    def _calculate_grammar_score(self, content: str) -> float:
        """Calculate grammar score"""
        
        # Simple grammar checks
        score = 1.0
        
        # Check for basic punctuation
        if not content.strip().endswith(('.', '!', '?')):
            score -= 0.1
        
        # Check for capitalization
        sentences = sent_tokenize(content)
        for sentence in sentences:
            if sentence.strip() and not sentence.strip()[0].isupper():
                score -= 0.1
        
        # Check for excessive punctuation
        if content.count('!') > len(sentences) * 0.5:
            score -= 0.2
        
        return max(0.0, score)


class EngagementMetrics:
    """Engagement prediction and analysis metrics"""
    
    def __init__(self, config: MetricsConfig):
        
    """__init__ function."""
self.config = config
        
        # Engagement keywords and patterns
        self.engagement_keywords = {
            "cta": ["click", "download", "sign up", "register", "subscribe", "buy", "order", "learn more"],
            "urgency": ["limited time", "offer ends", "act now", "hurry", "urgent", "expires", "last chance"],
            "personalization": ["you", "your", "personal", "customized", "exclusive", "special"],
            "social_proof": ["customers", "users", "people", "testimonials", "reviews", "trusted"],
            "benefits": ["benefit", "advantage", "feature", "improve", "enhance", "boost", "increase"]
        }
        
        logger.info("Engagement Metrics initialized")
    
    def evaluate_engagement(
        self,
        content: str,
        subject_line: str = "",
        target_audience: Optional[Subscriber] = None
    ) -> Dict[str, float]:
        """Evaluate engagement metrics"""
        
        metrics = {}
        
        # CTA analysis
        if self.config.enable_cta_analysis:
            metrics.update(self._cta_analysis(content, subject_line))
        
        # Urgency detection
        if self.config.enable_urgency_detection:
            metrics.update(self._urgency_analysis(content, subject_line))
        
        # Engagement keywords
        metrics.update(self._engagement_keyword_analysis(content))
        
        # Personalization effectiveness
        metrics.update(self._personalization_analysis(content, target_audience))
        
        # Calculate composite engagement score
        metrics["engagement_score"] = self._calculate_engagement_score(metrics)
        
        return metrics
    
    def _cta_analysis(self, content: str, subject_line: str = "") -> Dict[str, float]:
        """Analyze call-to-action effectiveness"""
        
        full_text = f"{subject_line} {content}".lower()
        cta_keywords = self.engagement_keywords["cta"]
        
        # Count CTA keywords
        cta_count = sum(1 for keyword in cta_keywords if keyword in full_text)
        
        # Check CTA placement
        sentences = sent_tokenize(content)
        cta_in_subject = any(keyword in subject_line.lower() for keyword in cta_keywords)
        cta_in_body = any(keyword in content.lower() for keyword in cta_keywords)
        
        metrics = {
            "cta_count": cta_count,
            "cta_in_subject": float(cta_in_subject),
            "cta_in_body": float(cta_in_body),
            "cta_placement_score": self._calculate_cta_placement_score(cta_in_subject, cta_in_body),
            "cta_effectiveness": min(1.0, cta_count / 3.0)  # Normalize to 0-1
        }
        
        return metrics
    
    def _urgency_analysis(self, content: str, subject_line: str = "") -> Dict[str, float]:
        """Analyze urgency indicators"""
        
        full_text = f"{subject_line} {content}".lower()
        urgency_keywords = self.engagement_keywords["urgency"]
        
        # Count urgency keywords
        urgency_count = sum(1 for keyword in urgency_keywords if keyword in full_text)
        
        # Check for time indicators
        time_indicators = ["today", "now", "immediately", "asap", "quickly"]
        time_count = sum(1 for indicator in time_indicators if indicator in full_text)
        
        metrics = {
            "urgency_count": urgency_count,
            "time_indicators": time_count,
            "urgency_score": min(1.0, (urgency_count + time_count) / 5.0)
        }
        
        return metrics
    
    def _engagement_keyword_analysis(self, content: str) -> Dict[str, float]:
        """Analyze engagement keywords"""
        
        content_lower = content.lower()
        keyword_scores = {}
        
        for category, keywords in self.engagement_keywords.items():
            count = sum(1 for keyword in keywords if keyword in content_lower)
            keyword_scores[f"{category}_count"] = count
            keyword_scores[f"{category}_score"] = min(1.0, count / len(keywords))
        
        # Calculate overall keyword diversity
        total_keywords = sum(keyword_scores[f"{category}_count"] for category in self.engagement_keywords.keys())
        keyword_scores["keyword_diversity"] = min(1.0, total_keywords / 10.0)
        
        return keyword_scores
    
    def _personalization_analysis(
        self,
        content: str,
        target_audience: Optional[Subscriber] = None
    ) -> Dict[str, float]:
        """Analyze personalization effectiveness"""
        
        metrics = {
            "personalization_keywords": 0,
            "audience_relevance": 0.5,
            "customization_level": 0.5
        }
        
        # Count personalization keywords
        personalization_keywords = self.engagement_keywords["personalization"]
        metrics["personalization_keywords"] = sum(
            1 for keyword in personalization_keywords if keyword in content.lower()
        )
        
        # Analyze audience relevance if target audience provided
        if target_audience:
            relevance_score = self._calculate_audience_relevance(content, target_audience)
            metrics["audience_relevance"] = relevance_score
            metrics["customization_level"] = relevance_score
        
        return metrics
    
    def _calculate_engagement_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite engagement score"""
        
        scores = []
        
        # CTA effectiveness
        if "cta_effectiveness" in metrics:
            scores.append(metrics["cta_effectiveness"])
        
        # Urgency score
        if "urgency_score" in metrics:
            scores.append(metrics["urgency_score"])
        
        # Keyword diversity
        if "keyword_diversity" in metrics:
            scores.append(metrics["keyword_diversity"])
        
        # Personalization
        if "customization_level" in metrics:
            scores.append(metrics["customization_level"])
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_cta_placement_score(self, cta_in_subject: bool, cta_in_body: bool) -> float:
        """Calculate CTA placement score"""
        
        if cta_in_subject and cta_in_body:
            return 1.0  # Optimal: CTA in both subject and body
        elif cta_in_body:
            return 0.8  # Good: CTA in body
        elif cta_in_subject:
            return 0.6  # Acceptable: CTA only in subject
        else:
            return 0.0  # Poor: No CTA
    
    def _calculate_audience_relevance(self, content: str, target_audience: Subscriber) -> float:
        """Calculate audience relevance score"""
        
        relevance_score = 0.5  # Base score
        
        # Check for company mentions
        if target_audience.company and target_audience.company.lower() in content.lower():
            relevance_score += 0.2
        
        # Check for industry-specific terms
        if target_audience.interests:
            interest_matches = sum(
                1 for interest in target_audience.interests 
                if interest.lower() in content.lower()
            )
            relevance_score += min(0.3, interest_matches * 0.1)
        
        return min(1.0, relevance_score)


class BusinessImpactMetrics:
    """Business impact and conversion metrics"""
    
    def __init__(self, config: MetricsConfig):
        
    """__init__ function."""
self.config = config
        
        # Conversion indicators
        self.conversion_keywords = [
            "buy", "purchase", "order", "sign up", "register", "subscribe",
            "download", "get started", "try", "demo", "free trial"
        ]
        
        logger.info("Business Impact Metrics initialized")
    
    def evaluate_business_impact(
        self,
        content: str,
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Evaluate business impact metrics"""
        
        metrics = {}
        
        # Conversion potential
        if self.config.enable_conversion:
            metrics.update(self._conversion_analysis(content))
        
        # Revenue potential
        if self.config.enable_revenue:
            metrics.update(self._revenue_analysis(content, historical_data))
        
        # ROI indicators
        metrics.update(self._roi_analysis(content))
        
        # Calculate composite business impact score
        metrics["business_impact_score"] = self._calculate_business_impact_score(metrics)
        
        return metrics
    
    def _conversion_analysis(self, content: str) -> Dict[str, float]:
        """Analyze conversion potential"""
        
        content_lower = content.lower()
        
        # Count conversion keywords
        conversion_count = sum(1 for keyword in self.conversion_keywords if keyword in content_lower)
        
        # Analyze conversion funnel
        funnel_stages = {
            "awareness": ["learn", "discover", "find out", "understand"],
            "interest": ["benefit", "advantage", "feature", "improve"],
            "desire": ["want", "need", "must have", "essential"],
            "action": self.conversion_keywords
        }
        
        funnel_scores = {}
        for stage, keywords in funnel_stages.items():
            count = sum(1 for keyword in keywords if keyword in content_lower)
            funnel_scores[f"{stage}_score"] = min(1.0, count / len(keywords))
        
        metrics = {
            "conversion_keywords": conversion_count,
            "conversion_potential": min(1.0, conversion_count / 5.0),
            **funnel_scores,
            "funnel_completeness": np.mean(list(funnel_scores.values()))
        }
        
        return metrics
    
    def _revenue_analysis(
        self,
        content: str,
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Analyze revenue potential"""
        
        metrics = {
            "revenue_potential": 0.5,
            "pricing_mentions": 0.0,
            "value_proposition": 0.5
        }
        
        # Check for pricing mentions
        pricing_keywords = ["price", "cost", "fee", "subscription", "plan", "package"]
        pricing_count = sum(1 for keyword in pricing_keywords if keyword in content.lower())
        metrics["pricing_mentions"] = min(1.0, pricing_count / 3.0)
        
        # Analyze value proposition
        value_keywords = ["save", "earn", "profit", "benefit", "advantage", "improve"]
        value_count = sum(1 for keyword in value_keywords if keyword in content.lower())
        metrics["value_proposition"] = min(1.0, value_count / 4.0)
        
        # Calculate revenue potential
        metrics["revenue_potential"] = (metrics["pricing_mentions"] + metrics["value_proposition"]) / 2
        
        return metrics
    
    def _roi_analysis(self, content: str) -> Dict[str, float]:
        """Analyze ROI indicators"""
        
        content_lower = content.lower()
        
        # ROI-related keywords
        roi_keywords = ["roi", "return", "investment", "profit", "revenue", "earnings"]
        roi_count = sum(1 for keyword in roi_keywords if keyword in content_lower)
        
        # Cost-benefit keywords
        cost_benefit_keywords = ["cost", "benefit", "savings", "efficiency", "productivity"]
        cb_count = sum(1 for keyword in cost_benefit_keywords if keyword in content_lower)
        
        metrics = {
            "roi_indicators": roi_count,
            "cost_benefit_analysis": cb_count,
            "roi_score": min(1.0, (roi_count + cb_count) / 8.0)
        }
        
        return metrics
    
    def _calculate_business_impact_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite business impact score"""
        
        scores = []
        
        # Conversion potential
        if "conversion_potential" in metrics:
            scores.append(metrics["conversion_potential"])
        
        # Revenue potential
        if "revenue_potential" in metrics:
            scores.append(metrics["revenue_potential"])
        
        # ROI score
        if "roi_score" in metrics:
            scores.append(metrics["roi_score"])
        
        # Funnel completeness
        if "funnel_completeness" in metrics:
            scores.append(metrics["funnel_completeness"])
        
        return np.mean(scores) if scores else 0.0


class TechnicalMetrics:
    """Technical performance and model metrics"""
    
    def __init__(self, config: MetricsConfig):
        
    """__init__ function."""
self.config = config
        self.metrics_history = defaultdict(list)
        
        logger.info("Technical Metrics initialized")
    
    def evaluate_technical_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """Evaluate technical metrics"""
        
        metrics = {}
        
        # Convert to numpy for sklearn metrics
        if isinstance(predictions, torch.Tensor):
            predictions_np = predictions.detach().cpu().numpy()
        else:
            predictions_np = np.array(predictions)
        
        if isinstance(targets, torch.Tensor):
            targets_np = targets.detach().cpu().numpy()
        else:
            targets_np = np.array(targets)
        
        # Regression metrics
        if len(predictions_np.shape) == 1 or predictions_np.shape[1] == 1:
            metrics.update(self._regression_metrics(predictions_np, targets_np))
        else:
            metrics.update(self._classification_metrics(predictions_np, targets_np))
        
        # Model performance metrics
        if model:
            metrics.update(self._model_performance_metrics(model))
        
        return metrics
    
    def _regression_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        
        # Flatten arrays if needed
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        metrics = {
            "mse": mean_squared_error(targets, predictions),
            "rmse": np.sqrt(mean_squared_error(targets, predictions)),
            "mae": mean_absolute_error(targets, predictions),
            "r2_score": r2_score(targets, predictions),
            "explained_variance": 1 - np.var(targets - predictions) / np.var(targets)
        }
        
        # Additional regression metrics
        metrics["mape"] = np.mean(np.abs((targets - predictions) / np.maximum(np.abs(targets), 1e-8))) * 100
        metrics["smape"] = 2.0 * np.mean(np.abs(predictions - targets) / (np.abs(predictions) + np.abs(targets))) * 100
        
        return metrics
    
    def _classification_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics"""
        
        # Handle multi-class classification
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Multi-class: use argmax for predictions
            pred_classes = np.argmax(predictions, axis=1)
            if len(targets.shape) > 1 and targets.shape[1] > 1:
                target_classes = np.argmax(targets, axis=1)
            else:
                target_classes = targets
        else:
            # Binary classification
            pred_classes = (predictions > 0.5).astype(int)
            target_classes = targets
        
        metrics = {
            "accuracy": accuracy_score(target_classes, pred_classes),
            "precision": precision_score(target_classes, pred_classes, average='weighted', zero_division=0),
            "recall": recall_score(target_classes, pred_classes, average='weighted', zero_division=0),
            "f1_score": f1_score(target_classes, pred_classes, average='weighted', zero_division=0)
        }
        
        # Additional classification metrics
        try:
            metrics["roc_auc"] = roc_auc_score(target_classes, predictions, multi_class='ovr' if len(np.unique(target_classes)) > 2 else 'raise')
        except:
            metrics["roc_auc"] = 0.0
        
        metrics["cohen_kappa"] = cohen_kappa_score(target_classes, pred_classes)
        
        try:
            metrics["matthews_corrcoef"] = matthews_corrcoef(target_classes, pred_classes)
        except:
            metrics["matthews_corrcoef"] = 0.0
        
        return metrics
    
    def _model_performance_metrics(self, model: nn.Module) -> Dict[str, float]:
        """Calculate model performance metrics"""
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate model size (approximate)
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        metrics = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "parameter_efficiency": trainable_params / max(total_params, 1)
        }
        
        return metrics


class EmailSequenceEvaluator:
    """Comprehensive email sequence evaluation system"""
    
    def __init__(self, config: MetricsConfig):
        
    """__init__ function."""
self.config = config
        self.content_metrics = ContentQualityMetrics(config)
        self.engagement_metrics = EngagementMetrics(config)
        self.business_metrics = BusinessImpactMetrics(config)
        self.technical_metrics = TechnicalMetrics(config)
        
        # Evaluation history
        self.evaluation_history = []
        
        logger.info("Email Sequence Evaluator initialized")
    
    async def evaluate_sequence(
        self,
        sequence: EmailSequence,
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """Evaluate complete email sequence"""
        
        evaluation_results = {
            "sequence_id": sequence.id,
            "sequence_name": sequence.name,
            "step_evaluations": [],
            "overall_metrics": {}
        }
        
        # Evaluate each step
        for step in sequence.steps:
            step_evaluation = await self._evaluate_step(step, subscribers, templates)
            evaluation_results["step_evaluations"].append(step_evaluation)
        
        # Calculate overall sequence metrics
        evaluation_results["overall_metrics"] = self._calculate_overall_metrics(
            evaluation_results["step_evaluations"]
        )
        
        # Add technical metrics if available
        if predictions is not None and targets is not None:
            technical_metrics = self.technical_metrics.evaluate_technical_metrics(
                predictions, targets, model
            )
            evaluation_results["technical_metrics"] = technical_metrics
        
        # Store evaluation history
        self.evaluation_history.append(evaluation_results)
        
        return evaluation_results
    
    async def _evaluate_step(
        self,
        step: SequenceStep,
        subscribers: List[Subscriber],
        templates: List[EmailTemplate]
    ) -> Dict[str, Any]:
        """Evaluate individual step"""
        
        # Find relevant subscriber and template
        subscriber = next((s for s in subscribers if s.id == step.subscriber_id), None)
        template = next((t for t in templates if t.id == step.template_id), None)
        
        step_evaluation = {
            "step_order": step.order,
            "content_metrics": {},
            "engagement_metrics": {},
            "business_metrics": {}
        }
        
        # Evaluate content quality
        if step.content:
            step_evaluation["content_metrics"] = self.content_metrics.evaluate_content_quality(
                step.content, subscriber
            )
            
            step_evaluation["engagement_metrics"] = self.engagement_metrics.evaluate_engagement(
                step.content, "", subscriber
            )
            
            step_evaluation["business_metrics"] = self.business_metrics.evaluate_business_impact(
                step.content
            )
        
        return step_evaluation
    
    def _calculate_overall_metrics(self, step_evaluations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall sequence metrics"""
        
        overall_metrics = {
            "content_quality_score": 0.0,
            "engagement_score": 0.0,
            "business_impact_score": 0.0,
            "sequence_coherence": 0.0,
            "progression_effectiveness": 0.0
        }
        
        if not step_evaluations:
            return overall_metrics
        
        # Aggregate step metrics
        content_scores = []
        engagement_scores = []
        business_scores = []
        
        for step_eval in step_evaluations:
            if "content_metrics" in step_eval and "content_quality_score" in step_eval["content_metrics"]:
                content_scores.append(step_eval["content_metrics"]["content_quality_score"])
            
            if "engagement_metrics" in step_eval and "engagement_score" in step_eval["engagement_metrics"]:
                engagement_scores.append(step_eval["engagement_metrics"]["engagement_score"])
            
            if "business_metrics" in step_eval and "business_impact_score" in step_eval["business_metrics"]:
                business_scores.append(step_eval["business_metrics"]["business_impact_score"])
        
        # Calculate overall scores
        overall_metrics["content_quality_score"] = np.mean(content_scores) if content_scores else 0.0
        overall_metrics["engagement_score"] = np.mean(engagement_scores) if engagement_scores else 0.0
        overall_metrics["business_impact_score"] = np.mean(business_scores) if business_scores else 0.0
        
        # Calculate sequence-specific metrics
        overall_metrics["sequence_coherence"] = self._calculate_sequence_coherence(step_evaluations)
        overall_metrics["progression_effectiveness"] = self._calculate_progression_effectiveness(step_evaluations)
        
        # Calculate composite score
        overall_metrics["overall_score"] = (
            self.config.content_weight * overall_metrics["content_quality_score"] +
            self.config.engagement_weight * overall_metrics["engagement_score"] +
            self.config.personalization_weight * overall_metrics["sequence_coherence"] +
            self.config.business_weight * overall_metrics["business_impact_score"]
        )
        
        return overall_metrics
    
    def _calculate_sequence_coherence(self, step_evaluations: List[Dict[str, Any]]) -> float:
        """Calculate sequence coherence score"""
        
        if len(step_evaluations) < 2:
            return 1.0
        
        # Analyze consistency across steps
        content_scores = []
        engagement_scores = []
        
        for step_eval in step_evaluations:
            if "content_metrics" in step_eval and "content_quality_score" in step_eval["content_metrics"]:
                content_scores.append(step_eval["content_metrics"]["content_quality_score"])
            
            if "engagement_metrics" in step_eval and "engagement_score" in step_eval["engagement_metrics"]:
                engagement_scores.append(step_eval["engagement_metrics"]["engagement_score"])
        
        # Calculate consistency (lower variance = higher coherence)
        if content_scores:
            content_consistency = 1.0 - min(1.0, np.std(content_scores))
        else:
            content_consistency = 0.0
        
        if engagement_scores:
            engagement_consistency = 1.0 - min(1.0, np.std(engagement_scores))
        else:
            engagement_consistency = 0.0
        
        return (content_consistency + engagement_consistency) / 2
    
    def _calculate_progression_effectiveness(self, step_evaluations: List[Dict[str, Any]]) -> float:
        """Calculate progression effectiveness score"""
        
        if len(step_evaluations) < 2:
            return 1.0
        
        # Analyze progression patterns
        engagement_scores = []
        for step_eval in step_evaluations:
            if "engagement_metrics" in step_eval and "engagement_score" in step_eval["engagement_metrics"]:
                engagement_scores.append(step_eval["engagement_metrics"]["engagement_score"])
        
        if not engagement_scores:
            return 0.0
        
        # Check for increasing engagement (good progression)
        increasing_count = 0
        for i in range(1, len(engagement_scores)):
            if engagement_scores[i] > engagement_scores[i-1]:
                increasing_count += 1
        
        progression_score = increasing_count / max(len(engagement_scores) - 1, 1)
        
        return progression_score
    
    def get_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        if not self.evaluation_history:
            return {"message": "No evaluations performed"}
        
        # Aggregate metrics across all evaluations
        all_overall_scores = [eval_result["overall_metrics"]["overall_score"] 
                             for eval_result in self.evaluation_history]
        
        report = {
            "total_evaluations": len(self.evaluation_history),
            "average_overall_score": np.mean(all_overall_scores),
            "score_distribution": {
                "min": np.min(all_overall_scores),
                "max": np.max(all_overall_scores),
                "std": np.std(all_overall_scores),
                "median": np.median(all_overall_scores)
            },
            "metric_breakdown": self._calculate_metric_breakdown(),
            "recent_evaluations": self.evaluation_history[-5:]  # Last 5 evaluations
        }
        
        return report
    
    def _calculate_metric_breakdown(self) -> Dict[str, float]:
        """Calculate breakdown of different metric types"""
        
        metric_sums = defaultdict(float)
        metric_counts = defaultdict(int)
        
        for evaluation in self.evaluation_history:
            overall_metrics = evaluation["overall_metrics"]
            for metric_name, value in overall_metrics.items():
                if metric_name != "overall_score":
                    metric_sums[metric_name] += value
                    metric_counts[metric_name] += 1
        
        breakdown = {}
        for metric_name in metric_sums:
            if metric_counts[metric_name] > 0:
                breakdown[metric_name] = metric_sums[metric_name] / metric_counts[metric_name]
        
        return breakdown
    
    def plot_evaluation_results(self, save_path: str = None):
        """Plot evaluation results"""
        
        if not self.evaluation_history:
            logger.warning("No evaluation history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        overall_scores = [eval_result["overall_metrics"]["overall_score"] 
                         for eval_result in self.evaluation_history]
        content_scores = [eval_result["overall_metrics"]["content_quality_score"] 
                         for eval_result in self.evaluation_history]
        engagement_scores = [eval_result["overall_metrics"]["engagement_score"] 
                           for eval_result in self.evaluation_history]
        business_scores = [eval_result["overall_metrics"]["business_impact_score"] 
                          for eval_result in self.evaluation_history]
        
        # Plot overall scores over time
        axes[0, 0].plot(overall_scores, 'b-', linewidth=2)
        axes[0, 0].set_title("Overall Evaluation Scores")
        axes[0, 0].set_xlabel("Evaluation")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot metric comparison
        x = range(len(overall_scores))
        axes[0, 1].plot(x, content_scores, 'g-', label='Content Quality', linewidth=2)
        axes[0, 1].plot(x, engagement_scores, 'r-', label='Engagement', linewidth=2)
        axes[0, 1].plot(x, business_scores, 'y-', label='Business Impact', linewidth=2)
        axes[0, 1].set_title("Metric Comparison")
        axes[0, 1].set_xlabel("Evaluation")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot score distribution
        axes[1, 0].hist(overall_scores, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=np.mean(overall_scores), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(overall_scores):.3f}')
        axes[1, 0].set_title("Score Distribution")
        axes[1, 0].set_xlabel("Overall Score")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot correlation matrix
        metrics_data = np.column_stack([content_scores, engagement_scores, business_scores])
        correlation_matrix = np.corrcoef(metrics_data.T)
        
        im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_title("Metric Correlations")
        axes[1, 1].set_xticks(range(3))
        axes[1, 1].set_yticks(range(3))
        axes[1, 1].set_xticklabels(['Content', 'Engagement', 'Business'])
        axes[1, 1].set_yticklabels(['Content', 'Engagement', 'Business'])
        
        # Add correlation values
        for i in range(3):
            for j in range(3):
                axes[1, 1].text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                               ha='center', va='center', color='black', fontweight='bold')
        
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show() 