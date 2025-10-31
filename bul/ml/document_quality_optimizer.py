"""
BUL ML Document Quality Optimizer
=================================

Machine learning pipeline for optimizing document quality and generation.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

from ..utils import get_logger, get_cache_manager, get_data_processor
from ..config import get_config
from ..core import BusinessArea, DocumentType

logger = get_logger(__name__)

class QualityMetric(str, Enum):
    """Document quality metrics"""
    READABILITY = "readability"
    COHERENCE = "coherence"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    PROFESSIONALISM = "professionalism"
    CLARITY = "clarity"
    STRUCTURE = "structure"

class ModelType(str, Enum):
    """ML model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LINEAR_REGRESSION = "linear_regression"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"

@dataclass
class DocumentFeatures:
    """Document features for ML analysis"""
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_word_length: float
    avg_sentence_length: float
    readability_score: float
    sentiment_score: float
    topic_coherence: float
    structure_score: float
    business_area: str
    document_type: str
    language: str
    complexity_score: float
    keyword_density: float
    section_count: int
    bullet_point_count: int
    table_count: int
    image_count: int
    link_count: int

@dataclass
class QualityScore:
    """Document quality score"""
    overall_score: float
    metric_scores: Dict[QualityMetric, float]
    confidence: float
    recommendations: List[str]
    model_used: str
    generated_at: datetime

class DocumentQualityOptimizer:
    """ML-powered document quality optimizer"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache_manager = get_cache_manager()
        self.data_processor = get_data_processor()
        self.config = get_config()
        
        # ML Models
        self.quality_models: Dict[QualityMetric, Any] = {}
        self.feature_scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Training data
        self.training_data: List[Tuple[DocumentFeatures, Dict[QualityMetric, float]]] = []
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        
        # Initialize NLTK
        self._initialize_nltk()
        
        # Load or initialize models
        self._initialize_models()
    
    def _initialize_nltk(self):
        """Initialize NLTK components"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            
            self.logger.info("NLTK initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize NLTK: {e}")
    
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Initialize models for each quality metric
            for metric in QualityMetric:
                self.quality_models[metric] = {
                    ModelType.RANDOM_FOREST: RandomForestRegressor(n_estimators=100, random_state=42),
                    ModelType.GRADIENT_BOOSTING: GradientBoostingRegressor(n_estimators=100, random_state=42),
                    ModelType.LINEAR_REGRESSION: LinearRegression()
                }
            
            # Initialize label encoders
            self.label_encoders['business_area'] = LabelEncoder()
            self.label_encoders['document_type'] = LabelEncoder()
            self.label_encoders['language'] = LabelEncoder()
            
            self.logger.info("ML models initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {e}")
    
    async def extract_features(self, content: str, metadata: Dict[str, Any]) -> DocumentFeatures:
        """Extract features from document content"""
        try:
            # Basic text statistics
            words = word_tokenize(content)
            sentences = nltk.sent_tokenize(content)
            paragraphs = content.split('\n\n')
            
            word_count = len(words)
            sentence_count = len(sentences)
            paragraph_count = len([p for p in paragraphs if p.strip()])
            
            # Average lengths
            avg_word_length = np.mean([len(word) for word in words if word.isalpha()]) if words else 0
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            # Readability score (simplified Flesch Reading Ease)
            readability_score = self._calculate_readability(content)
            
            # Sentiment analysis
            sentiment_scores = self.sentiment_analyzer.polarity_scores(content)
            sentiment_score = sentiment_scores['compound']
            
            # Topic coherence (simplified)
            topic_coherence = self._calculate_topic_coherence(content)
            
            # Structure score
            structure_score = self._calculate_structure_score(content)
            
            # Complexity score
            complexity_score = self._calculate_complexity(content)
            
            # Keyword density
            keyword_density = self._calculate_keyword_density(content, metadata.get('keywords', []))
            
            # Content elements
            section_count = len(re.findall(r'^#+\s', content, re.MULTILINE))
            bullet_point_count = len(re.findall(r'^\s*[-*+]\s', content, re.MULTILINE))
            table_count = len(re.findall(r'\|.*\|', content))
            image_count = len(re.findall(r'!\[.*\]\(.*\)', content))
            link_count = len(re.findall(r'\[.*\]\(.*\)', content))
            
            return DocumentFeatures(
                word_count=word_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                avg_word_length=avg_word_length,
                avg_sentence_length=avg_sentence_length,
                readability_score=readability_score,
                sentiment_score=sentiment_score,
                topic_coherence=topic_coherence,
                structure_score=structure_score,
                business_area=metadata.get('business_area', 'unknown'),
                document_type=metadata.get('document_type', 'unknown'),
                language=metadata.get('language', 'en'),
                complexity_score=complexity_score,
                keyword_density=keyword_density,
                section_count=section_count,
                bullet_point_count=bullet_point_count,
                table_count=table_count,
                image_count=image_count,
                link_count=link_count
            )
        
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            # Return default features
            return DocumentFeatures(
                word_count=0, sentence_count=0, paragraph_count=0,
                avg_word_length=0, avg_sentence_length=0,
                readability_score=0, sentiment_score=0, topic_coherence=0,
                structure_score=0, business_area='unknown', document_type='unknown',
                language='en', complexity_score=0, keyword_density=0,
                section_count=0, bullet_point_count=0, table_count=0,
                image_count=0, link_count=0
            )
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)"""
        try:
            sentences = nltk.sent_tokenize(content)
            words = word_tokenize(content)
            
            if not sentences or not words:
                return 0.0
            
            # Count syllables (simplified)
            syllables = sum(self._count_syllables(word) for word in words if word.isalpha())
            
            # Flesch Reading Ease formula
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = syllables / len(words)
            
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            return max(0, min(100, score))  # Clamp between 0 and 100
        
        except Exception:
            return 50.0  # Default moderate readability
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _calculate_topic_coherence(self, content: str) -> float:
        """Calculate topic coherence score"""
        try:
            # Simple coherence based on word repetition and topic consistency
            words = [word.lower() for word in word_tokenize(content) if word.isalpha()]
            
            if len(words) < 10:
                return 0.5
            
            # Calculate word frequency
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Calculate coherence based on repeated important words
            total_words = len(words)
            unique_words = len(word_freq)
            
            # Higher coherence for more repeated words (but not too repetitive)
            repetition_ratio = (total_words - unique_words) / total_words
            coherence = min(0.9, max(0.1, repetition_ratio * 2))
            
            return coherence
        
        except Exception:
            return 0.5
    
    def _calculate_structure_score(self, content: str) -> float:
        """Calculate document structure score"""
        try:
            score = 0.0
            
            # Check for headings
            headings = re.findall(r'^#+\s', content, re.MULTILINE)
            if headings:
                score += 0.3
            
            # Check for paragraphs
            paragraphs = [p for p in content.split('\n\n') if p.strip()]
            if len(paragraphs) > 1:
                score += 0.2
            
            # Check for lists
            lists = re.findall(r'^\s*[-*+]\s', content, re.MULTILINE)
            if lists:
                score += 0.2
            
            # Check for tables
            tables = re.findall(r'\|.*\|', content)
            if tables:
                score += 0.1
            
            # Check for links and images
            links = re.findall(r'\[.*\]\(.*\)', content)
            images = re.findall(r'!\[.*\]\(.*\)', content)
            if links or images:
                score += 0.2
            
            return min(1.0, score)
        
        except Exception:
            return 0.5
    
    def _calculate_complexity(self, content: str) -> float:
        """Calculate document complexity score"""
        try:
            words = word_tokenize(content)
            sentences = nltk.sent_tokenize(content)
            
            if not words or not sentences:
                return 0.5
            
            # Factors contributing to complexity
            avg_word_length = np.mean([len(word) for word in words if word.isalpha()])
            avg_sentence_length = len(words) / len(sentences)
            
            # Normalize complexity (0-1 scale)
            word_complexity = min(1.0, avg_word_length / 10.0)
            sentence_complexity = min(1.0, avg_sentence_length / 30.0)
            
            complexity = (word_complexity + sentence_complexity) / 2
            return complexity
        
        except Exception:
            return 0.5
    
    def _calculate_keyword_density(self, content: str, keywords: List[str]) -> float:
        """Calculate keyword density"""
        try:
            if not keywords:
                return 0.0
            
            words = [word.lower() for word in word_tokenize(content) if word.isalpha()]
            total_words = len(words)
            
            if total_words == 0:
                return 0.0
            
            keyword_count = sum(1 for word in words if word in [kw.lower() for kw in keywords])
            density = keyword_count / total_words
            
            return min(1.0, density * 10)  # Scale to 0-1 range
        
        except Exception:
            return 0.0
    
    async def predict_quality(self, features: DocumentFeatures) -> QualityScore:
        """Predict document quality using ML models"""
        try:
            # Convert features to array
            feature_array = self._features_to_array(features)
            
            # Predict quality for each metric
            metric_scores = {}
            model_used = "ensemble"
            
            for metric in QualityMetric:
                if metric in self.quality_models:
                    # Use ensemble of models
                    predictions = []
                    for model_type, model in self.quality_models[metric].items():
                        try:
                            if hasattr(model, 'predict'):
                                pred = model.predict([feature_array])[0]
                                predictions.append(pred)
                        except Exception as e:
                            self.logger.warning(f"Model {model_type} failed for {metric}: {e}")
                    
                    if predictions:
                        # Use average of predictions
                        metric_scores[metric] = np.mean(predictions)
                    else:
                        # Fallback to rule-based scoring
                        metric_scores[metric] = self._rule_based_quality_score(features, metric)
                else:
                    # Use rule-based scoring
                    metric_scores[metric] = self._rule_based_quality_score(features, metric)
            
            # Calculate overall score
            overall_score = np.mean(list(metric_scores.values()))
            
            # Calculate confidence based on model agreement
            confidence = self._calculate_confidence(metric_scores)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(features, metric_scores)
            
            return QualityScore(
                overall_score=overall_score,
                metric_scores=metric_scores,
                confidence=confidence,
                recommendations=recommendations,
                model_used=model_used,
                generated_at=datetime.now()
            )
        
        except Exception as e:
            self.logger.error(f"Error predicting quality: {e}")
            # Return default quality score
            return QualityScore(
                overall_score=0.5,
                metric_scores={metric: 0.5 for metric in QualityMetric},
                confidence=0.0,
                recommendations=["Unable to analyze document quality"],
                model_used="fallback",
                generated_at=datetime.now()
            )
    
    def _features_to_array(self, features: DocumentFeatures) -> np.ndarray:
        """Convert features to numpy array for ML models"""
        try:
            # Encode categorical features
            business_area_encoded = self._encode_categorical(features.business_area, 'business_area')
            document_type_encoded = self._encode_categorical(features.document_type, 'document_type')
            language_encoded = self._encode_categorical(features.language, 'language')
            
            # Create feature array
            feature_array = np.array([
                features.word_count,
                features.sentence_count,
                features.paragraph_count,
                features.avg_word_length,
                features.avg_sentence_length,
                features.readability_score,
                features.sentiment_score,
                features.topic_coherence,
                features.structure_score,
                business_area_encoded,
                document_type_encoded,
                language_encoded,
                features.complexity_score,
                features.keyword_density,
                features.section_count,
                features.bullet_point_count,
                features.table_count,
                features.image_count,
                features.link_count
            ])
            
            return feature_array
        
        except Exception as e:
            self.logger.error(f"Error converting features to array: {e}")
            return np.zeros(20)  # Return zero array as fallback
    
    def _encode_categorical(self, value: str, category: str) -> int:
        """Encode categorical value"""
        try:
            if category not in self.label_encoders:
                self.label_encoders[category] = LabelEncoder()
                # Fit with known values
                known_values = [area.value for area in BusinessArea] if category == 'business_area' else \
                              [doc_type.value for doc_type in DocumentType] if category == 'document_type' else \
                              ['en', 'es', 'pt', 'fr']
                self.label_encoders[category].fit(known_values + ['unknown'])
            
            return self.label_encoders[category].transform([value])[0]
        except Exception:
            return 0  # Default encoding
    
    def _rule_based_quality_score(self, features: DocumentFeatures, metric: QualityMetric) -> float:
        """Calculate rule-based quality score as fallback"""
        try:
            if metric == QualityMetric.READABILITY:
                # Good readability is between 60-80
                if 60 <= features.readability_score <= 80:
                    return 0.9
                elif 40 <= features.readability_score <= 100:
                    return 0.7
                else:
                    return 0.3
            
            elif metric == QualityMetric.COHERENCE:
                return features.topic_coherence
            
            elif metric == QualityMetric.COMPLETENESS:
                # Based on structure and content length
                completeness = 0.0
                if features.word_count > 100:
                    completeness += 0.3
                if features.section_count > 0:
                    completeness += 0.3
                if features.paragraph_count > 2:
                    completeness += 0.2
                if features.bullet_point_count > 0 or features.table_count > 0:
                    completeness += 0.2
                return completeness
            
            elif metric == QualityMetric.STRUCTURE:
                return features.structure_score
            
            elif metric == QualityMetric.CLARITY:
                # Based on readability and sentence length
                clarity = 0.0
                if features.readability_score > 60:
                    clarity += 0.5
                if features.avg_sentence_length < 20:
                    clarity += 0.3
                if features.avg_word_length < 6:
                    clarity += 0.2
                return clarity
            
            else:
                # Default scoring
                return 0.5
        
        except Exception:
            return 0.5
    
    def _calculate_confidence(self, metric_scores: Dict[QualityMetric, float]) -> float:
        """Calculate confidence in quality prediction"""
        try:
            # Confidence based on score consistency
            scores = list(metric_scores.values())
            if not scores:
                return 0.0
            
            # Higher confidence for more consistent scores
            score_std = np.std(scores)
            confidence = max(0.0, 1.0 - score_std)
            
            return confidence
        
        except Exception:
            return 0.5
    
    def _generate_recommendations(self, features: DocumentFeatures, metric_scores: Dict[QualityMetric, float]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        try:
            # Readability recommendations
            if metric_scores.get(QualityMetric.READABILITY, 0.5) < 0.6:
                if features.avg_sentence_length > 25:
                    recommendations.append("Consider shortening sentences for better readability")
                if features.avg_word_length > 6:
                    recommendations.append("Use simpler words to improve readability")
            
            # Structure recommendations
            if metric_scores.get(QualityMetric.STRUCTURE, 0.5) < 0.6:
                if features.section_count == 0:
                    recommendations.append("Add headings to improve document structure")
                if features.paragraph_count < 3:
                    recommendations.append("Break content into more paragraphs")
            
            # Completeness recommendations
            if metric_scores.get(QualityMetric.COMPLETENESS, 0.5) < 0.6:
                if features.word_count < 200:
                    recommendations.append("Add more content to make the document more comprehensive")
                if features.bullet_point_count == 0 and features.table_count == 0:
                    recommendations.append("Consider adding lists or tables to organize information")
            
            # Clarity recommendations
            if metric_scores.get(QualityMetric.CLARITY, 0.5) < 0.6:
                recommendations.append("Improve clarity by using shorter sentences and simpler language")
            
            # Coherence recommendations
            if metric_scores.get(QualityMetric.COHERENCE, 0.5) < 0.6:
                recommendations.append("Improve topic coherence by focusing on a single main theme")
            
            # Keyword recommendations
            if features.keyword_density < 0.02:
                recommendations.append("Consider adding more relevant keywords to improve SEO")
            elif features.keyword_density > 0.05:
                recommendations.append("Reduce keyword density to avoid over-optimization")
            
            if not recommendations:
                recommendations.append("Document quality is good. Consider minor improvements for optimization.")
        
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations = ["Unable to generate specific recommendations"]
        
        return recommendations
    
    async def train_models(self, training_data: List[Tuple[DocumentFeatures, Dict[QualityMetric, float]]]):
        """Train ML models with new data"""
        try:
            if not training_data:
                self.logger.warning("No training data provided")
                return
            
            self.training_data.extend(training_data)
            
            # Prepare training data
            X = []
            y = {metric: [] for metric in QualityMetric}
            
            for features, quality_scores in self.training_data:
                feature_array = self._features_to_array(features)
                X.append(feature_array)
                
                for metric in QualityMetric:
                    y[metric].append(quality_scores.get(metric, 0.5))
            
            X = np.array(X)
            
            # Train models for each metric
            for metric in QualityMetric:
                if len(y[metric]) < 10:  # Need minimum data
                    continue
                
                y_metric = np.array(y[metric])
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_metric, test_size=0.2, random_state=42
                )
                
                # Scale features
                X_train_scaled = self.feature_scaler.fit_transform(X_train)
                X_test_scaled = self.feature_scaler.transform(X_test)
                
                # Train and evaluate models
                best_model = None
                best_score = -float('inf')
                
                for model_type, model in self.quality_models[metric].items():
                    try:
                        # Train model
                        model.fit(X_train_scaled, y_train)
                        
                        # Evaluate model
                        y_pred = model.predict(X_test_scaled)
                        score = r2_score(y_test, y_pred)
                        
                        if score > best_score:
                            best_score = score
                            best_model = model_type
                        
                        # Store metrics
                        if metric.value not in self.model_metrics:
                            self.model_metrics[metric.value] = {}
                        
                        self.model_metrics[metric.value][model_type.value] = {
                            'r2_score': score,
                            'mse': mean_squared_error(y_test, y_pred),
                            'mae': mean_absolute_error(y_test, y_pred)
                        }
                    
                    except Exception as e:
                        self.logger.error(f"Error training {model_type} for {metric}: {e}")
                
                self.logger.info(f"Trained models for {metric}. Best: {best_model} (R²: {best_score:.3f})")
            
            # Save models
            await self._save_models()
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
    
    async def _save_models(self):
        """Save trained models to disk"""
        try:
            # Save models (in production, use proper model storage)
            model_data = {
                'models': self.quality_models,
                'scaler': self.feature_scaler,
                'encoders': self.label_encoders,
                'metrics': self.model_metrics
            }
            
            # In a real implementation, save to persistent storage
            self.cache_manager.set("ml_models", model_data, ttl=86400)  # 24 hours
            
            self.logger.info("Models saved successfully")
        
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    async def load_models(self):
        """Load trained models from disk"""
        try:
            model_data = self.cache_manager.get("ml_models")
            if model_data:
                self.quality_models = model_data.get('models', self.quality_models)
                self.feature_scaler = model_data.get('scaler', self.feature_scaler)
                self.label_encoders = model_data.get('encoders', self.label_encoders)
                self.model_metrics = model_data.get('metrics', self.model_metrics)
                
                self.logger.info("Models loaded successfully")
            else:
                self.logger.info("No saved models found, using default models")
        
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    async def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            return {
                "model_metrics": self.model_metrics,
                "training_data_size": len(self.training_data),
                "models_loaded": len(self.quality_models),
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting model performance: {e}")
            return {}

# Global quality optimizer
_quality_optimizer: Optional[DocumentQualityOptimizer] = None

def get_quality_optimizer() -> DocumentQualityOptimizer:
    """Get the global quality optimizer"""
    global _quality_optimizer
    if _quality_optimizer is None:
        _quality_optimizer = DocumentQualityOptimizer()
    return _quality_optimizer

# ML router
ml_router = APIRouter(prefix="/ml", tags=["Machine Learning"])

@ml_router.post("/analyze-quality")
async def analyze_document_quality(
    content: str = Field(..., description="Document content to analyze"),
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
):
    """Analyze document quality using ML models"""
    try:
        optimizer = get_quality_optimizer()
        
        # Extract features
        features = await optimizer.extract_features(content, metadata)
        
        # Predict quality
        quality_score = await optimizer.predict_quality(features)
        
        return {
            "quality_score": asdict(quality_score),
            "features": asdict(features),
            "success": True
        }
    
    except Exception as e:
        logger.error(f"Error analyzing document quality: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze document quality")

@ml_router.post("/train-models")
async def train_quality_models(
    training_data: List[Dict[str, Any]] = Field(..., description="Training data")
):
    """Train quality prediction models"""
    try:
        optimizer = get_quality_optimizer()
        
        # Convert training data
        converted_data = []
        for item in training_data:
            features = DocumentFeatures(**item['features'])
            quality_scores = {QualityMetric(k): v for k, v in item['quality_scores'].items()}
            converted_data.append((features, quality_scores))
        
        # Train models
        await optimizer.train_models(converted_data)
        
        return {"success": True, "message": "Models trained successfully"}
    
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise HTTPException(status_code=500, detail="Failed to train models")

@ml_router.get("/model-performance")
async def get_model_performance():
    """Get ML model performance metrics"""
    try:
        optimizer = get_quality_optimizer()
        performance = await optimizer.get_model_performance()
        return {"performance": performance}
    
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model performance")

@ml_router.post("/optimize-document")
async def optimize_document(
    content: str = Field(..., description="Document content to optimize"),
    target_quality: float = Field(0.8, description="Target quality score"),
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
):
    """Optimize document content for better quality"""
    try:
        optimizer = get_quality_optimizer()
        
        # Analyze current quality
        features = await optimizer.extract_features(content, metadata)
        quality_score = await optimizer.predict_quality(features)
        
        # Generate optimization suggestions
        optimization_suggestions = []
        
        if quality_score.overall_score < target_quality:
            optimization_suggestions.extend(quality_score.recommendations)
        
        return {
            "current_quality": quality_score.overall_score,
            "target_quality": target_quality,
            "optimization_suggestions": optimization_suggestions,
            "quality_breakdown": quality_score.metric_scores,
            "success": True
        }
    
    except Exception as e:
        logger.error(f"Error optimizing document: {e}")
        raise HTTPException(status_code=500, detail="Failed to optimize document")


