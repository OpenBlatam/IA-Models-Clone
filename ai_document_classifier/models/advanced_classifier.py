"""
Advanced AI Document Classifier Models
======================================

Enhanced classification models with advanced NLP capabilities, machine learning,
and multi-modal document analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import pickle
from pathlib import Path
import re
from collections import Counter, defaultdict
import hashlib
from datetime import datetime, timedelta

# Advanced ML libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Scikit-learn not available. Advanced ML features disabled.")

# NLP libraries
try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("SpaCy not available. Advanced NLP features disabled.")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Basic NLP features disabled.")

logger = logging.getLogger(__name__)

class ClassificationMethod(Enum):
    """Classification methods available"""
    PATTERN_MATCHING = "pattern_matching"
    TFIDF_SVM = "tfidf_svm"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NAIVE_BAYES = "naive_bayes"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    AI_GPT = "ai_gpt"

@dataclass
class DocumentFeatures:
    """Extracted features from document text"""
    # Text statistics
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    avg_sentence_length: float = 0.0
    avg_word_length: float = 0.0
    
    # Linguistic features
    pos_tags: Dict[str, int] = field(default_factory=dict)
    named_entities: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    
    # Content features
    keywords: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    
    # Structural features
    has_titles: bool = False
    has_sections: bool = False
    has_lists: bool = False
    has_tables: bool = False
    has_footnotes: bool = False
    
    # Domain-specific features
    legal_terms: int = 0
    technical_terms: int = 0
    business_terms: int = 0
    academic_terms: int = 0
    creative_terms: int = 0

@dataclass
class AdvancedClassificationResult:
    """Enhanced classification result with detailed analysis"""
    document_type: str
    confidence: float
    method_used: ClassificationMethod
    features: DocumentFeatures
    keywords: List[str]
    reasoning: str
    alternative_types: List[Tuple[str, float]] = field(default_factory=list)
    processing_time: float = 0.0
    model_version: str = "1.0"
    timestamp: datetime = field(default_factory=datetime.now)

class AdvancedDocumentClassifier:
    """
    Advanced document classifier with multiple ML models and NLP capabilities
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the advanced classifier
        
        Args:
            model_dir: Directory to save/load trained models
        """
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent / "saved_models"
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize NLP components
        self._init_nlp()
        
        # Initialize ML models
        self._init_ml_models()
        
        # Load or create training data
        self.training_data = self._load_training_data()
        
        # Document type patterns (enhanced)
        self.enhanced_patterns = self._create_enhanced_patterns()
        
        # Feature extractors
        self.feature_extractors = self._init_feature_extractors()
        
        # Model cache
        self.model_cache = {}
        self.feature_cache = {}
        
    def _init_nlp(self):
        """Initialize NLP components"""
        self.nlp = None
        self.lemmatizer = None
        self.stemmer = None
        self.stop_words = set()
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("SpaCy English model not found. Using basic tokenization.")
        
        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.stemmer = PorterStemmer()
                self.stop_words = set(stopwords.words('english'))
            except LookupError:
                logger.warning("NLTK data not found. Download required: nltk.download('all')")
    
    def _init_ml_models(self):
        """Initialize machine learning models"""
        if not ML_AVAILABLE:
            logger.warning("Scikit-learn not available. ML models disabled.")
            return
        
        self.models = {
            ClassificationMethod.TFIDF_SVM: Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            ClassificationMethod.RANDOM_FOREST: Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            ClassificationMethod.GRADIENT_BOOSTING: Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('classifier', GradientBoostingClassifier(random_state=42))
            ]),
            ClassificationMethod.NAIVE_BAYES: Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('classifier', MultinomialNB())
            ])
        }
        
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def _create_enhanced_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Create enhanced document type patterns with more sophisticated matching"""
        return {
            "novel": {
                "keywords": [
                    "novel", "fiction", "story", "character", "plot", "chapter", "narrative",
                    "protagonist", "antagonist", "dialogue", "scene", "setting", "theme",
                    "conflict", "resolution", "climax", "epilogue", "prologue", "genre",
                    "romance", "mystery", "thriller", "fantasy", "sci-fi", "horror"
                ],
                "patterns": [
                    r"\b(chapter|scene|character|plot|story|novel|fiction|narrative)\b",
                    r"\b(protagonist|antagonist|hero|villain|main character)\b",
                    r"\b(romance|mystery|thriller|fantasy|sci-fi|horror|genre)\b",
                    r"\b(epilogue|prologue|climax|resolution|conflict)\b"
                ],
                "structural_indicators": ["chapter", "scene", "dialogue", "narrative"],
                "domain_terms": ["character development", "plot twist", "story arc", "world building"]
            },
            "contract": {
                "keywords": [
                    "contract", "agreement", "terms", "conditions", "party", "parties",
                    "obligation", "liability", "indemnity", "clause", "section", "whereas",
                    "hereby", "witnesseth", "legal", "binding", "signature", "execution",
                    "breach", "remedy", "jurisdiction", "governing law", "force majeure"
                ],
                "patterns": [
                    r"\b(contract|agreement|terms|conditions|legal|binding)\b",
                    r"\b(party|parties|obligation|liability|indemnity)\b",
                    r"\b(whereas|hereby|witnesseth|clause|section)\b",
                    r"\b(breach|remedy|jurisdiction|governing law)\b"
                ],
                "structural_indicators": ["clause", "section", "whereas", "hereby"],
                "domain_terms": ["legal obligation", "breach of contract", "force majeure", "governing law"]
            },
            "design": {
                "keywords": [
                    "design", "blueprint", "specification", "technical", "drawing", "architecture",
                    "engineering", "dimensions", "materials", "components", "assembly", "CAD",
                    "model", "prototype", "wireframe", "mockup", "rendering", "sketch",
                    "technical drawing", "engineering drawing", "architectural plan"
                ],
                "patterns": [
                    r"\b(design|blueprint|specification|technical|engineering)\b",
                    r"\b(architecture|dimensions|materials|components|assembly)\b",
                    r"\b(CAD|model|prototype|wireframe|mockup|rendering)\b",
                    r"\b(technical drawing|engineering drawing|architectural plan)\b"
                ],
                "structural_indicators": ["specification", "drawing", "blueprint", "technical"],
                "domain_terms": ["technical specification", "engineering design", "architectural plan", "CAD model"]
            },
            "business_plan": {
                "keywords": [
                    "business", "plan", "strategy", "market", "revenue", "profit", "investment",
                    "funding", "executive", "summary", "financial", "projection", "milestone",
                    "objective", "goal", "mission", "vision", "SWOT", "analysis", "competitor",
                    "market analysis", "financial forecast", "business model", "value proposition"
                ],
                "patterns": [
                    r"\b(business plan|strategy|market analysis|financial projection)\b",
                    r"\b(revenue|profit|investment|funding|ROI|ROE)\b",
                    r"\b(executive summary|mission|vision|SWOT analysis)\b",
                    r"\b(business model|value proposition|competitive advantage)\b"
                ],
                "structural_indicators": ["executive summary", "market analysis", "financial projection"],
                "domain_terms": ["market opportunity", "competitive analysis", "financial forecast", "business model"]
            },
            "academic_paper": {
                "keywords": [
                    "research", "study", "analysis", "methodology", "results", "conclusion",
                    "abstract", "introduction", "literature", "review", "hypothesis", "data",
                    "findings", "citation", "reference", "peer-reviewed", "journal", "conference",
                    "empirical", "theoretical", "framework", "model", "statistical", "significance"
                ],
                "patterns": [
                    r"\b(research|study|analysis|methodology|empirical|theoretical)\b",
                    r"\b(abstract|introduction|conclusion|literature review)\b",
                    r"\b(hypothesis|findings|statistical significance|peer-reviewed)\b",
                    r"\b(citation|reference|journal|conference|academic)\b"
                ],
                "structural_indicators": ["abstract", "introduction", "methodology", "results", "conclusion"],
                "domain_terms": ["literature review", "research methodology", "statistical analysis", "peer review"]
            }
        }
    
    def _init_feature_extractors(self) -> Dict[str, callable]:
        """Initialize feature extraction functions"""
        return {
            "text_stats": self._extract_text_statistics,
            "linguistic": self._extract_linguistic_features,
            "structural": self._extract_structural_features,
            "domain_specific": self._extract_domain_features,
            "semantic": self._extract_semantic_features
        }
    
    def _extract_text_statistics(self, text: str) -> Dict[str, Any]:
        """Extract basic text statistics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\n\n')
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len([p for p in paragraphs if p.strip()]),
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "avg_word_length": sum(len(word) for word in words) / max(len(words), 1),
            "text_length": len(text),
            "unique_words": len(set(word.lower() for word in words))
        }
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features using NLP"""
        features = {
            "pos_tags": {},
            "named_entities": [],
            "sentiment_score": 0.0,
            "complexity_score": 0.0
        }
        
        if self.nlp:
            doc = self.nlp(text)
            
            # POS tags
            pos_counts = Counter([token.pos_ for token in doc])
            features["pos_tags"] = dict(pos_counts)
            
            # Named entities
            features["named_entities"] = [ent.text for ent in doc.ents]
            
            # Sentiment (if available)
            if hasattr(doc, 'sentiment'):
                features["sentiment_score"] = doc.sentiment
        
        # Complexity score (Flesch-like)
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        if words and sentences:
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
            features["complexity_score"] = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        
        return features
    
    def _extract_structural_features(self, text: str) -> Dict[str, bool]:
        """Extract structural features"""
        return {
            "has_titles": bool(re.search(r'^#+\s', text, re.MULTILINE)),
            "has_sections": bool(re.search(r'^\d+\.\s', text, re.MULTILINE)),
            "has_lists": bool(re.search(r'^\s*[-*+]\s', text, re.MULTILINE)),
            "has_tables": bool(re.search(r'\|.*\|', text)),
            "has_footnotes": bool(re.search(r'\[\d+\]', text)),
            "has_quotes": bool(re.search(r'["""]', text)),
            "has_numbers": bool(re.search(r'\d+', text)),
            "has_capitals": bool(re.search(r'[A-Z]{3,}', text))
        }
    
    def _extract_domain_features(self, text: str) -> Dict[str, int]:
        """Extract domain-specific term counts"""
        domain_terms = {
            "legal": ["contract", "agreement", "liability", "breach", "jurisdiction", "legal"],
            "technical": ["specification", "technical", "engineering", "system", "component"],
            "business": ["revenue", "profit", "market", "strategy", "business", "investment"],
            "academic": ["research", "study", "analysis", "methodology", "hypothesis", "citation"],
            "creative": ["story", "character", "plot", "narrative", "creative", "artistic"]
        }
        
        text_lower = text.lower()
        counts = {}
        for domain, terms in domain_terms.items():
            counts[f"{domain}_terms"] = sum(text_lower.count(term) for term in terms)
        
        return counts
    
    def _extract_semantic_features(self, text: str) -> Dict[str, Any]:
        """Extract semantic features"""
        features = {
            "key_phrases": [],
            "topics": [],
            "entities": []
        }
        
        if self.nlp:
            doc = self.nlp(text)
            
            # Key phrases (noun phrases)
            features["key_phrases"] = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
            
            # Named entities
            features["entities"] = [(ent.text, ent.label_) for ent in doc.ents]
        
        return features
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximate)"""
        word = word.lower()
        vowels = "aeiouy"
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
    
    def extract_features(self, text: str) -> DocumentFeatures:
        """Extract comprehensive features from document text"""
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.feature_cache:
            return self.feature_cache[text_hash]
        
        features = DocumentFeatures()
        
        # Extract all feature types
        for extractor_name, extractor_func in self.feature_extractors.items():
            try:
                extracted = extractor_func(text)
                for key, value in extracted.items():
                    setattr(features, key, value)
            except Exception as e:
                logger.warning(f"Feature extraction failed for {extractor_name}: {e}")
        
        # Cache the result
        self.feature_cache[text_hash] = features
        return features
    
    def classify_with_ml(self, text: str, method: ClassificationMethod = ClassificationMethod.ENSEMBLE) -> AdvancedClassificationResult:
        """Classify document using machine learning models"""
        if not ML_AVAILABLE or not self.is_trained:
            return self._fallback_classification(text)
        
        start_time = datetime.now()
        
        # Extract features
        features = self.extract_features(text)
        
        # Prepare text for ML models
        if method == ClassificationMethod.ENSEMBLE:
            # Use ensemble of models
            predictions = []
            confidences = []
            
            for model_method, model in self.models.items():
                try:
                    pred = model.predict([text])[0]
                    pred_proba = model.predict_proba([text])[0]
                    predictions.append(pred)
                    confidences.append(max(pred_proba))
                except Exception as e:
                    logger.warning(f"Model {model_method} failed: {e}")
            
            if predictions:
                # Majority vote
                final_prediction = max(set(predictions), key=predictions.count)
                final_confidence = sum(confidences) / len(confidences)
            else:
                return self._fallback_classification(text)
        else:
            # Use single model
            if method not in self.models:
                return self._fallback_classification(text)
            
            model = self.models[method]
            final_prediction = model.predict([text])[0]
            pred_proba = model.predict_proba([text])[0]
            final_confidence = max(pred_proba)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AdvancedClassificationResult(
            document_type=final_prediction,
            confidence=final_confidence,
            method_used=method,
            features=features,
            keywords=self._extract_keywords(text),
            reasoning=f"Classified using {method.value} with confidence {final_confidence:.2f}",
            processing_time=processing_time
        )
    
    def _fallback_classification(self, text: str) -> AdvancedClassificationResult:
        """Fallback to pattern-based classification"""
        start_time = datetime.now()
        
        features = self.extract_features(text)
        text_lower = text.lower()
        
        scores = {}
        for doc_type, patterns in self.enhanced_patterns.items():
            score = 0
            
            # Keyword matching
            for keyword in patterns["keywords"]:
                if keyword in text_lower:
                    score += 1
            
            # Pattern matching
            for pattern in patterns["patterns"]:
                matches = re.findall(pattern, text_lower)
                score += len(matches) * 0.5
            
            # Structural indicators
            for indicator in patterns["structural_indicators"]:
                if indicator in text_lower:
                    score += 0.3
            
            scores[doc_type] = score
        
        if not scores or max(scores.values()) == 0:
            best_type = "unknown"
            confidence = 0.0
        else:
            best_type = max(scores, key=scores.get)
            max_score = scores[best_type]
            total_possible = len(self.enhanced_patterns[best_type]["keywords"]) + len(self.enhanced_patterns[best_type]["patterns"])
            confidence = min(max_score / total_possible, 1.0)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AdvancedClassificationResult(
            document_type=best_type,
            confidence=confidence,
            method_used=ClassificationMethod.PATTERN_MATCHING,
            features=features,
            keywords=self._extract_keywords(text),
            reasoning=f"Pattern-based classification with {len(features.keywords)} keywords matched",
            processing_time=processing_time
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        if not self.nlp:
            # Simple keyword extraction
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq = Counter(words)
            return [word for word, freq in word_freq.most_common(10) if len(word) > 3]
        
        doc = self.nlp(text)
        keywords = []
        
        # Extract important words (nouns, adjectives, verbs)
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 3):
                keywords.append(token.lemma_.lower())
        
        # Return most common keywords
        keyword_freq = Counter(keywords)
        return [word for word, freq in keyword_freq.most_common(10)]
    
    def _load_training_data(self) -> pd.DataFrame:
        """Load or create training data for ML models"""
        training_file = self.model_dir / "training_data.pkl"
        
        if training_file.exists():
            try:
                return pd.read_pickle(training_file)
            except Exception as e:
                logger.warning(f"Failed to load training data: {e}")
        
        # Create synthetic training data
        return self._create_synthetic_training_data()
    
    def _create_synthetic_training_data(self) -> pd.DataFrame:
        """Create synthetic training data for initial model training"""
        training_examples = [
            # Novels
            ("I want to write a science fiction novel about space exploration", "novel"),
            ("Create a romance story with compelling characters", "novel"),
            ("Write a mystery novel with plot twists", "novel"),
            ("Fantasy novel with magic and adventure", "novel"),
            
            # Contracts
            ("Service agreement contract for web development", "contract"),
            ("Employment contract with benefits and terms", "contract"),
            ("Legal contract with liability clauses", "contract"),
            ("Partnership agreement with terms and conditions", "contract"),
            
            # Design
            ("Technical design document with specifications", "design"),
            ("Architectural blueprint for new building", "design"),
            ("Product design with CAD models", "design"),
            ("Engineering design with technical drawings", "design"),
            
            # Business Plans
            ("Business plan for startup company", "business_plan"),
            ("Market analysis and financial projections", "business_plan"),
            ("Strategic business plan with goals", "business_plan"),
            ("Investment proposal with revenue model", "business_plan"),
            
            # Academic Papers
            ("Research paper on machine learning", "academic_paper"),
            ("Scientific study with methodology", "academic_paper"),
            ("Academic analysis with citations", "academic_paper"),
            ("Empirical research with statistical analysis", "academic_paper")
        ]
        
        df = pd.DataFrame(training_examples, columns=['text', 'document_type'])
        
        # Save for future use
        try:
            df.to_pickle(self.model_dir / "training_data.pkl")
        except Exception as e:
            logger.warning(f"Failed to save training data: {e}")
        
        return df
    
    def train_models(self, training_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Train all ML models"""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available. Cannot train models.")
            return {}
        
        if training_data is None:
            training_data = self.training_data
        
        if training_data.empty:
            logger.warning("No training data available.")
            return {}
        
        # Prepare data
        X = training_data['text']
        y = training_data['document_type']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        results = {}
        
        # Train each model
        for method, model in self.models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = (y_pred == y_test).mean()
                results[method.value] = accuracy
                
                # Save model
                model_file = self.model_dir / f"{method.value}_model.pkl"
                joblib.dump(model, model_file)
                
                logger.info(f"Trained {method.value} with accuracy: {accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train {method.value}: {e}")
                results[method.value] = 0.0
        
        self.is_trained = True
        
        # Save label encoder
        encoder_file = self.model_dir / "label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_file)
        
        return results
    
    def load_trained_models(self) -> bool:
        """Load pre-trained models"""
        if not ML_AVAILABLE:
            return False
        
        try:
            # Load label encoder
            encoder_file = self.model_dir / "label_encoder.pkl"
            if encoder_file.exists():
                self.label_encoder = joblib.load(encoder_file)
            
            # Load models
            for method in self.models.keys():
                model_file = self.model_dir / f"{method.value}_model.pkl"
                if model_file.exists():
                    self.models[method] = joblib.load(model_file)
            
            self.is_trained = True
            logger.info("Successfully loaded trained models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        if not self.is_trained:
            return {"status": "Models not trained"}
        
        performance = {
            "models_loaded": len([m for m in self.models.values() if hasattr(m, 'predict')]),
            "training_data_size": len(self.training_data),
            "feature_extractors": len(self.feature_extractors),
            "cache_size": len(self.feature_cache)
        }
        
        return performance

# Example usage and testing
if __name__ == "__main__":
    # Initialize the advanced classifier
    classifier = AdvancedDocumentClassifier()
    
    # Train models (if training data available)
    if ML_AVAILABLE:
        results = classifier.train_models()
        print("Training results:", results)
    
    # Test classification
    test_queries = [
        "I want to write a science fiction novel about alien civilizations",
        "Create a service agreement contract with payment terms",
        "Design a mobile app interface with user experience focus",
        "Write a business plan for a tech startup with market analysis"
    ]
    
    print("\nAdvanced Classification Results:")
    print("=" * 50)
    
    for query in test_queries:
        result = classifier.classify_with_ml(query)
        print(f"\nQuery: {query}")
        print(f"Type: {result.document_type}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Method: {result.method_used.value}")
        print(f"Keywords: {', '.join(result.keywords[:5])}")
        print(f"Processing time: {result.processing_time:.3f}s")
        print(f"Features: {result.features.word_count} words, {result.features.sentence_count} sentences")



























