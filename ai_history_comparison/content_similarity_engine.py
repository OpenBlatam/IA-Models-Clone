"""
AI History Comparison System - Content Similarity Engine

This module provides advanced content similarity detection, plagiarism analysis,
and content originality scoring capabilities.
"""

import logging
import numpy as np
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import re
from collections import defaultdict, Counter
import difflib

# Advanced NLP libraries
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import spacy
    from spacy import displacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

from .ai_history_analyzer import AIHistoryAnalyzer, HistoryEntry

logger = logging.getLogger(__name__)

class SimilarityType(Enum):
    """Types of similarity analysis"""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    LEXICAL_SIMILARITY = "lexical_similarity"
    STRUCTURAL_SIMILARITY = "structural_similarity"
    STYLISTIC_SIMILARITY = "stylistic_similarity"
    CONTENT_OVERLAP = "content_overlap"

class PlagiarismLevel(Enum):
    """Plagiarism detection levels"""
    ORIGINAL = "original"
    PARAPHRASED = "paraphrased"
    HEAVILY_PARAPHRASED = "heavily_paraphrased"
    SUSPICIOUS = "suspicious"
    PLAGIARIZED = "plagiarized"

@dataclass
class SimilarityResult:
    """Result of similarity analysis"""
    similarity_score: float
    similarity_type: SimilarityType
    matched_segments: List[Dict[str, Any]]
    confidence: float
    analysis_timestamp: datetime

@dataclass
class PlagiarismDetection:
    """Plagiarism detection result"""
    plagiarism_level: PlagiarismLevel
    similarity_score: float
    matched_content: List[Dict[str, Any]]
    originality_score: float
    suspicious_patterns: List[str]
    recommendations: List[str]
    confidence: float
    detection_timestamp: datetime

@dataclass
class ContentFingerprint:
    """Content fingerprint for similarity detection"""
    content_hash: str
    semantic_vector: Optional[np.ndarray]
    lexical_features: Dict[str, Any]
    structural_features: Dict[str, Any]
    stylistic_features: Dict[str, Any]
    n_grams: List[str]
    created_at: datetime

class ContentSimilarityEngine:
    """
    Advanced content similarity and plagiarism detection engine
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize content similarity engine"""
        self.config = config or {}
        self.sentence_model = None
        self.nlp = None
        self.content_fingerprints: Dict[str, ContentFingerprint] = {}
        self.similarity_cache: Dict[str, float] = {}
        
        # Initialize sentence transformer if available
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer model loaded")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
        
        # Initialize spaCy if available
        if HAS_SPACY:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Similarity thresholds
        self.similarity_thresholds = {
            "high_similarity": 0.8,
            "medium_similarity": 0.6,
            "low_similarity": 0.4,
            "plagiarism_threshold": 0.85,
            "suspicious_threshold": 0.7
        }
        
        logger.info("Content Similarity Engine initialized")

    def create_content_fingerprint(self, content: str, content_id: str = None) -> ContentFingerprint:
        """Create a comprehensive content fingerprint"""
        try:
            # Generate content hash
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            # Generate semantic vector
            semantic_vector = None
            if self.sentence_model:
                try:
                    semantic_vector = self.sentence_model.encode([content])[0]
                except Exception as e:
                    logger.warning(f"Failed to generate semantic vector: {e}")
            
            # Extract lexical features
            lexical_features = self._extract_lexical_features(content)
            
            # Extract structural features
            structural_features = self._extract_structural_features(content)
            
            # Extract stylistic features
            stylistic_features = self._extract_stylistic_features(content)
            
            # Generate n-grams
            n_grams = self._generate_n_grams(content)
            
            fingerprint = ContentFingerprint(
                content_hash=content_hash,
                semantic_vector=semantic_vector,
                lexical_features=lexical_features,
                structural_features=structural_features,
                stylistic_features=stylistic_features,
                n_grams=n_grams,
                created_at=datetime.now()
            )
            
            if content_id:
                self.content_fingerprints[content_id] = fingerprint
            
            return fingerprint
            
        except Exception as e:
            logger.error(f"Error creating content fingerprint: {e}")
            return None

    def calculate_similarity(self, content1: str, content2: str, 
                           similarity_types: List[SimilarityType] = None) -> Dict[SimilarityType, SimilarityResult]:
        """Calculate comprehensive similarity between two content pieces"""
        try:
            if similarity_types is None:
                similarity_types = list(SimilarityType)
            
            results = {}
            
            for sim_type in similarity_types:
                try:
                    if sim_type == SimilarityType.SEMANTIC_SIMILARITY:
                        result = self._calculate_semantic_similarity(content1, content2)
                    elif sim_type == SimilarityType.LEXICAL_SIMILARITY:
                        result = self._calculate_lexical_similarity(content1, content2)
                    elif sim_type == SimilarityType.STRUCTURAL_SIMILARITY:
                        result = self._calculate_structural_similarity(content1, content2)
                    elif sim_type == SimilarityType.STYLISTIC_SIMILARITY:
                        result = self._calculate_stylistic_similarity(content1, content2)
                    elif sim_type == SimilarityType.CONTENT_OVERLAP:
                        result = self._calculate_content_overlap(content1, content2)
                    else:
                        continue
                    
                    results[sim_type] = result
                    
                except Exception as e:
                    logger.warning(f"Error calculating {sim_type.value} similarity: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return {}

    def detect_plagiarism(self, content: str, reference_content: str = None, 
                         content_id: str = None) -> PlagiarismDetection:
        """Detect plagiarism in content"""
        try:
            # If no reference content provided, compare against all stored fingerprints
            if reference_content is None:
                return self._detect_plagiarism_against_database(content, content_id)
            else:
                return self._detect_plagiarism_against_content(content, reference_content)
                
        except Exception as e:
            logger.error(f"Error detecting plagiarism: {e}")
            return PlagiarismDetection(
                plagiarism_level=PlagiarismLevel.ORIGINAL,
                similarity_score=0.0,
                matched_content=[],
                originality_score=1.0,
                suspicious_patterns=[],
                recommendations=["Analysis failed"],
                confidence=0.0,
                detection_timestamp=datetime.now()
            )

    def find_similar_content(self, content: str, threshold: float = 0.7, 
                           max_results: int = 10) -> List[Dict[str, Any]]:
        """Find similar content in the database"""
        try:
            similar_content = []
            
            # Create fingerprint for input content
            input_fingerprint = self.create_content_fingerprint(content)
            
            # Compare against all stored fingerprints
            for content_id, fingerprint in self.content_fingerprints.items():
                try:
                    # Calculate semantic similarity
                    semantic_sim = 0.0
                    if (input_fingerprint.semantic_vector is not None and 
                        fingerprint.semantic_vector is not None):
                        semantic_sim = cosine_similarity(
                            [input_fingerprint.semantic_vector],
                            [fingerprint.semantic_vector]
                        )[0][0]
                    
                    # Calculate lexical similarity
                    lexical_sim = self._calculate_lexical_similarity_score(
                        input_fingerprint.lexical_features,
                        fingerprint.lexical_features
                    )
                    
                    # Calculate n-gram overlap
                    ngram_sim = self._calculate_ngram_similarity(
                        input_fingerprint.n_grams,
                        fingerprint.n_grams
                    )
                    
                    # Combined similarity score
                    combined_score = (semantic_sim * 0.5 + lexical_sim * 0.3 + ngram_sim * 0.2)
                    
                    if combined_score >= threshold:
                        similar_content.append({
                            "content_id": content_id,
                            "similarity_score": combined_score,
                            "semantic_similarity": semantic_sim,
                            "lexical_similarity": lexical_sim,
                            "ngram_similarity": ngram_sim,
                            "created_at": fingerprint.created_at.isoformat()
                        })
                        
                except Exception as e:
                    logger.warning(f"Error comparing with content {content_id}: {e}")
                    continue
            
            # Sort by similarity score and return top results
            similar_content.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similar_content[:max_results]
            
        except Exception as e:
            logger.error(f"Error finding similar content: {e}")
            return []

    def calculate_originality_score(self, content: str) -> float:
        """Calculate originality score for content"""
        try:
            # Find similar content
            similar_content = self.find_similar_content(content, threshold=0.3, max_results=5)
            
            if not similar_content:
                return 1.0  # Completely original
            
            # Calculate originality based on highest similarity
            max_similarity = max(item["similarity_score"] for item in similar_content)
            originality_score = 1.0 - max_similarity
            
            return max(0.0, originality_score)
            
        except Exception as e:
            logger.error(f"Error calculating originality score: {e}")
            return 0.5  # Default neutral score

    def analyze_content_evolution(self, content_series: List[str], 
                                timestamps: List[datetime] = None) -> Dict[str, Any]:
        """Analyze how content evolves over time"""
        try:
            if len(content_series) < 2:
                return {"error": "Need at least 2 content pieces for evolution analysis"}
            
            if timestamps is None:
                timestamps = [datetime.now() - timedelta(hours=i) for i in range(len(content_series))]
            
            # Calculate similarities between consecutive content pieces
            evolution_similarities = []
            for i in range(len(content_series) - 1):
                similarity_results = self.calculate_similarity(
                    content_series[i], content_series[i + 1],
                    [SimilarityType.SEMANTIC_SIMILARITY, SimilarityType.LEXICAL_SIMILARITY]
                )
                
                semantic_sim = similarity_results.get(SimilarityType.SEMANTIC_SIMILARITY, SimilarityResult(0.0, SimilarityType.SEMANTIC_SIMILARITY, [], 0.0, datetime.now()))
                lexical_sim = similarity_results.get(SimilarityType.LEXICAL_SIMILARITY, SimilarityResult(0.0, SimilarityType.LEXICAL_SIMILARITY, [], 0.0, datetime.now()))
                
                evolution_similarities.append({
                    "from_index": i,
                    "to_index": i + 1,
                    "semantic_similarity": semantic_sim.similarity_score,
                    "lexical_similarity": lexical_sim.similarity_score,
                    "timestamp_from": timestamps[i].isoformat(),
                    "timestamp_to": timestamps[i + 1].isoformat()
                })
            
            # Analyze evolution patterns
            avg_semantic_sim = np.mean([item["semantic_similarity"] for item in evolution_similarities])
            avg_lexical_sim = np.mean([item["lexical_similarity"] for item in evolution_similarities])
            
            # Determine evolution pattern
            if avg_semantic_sim > 0.8 and avg_lexical_sim > 0.7:
                evolution_pattern = "stable"
            elif avg_semantic_sim < 0.5 or avg_lexical_sim < 0.4:
                evolution_pattern = "divergent"
            else:
                evolution_pattern = "gradual_change"
            
            return {
                "evolution_similarities": evolution_similarities,
                "average_semantic_similarity": avg_semantic_sim,
                "average_lexical_similarity": avg_lexical_sim,
                "evolution_pattern": evolution_pattern,
                "total_content_pieces": len(content_series),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content evolution: {e}")
            return {"error": str(e)}

    def _extract_lexical_features(self, content: str) -> Dict[str, Any]:
        """Extract lexical features from content"""
        features = {}
        
        # Basic text statistics
        words = content.split()
        sentences = content.split('.')
        
        features["word_count"] = len(words)
        features["sentence_count"] = len(sentences)
        features["avg_word_length"] = np.mean([len(word) for word in words]) if words else 0
        features["avg_sentence_length"] = np.mean([len(sentence.split()) for sentence in sentences]) if sentences else 0
        
        # Vocabulary features
        unique_words = set(word.lower() for word in words)
        features["vocabulary_size"] = len(unique_words)
        features["vocabulary_richness"] = len(unique_words) / len(words) if words else 0
        
        # Character features
        features["char_count"] = len(content)
        features["char_count_no_spaces"] = len(content.replace(' ', ''))
        features["punctuation_count"] = len(re.findall(r'[^\w\s]', content))
        
        # Word frequency features
        word_freq = Counter(word.lower() for word in words)
        features["most_common_word"] = word_freq.most_common(1)[0] if word_freq else ("", 0)
        features["word_frequency_std"] = np.std(list(word_freq.values())) if word_freq else 0
        
        return features

    def _extract_structural_features(self, content: str) -> Dict[str, Any]:
        """Extract structural features from content"""
        features = {}
        
        # Paragraph structure
        paragraphs = content.split('\n\n')
        features["paragraph_count"] = len(paragraphs)
        features["avg_paragraph_length"] = np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0
        
        # Sentence structure
        sentences = content.split('.')
        features["sentence_count"] = len(sentences)
        features["avg_sentence_length"] = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        features["sentence_length_std"] = np.std([len(s.split()) for s in sentences]) if sentences else 0
        
        # Capitalization patterns
        features["capitalization_ratio"] = sum(1 for c in content if c.isupper()) / len(content) if content else 0
        features["title_case_words"] = len(re.findall(r'\b[A-Z][a-z]+\b', content))
        
        # Punctuation patterns
        features["exclamation_count"] = content.count('!')
        features["question_count"] = content.count('?')
        features["comma_count"] = content.count(',')
        features["semicolon_count"] = content.count(';')
        features["colon_count"] = content.count(':')
        
        return features

    def _extract_stylistic_features(self, content: str) -> Dict[str, Any]:
        """Extract stylistic features from content"""
        features = {}
        
        # Readability features
        words = content.split()
        sentences = content.split('.')
        
        if words and sentences:
            # Flesch Reading Ease approximation
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = np.mean([self._count_syllables(word) for word in words])
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            features["flesch_score"] = max(0, min(100, flesch_score))
        
        # Formality features
        formal_words = ['therefore', 'however', 'furthermore', 'moreover', 'consequently', 'nevertheless']
        informal_words = ['yeah', 'okay', 'cool', 'awesome', 'gonna', 'wanna']
        
        features["formal_word_count"] = sum(1 for word in words if word.lower() in formal_words)
        features["informal_word_count"] = sum(1 for word in words if word.lower() in informal_words)
        features["formality_ratio"] = features["formal_word_count"] / (features["informal_word_count"] + 1)
        
        # Complexity features
        complex_words = [word for word in words if len(word) > 6]
        features["complex_word_ratio"] = len(complex_words) / len(words) if words else 0
        
        # Repetition features
        word_freq = Counter(word.lower() for word in words)
        repeated_words = [word for word, count in word_freq.items() if count > 1]
        features["repetition_ratio"] = len(repeated_words) / len(unique_words) if unique_words else 0
        
        return features

    def _generate_n_grams(self, content: str, n: int = 3) -> List[str]:
        """Generate n-grams from content"""
        words = content.lower().split()
        n_grams = []
        
        for i in range(len(words) - n + 1):
            n_gram = ' '.join(words[i:i + n])
            n_grams.append(n_gram)
        
        return n_grams

    def _calculate_semantic_similarity(self, content1: str, content2: str) -> SimilarityResult:
        """Calculate semantic similarity using sentence transformers"""
        try:
            if not self.sentence_model:
                return SimilarityResult(0.0, SimilarityType.SEMANTIC_SIMILARITY, [], 0.0, datetime.now())
            
            # Encode both content pieces
            embeddings = self.sentence_model.encode([content1, content2])
            
            # Calculate cosine similarity
            similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # Find similar segments (simplified)
            matched_segments = self._find_similar_segments(content1, content2, "semantic")
            
            return SimilarityResult(
                similarity_score=float(similarity_score),
                similarity_type=SimilarityType.SEMANTIC_SIMILARITY,
                matched_segments=matched_segments,
                confidence=0.9 if similarity_score > 0.8 else 0.7,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return SimilarityResult(0.0, SimilarityType.SEMANTIC_SIMILARITY, [], 0.0, datetime.now())

    def _calculate_lexical_similarity(self, content1: str, content2: str) -> SimilarityResult:
        """Calculate lexical similarity using TF-IDF"""
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([content1, content2])
            
            # Calculate cosine similarity
            similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Find similar segments
            matched_segments = self._find_similar_segments(content1, content2, "lexical")
            
            return SimilarityResult(
                similarity_score=float(similarity_score),
                similarity_type=SimilarityType.LEXICAL_SIMILARITY,
                matched_segments=matched_segments,
                confidence=0.8,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Error calculating lexical similarity: {e}")
            return SimilarityResult(0.0, SimilarityType.LEXICAL_SIMILARITY, [], 0.0, datetime.now())

    def _calculate_structural_similarity(self, content1: str, content2: str) -> SimilarityResult:
        """Calculate structural similarity"""
        try:
            # Extract structural features
            features1 = self._extract_structural_features(content1)
            features2 = self._extract_structural_features(content2)
            
            # Calculate similarity based on structural features
            similarity_score = self._calculate_lexical_similarity_score(features1, features2)
            
            return SimilarityResult(
                similarity_score=similarity_score,
                similarity_type=SimilarityType.STRUCTURAL_SIMILARITY,
                matched_segments=[],
                confidence=0.7,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Error calculating structural similarity: {e}")
            return SimilarityResult(0.0, SimilarityType.STRUCTURAL_SIMILARITY, [], 0.0, datetime.now())

    def _calculate_stylistic_similarity(self, content1: str, content2: str) -> SimilarityResult:
        """Calculate stylistic similarity"""
        try:
            # Extract stylistic features
            features1 = self._extract_stylistic_features(content1)
            features2 = self._extract_stylistic_features(content2)
            
            # Calculate similarity based on stylistic features
            similarity_score = self._calculate_lexical_similarity_score(features1, features2)
            
            return SimilarityResult(
                similarity_score=similarity_score,
                similarity_type=SimilarityType.STYLISTIC_SIMILARITY,
                matched_segments=[],
                confidence=0.7,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Error calculating stylistic similarity: {e}")
            return SimilarityResult(0.0, SimilarityType.STYLISTIC_SIMILARITY, [], 0.0, datetime.now())

    def _calculate_content_overlap(self, content1: str, content2: str) -> SimilarityResult:
        """Calculate content overlap using sequence matching"""
        try:
            # Use difflib to find common sequences
            matcher = difflib.SequenceMatcher(None, content1.lower(), content2.lower())
            similarity_score = matcher.ratio()
            
            # Find matching blocks
            matched_segments = []
            for block in matcher.get_matching_blocks():
                if block.size > 10:  # Only consider significant matches
                    matched_segments.append({
                        "content1_start": block.a,
                        "content1_end": block.a + block.size,
                        "content2_start": block.b,
                        "content2_end": block.b + block.size,
                        "matched_text": content1[block.a:block.a + block.size],
                        "size": block.size
                    })
            
            return SimilarityResult(
                similarity_score=similarity_score,
                similarity_type=SimilarityType.CONTENT_OVERLAP,
                matched_segments=matched_segments,
                confidence=0.8,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Error calculating content overlap: {e}")
            return SimilarityResult(0.0, SimilarityType.CONTENT_OVERLAP, [], 0.0, datetime.now())

    def _detect_plagiarism_against_content(self, content: str, reference_content: str) -> PlagiarismDetection:
        """Detect plagiarism against specific reference content"""
        try:
            # Calculate all types of similarity
            similarity_results = self.calculate_similarity(content, reference_content)
            
            # Get the highest similarity score
            max_similarity = 0.0
            for result in similarity_results.values():
                max_similarity = max(max_similarity, result.similarity_score)
            
            # Determine plagiarism level
            if max_similarity >= self.similarity_thresholds["plagiarism_threshold"]:
                plagiarism_level = PlagiarismLevel.PLAGIARIZED
            elif max_similarity >= self.similarity_thresholds["suspicious_threshold"]:
                plagiarism_level = PlagiarismLevel.SUSPICIOUS
            elif max_similarity >= 0.6:
                plagiarism_level = PlagiarismLevel.HEAVILY_PARAPHRASED
            elif max_similarity >= 0.4:
                plagiarism_level = PlagiarismLevel.PARAPHRASED
            else:
                plagiarism_level = PlagiarismLevel.ORIGINAL
            
            # Calculate originality score
            originality_score = 1.0 - max_similarity
            
            # Generate recommendations
            recommendations = self._generate_plagiarism_recommendations(plagiarism_level, max_similarity)
            
            # Find suspicious patterns
            suspicious_patterns = self._identify_suspicious_patterns(content, reference_content)
            
            return PlagiarismDetection(
                plagiarism_level=plagiarism_level,
                similarity_score=max_similarity,
                matched_content=[],  # Would be populated with actual matches
                originality_score=originality_score,
                suspicious_patterns=suspicious_patterns,
                recommendations=recommendations,
                confidence=0.9 if max_similarity > 0.8 else 0.7,
                detection_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error detecting plagiarism against content: {e}")
            return PlagiarismDetection(
                plagiarism_level=PlagiarismLevel.ORIGINAL,
                similarity_score=0.0,
                matched_content=[],
                originality_score=1.0,
                suspicious_patterns=[],
                recommendations=["Analysis failed"],
                confidence=0.0,
                detection_timestamp=datetime.now()
            )

    def _detect_plagiarism_against_database(self, content: str, content_id: str = None) -> PlagiarismDetection:
        """Detect plagiarism against all content in database"""
        try:
            # Find similar content
            similar_content = self.find_similar_content(content, threshold=0.3, max_results=5)
            
            if not similar_content:
                return PlagiarismDetection(
                    plagiarism_level=PlagiarismLevel.ORIGINAL,
                    similarity_score=0.0,
                    matched_content=[],
                    originality_score=1.0,
                    suspicious_patterns=[],
                    recommendations=["Content appears to be original"],
                    confidence=0.8,
                    detection_timestamp=datetime.now()
                )
            
            # Get the highest similarity
            max_similarity = max(item["similarity_score"] for item in similar_content)
            
            # Determine plagiarism level
            if max_similarity >= self.similarity_thresholds["plagiarism_threshold"]:
                plagiarism_level = PlagiarismLevel.PLAGIARIZED
            elif max_similarity >= self.similarity_thresholds["suspicious_threshold"]:
                plagiarism_level = PlagiarismLevel.SUSPICIOUS
            elif max_similarity >= 0.6:
                plagiarism_level = PlagiarismLevel.HEAVILY_PARAPHRASED
            elif max_similarity >= 0.4:
                plagiarism_level = PlagiarismLevel.PARAPHRASED
            else:
                plagiarism_level = PlagiarismLevel.ORIGINAL
            
            # Calculate originality score
            originality_score = 1.0 - max_similarity
            
            # Generate recommendations
            recommendations = self._generate_plagiarism_recommendations(plagiarism_level, max_similarity)
            
            return PlagiarismDetection(
                plagiarism_level=plagiarism_level,
                similarity_score=max_similarity,
                matched_content=similar_content,
                originality_score=originality_score,
                suspicious_patterns=[],
                recommendations=recommendations,
                confidence=0.9 if max_similarity > 0.8 else 0.7,
                detection_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error detecting plagiarism against database: {e}")
            return PlagiarismDetection(
                plagiarism_level=PlagiarismLevel.ORIGINAL,
                similarity_score=0.0,
                matched_content=[],
                originality_score=1.0,
                suspicious_patterns=[],
                recommendations=["Analysis failed"],
                confidence=0.0,
                detection_timestamp=datetime.now()
            )

    def _find_similar_segments(self, content1: str, content2: str, method: str) -> List[Dict[str, Any]]:
        """Find similar segments between two content pieces"""
        # Simplified implementation - would be more sophisticated in practice
        segments = []
        
        # Split into sentences
        sentences1 = content1.split('.')
        sentences2 = content2.split('.')
        
        # Find similar sentences
        for i, sent1 in enumerate(sentences1):
            for j, sent2 in enumerate(sentences2):
                if len(sent1) > 20 and len(sent2) > 20:  # Only consider substantial sentences
                    similarity = difflib.SequenceMatcher(None, sent1.lower(), sent2.lower()).ratio()
                    if similarity > 0.7:
                        segments.append({
                            "content1_sentence": i,
                            "content2_sentence": j,
                            "similarity": similarity,
                            "content1_text": sent1.strip(),
                            "content2_text": sent2.strip()
                        })
        
        return segments

    def _calculate_lexical_similarity_score(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity score between two feature dictionaries"""
        try:
            # Get common features
            common_features = set(features1.keys()) & set(features2.keys())
            
            if not common_features:
                return 0.0
            
            similarities = []
            for feature in common_features:
                val1 = features1[feature]
                val2 = features2[feature]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if val1 == 0 and val2 == 0:
                        similarity = 1.0
                    elif val1 == 0 or val2 == 0:
                        similarity = 0.0
                    else:
                        similarity = 1.0 - abs(val1 - val2) / max(abs(val1), abs(val2))
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating lexical similarity score: {e}")
            return 0.0

    def _calculate_ngram_similarity(self, ngrams1: List[str], ngrams2: List[str]) -> float:
        """Calculate similarity based on n-gram overlap"""
        try:
            if not ngrams1 or not ngrams2:
                return 0.0
            
            set1 = set(ngrams1)
            set2 = set(ngrams2)
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating n-gram similarity: {e}")
            return 0.0

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)"""
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

    def _generate_plagiarism_recommendations(self, plagiarism_level: PlagiarismLevel, similarity_score: float) -> List[str]:
        """Generate recommendations based on plagiarism detection"""
        recommendations = []
        
        if plagiarism_level == PlagiarismLevel.PLAGIARIZED:
            recommendations.extend([
                "Content shows high similarity to existing content",
                "Consider rewriting with original ideas and phrasing",
                "Ensure proper attribution if using reference material",
                "Review content for originality and uniqueness"
            ])
        elif plagiarism_level == PlagiarismLevel.SUSPICIOUS:
            recommendations.extend([
                "Content shows concerning similarity to existing content",
                "Review and revise similar sections",
                "Add more original analysis and insights",
                "Consider different approaches to the topic"
            ])
        elif plagiarism_level == PlagiarismLevel.HEAVILY_PARAPHRASED:
            recommendations.extend([
                "Content appears to be heavily paraphrased",
                "Add more original content and unique perspectives",
                "Ensure the content adds value beyond the source material"
            ])
        elif plagiarism_level == PlagiarismLevel.PARAPHRASED:
            recommendations.extend([
                "Content shows some similarity to existing content",
                "Consider adding more original elements",
                "Ensure unique value proposition"
            ])
        else:
            recommendations.append("Content appears to be original")
        
        return recommendations

    def _identify_suspicious_patterns(self, content: str, reference_content: str) -> List[str]:
        """Identify suspicious patterns that might indicate plagiarism"""
        patterns = []
        
        # Check for identical phrases
        words1 = content.lower().split()
        words2 = reference_content.lower().split()
        
        # Find common phrases of length 5 or more
        for i in range(len(words1) - 4):
            phrase1 = ' '.join(words1[i:i+5])
            if phrase1 in reference_content.lower():
                patterns.append(f"Identical phrase found: '{phrase1}'")
        
        # Check for similar sentence structures
        sentences1 = content.split('.')
        sentences2 = reference_content.split('.')
        
        for sent1 in sentences1:
            for sent2 in sentences2:
                if len(sent1) > 20 and len(sent2) > 20:
                    similarity = difflib.SequenceMatcher(None, sent1.lower(), sent2.lower()).ratio()
                    if similarity > 0.8:
                        patterns.append(f"Very similar sentence structure detected")
                        break
        
        return patterns

    def get_similarity_statistics(self) -> Dict[str, Any]:
        """Get statistics about similarity analysis"""
        return {
            "total_fingerprints": len(self.content_fingerprints),
            "similarity_cache_size": len(self.similarity_cache),
            "sentence_transformer_available": self.sentence_model is not None,
            "spacy_available": self.nlp is not None,
            "similarity_thresholds": self.similarity_thresholds
        }

    def clear_cache(self):
        """Clear similarity cache"""
        self.similarity_cache.clear()
        logger.info("Similarity cache cleared")


# Global similarity engine instance
similarity_engine = ContentSimilarityEngine()

# Convenience functions
def create_content_fingerprint(content: str, content_id: str = None) -> ContentFingerprint:
    """Create content fingerprint"""
    return similarity_engine.create_content_fingerprint(content, content_id)

def calculate_similarity(content1: str, content2: str, similarity_types: List[SimilarityType] = None) -> Dict[SimilarityType, SimilarityResult]:
    """Calculate similarity between content pieces"""
    return similarity_engine.calculate_similarity(content1, content2, similarity_types)

def detect_plagiarism(content: str, reference_content: str = None, content_id: str = None) -> PlagiarismDetection:
    """Detect plagiarism in content"""
    return similarity_engine.detect_plagiarism(content, reference_content, content_id)

def find_similar_content(content: str, threshold: float = 0.7, max_results: int = 10) -> List[Dict[str, Any]]:
    """Find similar content in database"""
    return similarity_engine.find_similar_content(content, threshold, max_results)

def calculate_originality_score(content: str) -> float:
    """Calculate originality score for content"""
    return similarity_engine.calculate_originality_score(content)



























