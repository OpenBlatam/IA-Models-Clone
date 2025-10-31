"""
Content Redundancy Detector
==========================

Advanced content redundancy detection and similarity analysis for document management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import json
import numpy as np
from collections import defaultdict, Counter
import hashlib
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class SimilarityType(str, Enum):
    """Similarity type."""
    EXACT_MATCH = "exact_match"
    NEAR_DUPLICATE = "near_duplicate"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    STRUCTURAL_SIMILARITY = "structural_similarity"
    CONTENT_OVERLAP = "content_overlap"


class RedundancyLevel(str, Enum):
    """Redundancy level."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SimilarityResult:
    """Similarity analysis result."""
    result_id: str
    document_id_1: str
    document_id_2: str
    similarity_type: SimilarityType
    similarity_score: float
    confidence: float
    matched_sections: List[Dict[str, Any]] = field(default_factory=list)
    differences: List[Dict[str, Any]] = field(default_factory=list)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RedundancyReport:
    """Redundancy analysis report."""
    report_id: str
    document_id: str
    redundancy_level: RedundancyLevel
    similar_documents: List[SimilarityResult] = field(default_factory=list)
    duplicate_sections: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    analysis_summary: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ContentFingerprint:
    """Content fingerprint for fast comparison."""
    fingerprint_id: str
    document_id: str
    content_hash: str
    semantic_hash: str
    structural_hash: str
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: str
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class ContentRedundancyDetector:
    """Advanced content redundancy detector."""
    
    def __init__(self):
        self.similarity_results: Dict[str, SimilarityResult] = {}
        self.redundancy_reports: Dict[str, RedundancyReport] = {}
        self.content_fingerprints: Dict[str, ContentFingerprint] = {}
        self.similarity_cache: Dict[str, float] = {}
        
        # Initialize text processing tools
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Language-specific stop words
        self.stop_words = {
            'en': set(stopwords.words('english')),
            'es': set(stopwords.words('spanish')),
            'fr': set(stopwords.words('french')),
            'de': set(stopwords.words('german')),
            'it': set(stopwords.words('italian')),
            'pt': set(stopwords.words('portuguese')),
            'ru': set(stopwords.words('russian')),
            'zh': set()  # Chinese doesn't use stop words in the same way
        }
    
    async def create_content_fingerprint(
        self,
        document_id: str,
        content: str,
        language: str = "en"
    ) -> ContentFingerprint:
        """Create content fingerprint for fast comparison."""
        
        # Calculate various hashes
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        semantic_hash = self._calculate_semantic_hash(content, language)
        structural_hash = self._calculate_structural_hash(content)
        
        # Extract content statistics
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # Extract topics and entities (simplified)
        topics = self._extract_topics(content, language)
        entities = self._extract_entities(content, language)
        
        fingerprint = ContentFingerprint(
            fingerprint_id=str(uuid4()),
            document_id=document_id,
            content_hash=content_hash,
            semantic_hash=semantic_hash,
            structural_hash=structural_hash,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            language=language,
            topics=topics,
            entities=entities
        )
        
        self.content_fingerprints[document_id] = fingerprint
        
        logger.info(f"Created content fingerprint for document {document_id}")
        
        return fingerprint
    
    def _calculate_semantic_hash(self, content: str, language: str) -> str:
        """Calculate semantic hash of content."""
        
        # Clean and tokenize content
        cleaned_content = self._clean_text(content, language)
        tokens = self._tokenize_text(cleaned_content, language)
        
        # Create semantic representation
        semantic_tokens = [token for token in tokens if token not in self.stop_words.get(language, set())]
        semantic_text = ' '.join(semantic_tokens)
        
        return hashlib.sha256(semantic_text.encode('utf-8')).hexdigest()
    
    def _calculate_structural_hash(self, content: str) -> str:
        """Calculate structural hash of content."""
        
        # Extract structural elements
        lines = content.split('\n')
        paragraphs = content.split('\n\n')
        
        # Create structural representation
        structure = {
            'line_count': len(lines),
            'paragraph_count': len([p for p in paragraphs if p.strip()]),
            'avg_line_length': np.mean([len(line) for line in lines if line.strip()]),
            'has_headers': any(line.strip().endswith(':') for line in lines),
            'has_lists': any(line.strip().startswith(('-', '*', '1.', '2.')) for line in lines),
            'has_tables': '|' in content and content.count('|') > 5
        }
        
        structure_text = json.dumps(structure, sort_keys=True)
        return hashlib.sha256(structure_text.encode('utf-8')).hexdigest()
    
    def _clean_text(self, text: str, language: str) -> str:
        """Clean text for processing."""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text.strip()
    
    def _tokenize_text(self, text: str, language: str) -> List[str]:
        """Tokenize text based on language."""
        
        if language == 'zh':
            # Chinese tokenization using jieba
            return list(jieba.cut(text))
        else:
            # Use NLTK for other languages
            return word_tokenize(text)
    
    def _extract_topics(self, content: str, language: str) -> List[str]:
        """Extract topics from content (simplified)."""
        
        # Simple topic extraction based on frequent words
        cleaned_content = self._clean_text(content, language)
        tokens = self._tokenize_text(cleaned_content, language)
        
        # Filter out stop words and short words
        filtered_tokens = [
            token for token in tokens
            if len(token) > 3 and token not in self.stop_words.get(language, set())
        ]
        
        # Count word frequencies
        word_freq = Counter(filtered_tokens)
        
        # Return top 5 most frequent words as topics
        return [word for word, _ in word_freq.most_common(5)]
    
    def _extract_entities(self, content: str, language: str) -> List[str]:
        """Extract entities from content (simplified)."""
        
        # Simple entity extraction based on capitalization patterns
        entities = []
        
        # Find capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', content)
        entities.extend(capitalized_words)
        
        # Find email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        entities.extend(emails)
        
        # Find URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        entities.extend(urls)
        
        return list(set(entities))  # Remove duplicates
    
    async def calculate_similarity(
        self,
        document_id_1: str,
        document_id_2: str,
        content_1: str,
        content_2: str,
        similarity_types: List[SimilarityType] = None
    ) -> SimilarityResult:
        """Calculate similarity between two documents."""
        
        if similarity_types is None:
            similarity_types = [SimilarityType.EXACT_MATCH, SimilarityType.SEMANTIC_SIMILARITY]
        
        # Check cache first
        cache_key = f"{document_id_1}_{document_id_2}_{hash(tuple(similarity_types))}"
        if cache_key in self.similarity_cache:
            cached_score = self.similarity_cache[cache_key]
        else:
            cached_score = None
        
        # Calculate similarities
        similarities = {}
        matched_sections = []
        differences = []
        
        for sim_type in similarity_types:
            if sim_type == SimilarityType.EXACT_MATCH:
                score = self._calculate_exact_match_similarity(content_1, content_2)
            elif sim_type == SimilarityType.NEAR_DUPLICATE:
                score = self._calculate_near_duplicate_similarity(content_1, content_2)
            elif sim_type == SimilarityType.SEMANTIC_SIMILARITY:
                score = self._calculate_semantic_similarity(content_1, content_2)
            elif sim_type == SimilarityType.STRUCTURAL_SIMILARITY:
                score = self._calculate_structural_similarity(content_1, content_2)
            elif sim_type == SimilarityType.CONTENT_OVERLAP:
                score = self._calculate_content_overlap(content_1, content_2)
            else:
                score = 0.0
            
            similarities[sim_type.value] = score
        
        # Determine overall similarity
        overall_score = max(similarities.values()) if similarities else 0.0
        similarity_type = max(similarities.items(), key=lambda x: x[1])[0]
        
        # Calculate confidence
        confidence = self._calculate_confidence(similarities, content_1, content_2)
        
        # Find matched sections and differences
        if overall_score > 0.5:
            matched_sections = self._find_matched_sections(content_1, content_2)
            differences = self._find_differences(content_1, content_2)
        
        result = SimilarityResult(
            result_id=str(uuid4()),
            document_id_1=document_id_1,
            document_id_2=document_id_2,
            similarity_type=SimilarityType(similarity_type),
            similarity_score=overall_score,
            confidence=confidence,
            matched_sections=matched_sections,
            differences=differences,
            analysis_metadata={
                "similarities": similarities,
                "content_length_1": len(content_1),
                "content_length_2": len(content_2),
                "analysis_timestamp": datetime.now().isoformat()
            }
        )
        
        self.similarity_results[result.result_id] = result
        self.similarity_cache[cache_key] = overall_score
        
        return result
    
    def _calculate_exact_match_similarity(self, content_1: str, content_2: str) -> float:
        """Calculate exact match similarity."""
        
        if content_1 == content_2:
            return 1.0
        
        # Normalize whitespace
        norm_content_1 = re.sub(r'\s+', ' ', content_1.strip())
        norm_content_2 = re.sub(r'\s+', ' ', content_2.strip())
        
        if norm_content_1 == norm_content_2:
            return 0.95
        
        return 0.0
    
    def _calculate_near_duplicate_similarity(self, content_1: str, content_2: str) -> float:
        """Calculate near duplicate similarity."""
        
        # Use SequenceMatcher for character-level similarity
        matcher = SequenceMatcher(None, content_1, content_2)
        return matcher.ratio()
    
    def _calculate_semantic_similarity(self, content_1: str, content_2: str) -> float:
        """Calculate semantic similarity using TF-IDF."""
        
        try:
            # Create TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([content_1, content_2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
        except Exception as e:
            logger.warning(f"Failed to calculate semantic similarity: {str(e)}")
            return 0.0
    
    def _calculate_structural_similarity(self, content_1: str, content_2: str) -> float:
        """Calculate structural similarity."""
        
        # Extract structural features
        features_1 = self._extract_structural_features(content_1)
        features_2 = self._extract_structural_features(content_2)
        
        # Calculate similarity based on features
        similarities = []
        
        for feature in features_1:
            if feature in features_2:
                val_1 = features_1[feature]
                val_2 = features_2[feature]
                
                if isinstance(val_1, (int, float)) and isinstance(val_2, (int, float)):
                    # Numerical similarity
                    if val_1 == 0 and val_2 == 0:
                        sim = 1.0
                    else:
                        sim = 1.0 - abs(val_1 - val_2) / max(val_1, val_2)
                else:
                    # Boolean similarity
                    sim = 1.0 if val_1 == val_2 else 0.0
                
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _extract_structural_features(self, content: str) -> Dict[str, Any]:
        """Extract structural features from content."""
        
        lines = content.split('\n')
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        
        return {
            'line_count': len(lines),
            'paragraph_count': len(paragraphs),
            'avg_line_length': np.mean([len(line) for line in lines if line.strip()]) if lines else 0,
            'has_headers': any(line.strip().endswith(':') for line in lines),
            'has_lists': any(line.strip().startswith(('-', '*', '1.', '2.')) for line in lines),
            'has_tables': '|' in content and content.count('|') > 5,
            'has_numbers': bool(re.search(r'\d+', content)),
            'has_emails': bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)),
            'has_urls': bool(re.search(r'http[s]?://', content))
        }
    
    def _calculate_content_overlap(self, content_1: str, content_2: str) -> float:
        """Calculate content overlap similarity."""
        
        # Split into sentences
        sentences_1 = re.split(r'[.!?]+', content_1)
        sentences_2 = re.split(r'[.!?]+', content_2)
        
        # Clean sentences
        sentences_1 = [s.strip() for s in sentences_1 if s.strip()]
        sentences_2 = [s.strip() for s in sentences_2 if s.strip()]
        
        if not sentences_1 or not sentences_2:
            return 0.0
        
        # Find overlapping sentences
        overlap_count = 0
        for sent_1 in sentences_1:
            for sent_2 in sentences_2:
                if self._sentences_similar(sent_1, sent_2):
                    overlap_count += 1
                    break
        
        # Calculate overlap ratio
        total_sentences = len(sentences_1) + len(sentences_2)
        overlap_ratio = (2 * overlap_count) / total_sentences if total_sentences > 0 else 0.0
        
        return overlap_ratio
    
    def _sentences_similar(self, sent_1: str, sent_2: str, threshold: float = 0.8) -> bool:
        """Check if two sentences are similar."""
        
        # Normalize sentences
        norm_sent_1 = re.sub(r'\s+', ' ', sent_1.lower().strip())
        norm_sent_2 = re.sub(r'\s+', ' ', sent_2.lower().strip())
        
        # Use SequenceMatcher for similarity
        matcher = SequenceMatcher(None, norm_sent_1, norm_sent_2)
        return matcher.ratio() >= threshold
    
    def _calculate_confidence(
        self,
        similarities: Dict[str, float],
        content_1: str,
        content_2: str
    ) -> float:
        """Calculate confidence in similarity analysis."""
        
        # Base confidence on similarity consistency
        if not similarities:
            return 0.0
        
        similarity_values = list(similarities.values())
        mean_similarity = np.mean(similarity_values)
        std_similarity = np.std(similarity_values)
        
        # Higher confidence for consistent similarities
        consistency_factor = 1.0 - min(std_similarity, 0.5)
        
        # Higher confidence for longer content
        length_factor = min(len(content_1) + len(content_2), 10000) / 10000
        
        # Higher confidence for higher similarities
        similarity_factor = mean_similarity
        
        confidence = (consistency_factor * 0.4 + length_factor * 0.3 + similarity_factor * 0.3)
        
        return min(confidence, 1.0)
    
    def _find_matched_sections(self, content_1: str, content_2: str) -> List[Dict[str, Any]]:
        """Find matched sections between documents."""
        
        matched_sections = []
        
        # Split into paragraphs
        paragraphs_1 = [p.strip() for p in content_1.split('\n\n') if p.strip()]
        paragraphs_2 = [p.strip() for p in content_2.split('\n\n') if p.strip()]
        
        for i, para_1 in enumerate(paragraphs_1):
            for j, para_2 in enumerate(paragraphs_2):
                if self._sentences_similar(para_1, para_2, threshold=0.7):
                    matched_sections.append({
                        "section_1": {
                            "index": i,
                            "content": para_1[:200] + "..." if len(para_1) > 200 else para_1
                        },
                        "section_2": {
                            "index": j,
                            "content": para_2[:200] + "..." if len(para_2) > 200 else para_2
                        },
                        "similarity": SequenceMatcher(None, para_1, para_2).ratio()
                    })
        
        return matched_sections
    
    def _find_differences(self, content_1: str, content_2: str) -> List[Dict[str, Any]]:
        """Find differences between documents."""
        
        differences = []
        
        # Use difflib to find differences
        differ = SequenceMatcher(None, content_1, content_2)
        
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag in ['replace', 'delete', 'insert']:
                differences.append({
                    "type": tag,
                    "content_1": content_1[i1:i2] if tag in ['replace', 'delete'] else "",
                    "content_2": content_2[j1:j2] if tag in ['replace', 'insert'] else "",
                    "position_1": (i1, i2),
                    "position_2": (j1, j2)
                })
        
        return differences
    
    async def analyze_redundancy(
        self,
        document_id: str,
        content: str,
        comparison_documents: List[Tuple[str, str]] = None,
        threshold: float = 0.7
    ) -> RedundancyReport:
        """Analyze redundancy for a document."""
        
        similar_documents = []
        duplicate_sections = []
        
        # Compare with provided documents
        if comparison_documents:
            for comp_doc_id, comp_content in comparison_documents:
                similarity_result = await self.calculate_similarity(
                    document_id, comp_doc_id, content, comp_content
                )
                
                if similarity_result.similarity_score >= threshold:
                    similar_documents.append(similarity_result)
        
        # Compare with existing fingerprints
        for doc_id, fingerprint in self.content_fingerprints.items():
            if doc_id != document_id:
                # Quick hash comparison first
                if fingerprint.content_hash == hashlib.md5(content.encode('utf-8')).hexdigest():
                    # Exact match
                    similarity_result = SimilarityResult(
                        result_id=str(uuid4()),
                        document_id_1=document_id,
                        document_id_2=doc_id,
                        similarity_type=SimilarityType.EXACT_MATCH,
                        similarity_score=1.0,
                        confidence=1.0
                    )
                    similar_documents.append(similarity_result)
        
        # Determine redundancy level
        redundancy_level = self._determine_redundancy_level(similar_documents)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(redundancy_level, similar_documents)
        
        # Create analysis summary
        analysis_summary = {
            "total_comparisons": len(comparison_documents) if comparison_documents else len(self.content_fingerprints) - 1,
            "similar_documents_count": len(similar_documents),
            "highest_similarity": max([s.similarity_score for s in similar_documents]) if similar_documents else 0.0,
            "average_similarity": np.mean([s.similarity_score for s in similar_documents]) if similar_documents else 0.0,
            "threshold_used": threshold
        }
        
        report = RedundancyReport(
            report_id=str(uuid4()),
            document_id=document_id,
            redundancy_level=redundancy_level,
            similar_documents=similar_documents,
            duplicate_sections=duplicate_sections,
            recommendations=recommendations,
            analysis_summary=analysis_summary
        )
        
        self.redundancy_reports[report.report_id] = report
        
        logger.info(f"Created redundancy report for document {document_id}: {redundancy_level.value}")
        
        return report
    
    def _determine_redundancy_level(self, similar_documents: List[SimilarityResult]) -> RedundancyLevel:
        """Determine redundancy level based on similar documents."""
        
        if not similar_documents:
            return RedundancyLevel.NONE
        
        # Count documents by similarity type
        exact_matches = len([s for s in similar_documents if s.similarity_type == SimilarityType.EXACT_MATCH])
        near_duplicates = len([s for s in similar_documents if s.similarity_score >= 0.9])
        high_similarity = len([s for s in similar_documents if s.similarity_score >= 0.8])
        
        if exact_matches > 0:
            return RedundancyLevel.CRITICAL
        elif near_duplicates > 2:
            return RedundancyLevel.HIGH
        elif high_similarity > 3:
            return RedundancyLevel.MEDIUM
        elif len(similar_documents) > 1:
            return RedundancyLevel.LOW
        else:
            return RedundancyLevel.NONE
    
    def _generate_recommendations(
        self,
        redundancy_level: RedundancyLevel,
        similar_documents: List[SimilarityResult]
    ) -> List[str]:
        """Generate recommendations based on redundancy analysis."""
        
        recommendations = []
        
        if redundancy_level == RedundancyLevel.CRITICAL:
            recommendations.extend([
                "Remove duplicate documents immediately",
                "Implement duplicate detection in document creation workflow",
                "Review document management processes"
            ])
        elif redundancy_level == RedundancyLevel.HIGH:
            recommendations.extend([
                "Review similar documents for consolidation",
                "Consider merging related content",
                "Implement content versioning system"
            ])
        elif redundancy_level == RedundancyLevel.MEDIUM:
            recommendations.extend([
                "Review documents for potential consolidation",
                "Consider creating content templates",
                "Implement content reuse strategies"
            ])
        elif redundancy_level == RedundancyLevel.LOW:
            recommendations.extend([
                "Monitor for increasing similarity",
                "Consider content organization improvements"
            ])
        
        # Add specific recommendations based on similarity types
        similarity_types = [s.similarity_type for s in similar_documents]
        
        if SimilarityType.EXACT_MATCH in similarity_types:
            recommendations.append("Investigate exact matches for potential duplicates")
        
        if SimilarityType.SEMANTIC_SIMILARITY in similarity_types:
            recommendations.append("Review semantically similar content for consolidation opportunities")
        
        if SimilarityType.STRUCTURAL_SIMILARITY in similarity_types:
            recommendations.append("Consider creating templates for structurally similar documents")
        
        return recommendations
    
    async def get_redundancy_analytics(self) -> Dict[str, Any]:
        """Get redundancy detection analytics."""
        
        total_reports = len(self.redundancy_reports)
        total_similarities = len(self.similarity_results)
        total_fingerprints = len(self.content_fingerprints)
        
        # Redundancy level distribution
        redundancy_distribution = Counter(
            report.redundancy_level.value for report in self.redundancy_reports.values()
        )
        
        # Similarity type distribution
        similarity_type_distribution = Counter(
            result.similarity_type.value for result in self.similarity_results.values()
        )
        
        # Average similarity scores
        similarity_scores = [result.similarity_score for result in self.similarity_results.values()]
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        # Cache statistics
        cache_hit_rate = len(self.similarity_cache) / max(total_similarities, 1)
        
        return {
            "total_reports": total_reports,
            "total_similarities": total_similarities,
            "total_fingerprints": total_fingerprints,
            "redundancy_distribution": dict(redundancy_distribution),
            "similarity_type_distribution": dict(similarity_type_distribution),
            "average_similarity_score": avg_similarity,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.similarity_cache)
        }
    
    async def export_similarity_analysis(
        self,
        format: str = "json"
    ) -> Union[str, bytes]:
        """Export similarity analysis results."""
        
        data = {
            "similarity_results": [
                {
                    "result_id": result.result_id,
                    "document_id_1": result.document_id_1,
                    "document_id_2": result.document_id_2,
                    "similarity_type": result.similarity_type.value,
                    "similarity_score": result.similarity_score,
                    "confidence": result.confidence,
                    "matched_sections": result.matched_sections,
                    "differences": result.differences,
                    "timestamp": result.timestamp.isoformat()
                }
                for result in self.similarity_results.values()
            ],
            "redundancy_reports": [
                {
                    "report_id": report.report_id,
                    "document_id": report.document_id,
                    "redundancy_level": report.redundancy_level.value,
                    "similar_documents_count": len(report.similar_documents),
                    "recommendations": report.recommendations,
                    "analysis_summary": report.analysis_summary,
                    "created_at": report.created_at.isoformat()
                }
                for report in self.redundancy_reports.values()
            ]
        }
        
        if format == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")



























