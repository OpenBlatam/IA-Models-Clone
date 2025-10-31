"""
Content Redundancy Detector for Export IA
=========================================

Advanced content redundancy detection system that identifies and removes
redundant content while preserving document quality and meaning.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import re
from collections import defaultdict, Counter
import hashlib
from difflib import SequenceMatcher
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import spacy
from transformers import AutoTokenizer, AutoModel, pipeline
import sentence_transformers
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class RedundancyType(Enum):
    """Types of content redundancy."""
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    SEMANTIC_SIMILAR = "semantic_similar"
    PARAPHRASE = "paraphrase"
    REPETITIVE_PHRASES = "repetitive_phrases"
    REDUNDANT_SECTIONS = "redundant_sections"
    SIMILAR_STRUCTURE = "similar_structure"

class RedundancyLevel(Enum):
    """Levels of redundancy detection."""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"

@dataclass
class RedundancyMatch:
    """Represents a redundancy match between content segments."""
    id: str
    redundancy_type: RedundancyType
    similarity_score: float
    content_segments: List[str]
    positions: List[Tuple[int, int]]  # (start, end) positions
    confidence: float
    suggested_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentSegment:
    """Represents a segment of content for analysis."""
    id: str
    text: str
    position: Tuple[int, int]
    segment_type: str  # sentence, paragraph, section
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RedundancyReport:
    """Comprehensive redundancy analysis report."""
    document_id: str
    total_segments: int
    redundant_segments: int
    redundancy_percentage: float
    redundancy_matches: List[RedundancyMatch]
    optimization_suggestions: List[str]
    quality_impact: float
    processing_time: float
    created_at: datetime = field(default_factory=datetime.now)

class ContentRedundancyDetector:
    """Advanced content redundancy detection system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize NLP models
        self.nlp = None
        self.sentence_transformer = None
        self.tokenizer = None
        self.similarity_model = None
        
        # Initialize analysis components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        # Redundancy detection settings
        self.similarity_thresholds = {
            RedundancyType.EXACT_DUPLICATE: 1.0,
            RedundancyType.NEAR_DUPLICATE: 0.95,
            RedundancyType.SEMANTIC_SIMILAR: 0.85,
            RedundancyType.PARAPHRASE: 0.80,
            RedundancyType.REPETITIVE_PHRASES: 0.90,
            RedundancyType.REDUNDANT_SECTIONS: 0.75,
            RedundancyType.SIMILAR_STRUCTURE: 0.70
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Content Redundancy Detector initialized")
    
    def _initialize_models(self):
        """Initialize NLP and ML models."""
        try:
            # Initialize spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using basic tokenization")
                self.nlp = None
            
            # Initialize sentence transformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize similarity model
            self.similarity_model = pipeline(
                "feature-extraction",
                model="sentence-transformers/all-MiniLM-L6-v2",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("NLP models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
    
    async def analyze_document_redundancy(
        self,
        content: str,
        document_id: str = None,
        redundancy_level: RedundancyLevel = RedundancyLevel.STANDARD
    ) -> RedundancyReport:
        """Analyze document for content redundancy."""
        
        if not document_id:
            document_id = str(uuid.uuid4())
        
        start_time = datetime.now()
        
        logger.info(f"Starting redundancy analysis for document: {document_id}")
        
        try:
            # Segment content
            segments = self._segment_content(content)
            logger.info(f"Content segmented into {len(segments)} segments")
            
            # Detect different types of redundancy
            redundancy_matches = []
            
            if redundancy_level in [RedundancyLevel.STANDARD, RedundancyLevel.ADVANCED, RedundancyLevel.ENTERPRISE]:
                # Exact and near duplicates
                exact_matches = await self._detect_exact_duplicates(segments)
                near_matches = await self._detect_near_duplicates(segments)
                redundancy_matches.extend(exact_matches + near_matches)
            
            if redundancy_level in [RedundancyLevel.ADVANCED, RedundancyLevel.ENTERPRISE]:
                # Semantic similarity
                semantic_matches = await self._detect_semantic_similarity(segments)
                paraphrase_matches = await self._detect_paraphrases(segments)
                redundancy_matches.extend(semantic_matches + paraphrase_matches)
            
            if redundancy_level == RedundancyLevel.ENTERPRISE:
                # Advanced redundancy detection
                phrase_matches = await self._detect_repetitive_phrases(segments)
                section_matches = await self._detect_redundant_sections(segments)
                structure_matches = await self._detect_similar_structures(segments)
                redundancy_matches.extend(phrase_matches + section_matches + structure_matches)
            
            # Remove overlapping matches
            redundancy_matches = self._remove_overlapping_matches(redundancy_matches)
            
            # Calculate metrics
            total_segments = len(segments)
            redundant_segments = len(set(
                match.content_segments[0] for match in redundancy_matches
            ))
            redundancy_percentage = (redundant_segments / total_segments) * 100 if total_segments > 0 else 0
            
            # Generate optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(redundancy_matches)
            
            # Calculate quality impact
            quality_impact = self._calculate_quality_impact(redundancy_matches, total_segments)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create report
            report = RedundancyReport(
                document_id=document_id,
                total_segments=total_segments,
                redundant_segments=redundant_segments,
                redundancy_percentage=redundancy_percentage,
                redundancy_matches=redundancy_matches,
                optimization_suggestions=optimization_suggestions,
                quality_impact=quality_impact,
                processing_time=processing_time
            )
            
            logger.info(f"Redundancy analysis completed: {redundancy_percentage:.1f}% redundant content found")
            
            return report
            
        except Exception as e:
            logger.error(f"Redundancy analysis failed: {e}")
            raise
    
    def _segment_content(self, content: str) -> List[ContentSegment]:
        """Segment content into analyzable segments."""
        segments = []
        
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        paragraph_id = 0
        
        for para_text in paragraphs:
            if para_text.strip():
                # Split paragraphs into sentences
                sentences = self._split_into_sentences(para_text)
                
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        segment = ContentSegment(
                            id=f"para_{paragraph_id}_sent_{i}",
                            text=sentence.strip(),
                            position=(content.find(sentence), content.find(sentence) + len(sentence)),
                            segment_type="sentence",
                            metadata={
                                "paragraph_id": paragraph_id,
                                "sentence_id": i,
                                "word_count": len(sentence.split()),
                                "char_count": len(sentence)
                            }
                        )
                        segments.append(segment)
                
                paragraph_id += 1
        
        return segments
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text for sent in doc.sents]
        else:
            # Basic sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    async def _detect_exact_duplicates(self, segments: List[ContentSegment]) -> List[RedundancyMatch]:
        """Detect exact duplicate content."""
        matches = []
        seen_hashes = {}
        
        for segment in segments:
            # Create hash of normalized text
            normalized_text = self._normalize_text(segment.text)
            text_hash = hashlib.md5(normalized_text.encode()).hexdigest()
            
            if text_hash in seen_hashes:
                # Found exact duplicate
                original_segment = seen_hashes[text_hash]
                match = RedundancyMatch(
                    id=str(uuid.uuid4()),
                    redundancy_type=RedundancyType.EXACT_DUPLICATE,
                    similarity_score=1.0,
                    content_segments=[original_segment.text, segment.text],
                    positions=[original_segment.position, segment.position],
                    confidence=1.0,
                    suggested_action="remove_duplicate",
                    metadata={
                        "hash": text_hash,
                        "original_id": original_segment.id,
                        "duplicate_id": segment.id
                    }
                )
                matches.append(match)
            else:
                seen_hashes[text_hash] = segment
        
        return matches
    
    async def _detect_near_duplicates(self, segments: List[ContentSegment]) -> List[RedundancyMatch]:
        """Detect near-duplicate content using similarity."""
        matches = []
        
        # Use TF-IDF for similarity
        texts = [segment.text for segment in segments]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                similarity = similarity_matrix[i][j]
                
                if similarity >= self.similarity_thresholds[RedundancyType.NEAR_DUPLICATE]:
                    match = RedundancyMatch(
                        id=str(uuid.uuid4()),
                        redundancy_type=RedundancyType.NEAR_DUPLICATE,
                        similarity_score=similarity,
                        content_segments=[segments[i].text, segments[j].text],
                        positions=[segments[i].position, segments[j].position],
                        confidence=similarity,
                        suggested_action="merge_or_remove",
                        metadata={
                            "segment_1_id": segments[i].id,
                            "segment_2_id": segments[j].id,
                            "method": "tfidf_cosine"
                        }
                    )
                    matches.append(match)
        
        return matches
    
    async def _detect_semantic_similarity(self, segments: List[ContentSegment]) -> List[RedundancyMatch]:
        """Detect semantically similar content."""
        matches = []
        
        if not self.sentence_transformer:
            return matches
        
        try:
            # Get sentence embeddings
            texts = [segment.text for segment in segments]
            embeddings = self.sentence_transformer.encode(texts)
            
            # Calculate semantic similarity
            similarity_matrix = cosine_similarity(embeddings)
            
            for i in range(len(segments)):
                for j in range(i + 1, len(segments)):
                    similarity = similarity_matrix[i][j]
                    
                    if similarity >= self.similarity_thresholds[RedundancyType.SEMANTIC_SIMILAR]:
                        match = RedundancyMatch(
                            id=str(uuid.uuid4()),
                            redundancy_type=RedundancyType.SEMANTIC_SIMILAR,
                            similarity_score=similarity,
                            content_segments=[segments[i].text, segments[j].text],
                            positions=[segments[i].position, segments[j].position],
                            confidence=similarity,
                            suggested_action="consolidate_content",
                            metadata={
                                "segment_1_id": segments[i].id,
                                "segment_2_id": segments[j].id,
                                "method": "sentence_transformer"
                            }
                        )
                        matches.append(match)
        
        except Exception as e:
            logger.error(f"Semantic similarity detection failed: {e}")
        
        return matches
    
    async def _detect_paraphrases(self, segments: List[ContentSegment]) -> List[RedundancyMatch]:
        """Detect paraphrased content."""
        matches = []
        
        # Use sequence matching for paraphrase detection
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                similarity = SequenceMatcher(None, segments[i].text, segments[j].text).ratio()
                
                # Check if it's a paraphrase (high similarity but not exact)
                if (self.similarity_thresholds[RedundancyType.PARAPHRASE] <= similarity < 
                    self.similarity_thresholds[RedundancyType.NEAR_DUPLICATE]):
                    
                    match = RedundancyMatch(
                        id=str(uuid.uuid4()),
                        redundancy_type=RedundancyType.PARAPHRASE,
                        similarity_score=similarity,
                        content_segments=[segments[i].text, segments[j].text],
                        positions=[segments[i].position, segments[j].position],
                        confidence=similarity,
                        suggested_action="choose_best_version",
                        metadata={
                            "segment_1_id": segments[i].id,
                            "segment_2_id": segments[j].id,
                            "method": "sequence_matcher"
                        }
                    )
                    matches.append(match)
        
        return matches
    
    async def _detect_repetitive_phrases(self, segments: List[ContentSegment]) -> List[RedundancyMatch]:
        """Detect repetitive phrases and expressions."""
        matches = []
        
        # Extract n-grams from all segments
        all_phrases = []
        phrase_positions = {}
        
        for segment in segments:
            phrases = self._extract_phrases(segment.text)
            for phrase in phrases:
                all_phrases.append(phrase)
                if phrase not in phrase_positions:
                    phrase_positions[phrase] = []
                phrase_positions[phrase].append((segment.id, segment.position))
        
        # Find repeated phrases
        phrase_counts = Counter(all_phrases)
        repeated_phrases = {phrase: count for phrase, count in phrase_counts.items() 
                          if count > 2 and len(phrase.split()) >= 3}
        
        for phrase, count in repeated_phrases.items():
            positions = phrase_positions[phrase]
            if len(positions) > 2:  # Only flag if used more than twice
                match = RedundancyMatch(
                    id=str(uuid.uuid4()),
                    redundancy_type=RedundancyType.REPETITIVE_PHRASES,
                    similarity_score=1.0,
                    content_segments=[phrase] * len(positions),
                    positions=[pos[1] for pos in positions],
                    confidence=min(1.0, count / 5.0),  # Higher confidence for more repetitions
                    suggested_action="reduce_repetition",
                    metadata={
                        "phrase": phrase,
                        "occurrence_count": count,
                        "segment_ids": [pos[0] for pos in positions]
                    }
                )
                matches.append(match)
        
        return matches
    
    async def _detect_redundant_sections(self, segments: List[ContentSegment]) -> List[RedundancyMatch]:
        """Detect redundant sections or paragraphs."""
        matches = []
        
        # Group segments by paragraph
        paragraphs = defaultdict(list)
        for segment in segments:
            para_id = segment.metadata.get("paragraph_id", 0)
            paragraphs[para_id].append(segment)
        
        # Compare paragraphs
        para_texts = []
        para_positions = []
        para_ids = []
        
        for para_id, para_segments in paragraphs.items():
            para_text = " ".join([seg.text for seg in para_segments])
            para_texts.append(para_text)
            para_positions.append((para_segments[0].position[0], para_segments[-1].position[1]))
            para_ids.append(para_id)
        
        # Calculate paragraph similarities
        if len(para_texts) > 1:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(para_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            for i in range(len(para_texts)):
                for j in range(i + 1, len(para_texts)):
                    similarity = similarity_matrix[i][j]
                    
                    if similarity >= self.similarity_thresholds[RedundancyType.REDUNDANT_SECTIONS]:
                        match = RedundancyMatch(
                            id=str(uuid.uuid4()),
                            redundancy_type=RedundancyType.REDUNDANT_SECTIONS,
                            similarity_score=similarity,
                            content_segments=[para_texts[i], para_texts[j]],
                            positions=[para_positions[i], para_positions[j]],
                            confidence=similarity,
                            suggested_action="merge_sections",
                            metadata={
                                "paragraph_1_id": para_ids[i],
                                "paragraph_2_id": para_ids[j],
                                "method": "paragraph_tfidf"
                            }
                        )
                        matches.append(match)
        
        return matches
    
    async def _detect_similar_structures(self, segments: List[ContentSegment]) -> List[RedundancyMatch]:
        """Detect similar structural patterns."""
        matches = []
        
        # Extract structural patterns
        patterns = []
        for segment in segments:
            pattern = self._extract_structural_pattern(segment.text)
            patterns.append((segment, pattern))
        
        # Group by similar patterns
        pattern_groups = defaultdict(list)
        for segment, pattern in patterns:
            pattern_groups[pattern].append(segment)
        
        # Find groups with similar structures
        for pattern, segments_with_pattern in pattern_groups.items():
            if len(segments_with_pattern) > 2:  # Only flag if pattern appears multiple times
                match = RedundancyMatch(
                    id=str(uuid.uuid4()),
                    redundancy_type=RedundancyType.SIMILAR_STRUCTURE,
                    similarity_score=1.0,
                    content_segments=[seg.text for seg in segments_with_pattern],
                    positions=[seg.position for seg in segments_with_pattern],
                    confidence=min(1.0, len(segments_with_pattern) / 5.0),
                    suggested_action="vary_structure",
                    metadata={
                        "pattern": pattern,
                        "occurrence_count": len(segments_with_pattern),
                        "segment_ids": [seg.id for seg in segments_with_pattern]
                    }
                )
                matches.append(match)
        
        return matches
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def _extract_phrases(self, text: str, min_length: int = 3, max_length: int = 6) -> List[str]:
        """Extract n-gram phrases from text."""
        words = text.lower().split()
        phrases = []
        
        for n in range(min_length, min(max_length + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                phrases.append(phrase)
        
        return phrases
    
    def _extract_structural_pattern(self, text: str) -> str:
        """Extract structural pattern from text."""
        # Simple pattern: sentence length + punctuation pattern
        sentences = self._split_into_sentences(text)
        pattern = []
        
        for sentence in sentences:
            word_count = len(sentence.split())
            if word_count <= 5:
                pattern.append("S")  # Short
            elif word_count <= 15:
                pattern.append("M")  # Medium
            else:
                pattern.append("L")  # Long
            
            # Add punctuation pattern
            if sentence.endswith('.'):
                pattern.append(".")
            elif sentence.endswith('!'):
                pattern.append("!")
            elif sentence.endswith('?'):
                pattern.append("?")
            else:
                pattern.append("_")
        
        return "".join(pattern)
    
    def _remove_overlapping_matches(self, matches: List[RedundancyMatch]) -> List[RedundancyMatch]:
        """Remove overlapping redundancy matches."""
        if not matches:
            return matches
        
        # Sort by confidence (keep higher confidence matches)
        matches.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered_matches = []
        used_positions = set()
        
        for match in matches:
            # Check if any position overlaps with already used positions
            overlap = False
            for pos in match.positions:
                for used_pos in used_positions:
                    if self._positions_overlap(pos, used_pos):
                        overlap = True
                        break
                if overlap:
                    break
            
            if not overlap:
                filtered_matches.append(match)
                used_positions.update(match.positions)
        
        return filtered_matches
    
    def _positions_overlap(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two position ranges overlap."""
        return not (pos1[1] <= pos2[0] or pos2[1] <= pos1[0])
    
    def _generate_optimization_suggestions(self, matches: List[RedundancyMatch]) -> List[str]:
        """Generate optimization suggestions based on redundancy matches."""
        suggestions = []
        
        if not matches:
            return ["No redundancy detected. Content is well-optimized."]
        
        # Count redundancy types
        type_counts = Counter(match.redundancy_type for match in matches)
        
        if type_counts[RedundancyType.EXACT_DUPLICATE] > 0:
            suggestions.append(f"Remove {type_counts[RedundancyType.EXACT_DUPLICATE]} exact duplicate(s)")
        
        if type_counts[RedundancyType.NEAR_DUPLICATE] > 0:
            suggestions.append(f"Consolidate {type_counts[RedundancyType.NEAR_DUPLICATE]} near-duplicate section(s)")
        
        if type_counts[RedundancyType.SEMANTIC_SIMILAR] > 0:
            suggestions.append(f"Merge {type_counts[RedundancyType.SEMANTIC_SIMILAR]} semantically similar section(s)")
        
        if type_counts[RedundancyType.REPETITIVE_PHRASES] > 0:
            suggestions.append(f"Reduce repetition of {type_counts[RedundancyType.REPETITIVE_PHRASES]} phrase(s)")
        
        if type_counts[RedundancyType.SIMILAR_STRUCTURE] > 0:
            suggestions.append(f"Vary structure in {type_counts[RedundancyType.SIMILAR_STRUCTURE]} section(s)")
        
        # General suggestions
        total_redundancy = len(matches)
        if total_redundancy > 10:
            suggestions.append("Consider comprehensive content restructuring")
        elif total_redundancy > 5:
            suggestions.append("Review content organization and flow")
        
        return suggestions
    
    def _calculate_quality_impact(self, matches: List[RedundancyMatch], total_segments: int) -> float:
        """Calculate the impact of redundancy on content quality."""
        if total_segments == 0:
            return 0.0
        
        # Weight different types of redundancy
        type_weights = {
            RedundancyType.EXACT_DUPLICATE: 1.0,
            RedundancyType.NEAR_DUPLICATE: 0.8,
            RedundancyType.SEMANTIC_SIMILAR: 0.6,
            RedundancyType.PARAPHRASE: 0.4,
            RedundancyType.REPETITIVE_PHRASES: 0.3,
            RedundancyType.REDUNDANT_SECTIONS: 0.7,
            RedundancyType.SIMILAR_STRUCTURE: 0.2
        }
        
        weighted_impact = 0.0
        for match in matches:
            weight = type_weights.get(match.redundancy_type, 0.5)
            weighted_impact += weight * match.confidence
        
        # Normalize to 0-1 scale
        max_possible_impact = total_segments * 1.0  # Assuming all segments could be redundant
        quality_impact = min(1.0, weighted_impact / max_possible_impact)
        
        return quality_impact
    
    async def optimize_content(
        self,
        content: str,
        redundancy_report: RedundancyReport,
        optimization_level: str = "standard"
    ) -> str:
        """Optimize content by removing redundancy."""
        
        logger.info(f"Optimizing content with {len(redundancy_report.redundancy_matches)} redundancy matches")
        
        # Sort matches by position (process from end to beginning to maintain positions)
        matches = sorted(redundancy_report.redundancy_matches, 
                        key=lambda x: x.positions[0][0], reverse=True)
        
        optimized_content = content
        
        for match in matches:
            if optimization_level == "aggressive" or match.confidence > 0.8:
                optimized_content = self._apply_optimization(optimized_content, match)
        
        return optimized_content
    
    def _apply_optimization(self, content: str, match: RedundancyMatch) -> str:
        """Apply optimization to content based on redundancy match."""
        
        if match.suggested_action == "remove_duplicate":
            # Remove the second occurrence
            if len(match.positions) >= 2:
                start, end = match.positions[1]
                content = content[:start] + content[end:]
        
        elif match.suggested_action == "merge_or_remove":
            # Keep the first occurrence, remove the second
            if len(match.positions) >= 2:
                start, end = match.positions[1]
                content = content[:start] + content[end:]
        
        elif match.suggested_action == "consolidate_content":
            # Merge similar content
            if len(match.positions) >= 2:
                start1, end1 = match.positions[0]
                start2, end2 = match.positions[1]
                
                # Keep the first segment, remove the second
                content = content[:start2] + content[end2:]
        
        elif match.suggested_action == "choose_best_version":
            # Keep the longer/more detailed version
            if len(match.content_segments) >= 2:
                if len(match.content_segments[0]) >= len(match.content_segments[1]):
                    # Keep first, remove second
                    start, end = match.positions[1]
                    content = content[:start] + content[end:]
                else:
                    # Keep second, remove first
                    start, end = match.positions[0]
                    content = content[:start] + content[end:]
        
        return content

# Global redundancy detector instance
_global_redundancy_detector: Optional[ContentRedundancyDetector] = None

def get_global_redundancy_detector() -> ContentRedundancyDetector:
    """Get the global redundancy detector instance."""
    global _global_redundancy_detector
    if _global_redundancy_detector is None:
        _global_redundancy_detector = ContentRedundancyDetector()
    return _global_redundancy_detector



























