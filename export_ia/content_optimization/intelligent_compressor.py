"""
Intelligent Content Compressor for Export IA
============================================

Advanced content compression and summarization system that intelligently
reduces content size while preserving meaning and quality.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
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
import heapq
from transformers import AutoTokenizer, AutoModel, pipeline, BartForConditionalGeneration, BartTokenizer
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

logger = logging.getLogger(__name__)

class CompressionLevel(Enum):
    """Levels of content compression."""
    LIGHT = "light"        # 10-20% reduction
    MODERATE = "moderate"  # 20-40% reduction
    AGGRESSIVE = "aggressive"  # 40-60% reduction
    MAXIMUM = "maximum"    # 60-80% reduction

class CompressionMethod(Enum):
    """Methods of content compression."""
    EXTRACTIVE = "extractive"      # Extract key sentences
    ABSTRACTIVE = "abstractive"    # Generate new summaries
    HYBRID = "hybrid"             # Combine both methods
    SEMANTIC = "semantic"         # Semantic compression
    STRUCTURAL = "structural"     # Structural optimization

@dataclass
class CompressionResult:
    """Result of content compression."""
    id: str
    original_content: str
    compressed_content: str
    compression_ratio: float
    compression_level: CompressionLevel
    compression_method: CompressionMethod
    quality_score: float
    key_points: List[str]
    removed_content: List[str]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ContentSegment:
    """Represents a segment of content for compression."""
    id: str
    text: str
    importance_score: float
    position: Tuple[int, int]
    segment_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class IntelligentCompressor:
    """Advanced intelligent content compression system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.nlp = None
        self.sentence_transformer = None
        self.summarization_model = None
        self.summarization_tokenizer = None
        self.tfidf_vectorizer = None
        
        # Compression settings
        self.compression_ratios = {
            CompressionLevel.LIGHT: 0.15,
            CompressionLevel.MODERATE: 0.30,
            CompressionLevel.AGGRESSIVE: 0.50,
            CompressionLevel.MAXIMUM: 0.70
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Intelligent Content Compressor initialized")
    
    def _initialize_models(self):
        """Initialize NLP and ML models for compression."""
        try:
            # Initialize spaCy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using basic tokenization")
                self.nlp = None
            
            # Initialize sentence transformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize summarization model
            self.summarization_model = BartForConditionalGeneration.from_pretrained(
                "facebook/bart-large-cnn"
            )
            self.summarization_tokenizer = BartTokenizer.from_pretrained(
                "facebook/bart-large-cnn"
            )
            
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logger.info("Compression models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize compression models: {e}")
    
    async def compress_content(
        self,
        content: str,
        compression_level: CompressionLevel = CompressionLevel.MODERATE,
        compression_method: CompressionMethod = CompressionMethod.HYBRID,
        preserve_structure: bool = True,
        target_ratio: Optional[float] = None
    ) -> CompressionResult:
        """Compress content intelligently while preserving quality."""
        
        start_time = datetime.now()
        result_id = str(uuid.uuid4())
        
        logger.info(f"Starting content compression: {compression_level.value} level, {compression_method.value} method")
        
        try:
            # Determine target compression ratio
            if target_ratio is None:
                target_ratio = self.compression_ratios[compression_level]
            
            # Segment content
            segments = self._segment_content(content)
            logger.info(f"Content segmented into {len(segments)} segments")
            
            # Calculate importance scores
            segments = await self._calculate_importance_scores(segments, content)
            
            # Apply compression based on method
            if compression_method == CompressionMethod.EXTRACTIVE:
                compressed_content, removed_content = await self._extractive_compression(
                    segments, target_ratio, preserve_structure
                )
            elif compression_method == CompressionMethod.ABSTRACTIVE:
                compressed_content, removed_content = await self._abstractive_compression(
                    content, target_ratio
                )
            elif compression_method == CompressionMethod.HYBRID:
                compressed_content, removed_content = await self._hybrid_compression(
                    segments, content, target_ratio, preserve_structure
                )
            elif compression_method == CompressionMethod.SEMANTIC:
                compressed_content, removed_content = await self._semantic_compression(
                    segments, target_ratio
                )
            elif compression_method == CompressionMethod.STRUCTURAL:
                compressed_content, removed_content = await self._structural_compression(
                    segments, target_ratio
                )
            else:
                raise ValueError(f"Unknown compression method: {compression_method}")
            
            # Calculate actual compression ratio
            original_length = len(content)
            compressed_length = len(compressed_content)
            actual_ratio = (original_length - compressed_length) / original_length
            
            # Extract key points
            key_points = self._extract_key_points(segments)
            
            # Calculate quality score
            quality_score = await self._calculate_quality_score(content, compressed_content)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = CompressionResult(
                id=result_id,
                original_content=content,
                compressed_content=compressed_content,
                compression_ratio=actual_ratio,
                compression_level=compression_level,
                compression_method=compression_method,
                quality_score=quality_score,
                key_points=key_points,
                removed_content=removed_content,
                processing_time=processing_time,
                metadata={
                    "original_length": original_length,
                    "compressed_length": compressed_length,
                    "target_ratio": target_ratio,
                    "preserve_structure": preserve_structure
                }
            )
            
            logger.info(f"Content compression completed: {actual_ratio:.1%} reduction, quality score: {quality_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Content compression failed: {e}")
            raise
    
    def _segment_content(self, content: str) -> List[ContentSegment]:
        """Segment content into analyzable segments."""
        segments = []
        
        # Split into paragraphs first
        paragraphs = content.split('\n\n')
        segment_id = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                # Split paragraph into sentences
                sentences = self._split_into_sentences(paragraph)
                
                for sent_idx, sentence in enumerate(sentences):
                    if sentence.strip():
                        segment = ContentSegment(
                            id=f"seg_{segment_id}",
                            text=sentence.strip(),
                            importance_score=0.0,
                            position=(content.find(sentence), content.find(sentence) + len(sentence)),
                            segment_type="sentence",
                            metadata={
                                "paragraph_id": para_idx,
                                "sentence_id": sent_idx,
                                "word_count": len(sentence.split()),
                                "char_count": len(sentence)
                            }
                        )
                        segments.append(segment)
                        segment_id += 1
        
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
    
    async def _calculate_importance_scores(self, segments: List[ContentSegment], full_content: str) -> List[ContentSegment]:
        """Calculate importance scores for content segments."""
        
        # Method 1: TF-IDF based scoring
        texts = [seg.text for seg in segments]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Method 2: Position-based scoring (first and last sentences are more important)
        position_scores = []
        for i, seg in enumerate(segments):
            para_id = seg.metadata.get("paragraph_id", 0)
            sent_id = seg.metadata.get("sentence_id", 0)
            
            # Higher score for first and last sentences of paragraphs
            if sent_id == 0 or sent_id == len([s for s in segments if s.metadata.get("paragraph_id") == para_id]) - 1:
                position_scores.append(1.0)
            else:
                position_scores.append(0.5)
        
        # Method 3: Length-based scoring (very short or very long sentences might be less important)
        length_scores = []
        for seg in segments:
            word_count = seg.metadata.get("word_count", 0)
            if 5 <= word_count <= 25:  # Optimal sentence length
                length_scores.append(1.0)
            elif word_count < 5:
                length_scores.append(0.3)
            else:
                length_scores.append(0.7)
        
        # Method 4: Keyword-based scoring
        keyword_scores = self._calculate_keyword_scores(segments, full_content)
        
        # Combine scores
        for i, seg in enumerate(segments):
            # Normalize TF-IDF scores
            normalized_tfidf = tfidf_scores[i] / (tfidf_scores.max() + 1e-8)
            
            # Weighted combination
            importance_score = (
                normalized_tfidf * 0.4 +
                position_scores[i] * 0.3 +
                length_scores[i] * 0.2 +
                keyword_scores[i] * 0.1
            )
            
            seg.importance_score = importance_score
        
        return segments
    
    def _calculate_keyword_scores(self, segments: List[ContentSegment], full_content: str) -> List[float]:
        """Calculate keyword-based importance scores."""
        # Extract keywords from full content
        if self.nlp:
            doc = self.nlp(full_content)
            keywords = [token.lemma_.lower() for token in doc 
                       if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop and len(token.text) > 3]
        else:
            # Basic keyword extraction
            words = re.findall(r'\b[a-zA-Z]{4,}\b', full_content.lower())
            keywords = [word for word in words if word not in ['this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'there', 'could', 'other', 'after', 'first', 'well', 'also', 'where', 'much', 'some', 'very', 'when', 'here', 'just', 'into', 'over', 'think', 'also', 'back', 'then', 'only', 'come', 'right', 'too', 'any', 'new', 'want', 'because', 'these', 'give', 'day', 'most', 'us', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall']]
        
        # Count keyword frequency
        keyword_counts = Counter(keywords)
        top_keywords = [word for word, count in keyword_counts.most_common(20)]
        
        # Score segments based on keyword presence
        scores = []
        for seg in segments:
            seg_words = re.findall(r'\b[a-zA-Z]{4,}\b', seg.text.lower())
            keyword_matches = sum(1 for word in seg_words if word in top_keywords)
            score = min(1.0, keyword_matches / 5.0)  # Normalize to 0-1
            scores.append(score)
        
        return scores
    
    async def _extractive_compression(
        self, 
        segments: List[ContentSegment], 
        target_ratio: float, 
        preserve_structure: bool
    ) -> Tuple[str, List[str]]:
        """Extract key sentences to create compressed content."""
        
        # Sort segments by importance score
        sorted_segments = sorted(segments, key=lambda x: x.importance_score, reverse=True)
        
        # Calculate how many segments to keep
        total_segments = len(segments)
        segments_to_keep = max(1, int(total_segments * (1 - target_ratio)))
        
        if preserve_structure:
            # Keep segments in original order but select most important ones
            selected_segments = []
            used_positions = set()
            
            # First, select the most important segments
            for seg in sorted_segments:
                if len(selected_segments) >= segments_to_keep:
                    break
                
                # Check if this segment is already selected
                if seg.id not in used_positions:
                    selected_segments.append(seg)
                    used_positions.add(seg.id)
            
            # Sort selected segments by original position
            selected_segments.sort(key=lambda x: x.position[0])
            
        else:
            # Just take the most important segments
            selected_segments = sorted_segments[:segments_to_keep]
            selected_segments.sort(key=lambda x: x.position[0])
        
        # Build compressed content
        compressed_content = ""
        removed_content = []
        
        for seg in segments:
            if seg in selected_segments:
                compressed_content += seg.text + " "
            else:
                removed_content.append(seg.text)
        
        return compressed_content.strip(), removed_content
    
    async def _abstractive_compression(self, content: str, target_ratio: float) -> Tuple[str, List[str]]:
        """Generate abstractive summary of content."""
        
        try:
            # Prepare content for BART model (max 1024 tokens)
            max_length = int(len(content.split()) * (1 - target_ratio))
            max_length = max(50, min(max_length, 500))  # Reasonable bounds
            
            # Tokenize input
            inputs = self.summarization_tokenizer(
                content,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            )
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.summarization_model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    min_length=max_length // 2,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode summary
            compressed_content = self.summarization_tokenizer.decode(
                summary_ids[0], 
                skip_special_tokens=True
            )
            
            # For removed content, we can't easily identify what was removed in abstractive summarization
            removed_content = ["[Abstractive summarization - original content transformed]"]
            
            return compressed_content, removed_content
            
        except Exception as e:
            logger.error(f"Abstractive compression failed: {e}")
            # Fallback to extractive compression
            segments = self._segment_content(content)
            segments = await self._calculate_importance_scores(segments, content)
            return await self._extractive_compression(segments, target_ratio, True)
    
    async def _hybrid_compression(
        self, 
        segments: List[ContentSegment], 
        content: str, 
        target_ratio: float, 
        preserve_structure: bool
    ) -> Tuple[str, List[str]]:
        """Combine extractive and abstractive compression."""
        
        # Use extractive compression for initial reduction
        extractive_ratio = target_ratio * 0.6  # 60% of target reduction
        compressed_content, removed_content = await self._extractive_compression(
            segments, extractive_ratio, preserve_structure
        )
        
        # Apply abstractive compression to further reduce
        remaining_ratio = target_ratio - extractive_ratio
        if remaining_ratio > 0.1:  # Only if significant reduction still needed
            try:
                final_compressed, _ = await self._abstractive_compression(
                    compressed_content, remaining_ratio
                )
                compressed_content = final_compressed
            except Exception as e:
                logger.warning(f"Abstractive step failed, using extractive result: {e}")
        
        return compressed_content, removed_content
    
    async def _semantic_compression(self, segments: List[ContentSegment], target_ratio: float) -> Tuple[str, List[str]]:
        """Compress content using semantic similarity clustering."""
        
        if not self.sentence_transformer:
            # Fallback to extractive compression
            return await self._extractive_compression(segments, target_ratio, True)
        
        try:
            # Get sentence embeddings
            texts = [seg.text for seg in segments]
            embeddings = self.sentence_transformer.encode(texts)
            
            # Cluster similar sentences
            from sklearn.cluster import KMeans
            n_clusters = max(2, int(len(segments) * (1 - target_ratio)))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Select representative sentence from each cluster
            selected_segments = []
            removed_content = []
            
            for cluster_id in range(n_clusters):
                cluster_segments = [seg for i, seg in enumerate(segments) if cluster_labels[i] == cluster_id]
                
                if cluster_segments:
                    # Select the most important sentence from this cluster
                    best_segment = max(cluster_segments, key=lambda x: x.importance_score)
                    selected_segments.append(best_segment)
                    
                    # Mark others as removed
                    for seg in cluster_segments:
                        if seg != best_segment:
                            removed_content.append(seg.text)
            
            # Sort selected segments by original position
            selected_segments.sort(key=lambda x: x.position[0])
            
            # Build compressed content
            compressed_content = " ".join([seg.text for seg in selected_segments])
            
            return compressed_content, removed_content
            
        except Exception as e:
            logger.error(f"Semantic compression failed: {e}")
            # Fallback to extractive compression
            return await self._extractive_compression(segments, target_ratio, True)
    
    async def _structural_compression(self, segments: List[ContentSegment], target_ratio: float) -> Tuple[str, List[str]]:
        """Compress content by optimizing structure and removing redundancy."""
        
        # Group segments by paragraph
        paragraphs = defaultdict(list)
        for seg in segments:
            para_id = seg.metadata.get("paragraph_id", 0)
            paragraphs[para_id].append(seg)
        
        selected_segments = []
        removed_content = []
        
        for para_id, para_segments in paragraphs.items():
            # Sort sentences in paragraph by importance
            para_segments.sort(key=lambda x: x.importance_score, reverse=True)
            
            # Keep only the most important sentences from each paragraph
            sentences_to_keep = max(1, int(len(para_segments) * (1 - target_ratio)))
            kept_sentences = para_segments[:sentences_to_keep]
            
            # Sort kept sentences by original position
            kept_sentences.sort(key=lambda x: x.position[0])
            
            selected_segments.extend(kept_sentences)
            
            # Mark removed sentences
            for seg in para_segments[sentences_to_keep:]:
                removed_content.append(seg.text)
        
        # Sort all selected segments by position
        selected_segments.sort(key=lambda x: x.position[0])
        
        # Build compressed content
        compressed_content = " ".join([seg.text for seg in selected_segments])
        
        return compressed_content, removed_content
    
    def _extract_key_points(self, segments: List[ContentSegment]) -> List[str]:
        """Extract key points from the most important segments."""
        
        # Sort by importance and take top segments
        top_segments = sorted(segments, key=lambda x: x.importance_score, reverse=True)[:5]
        
        key_points = []
        for seg in top_segments:
            # Clean and format the text
            text = seg.text.strip()
            if len(text) > 10:  # Only include substantial sentences
                key_points.append(text)
        
        return key_points
    
    async def _calculate_quality_score(self, original_content: str, compressed_content: str) -> float:
        """Calculate quality score for compressed content."""
        
        try:
            # Method 1: Semantic similarity
            if self.sentence_transformer:
                original_embedding = self.sentence_transformer.encode([original_content])
                compressed_embedding = self.sentence_transformer.encode([compressed_content])
                semantic_similarity = cosine_similarity(original_embedding, compressed_embedding)[0][0]
            else:
                semantic_similarity = 0.5  # Default score
            
            # Method 2: Information retention (based on keyword overlap)
            original_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', original_content.lower()))
            compressed_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', compressed_content.lower()))
            
            if original_words:
                keyword_overlap = len(original_words.intersection(compressed_words)) / len(original_words)
            else:
                keyword_overlap = 0.0
            
            # Method 3: Readability (simple heuristic)
            original_sentences = len(self._split_into_sentences(original_content))
            compressed_sentences = len(self._split_into_sentences(compressed_content))
            
            if original_sentences > 0:
                sentence_ratio = min(1.0, compressed_sentences / original_sentences)
            else:
                sentence_ratio = 1.0
            
            # Combine scores
            quality_score = (
                semantic_similarity * 0.5 +
                keyword_overlap * 0.3 +
                sentence_ratio * 0.2
            )
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.5  # Default score
    
    async def batch_compress(
        self,
        contents: List[str],
        compression_level: CompressionLevel = CompressionLevel.MODERATE,
        compression_method: CompressionMethod = CompressionMethod.HYBRID
    ) -> List[CompressionResult]:
        """Compress multiple contents in batch."""
        
        logger.info(f"Starting batch compression of {len(contents)} documents")
        
        results = []
        for i, content in enumerate(contents):
            try:
                result = await self.compress_content(
                    content=content,
                    compression_level=compression_level,
                    compression_method=compression_method
                )
                results.append(result)
                logger.info(f"Compressed document {i+1}/{len(contents)}")
                
            except Exception as e:
                logger.error(f"Failed to compress document {i+1}: {e}")
                # Create error result
                error_result = CompressionResult(
                    id=str(uuid.uuid4()),
                    original_content=content,
                    compressed_content=content,  # Keep original on error
                    compression_ratio=0.0,
                    compression_level=compression_level,
                    compression_method=compression_method,
                    quality_score=0.0,
                    key_points=[],
                    removed_content=[],
                    processing_time=0.0,
                    metadata={"error": str(e)}
                )
                results.append(error_result)
        
        logger.info(f"Batch compression completed: {len(results)} results")
        return results

# Global intelligent compressor instance
_global_intelligent_compressor: Optional[IntelligentCompressor] = None

def get_global_intelligent_compressor() -> IntelligentCompressor:
    """Get the global intelligent compressor instance."""
    global _global_intelligent_compressor
    if _global_intelligent_compressor is None:
        _global_intelligent_compressor = IntelligentCompressor()
    return _global_intelligent_compressor



























