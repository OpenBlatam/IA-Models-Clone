"""
Advanced Analytics Engine - Functional approach for content analysis
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import asyncio
import hashlib
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
from dataclasses import dataclass, asdict
from decimal import Decimal
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import aiofiles
import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class ContentMetrics:
    """Content analysis metrics using dataclass for immutability"""
    content_id: str
    word_count: int
    character_count: int
    unique_words: int
    readability_score: float
    sentiment_score: float
    language: str
    created_at: datetime
    updated_at: datetime


@dataclass
class SimilarityResult:
    """Similarity analysis result"""
    content_id_1: str
    content_id_2: str
    similarity_score: float
    similarity_type: str
    matched_phrases: List[str]
    analysis_timestamp: datetime


@dataclass
class RedundancyReport:
    """Comprehensive redundancy analysis report"""
    report_id: str
    total_content_items: int
    duplicate_groups: List[List[str]]
    similarity_threshold: float
    analysis_duration: float
    generated_at: datetime
    recommendations: List[str]


# Redis connection pool for caching
redis_pool: Optional[redis.ConnectionPool] = None


async def get_redis_client() -> redis.Redis:
    """Get Redis client with connection pooling"""
    global redis_pool
    if redis_pool is None:
        redis_pool = redis.ConnectionPool(
            host="localhost", port=6379, db=0, decode_responses=True
        )
    return redis.Redis(connection_pool=redis_pool)


def calculate_content_hash(content: str) -> str:
    """Calculate SHA-256 hash for content deduplication"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def extract_text_features(content: str) -> Dict[str, Any]:
    """Extract text features for analysis"""
    words = content.split()
    sentences = content.split('.')
    
    return {
        "word_count": len(words),
        "character_count": len(content),
        "unique_words": len(set(words)),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
        "avg_sentence_length": len(words) / len(sentences) if sentences else 0
    }


def calculate_readability_score(content: str) -> float:
    """Calculate Flesch Reading Ease score"""
    words = content.split()
    sentences = [s for s in content.split('.') if s.strip()]
    
    if not words or not sentences:
        return 0.0
    
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = sum(count_syllables(word) for word in words) / len(words)
    
    # Flesch Reading Ease formula
    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    return max(0.0, min(100.0, score))


def count_syllables(word: str) -> int:
    """Count syllables in a word"""
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


async def analyze_content_similarity(
    content_1: str, 
    content_2: str, 
    similarity_type: str = "tfidf"
) -> SimilarityResult:
    """Analyze similarity between two content pieces"""
    
    # Early return for identical content
    if content_1 == content_2:
        return SimilarityResult(
            content_id_1="",
            content_id_2="",
            similarity_score=1.0,
            similarity_type=similarity_type,
            matched_phrases=[],
            analysis_timestamp=datetime.now()
        )
    
    # Calculate similarity based on type
    if similarity_type == "tfidf":
        similarity_score, matched_phrases = await _calculate_tfidf_similarity(content_1, content_2)
    elif similarity_type == "jaccard":
        similarity_score, matched_phrases = await _calculate_jaccard_similarity(content_1, content_2)
    elif similarity_type == "cosine":
        similarity_score, matched_phrases = await _calculate_cosine_similarity(content_1, content_2)
    else:
        raise ValueError(f"Unsupported similarity type: {similarity_type}")
    
    return SimilarityResult(
        content_id_1="",
        content_id_2="",
        similarity_score=similarity_score,
        similarity_type=similarity_type,
        matched_phrases=matched_phrases,
        analysis_timestamp=datetime.now()
    )


async def _calculate_tfidf_similarity(content_1: str, content_2: str) -> Tuple[float, List[str]]:
    """Calculate TF-IDF based similarity"""
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),
            max_features=1000
        )
        
        tfidf_matrix = vectorizer.fit_transform([content_1, content_2])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Extract common phrases
        feature_names = vectorizer.get_feature_names_out()
        matched_phrases = feature_names[:10].tolist()  # Top 10 features
        
        return float(similarity_score), matched_phrases
        
    except Exception as e:
        logger.error(f"Error calculating TF-IDF similarity: {e}")
        return 0.0, []


async def _calculate_jaccard_similarity(content_1: str, content_2: str) -> Tuple[float, List[str]]:
    """Calculate Jaccard similarity"""
    try:
        words_1 = set(content_1.lower().split())
        words_2 = set(content_2.lower().split())
        
        intersection = words_1.intersection(words_2)
        union = words_1.union(words_2)
        
        similarity_score = len(intersection) / len(union) if union else 0.0
        matched_phrases = list(intersection)[:10]  # Top 10 common words
        
        return similarity_score, matched_phrases
        
    except Exception as e:
        logger.error(f"Error calculating Jaccard similarity: {e}")
        return 0.0, []


async def _calculate_cosine_similarity(content_1: str, content_2: str) -> Tuple[float, List[str]]:
    """Calculate cosine similarity"""
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=500
        )
        
        tfidf_matrix = vectorizer.fit_transform([content_1, content_2])
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        feature_names = vectorizer.get_feature_names_out()
        matched_phrases = feature_names[:10].tolist()
        
        return float(similarity_score), matched_phrases
        
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0, []


async def detect_content_clusters(
    content_list: List[Dict[str, Any]], 
    similarity_threshold: float = 0.7
) -> List[List[str]]:
    """Detect content clusters using DBSCAN clustering"""
    
    if len(content_list) < 2:
        return []
    
    try:
        # Prepare content for clustering
        texts = [item["content"] for item in content_list]
        content_ids = [item["id"] for item in content_list]
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=1 - similarity_threshold,  # Convert similarity to distance
            min_samples=2,
            metric='cosine'
        )
        
        cluster_labels = clustering.fit_predict(tfidf_matrix)
        
        # Group content by cluster
        clusters = defaultdict(list)
        for content_id, label in zip(content_ids, cluster_labels):
            if label != -1:  # -1 indicates noise/outliers
                clusters[label].append(content_id)
        
        return [cluster for cluster in clusters.values() if len(cluster) > 1]
        
    except Exception as e:
        logger.error(f"Error in content clustering: {e}")
        return []


async def generate_redundancy_report(
    content_list: List[Dict[str, Any]],
    similarity_threshold: float = 0.7,
    analysis_type: str = "comprehensive"
) -> RedundancyReport:
    """Generate comprehensive redundancy analysis report"""
    
    start_time = datetime.now()
    
    # Early return for insufficient content
    if len(content_list) < 2:
        return RedundancyReport(
            report_id="",
            total_content_items=len(content_list),
            duplicate_groups=[],
            similarity_threshold=similarity_threshold,
            analysis_duration=0.0,
            generated_at=start_time,
            recommendations=["Insufficient content for analysis"]
        )
    
    try:
        # Detect duplicate groups
        duplicate_groups = await detect_content_clusters(content_list, similarity_threshold)
        
        # Generate recommendations
        recommendations = await _generate_recommendations(content_list, duplicate_groups)
        
        analysis_duration = (datetime.now() - start_time).total_seconds()
        
        return RedundancyReport(
            report_id=f"report_{int(datetime.now().timestamp())}",
            total_content_items=len(content_list),
            duplicate_groups=duplicate_groups,
            similarity_threshold=similarity_threshold,
            analysis_duration=analysis_duration,
            generated_at=start_time,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error generating redundancy report: {e}")
        return RedundancyReport(
            report_id="",
            total_content_items=len(content_list),
            duplicate_groups=[],
            similarity_threshold=similarity_threshold,
            analysis_duration=0.0,
            generated_at=start_time,
            recommendations=[f"Analysis failed: {str(e)}"]
        )


async def _generate_recommendations(
    content_list: List[Dict[str, Any]], 
    duplicate_groups: List[List[str]]
) -> List[str]:
    """Generate actionable recommendations based on analysis"""
    
    recommendations = []
    
    if not duplicate_groups:
        recommendations.append("No significant redundancy detected. Content appears unique.")
        return recommendations
    
    # Calculate redundancy percentage
    total_duplicates = sum(len(group) for group in duplicate_groups)
    redundancy_percentage = (total_duplicates / len(content_list)) * 100
    
    recommendations.append(f"Found {len(duplicate_groups)} duplicate groups affecting {redundancy_percentage:.1f}% of content.")
    
    # Specific recommendations
    if redundancy_percentage > 50:
        recommendations.append("High redundancy detected. Consider content consolidation strategy.")
    elif redundancy_percentage > 25:
        recommendations.append("Moderate redundancy detected. Review duplicate content for optimization.")
    else:
        recommendations.append("Low redundancy detected. Current content strategy appears effective.")
    
    # Group-specific recommendations
    for i, group in enumerate(duplicate_groups):
        if len(group) > 3:
            recommendations.append(f"Group {i+1}: {len(group)} similar items - consider merging or archiving.")
        else:
            recommendations.append(f"Group {i+1}: {len(group)} similar items - review for minor adjustments.")
    
    return recommendations


async def cache_analysis_result(
    content_hash: str, 
    result: Dict[str, Any], 
    ttl: int = 3600
) -> None:
    """Cache analysis result in Redis"""
    try:
        redis_client = await get_redis_client()
        cache_key = f"analysis:{content_hash}"
        await redis_client.setex(
            cache_key, 
            ttl, 
            json.dumps(result, default=str)
        )
    except Exception as e:
        logger.warning(f"Failed to cache analysis result: {e}")


async def get_cached_analysis_result(content_hash: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached analysis result from Redis"""
    try:
        redis_client = await get_redis_client()
        cache_key = f"analysis:{content_hash}"
        cached_result = await redis_client.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        return None
        
    except Exception as e:
        logger.warning(f"Failed to retrieve cached analysis result: {e}")
        return None


async def get_content_analytics(
    content_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Get comprehensive content analytics"""
    
    if not content_list:
        return {"error": "No content provided for analysis"}
    
    try:
        # Calculate aggregate metrics
        total_items = len(content_list)
        total_words = sum(len(item["content"].split()) for item in content_list)
        total_characters = sum(len(item["content"]) for item in content_list)
        
        # Calculate readability scores
        readability_scores = [
            calculate_readability_score(item["content"]) 
            for item in content_list
        ]
        
        avg_readability = sum(readability_scores) / len(readability_scores)
        
        # Content length distribution
        word_counts = [len(item["content"].split()) for item in content_list]
        
        return {
            "total_content_items": total_items,
            "total_words": total_words,
            "total_characters": total_characters,
            "average_words_per_item": total_words / total_items,
            "average_readability_score": avg_readability,
            "readability_distribution": {
                "min": min(readability_scores),
                "max": max(readability_scores),
                "median": sorted(readability_scores)[len(readability_scores) // 2]
            },
            "word_count_distribution": {
                "min": min(word_counts),
                "max": max(word_counts),
                "median": sorted(word_counts)[len(word_counts) // 2]
            },
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating content analytics: {e}")
        return {"error": f"Analytics generation failed: {str(e)}"}


async def stream_analysis_results(
    content_list: List[Dict[str, Any]],
    batch_size: int = 10
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream analysis results in batches for large datasets"""
    
    total_batches = (len(content_list) + batch_size - 1) // batch_size
    
    for i in range(0, len(content_list), batch_size):
        batch = content_list[i:i + batch_size]
        batch_number = (i // batch_size) + 1
        
        try:
            # Analyze batch
            batch_results = []
            for item in batch:
                features = extract_text_features(item["content"])
                readability = calculate_readability_score(item["content"])
                
                batch_results.append({
                    "content_id": item["id"],
                    "features": features,
                    "readability_score": readability,
                    "content_hash": calculate_content_hash(item["content"])
                })
            
            yield {
                "batch_number": batch_number,
                "total_batches": total_batches,
                "batch_size": len(batch),
                "results": batch_results,
                "progress_percentage": (batch_number / total_batches) * 100
            }
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_number}: {e}")
            yield {
                "batch_number": batch_number,
                "error": str(e),
                "progress_percentage": (batch_number / total_batches) * 100
            }
        
        # Small delay to prevent overwhelming the system
        await asyncio.sleep(0.1)




