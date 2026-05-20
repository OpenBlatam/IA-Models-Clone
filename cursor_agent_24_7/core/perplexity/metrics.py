"""
Metrics Collector - Collects metrics for Perplexity queries
===========================================================

Tracks performance metrics, query types, and usage statistics.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from .types import QueryType

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_type: str
    processing_time_ms: float
    search_result_count: int
    answer_length: int
    citation_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    has_errors: bool = False


class PerplexityMetrics:
    """Collects and aggregates metrics for Perplexity queries."""
    
    def __init__(self):
        self.metrics: List[QueryMetrics] = []
        self.query_type_counts: Dict[str, int] = defaultdict(int)
        self.total_queries: int = 0
        self.total_processing_time: float = 0.0
        self.error_count: int = 0
    
    def record_query(
        self,
        query_type: QueryType,
        processing_time_ms: float,
        search_result_count: int,
        answer_length: int,
        citation_count: int,
        has_errors: bool = False
    ) -> None:
        """
        Record metrics for a processed query.
        
        Args:
            query_type: The detected query type
            processing_time_ms: Processing time in milliseconds
            search_result_count: Number of search results used
            answer_length: Length of the answer
            citation_count: Number of citations in answer
            has_errors: Whether there were validation errors
        """
        metric = QueryMetrics(
            query_type=query_type.value,
            processing_time_ms=processing_time_ms,
            search_result_count=search_result_count,
            answer_length=answer_length,
            citation_count=citation_count,
            has_errors=has_errors
        )
        
        self.metrics.append(metric)
        self.query_type_counts[query_type.value] += 1
        self.total_queries += 1
        self.total_processing_time += processing_time_ms
        
        if has_errors:
            self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.metrics:
            return {
                'total_queries': 0,
                'average_processing_time_ms': 0.0,
                'query_type_distribution': {},
                'error_rate': 0.0
            }
        
        avg_processing_time = self.total_processing_time / self.total_queries
        error_rate = self.error_count / self.total_queries if self.total_queries > 0 else 0.0
        
        # Calculate average answer length
        avg_answer_length = sum(m.answer_length for m in self.metrics) / len(self.metrics)
        
        # Calculate average citation count
        avg_citations = sum(m.citation_count for m in self.metrics) / len(self.metrics)
        
        return {
            'total_queries': self.total_queries,
            'average_processing_time_ms': avg_processing_time,
            'average_answer_length': avg_answer_length,
            'average_citation_count': avg_citations,
            'query_type_distribution': dict(self.query_type_counts),
            'error_rate': error_rate,
            'error_count': self.error_count
        }
    
    def get_query_type_stats(self, query_type: str) -> Dict[str, Any]:
        """
        Get statistics for a specific query type.
        
        Args:
            query_type: The query type to get stats for
            
        Returns:
            Statistics for the query type
        """
        type_metrics = [m for m in self.metrics if m.query_type == query_type]
        
        if not type_metrics:
            return {
                'count': 0,
                'average_processing_time_ms': 0.0,
                'average_answer_length': 0,
                'average_citation_count': 0
            }
        
        return {
            'count': len(type_metrics),
            'average_processing_time_ms': sum(m.processing_time_ms for m in type_metrics) / len(type_metrics),
            'average_answer_length': sum(m.answer_length for m in type_metrics) / len(type_metrics),
            'average_citation_count': sum(m.citation_count for m in type_metrics) / len(type_metrics)
        }
    
    def clear(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()
        self.query_type_counts.clear()
        self.total_queries = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        logger.info("Metrics cleared")




