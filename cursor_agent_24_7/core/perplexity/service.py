"""
Perplexity Service - High-level service layer
==============================================

Provides a high-level service interface for Perplexity query processing
with better error handling and convenience methods.
"""

import logging
from typing import List, Dict, Any, Optional
from .processor import PerplexityProcessor
from .types import ProcessedQuery, QueryType
from .config import PerplexityConfig
from .helpers import (
    convert_search_results_to_dict,
    sanitize_query_text,
    validate_search_result,
    calculate_answer_quality_score
)
from .utils import count_citations
from .exceptions import (
    PerplexityError,
    QueryProcessingError,
    LLMProviderError,
    ValidationError
)

logger = logging.getLogger(__name__)


class PerplexityService:
    """
    High-level service for Perplexity query processing.
    
    Provides convenient methods with better error handling and
    additional features like batch processing.
    """
    
    def __init__(
        self,
        config: Optional[PerplexityConfig] = None,
        system_prompt_path: Optional[str] = None,
        enable_validation: bool = True,
        enable_cache: bool = True,
        enable_metrics: bool = True,
        cache_ttl: int = 3600
    ):
        """
        Initialize service.
        
        Args:
            config: Optional configuration object (overrides other params)
            system_prompt_path: Path to SYSTEM_PROMPT.md
            enable_validation: Enable response validation
            enable_cache: Enable response caching
            enable_metrics: Enable metrics collection
            cache_ttl: Cache TTL in seconds
        """
        if config:
            self.processor = PerplexityProcessor(
                system_prompt_path=config.system_prompt_path,
                enable_validation=config.enable_validation,
                enable_cache=config.enable_cache,
                enable_metrics=config.enable_metrics,
                cache_ttl=config.cache_ttl
            )
            self.config = config
        else:
            self.processor = PerplexityProcessor(
                system_prompt_path=system_prompt_path,
                enable_validation=enable_validation,
                enable_cache=enable_cache,
                enable_metrics=enable_metrics,
                cache_ttl=cache_ttl
            )
            self.config = PerplexityConfig(
                system_prompt_path=system_prompt_path,
                enable_validation=enable_validation,
                enable_cache=enable_cache,
                enable_metrics=enable_metrics,
                cache_ttl=cache_ttl
            )
    
    async def answer_query(
        self,
        query: str,
        search_results: Optional[List[Dict[str, Any]]] = None,
        llm_provider: Optional[Any] = None,
        use_llm: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a query with full processing.
        
        Args:
            query: The query string
            search_results: Optional search results
            llm_provider: Optional LLM provider
            use_llm: Whether to use LLM (if provider available)
            
        Returns:
            Dictionary with answer and metadata
            
        Raises:
            QueryProcessingError: If processing fails
        """
        try:
            # Sanitize query
            query = sanitize_query_text(query)
            if not query:
                raise QueryProcessingError("Query cannot be empty")
            
            # Validate and convert search results
            search_results_dict = convert_search_results_to_dict(search_results)
            if search_results_dict:
                for result in search_results_dict:
                    is_valid, error = validate_search_result(result)
                    if not is_valid:
                        logger.warning(f"Invalid search result: {error}")
            
            # Process query
            processed = self.processor.process_query(query, search_results_dict)
            
            # Generate answer
            provider = llm_provider if use_llm else None
            answer = await self.processor.generate_answer(
                processed,
                provider,
                search_results_dict
            )
            
            # Calculate quality score
            citation_count = count_citations(answer)
            quality_score = calculate_answer_quality_score(
                answer,
                len(processed.search_results),
                citation_count
            )
            
            return {
                'query': processed.original_query,
                'query_type': processed.query_type.value,
                'answer': answer,
                'metadata': {
                    **processed.metadata,
                    'quality_score': quality_score,
                    'citation_count': citation_count
                },
                'search_result_count': len(processed.search_results)
            }
            
        except QueryProcessingError:
            raise
        except Exception as e:
            logger.error(f"Error answering query: {e}", exc_info=True)
            raise QueryProcessingError(f"Failed to answer query: {str(e)}") from e
    
    async def batch_process(
        self,
        queries: List[Dict[str, Any]],
        llm_provider: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of query dictionaries with 'query' and optional 'search_results'
            llm_provider: Optional LLM provider
            
        Returns:
            List of results for each query
        """
        results = []
        
        for query_data in queries:
            try:
                result = await self.answer_query(
                    query=query_data['query'],
                    search_results=query_data.get('search_results'),
                    llm_provider=llm_provider,
                    use_llm=llm_provider is not None
                )
                result['success'] = True
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query in batch: {e}")
                results.append({
                    'query': query_data.get('query', ''),
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.processor.cache:
            return {'enabled': False}
        
        stats = self.processor.cache.get_stats()
        stats['enabled'] = True
        return stats
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics statistics."""
        if not self.processor.metrics:
            return {'enabled': False}
        
        return self.processor.metrics.get_stats()
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        if self.processor.cache:
            self.processor.cache.clear()
    
    def clear_metrics(self) -> None:
        """Clear metrics."""
        if self.processor.metrics:
            self.processor.metrics.clear()
    
    def validate_answer(self, answer: str, query_type: str = "general") -> Dict[str, Any]:
        """
        Validate an answer.
        
        Args:
            answer: Answer text to validate
            query_type: Query type
            
        Returns:
            Validation results
        """
        if not self.processor.validator:
            return {'enabled': False, 'valid': True}
        
        is_valid, issues = self.processor.validate_response(answer, query_type)
        
        return {
            'enabled': True,
            'valid': is_valid,
            'issue_count': len(issues),
            'issues': [
                {
                    'level': issue.level.value if hasattr(issue.level, 'value') else str(issue.level),
                    'rule': issue.rule,
                    'message': issue.message,
                    'location': issue.location,
                    'suggestion': issue.suggestion
                }
                for issue in issues
            ]
        }

