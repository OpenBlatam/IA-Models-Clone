"""
Perplexity Processor - Main orchestrator for query processing
============================================================

Coordinates all components to process queries and generate formatted answers.
"""

import re
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from .types import QueryType, SearchResult, ProcessedQuery
from .detector import QueryTypeDetector
from .formatter import ResponseFormatter
from .prompt_builder import PromptBuilder
from .cache import PerplexityCache
from .metrics import PerplexityMetrics

try:
    from .validator import PerplexityValidator, ValidationIssue
except ImportError:
    PerplexityValidator = None
    ValidationIssue = None

logger = logging.getLogger(__name__)


class PerplexityProcessor:
    """Main processor for Perplexity-style query handling."""
    
    def __init__(
        self,
        system_prompt_path: Optional[str] = None,
        enable_validation: bool = True,
        enable_cache: bool = True,
        enable_metrics: bool = True,
        cache_ttl: int = 3600
    ):
        """
        Initialize processor.
        
        Args:
            system_prompt_path: Path to SYSTEM_PROMPT.md file
            enable_validation: Whether to enable response validation
            enable_cache: Whether to enable response caching
            enable_metrics: Whether to enable metrics collection
            cache_ttl: Cache time-to-live in seconds
        """
        self.type_detector = QueryTypeDetector()
        self.formatter = ResponseFormatter()
        self.prompt_builder = PromptBuilder(system_prompt_path)
        self.validator = PerplexityValidator() if enable_validation and PerplexityValidator else None
        self.cache = PerplexityCache(ttl_seconds=cache_ttl) if enable_cache else None
        self.metrics = PerplexityMetrics() if enable_metrics else None
    
    def process_query(
        self,
        query: str,
        search_results: Optional[List[Dict[str, Any]]] = None
    ) -> ProcessedQuery:
        """
        Process a query and prepare it for answering.
        
        Args:
            query: User's query string
            search_results: Optional list of search result dictionaries
            
        Returns:
            ProcessedQuery object
        """
        # Convert search results to SearchResult objects
        search_result_objects = []
        if search_results:
            for idx, result in enumerate(search_results, start=1):
                search_result_objects.append(SearchResult.from_dict(result, idx))
        
        # Detect query type
        query_type = self.type_detector.detect(query, search_result_objects)
        
        # Determine if citations are needed
        requires_citations = query_type not in [
            QueryType.TRANSLATION,
            QueryType.CREATIVE_WRITING
        ]
        
        return ProcessedQuery(
            original_query=query,
            query_type=query_type,
            search_results=search_result_objects,
            requires_citations=requires_citations,
            metadata={
                'current_date': self.formatter.current_date.isoformat(),
                'result_count': len(search_result_objects)
            }
        )
    
    def format_response(
        self,
        processed_query: ProcessedQuery,
        answer_content: str
    ) -> str:
        """
        Format a response according to Perplexity guidelines.
        
        Args:
            processed_query: ProcessedQuery object
            answer_content: The answer content to format
            
        Returns:
            Formatted response string
        """
        return self.formatter.format_answer(
            query=processed_query.original_query,
            query_type=processed_query.query_type,
            search_results=processed_query.search_results,
            answer_content=answer_content
        )
    
    def build_llm_prompt(
        self,
        processed_query: ProcessedQuery,
        include_planning: bool = True
    ) -> str:
        """
        Build a prompt for LLM answer generation.
        
        Args:
            processed_query: ProcessedQuery object
            include_planning: Whether to include planning instructions
            
        Returns:
            Complete prompt string for LLM
        """
        return self.prompt_builder.build_prompt(processed_query, include_planning)
    
    async def generate_answer(
        self,
        processed_query: ProcessedQuery,
        llm_provider: Optional[Any] = None,
        search_results: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate an answer using LLM.
        
        Args:
            processed_query: ProcessedQuery object
            llm_provider: Optional LLM provider (LLMPipeline, OpenAI client, etc.)
            search_results: Original search results for caching
            
        Returns:
            Generated answer string
        """
        import time
        start_time = time.time()
        
        # Check cache first
        if self.cache:
            cached_answer = self.cache.get(
                processed_query.original_query,
                search_results
            )
            if cached_answer:
                logger.debug("Returning cached answer")
                return cached_answer
        
        # Build prompt
        prompt = self.build_llm_prompt(processed_query)
        
        # If no LLM provider, return placeholder
        if llm_provider is None:
            logger.warning("No LLM provider specified, returning placeholder answer")
            answer = self._generate_placeholder_answer(processed_query)
        else:
            # Try to use LLM provider
            try:
                llm_answer = await self._call_llm(llm_provider, prompt)
                answer = self.format_response(processed_query, llm_answer)
                
                # Validate if validator is enabled
                has_errors = False
                if self.validator:
                    is_valid, issues = self.validator.validate(
                        answer,
                        processed_query.query_type.value
                    )
                    has_errors = not is_valid
                    if has_errors:
                        logger.warning(f"Answer validation found {len(issues)} issues")
                        for issue in issues:
                            if issue.level.value == "error":
                                logger.error(f"Validation error: {issue.message} - {issue.suggestion}")
                            else:
                                logger.warning(f"Validation warning: {issue.message}")
                
                # Record metrics
                if self.metrics:
                    processing_time_ms = (time.time() - start_time) * 1000
                    citation_count = len([m for m in re.finditer(r'\[\d+\]', answer)])
                    self.metrics.record_query(
                        query_type=processed_query.query_type,
                        processing_time_ms=processing_time_ms,
                        search_result_count=len(processed_query.search_results),
                        answer_length=len(answer),
                        citation_count=citation_count,
                        has_errors=has_errors
                    )
                
            except Exception as e:
                logger.error(f"Error generating answer with LLM: {e}", exc_info=True)
                answer = self._generate_placeholder_answer(processed_query)
        
        # Cache the answer
        if self.cache:
            self.cache.set(
                processed_query.original_query,
                search_results,
                processed_query,
                answer
            )
        
        return answer
    
    async def _call_llm(self, llm_provider: Any, prompt: str) -> str:
        """Call LLM provider with prompt."""
        # Check if it's an LLMPipeline with async generate
        if hasattr(llm_provider, 'generate'):
            if asyncio.iscoroutinefunction(llm_provider.generate):
                return await llm_provider.generate(prompt)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, llm_provider.generate, prompt)
        
        # Check if it's an async callable
        if callable(llm_provider) and asyncio.iscoroutinefunction(llm_provider):
            return await llm_provider(prompt)
        
        # Try OpenAI-style client
        if hasattr(llm_provider, 'chat') and hasattr(llm_provider.chat, 'completions'):
            create_method = llm_provider.chat.completions.create
            system_prompt = self.prompt_builder.load_system_prompt()
            
            if asyncio.iscoroutinefunction(create_method):
                response = await create_method(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
            else:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: create_method(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                    )
                )
            return response.choices[0].message.content
        
        raise ValueError("Unsupported LLM provider type")
    
    def _generate_placeholder_answer(self, processed_query: ProcessedQuery) -> str:
        """Generate a placeholder answer when LLM is not available."""
        answer = (
            f"This query has been processed as a {processed_query.query_type.value} query. "
            f"To generate a complete answer, please provide an LLM provider. "
            f"Found {len(processed_query.search_results)} search results to use."
        )
        formatted = self.format_response(processed_query, answer)
        
        # Validate placeholder too
        if self.validator:
            is_valid, issues = self.validator.validate(
                formatted,
                processed_query.query_type.value
            )
            if not is_valid:
                logger.debug(f"Placeholder validation found {len(issues)} issues")
        
        return formatted
    
    def validate_response(
        self,
        answer: str,
        query_type: str = "general"
    ) -> Tuple[bool, List[Any]]:
        """
        Validate a response against Perplexity formatting rules.
        
        Args:
            answer: The answer text to validate
            query_type: The query type
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        if not self.validator:
            logger.warning("Validator not enabled")
            return True, []
        
        return self.validator.validate(answer, query_type)

