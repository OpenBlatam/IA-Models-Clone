"""
Response Formatter - Formats answers according to Perplexity guidelines
=======================================================================

Handles all formatting rules including LaTeX, citations, headers, and query-type specific formatting.
"""

import re
import logging
from typing import List
from datetime import datetime
from .types import QueryType, SearchResult
from .citations import CitationManager

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Formats responses according to Perplexity-style guidelines."""
    
    def __init__(self):
        self.current_date = datetime(2025, 5, 13, 4, 31, 29)  # From prompt
        self.citation_manager = CitationManager()
    
    def format_answer(
        self,
        query: str,
        query_type: QueryType,
        search_results: List[SearchResult],
        answer_content: str
    ) -> str:
        """
        Format the answer according to Perplexity guidelines.
        
        Args:
            query: Original query
            query_type: Detected query type
            search_results: List of search results
            answer_content: The answer content to format
            
        Returns:
            Formatted answer string
        """
        # Normalize LaTeX math expressions
        formatted = self._normalize_latex(answer_content)
        
        # Ensure answer doesn't start with header
        formatted = self._ensure_no_leading_header(formatted)
        
        # Add citations
        formatted = self.citation_manager.add_citations(formatted, search_results)
        
        # Normalize citations based on query type
        formatted = self.citation_manager.normalize_citations(formatted, query_type.value)
        
        # Apply query-type specific formatting
        formatted = self._apply_query_type_formatting(formatted, query_type, search_results)
        
        # Final cleanup
        formatted = self._remove_references_section(formatted)
        formatted = self._ensure_no_ending_question(formatted)
        
        return formatted
    
    def _normalize_latex(self, content: str) -> str:
        """Convert $ and $$ LaTeX delimiters to \( and \[ format."""
        # Convert $$...$$ to \[...\]
        content = re.sub(r'\$\$(.*?)\$\$', r'\\[\1\\]', content, flags=re.DOTALL)
        # Convert $...$ to \(...\)
        content = re.sub(r'\$(.*?)\$', r'\\(\1\\)', content, flags=re.DOTALL)
        # Remove \label instructions
        content = re.sub(r'\\label\{[^}]+\}', '', content)
        return content
    
    def _ensure_no_leading_header(self, content: str) -> str:
        """Ensure answer doesn't start with a header."""
        lines = content.split('\n')
        if lines and (lines[0].startswith('##') or lines[0].startswith('###')):
            content = '\n'.join(lines[1:])
            if not content.strip() or content.strip().startswith('##'):
                content = "Here's a comprehensive answer to your query.\n\n" + content
        return content
    
    def _ensure_no_ending_question(self, content: str) -> str:
        """Ensure answer doesn't end with a question mark."""
        content = content.rstrip()
        if content.endswith('?'):
            content = content.rstrip('?').rstrip() + '.'
        return content
    
    def _remove_references_section(self, content: str) -> str:
        """Remove References, Sources, or similar sections from the end."""
        content = re.sub(r'\n##?\s*References?\s*\n.*$', '', content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'\n##?\s*Sources?\s*\n.*$', '', content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'\n##?\s*Citations?\s*\n.*$', '', content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'\n\n\[\d+\]:\s*.*$', '', content, flags=re.MULTILINE | re.DOTALL)
        return content
    
    def _apply_query_type_formatting(
        self,
        content: str,
        query_type: QueryType,
        search_results: List[SearchResult]
    ) -> str:
        """Apply query-type specific formatting."""
        formatters = {
            QueryType.ACADEMIC_RESEARCH: self._format_academic,
            QueryType.RECENT_NEWS: lambda c: self._format_news(c, search_results),
            QueryType.WEATHER: self._format_weather,
            QueryType.PEOPLE: self._format_people,
            QueryType.CODING: self._format_coding,
            QueryType.COOKING_RECIPES: self._format_cooking,
            QueryType.SCIENCE_MATH: self._format_math,
            QueryType.URL_LOOKUP: lambda c: self._format_url_lookup(c, search_results),
            QueryType.TRANSLATION: self._format_translation,
            QueryType.CREATIVE_WRITING: self._format_creative_writing,
        }
        
        formatter = formatters.get(query_type)
        return formatter(content) if formatter else content
    
    def _format_academic(self, content: str) -> str:
        """Format academic research answers."""
        if not content.startswith('##'):
            lines = content.split('\n')
            if lines and not lines[0].startswith('##'):
                content = f"## Introduction\n\n{content}"
        return content
    
    def _format_news(self, content: str, search_results: List[SearchResult]) -> str:
        """Format news answers with lists and timestamps."""
        # Simplified - real implementation would parse and reorganize
        return content
    
    def _format_weather(self, content: str) -> str:
        """Format weather answers - keep it short."""
        lines = content.split('\n')
        if len(lines) > 5:
            content = '\n'.join(lines[:5])
        return content
    
    def _format_people(self, content: str) -> str:
        """Format people/biography answers."""
        if content.startswith('##'):
            content = content.split('\n', 1)[1] if '\n' in content else content
        return content
    
    def _format_coding(self, content: str) -> str:
        """Format coding answers - code first, then explanation."""
        return content
    
    def _format_cooking(self, content: str) -> str:
        """Format cooking recipe answers."""
        return content
    
    def _format_math(self, content: str) -> str:
        """Format math answers - just the result for simple calculations."""
        result_match = re.search(r'(?:=|is|equals?|result:?)\s*([0-9.+\-*/()]+)', content, re.IGNORECASE)
        if result_match:
            return result_match.group(1)
        return content
    
    def _format_url_lookup(self, content: str, search_results: List[SearchResult]) -> str:
        """Format URL lookup answers - only cite first result [1]."""
        # Already handled by citation_manager.normalize_citations
        return content
    
    def _format_translation(self, content: str) -> str:
        """Format translation answers - remove all citations."""
        # Already handled by citation_manager.normalize_citations
        return content
    
    def _format_creative_writing(self, content: str) -> str:
        """Format creative writing answers - remove all citations."""
        # Already handled by citation_manager.normalize_citations
        return content




