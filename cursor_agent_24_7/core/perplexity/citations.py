"""
Citation Manager - Handles citation matching and formatting
===========================================================

Manages intelligent citation placement based on content matching.
"""

import re
import logging
from typing import List, Tuple
from .types import SearchResult

logger = logging.getLogger(__name__)


class CitationManager:
    """Manages citation matching and formatting."""
    
    def add_citations(
        self,
        content: str,
        search_results: List[SearchResult]
    ) -> str:
        """
        Add citations to content based on search results.
        
        Citations are added as [1][2], etc. at the end of sentences with NO SPACE.
        
        Args:
            content: The content to add citations to
            search_results: List of search results to cite
            
        Returns:
            Content with citations added
        """
        if not search_results:
            return content
        
        # Remove any existing citation markers
        content = re.sub(r'\[\d+\]', '', content)
        
        # Split content into sentences (preserving LaTeX math)
        sentences = self._split_sentences_preserving_math(content)
        cited_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                cited_sentences.append(sentence)
                continue
            
            # Find relevant search results for this sentence
            relevant_indices = self._find_relevant_sources(sentence, search_results)
            
            # Add citations if found
            if relevant_indices:
                # Format citations: [1][2][3] (max 3 per sentence, no space before)
                citations = ''.join(f'[{idx}]' for idx in relevant_indices[:3])
                sentence_clean = sentence.rstrip()
                cited_sentence = f"{sentence_clean}{citations}"
                cited_sentences.append(cited_sentence)
            else:
                cited_sentences.append(sentence)
        
        return ' '.join(cited_sentences)
    
    def _split_sentences_preserving_math(self, content: str) -> List[str]:
        """Split content into sentences while preserving LaTeX math expressions."""
        # Protect LaTeX expressions during splitting
        math_pattern = r'\\(?:\(|\[).*?\\(?:\)|\])'
        math_expressions = []
        
        def replace_math(match):
            idx = len(math_expressions)
            math_expressions.append(match.group(0))
            return f"__MATH_{idx}__"
        
        # Replace math expressions with placeholders
        protected_content = re.sub(math_pattern, replace_math, content)
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', protected_content)
        
        # Restore math expressions
        restored = []
        for sentence in sentences:
            for idx, expr in enumerate(math_expressions):
                sentence = sentence.replace(f"__MATH_{idx}__", expr)
            restored.append(sentence)
        
        return restored
    
    def _find_relevant_sources(
        self,
        sentence: str,
        search_results: List[SearchResult]
    ) -> List[int]:
        """
        Find relevant search result indices for a sentence.
        
        Uses keyword matching and content similarity.
        
        Args:
            sentence: The sentence to find sources for
            search_results: List of search results
            
        Returns:
            List of relevant search result indices (sorted by relevance)
        """
        sentence_lower = sentence.lower()
        sentence_words = set(sentence_lower.split())
        relevant = []
        
        for result in search_results:
            score = 0
            
            # Check title keywords
            title_words = set(result.title.lower().split())
            title_overlap = len(title_words & sentence_words)
            if title_overlap > 0:
                score += title_overlap * 2
            
            # Check snippet/content keywords
            content_text = (result.content or result.snippet).lower()
            content_words = set(content_text.split())
            content_overlap = len(content_words & sentence_words)
            if content_overlap > 0:
                score += content_overlap
            
            # Check for exact phrase matches (higher weight)
            for phrase in result.snippet.lower().split('. '):
                if phrase in sentence_lower or sentence_lower in phrase:
                    score += 5
            
            # If score is significant, include this result
            if score >= 2:
                relevant.append((result.index, score))
        
        # Sort by score and return indices
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in relevant]
    
    def normalize_citations(self, content: str, query_type: str) -> str:
        """
        Normalize citations based on query type.
        
        Args:
            content: Content with citations
            query_type: The query type
            
        Returns:
            Normalized content
        """
        if query_type == "url_lookup":
            # URL lookup should only have [1] citations
            content = re.sub(r'\[\d+\]', '[1]', content)
            # Remove duplicate [1] citations
            content = re.sub(r'\[1\]\[1\]+', '[1]', content)
        elif query_type in ["translation", "creative_writing"]:
            # Remove all citations
            content = re.sub(r'\[\d+\]', '', content)
        
        return content




