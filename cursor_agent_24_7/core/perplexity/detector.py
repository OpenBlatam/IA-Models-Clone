"""
Query Type Detector - Detects query types from content
======================================================

Uses pattern matching and keyword analysis to detect query types.
"""

import re
import logging
from typing import Optional, List
from .types import QueryType, SearchResult

logger = logging.getLogger(__name__)


class QueryTypeDetector:
    """Detects the type of query based on patterns and context."""
    
    # Pattern groups for different query types
    PATTERNS = {
        QueryType.URL_LOOKUP: [
            r'https?://[^\s]+',
            r'www\.[^\s]+',
            r'[a-z0-9-]+\.(com|org|net|edu|gov|io|co)[^\s]*'
        ],
        QueryType.TRANSLATION: [
            r'\btranslate\b',
            r'\btranslation\b',
            r'\bto\s+(english|spanish|french|german|chinese|japanese)',
            r'\bfrom\s+(english|spanish|french|german|chinese|japanese)'
        ],
        QueryType.WEATHER: [
            r'\bweather\b',
            r'\btemperature\b',
            r'\bforecast\b',
            r'\bclimate\b',
            r'\bprecipitation\b',
            r'\bhumidity\b',
            r'\bwind\b',
            r'\bsunny\b',
            r'\brainy\b',
            r'\bsnow\b'
        ],
        QueryType.RECENT_NEWS: [
            r'\bnews\b',
            r'\brecent\b',
            r'\blatest\b',
            r'\bbreaking\b',
            r'\bheadlines\b',
            r'\btoday\b',
            r'\byesterday\b',
            r'\bthis week\b'
        ],
        QueryType.PEOPLE: [
            r'\bwho is\b',
            r'\bwho was\b',
            r'\bbiography\b',
            r'\bbio\b',
            r'\blife of\b',
            r'\babout\s+[A-Z][a-z]+\s+[A-Z][a-z]+'
        ],
        QueryType.CODING: [
            r'\bcode\b',
            r'\bprogramming\b',
            r'\bhow to\s+(implement|write|create|build|code)',
            r'\bpython\b',
            r'\bjavascript\b',
            r'\bjava\b',
            r'\bc\+\+\b',
            r'\bfunction\b',
            r'\balgorithm\b',
            r'\bexample\b.*\bcode\b'
        ],
        QueryType.COOKING_RECIPES: [
            r'\brecipe\b',
            r'\bcooking\b',
            r'\bhow to cook\b',
            r'\bhow to make\b',
            r'\bingredients\b',
            r'\bkitchen\b'
        ],
        QueryType.ACADEMIC_RESEARCH: [
            r'\bresearch\b',
            r'\bacademic\b',
            r'\bstudy\b',
            r'\bpaper\b',
            r'\bthesis\b',
            r'\bdissertation\b',
            r'\bscholar\b',
            r'\bpeer-reviewed\b'
        ],
        QueryType.SCIENCE_MATH: [
            r'\bcalculate\b',
            r'\bcompute\b',
            r'\bsolve\b',
            r'\bmath\b',
            r'\bmathematical\b',
            r'\bequation\b',
            r'\bformula\b',
            r'[0-9]+\s*[+\-*/]\s*[0-9]+'
        ],
    }
    
    CREATIVE_WRITING_KEYWORDS = [
        'write a', 'create a story', 'poem', 'essay', 'fiction'
    ]
    
    def detect(
        self,
        query: str,
        search_results: Optional[List[SearchResult]] = None
    ) -> QueryType:
        """
        Detect the query type based on patterns and context.
        
        Args:
            query: The user's query string
            search_results: Optional search results for context
            
        Returns:
            Detected QueryType
        """
        query_lower = query.lower().strip()
        
        # Check patterns in priority order
        for query_type, patterns in self.PATTERNS.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                return query_type
        
        # Check for creative writing
        if any(keyword in query_lower for keyword in self.CREATIVE_WRITING_KEYWORDS):
            return QueryType.CREATIVE_WRITING
        
        return QueryType.GENERAL
    
    def get_confidence(self, query: str, query_type: QueryType) -> float:
        """
        Get confidence score for detected query type.
        
        Args:
            query: The query string
            query_type: The detected query type
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if query_type == QueryType.GENERAL:
            return 0.5
        
        query_lower = query.lower()
        patterns = self.PATTERNS.get(query_type, [])
        
        if not patterns:
            return 0.5
        
        matches = sum(1 for pattern in patterns if re.search(pattern, query_lower))
        return min(1.0, matches / len(patterns) * 2.0)




