"""
Perplexity Query Processor - Processes queries and formats responses
====================================================================

Handles query type detection, search result processing, and response formatting
following Perplexity-style guidelines.
"""

import re
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from .perplexity_validator import PerplexityValidator, ValidationIssue
except ImportError:
    PerplexityValidator = None
    ValidationIssue = None

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries supported."""
    ACADEMIC_RESEARCH = "academic_research"
    RECENT_NEWS = "recent_news"
    WEATHER = "weather"
    PEOPLE = "people"
    CODING = "coding"
    COOKING_RECIPES = "cooking_recipes"
    TRANSLATION = "translation"
    CREATIVE_WRITING = "creative_writing"
    SCIENCE_MATH = "science_math"
    URL_LOOKUP = "url_lookup"
    GENERAL = "general"


@dataclass
class SearchResult:
    """Represents a search result from external sources."""
    index: int
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    timestamp: Optional[datetime] = None
    source: Optional[str] = None


@dataclass
class ProcessedQuery:
    """Processed query with metadata."""
    original_query: str
    query_type: QueryType
    search_results: List[SearchResult]
    requires_citations: bool = True
    metadata: Dict[str, Any] = None


class QueryTypeDetector:
    """Detects the type of query based on content and patterns."""
    
    # Patterns for different query types
    WEATHER_PATTERNS = [
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
    ]
    
    NEWS_PATTERNS = [
        r'\bnews\b',
        r'\brecent\b',
        r'\blatest\b',
        r'\bbreaking\b',
        r'\bheadlines\b',
        r'\btoday\b',
        r'\byesterday\b',
        r'\bthis week\b'
    ]
    
    PEOPLE_PATTERNS = [
        r'\bwho is\b',
        r'\bwho was\b',
        r'\bbiography\b',
        r'\bbio\b',
        r'\blife of\b',
        r'\babout\s+[A-Z][a-z]+\s+[A-Z][a-z]+'  # "about John Doe"
    ]
    
    CODING_PATTERNS = [
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
    ]
    
    COOKING_PATTERNS = [
        r'\brecipe\b',
        r'\bcooking\b',
        r'\bhow to cook\b',
        r'\bhow to make\b',
        r'\bingredients\b',
        r'\bkitchen\b'
    ]
    
    TRANSLATION_PATTERNS = [
        r'\btranslate\b',
        r'\btranslation\b',
        r'\bto\s+(english|spanish|french|german|chinese|japanese)',
        r'\bfrom\s+(english|spanish|french|german|chinese|japanese)'
    ]
    
    URL_PATTERNS = [
        r'https?://[^\s]+',
        r'www\.[^\s]+',
        r'[a-z0-9-]+\.(com|org|net|edu|gov|io|co)[^\s]*'
    ]
    
    ACADEMIC_PATTERNS = [
        r'\bresearch\b',
        r'\bacademic\b',
        r'\bstudy\b',
        r'\bpaper\b',
        r'\bthesis\b',
        r'\bdissertation\b',
        r'\bscholar\b',
        r'\bpeer-reviewed\b'
    ]
    
    MATH_PATTERNS = [
        r'\bcalculate\b',
        r'\bcompute\b',
        r'\bsolve\b',
        r'\bmath\b',
        r'\bmathematical\b',
        r'\bequation\b',
        r'\bformula\b',
        r'[0-9]+\s*[+\-*/]\s*[0-9]+'  # Simple arithmetic
    ]
    
    def detect(self, query: str, search_results: Optional[List[SearchResult]] = None) -> QueryType:
        """
        Detect the query type based on patterns and context.
        
        Args:
            query: The user's query string
            search_results: Optional search results for context
            
        Returns:
            Detected QueryType
        """
        query_lower = query.lower().strip()
        
        # Check for URL lookup first (highest priority)
        if re.search('|'.join(self.URL_PATTERNS), query_lower):
            return QueryType.URL_LOOKUP
        
        # Check for translation
        if any(re.search(pattern, query_lower) for pattern in self.TRANSLATION_PATTERNS):
            return QueryType.TRANSLATION
        
        # Check for weather
        if any(re.search(pattern, query_lower) for pattern in self.WEATHER_PATTERNS):
            return QueryType.WEATHER
        
        # Check for news
        if any(re.search(pattern, query_lower) for pattern in self.NEWS_PATTERNS):
            return QueryType.RECENT_NEWS
        
        # Check for people/biography
        if any(re.search(pattern, query_lower) for pattern in self.PEOPLE_PATTERNS):
            return QueryType.PEOPLE
        
        # Check for coding
        if any(re.search(pattern, query_lower) for pattern in self.CODING_PATTERNS):
            return QueryType.CODING
        
        # Check for cooking
        if any(re.search(pattern, query_lower) for pattern in self.COOKING_PATTERNS):
            return QueryType.COOKING_RECIPES
        
        # Check for academic research
        if any(re.search(pattern, query_lower) for pattern in self.ACADEMIC_PATTERNS):
            return QueryType.ACADEMIC_RESEARCH
        
        # Check for math/calculations
        if any(re.search(pattern, query_lower) for pattern in self.MATH_PATTERNS):
            return QueryType.SCIENCE_MATH
        
        # Check for creative writing indicators
        if any(keyword in query_lower for keyword in ['write a', 'create a story', 'poem', 'essay', 'fiction']):
            return QueryType.CREATIVE_WRITING
        
        return QueryType.GENERAL


class ResponseFormatter:
    """Formats responses according to Perplexity-style guidelines."""
    
    def __init__(self):
        self.current_date = datetime(2025, 5, 13, 4, 31, 29)  # From prompt
    
    def format_answer(
        self,
        query: str,
        query_type: QueryType,
        search_results: List[SearchResult],
        answer_content: str
    ) -> str:
        """
        Format the answer according to Perplexity guidelines.
        
        Ensures:
        - Answer never starts with a header
        - Citations have no space before them: [1][2]
        - LaTeX math uses \( and \[ (not $ or $$)
        - No References section at end
        - Query-type specific formatting applied
        
        Args:
            query: Original query
            query_type: Detected query type
            search_results: List of search results
            answer_content: The answer content to format
            
        Returns:
            Formatted answer string
        """
        # Normalize LaTeX math expressions (convert $ to \( and \[)
        formatted = self._normalize_latex(answer_content)
        
        # Ensure answer doesn't start with header
        formatted = self._ensure_no_leading_header(formatted)
        
        # Add citations to the answer content
        formatted = self._add_citations(formatted, search_results)
        
        # Apply query-type specific formatting
        if query_type == QueryType.ACADEMIC_RESEARCH:
            formatted = self._format_academic(formatted)
        elif query_type == QueryType.RECENT_NEWS:
            formatted = self._format_news(formatted, search_results)
        elif query_type == QueryType.WEATHER:
            formatted = self._format_weather(formatted)
        elif query_type == QueryType.PEOPLE:
            formatted = self._format_people(formatted)
        elif query_type == QueryType.CODING:
            formatted = self._format_coding(formatted)
        elif query_type == QueryType.COOKING_RECIPES:
            formatted = self._format_cooking(formatted)
        elif query_type == QueryType.SCIENCE_MATH:
            formatted = self._format_math(formatted)
        elif query_type == QueryType.URL_LOOKUP:
            formatted = self._format_url_lookup(formatted, search_results)
        elif query_type == QueryType.TRANSLATION:
            formatted = self._format_translation(formatted)
        elif query_type == QueryType.CREATIVE_WRITING:
            formatted = self._format_creative_writing(formatted)
        
        # Final cleanup: ensure no References section
        formatted = self._remove_references_section(formatted)
        
        # Ensure answer doesn't end with a question
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
        """Ensure answer doesn't start with a header (## or ###)."""
        lines = content.split('\n')
        if lines and (lines[0].startswith('##') or lines[0].startswith('###')):
            # Remove leading header, add intro text if needed
            content = '\n'.join(lines[1:])
            # If content is now empty or starts with another header, add placeholder intro
            if not content.strip() or content.strip().startswith('##'):
                content = "Here's a comprehensive answer to your query.\n\n" + content
        return content
    
    def _ensure_no_ending_question(self, content: str) -> str:
        """Ensure answer doesn't end with a question mark."""
        content = content.rstrip()
        if content.endswith('?'):
            # Remove trailing question and add period or summary
            content = content.rstrip('?').rstrip() + '.'
        return content
    
    def _add_citations(self, content: str, search_results: List[SearchResult]) -> str:
        """
        Add citations to content based on search results.
        Citations are added as [1][2], etc. at the end of sentences with NO SPACE.
        
        Uses keyword matching and similarity to determine which search results
        are relevant to each sentence.
        
        Ensures:
        - No space between last word and citation: "water12" not "water 12"
        - Each citation in separate brackets: [1][2] not [1,2]
        - Max 3 citations per sentence
        - Removes any References section at the end
        """
        if not search_results:
            return self._remove_references_section(content)
        
        # Remove any existing citation markers that might be incorrect
        content = re.sub(r'\[\d+\]', '', content)
        
        # Remove References section if present
        content = self._remove_references_section(content)
        
        # Split content into sentences (handle LaTeX math expressions)
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
                # Ensure no space between sentence and citation
                sentence_clean = sentence.rstrip()
                # Remove trailing punctuation if needed, then add citation
                cited_sentence = f"{sentence_clean}{citations}"
                cited_sentences.append(cited_sentence)
            else:
                cited_sentences.append(sentence)
        
        return ' '.join(cited_sentences)
    
    def _split_sentences_preserving_math(self, content: str) -> List[str]:
        """Split content into sentences while preserving LaTeX math expressions."""
        # Protect LaTeX expressions during splitting
        math_pattern = r'\\(?:\(|\[).*?\\(?:\)|\])'
        protected = []
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
    
    def _remove_references_section(self, content: str) -> str:
        """Remove References, Sources, or similar sections from the end."""
        # Remove References section
        content = re.sub(r'\n##?\s*References?\s*\n.*$', '', content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'\n##?\s*Sources?\s*\n.*$', '', content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'\n##?\s*Citations?\s*\n.*$', '', content, flags=re.IGNORECASE | re.DOTALL)
        # Remove any list of citations at the end
        content = re.sub(r'\n\n\[\d+\]:\s*.*$', '', content, flags=re.MULTILINE | re.DOTALL)
        return content
    
    def _find_relevant_sources(self, sentence: str, search_results: List[SearchResult]) -> List[int]:
        """
        Find relevant search result indices for a sentence.
        
        Uses keyword matching and content similarity.
        """
        sentence_lower = sentence.lower()
        relevant = []
        
        for result in search_results:
            score = 0
            
            # Check title keywords
            title_words = set(result.title.lower().split())
            sentence_words = set(sentence_lower.split())
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
    
    def _format_academic(self, content: str) -> str:
        """Format academic research answers."""
        # Ensure proper section headers and scientific formatting
        if not content.startswith('##'):
            # Add introduction if missing
            lines = content.split('\n')
            if lines and not lines[0].startswith('##'):
                content = f"## Introduction\n\n{content}"
        return content
    
    def _format_news(self, content: str, search_results: List[SearchResult]) -> str:
        """Format news answers with lists and timestamps."""
        # Group by topics and use lists
        # This is simplified - real implementation would parse and reorganize
        return content
    
    def _format_weather(self, content: str) -> str:
        """Format weather answers - keep it short."""
        # Truncate to essential forecast information
        lines = content.split('\n')
        if len(lines) > 5:
            content = '\n'.join(lines[:5])
        return content
    
    def _format_people(self, content: str) -> str:
        """Format people/biography answers."""
        # Ensure it doesn't start with header
        if content.startswith('##'):
            content = content.split('\n', 1)[1] if '\n' in content else content
        return content
    
    def _format_coding(self, content: str) -> str:
        """Format coding answers - code first, then explanation."""
        # Ensure code blocks are properly formatted
        return content
    
    def _format_cooking(self, content: str) -> str:
        """Format cooking recipe answers."""
        # Ensure step-by-step format with ingredients
        return content
    
    def _format_math(self, content: str) -> str:
        """Format math answers - just the result for simple calculations."""
        # For simple calculations, extract just the final result
        # Look for patterns like "= 42" or "The answer is 42"
        result_match = re.search(r'(?:=|is|equals?|result:?)\s*([0-9.+\-*/()]+)', content, re.IGNORECASE)
        if result_match:
            return result_match.group(1)
        return content
    
    def _format_url_lookup(self, content: str, search_results: List[SearchResult]) -> str:
        """Format URL lookup answers - only cite first result [1]."""
        if search_results:
            # Ensure all citations are [1] for URL lookup
            content = re.sub(r'\[\d+\]', '[1]', content)
            # Remove duplicate [1] citations
            content = re.sub(r'\[1\]\[1\]+', '[1]', content)
        return content
    
    def _format_translation(self, content: str) -> str:
        """Format translation answers - remove all citations."""
        # Remove all citations for translation queries
        content = re.sub(r'\[\d+\]', '', content)
        return content
    
    def _format_creative_writing(self, content: str) -> str:
        """Format creative writing answers - remove all citations."""
        # Remove all citations for creative writing queries
        content = re.sub(r'\[\d+\]', '', content)
        return content


class PromptBuilder:
    """Builds prompts for LLM based on Perplexity system prompt."""
    
    def __init__(self, system_prompt_path: Optional[str] = None):
        """
        Initialize prompt builder.
        
        Args:
            system_prompt_path: Path to SYSTEM_PROMPT.md file
        """
        if system_prompt_path is None:
            # Default to the SYSTEM_PROMPT.md in the same directory structure
            current_file = Path(__file__)
            system_prompt_path = current_file.parent.parent / "SYSTEM_PROMPT.md"
        
        self.system_prompt_path = Path(system_prompt_path)
        self._system_prompt = None
    
    def load_system_prompt(self) -> str:
        """Load the system prompt from file."""
        if self._system_prompt is None:
            if self.system_prompt_path.exists():
                self._system_prompt = self.system_prompt_path.read_text(encoding='utf-8')
            else:
                logger.warning(f"System prompt not found at {self.system_prompt_path}, using default")
                self._system_prompt = self._get_default_prompt()
        return self._system_prompt
    
    def _get_default_prompt(self) -> str:
        """Get default system prompt if file not found."""
        return """You are Perplexity, a helpful search assistant. Your goal is to write accurate, detailed, and comprehensive answers using provided search results. Format your answers with proper citations and follow Perplexity-style guidelines."""
    
    def build_prompt(
        self,
        query: str,
        processed_query: ProcessedQuery,
        include_planning: bool = True
    ) -> str:
        """
        Build a complete prompt for LLM answer generation.
        
        Args:
            query: Original user query
            processed_query: Processed query object
            include_planning: Whether to include planning instructions
            
        Returns:
            Complete prompt string
        """
        system_prompt = self.load_system_prompt()
        
        # Format search results
        search_results_text = self._format_search_results(processed_query.search_results)
        
        # Build query type instructions
        query_type_instructions = self._get_query_type_instructions(processed_query.query_type)
        
        # Build the prompt
        prompt_parts = [
            system_prompt,
            "",
            "## Query",
            query,
            "",
            "## Query Type",
            f"Detected type: {processed_query.query_type.value}",
            query_type_instructions,
            "",
            "## Search Results",
            search_results_text if search_results_text else "No search results provided.",
            "",
            "## Instructions",
            "Generate a comprehensive answer to the query using the search results above.",
            "Follow all formatting rules from the system prompt.",
            "",
            "**Citation Rules:**",
            "- Add citations [1][2][3] at the end of sentences (NO SPACE before citation)",
            "- Example: 'Ice is less dense than water12.' (not 'water 12')",
            "- Each citation in separate brackets: [1][2] not [1,2]",
            "- Maximum 3 citations per sentence",
            "- DO NOT include a References section, Sources list, or citations list at the end",
            "",
            "**LaTeX Math Rules:**",
            "- Use \\( for inline math and \\[ for block math",
            "- Example: \\(x^4=x-3\\) for inline, \\[x^2+y^2=z^2\\] for block",
            "- NEVER use $ or $$ even if present in the query",
            "- NEVER use unicode for math, ALWAYS use LaTeX",
            "- NEVER use \\label instruction",
            "",
            "**Formatting Rules:**",
            "- NEVER start answer with a header (## or ###)",
            "- Begin with a few sentences summary",
            "- Use Level 2 headers (##) for sections",
            "- Use flat lists, prefer unordered lists",
            "- NEVER have a list with only one bullet",
            "- NEVER end answer with a question",
            "- NEVER use emojis",
            "",
            "**Restrictions:**",
            "- NEVER use moralization or hedging language",
            "- AVOID: 'It is important to...', 'It is inappropriate...', 'It is subjective...'",
            "- NEVER say 'based on search results' or 'based on browser history'",
            "- NEVER refer to knowledge cutoff date or who trained you",
        ]
        
        if include_planning:
            prompt_parts.extend([
                "",
                "## Planning",
                "Before writing your answer, briefly plan:",
                "1. Determine which search results are most relevant",
                "2. Identify key information to include",
                "3. Structure your answer according to the query type",
                "4. Plan where to add citations",
            ])
        
        return "\n".join(prompt_parts)
    
    def _format_search_results(self, search_results: List[SearchResult]) -> str:
        """Format search results for the prompt."""
        if not search_results:
            return ""
        
        formatted = []
        for result in search_results:
            formatted.append(f"[{result.index}] {result.title}")
            formatted.append(f"URL: {result.url}")
            formatted.append(f"Snippet: {result.snippet}")
            if result.content:
                # Truncate long content
                content = result.content[:500] + "..." if len(result.content) > 500 else result.content
                formatted.append(f"Content: {content}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _get_query_type_instructions(self, query_type: QueryType) -> str:
        """Get specific instructions for query type."""
        instructions = {
            QueryType.ACADEMIC_RESEARCH: (
                "Format as a scientific write-up with sections and detailed paragraphs. "
                "Provide long and detailed answers with proper markdown headings."
            ),
            QueryType.RECENT_NEWS: (
                "Concisely summarize recent news events, grouping them by topics. "
                "Always use lists and highlight the news title at the beginning of each list item. "
                "Select news from diverse perspectives while prioritizing trustworthy sources. "
                "If several search results mention the same event, combine them and cite all. "
                "Prioritize more recent events, comparing timestamps."
            ),
            QueryType.WEATHER: (
                "Keep answer very short, only provide the weather forecast. "
                "If search results don't contain relevant weather information, state that you don't have the answer."
            ),
            QueryType.PEOPLE: (
                "Write a short, comprehensive biography. "
                "NEVER start your answer with the person's name as a header. "
                "If search results refer to different people, describe each person individually and avoid mixing information."
            ),
            QueryType.CODING: (
                "Write code first in markdown code blocks with language identifier (e.g., ```python), then explain. "
                "Use appropriate syntax highlighting."
            ),
            QueryType.COOKING_RECIPES: (
                "Provide step-by-step cooking recipes. "
                "Clearly specify each ingredient, the amount, and precise instructions during each step."
            ),
            QueryType.TRANSLATION: (
                "Do not cite any search results. Just provide the translation."
            ),
            QueryType.CREATIVE_WRITING: (
                "Do not use or cite search results. Ignore general instructions pertaining only to search. "
                "Follow the user's instructions precisely to write exactly what they need."
            ),
            QueryType.SCIENCE_MATH: (
                "For simple calculations, only answer with the final result."
            ),
            QueryType.URL_LOOKUP: (
                "Rely solely on information from the first search result. "
                "DO NOT cite other search results. ALWAYS cite with [1] only. "
                "If query consists only of a URL, summarize the content of that URL."
            ),
            QueryType.GENERAL: "Follow general formatting guidelines.",
        }
        return instructions.get(query_type, instructions[QueryType.GENERAL])


class PerplexityProcessor:
    """Main processor for Perplexity-style query handling."""
    
    def __init__(self, system_prompt_path: Optional[str] = None, enable_validation: bool = True):
        """
        Initialize processor.
        
        Args:
            system_prompt_path: Path to SYSTEM_PROMPT.md file
            enable_validation: Whether to enable response validation
        """
        self.type_detector = QueryTypeDetector()
        self.formatter = ResponseFormatter()
        self.prompt_builder = PromptBuilder(system_prompt_path)
        self.validator = PerplexityValidator() if enable_validation and PerplexityValidator else None
    
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
                search_result_objects.append(SearchResult(
                    index=idx,
                    title=result.get('title', ''),
                    url=result.get('url', ''),
                    snippet=result.get('snippet', ''),
                    content=result.get('content'),
                    source=result.get('source')
                ))
        
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
        return self.prompt_builder.build_prompt(
            query=processed_query.original_query,
            processed_query=processed_query,
            include_planning=include_planning
        )
    
    async def generate_answer(
        self,
        processed_query: ProcessedQuery,
        llm_provider: Optional[Any] = None
    ) -> str:
        """
        Generate an answer using LLM.
        
        Args:
            processed_query: ProcessedQuery object
            llm_provider: Optional LLM provider (LLMPipeline, OpenAI client, etc.)
            
        Returns:
            Generated answer string
        """
        # Build prompt
        prompt = self.build_llm_prompt(processed_query)
        
        # If no LLM provider, return placeholder
        if llm_provider is None:
            logger.warning("No LLM provider specified, returning placeholder answer")
            return self._generate_placeholder_answer(processed_query)
        
        # Try to use LLM provider
        try:
            import asyncio
            
            # Check if it's an LLMPipeline with async generate
            if hasattr(llm_provider, 'generate'):
                if asyncio.iscoroutinefunction(llm_provider.generate):
                    answer = await llm_provider.generate(prompt)
                else:
                    # Sync method, run in executor
                    loop = asyncio.get_event_loop()
                    answer = await loop.run_in_executor(None, llm_provider.generate, prompt)
            # Check if it's an async callable
            elif callable(llm_provider) and asyncio.iscoroutinefunction(llm_provider):
                answer = await llm_provider(prompt)
            else:
                # Try OpenAI-style client
                if hasattr(llm_provider, 'chat') and hasattr(llm_provider.chat, 'completions'):
                    if hasattr(llm_provider.chat.completions, 'create'):
                        # Check if create is async
                        create_method = llm_provider.chat.completions.create
                        if asyncio.iscoroutinefunction(create_method):
                            response = await create_method(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": self.prompt_builder.load_system_prompt()},
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
                                        {"role": "system", "content": self.prompt_builder.load_system_prompt()},
                                        {"role": "user", "content": prompt}
                                    ]
                                )
                            )
                        answer = response.choices[0].message.content
                    else:
                        raise ValueError("Unsupported LLM provider: no create method")
                else:
                    raise ValueError("Unsupported LLM provider type")
            
            # Format the answer
            formatted = self.format_response(processed_query, answer)
            
            # Validate if validator is enabled
            if self.validator:
                is_valid, issues = self.validator.validate(formatted, processed_query.query_type.value)
                if not is_valid:
                    logger.warning(f"Answer validation found {len(issues)} issues")
                    # Log issues but don't fail - formatting should have caught most
                    for issue in issues:
                        if issue.level.value == "error":
                            logger.error(f"Validation error: {issue.message} - {issue.suggestion}")
                        else:
                            logger.warning(f"Validation warning: {issue.message}")
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}", exc_info=True)
            return self._generate_placeholder_answer(processed_query)
    
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
            is_valid, issues = self.validator.validate(formatted, processed_query.query_type.value)
            if not is_valid:
                logger.debug(f"Placeholder validation found {len(issues)} issues")
        
        return formatted
    
    def validate_response(self, answer: str, query_type: str = "general") -> Tuple[bool, List[Any]]:
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

