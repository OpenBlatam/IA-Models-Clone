"""
Prompt Builder - Builds LLM prompts from system prompt and query context
========================================================================

Loads system prompt and builds complete prompts for LLM answer generation.
"""

import logging
from pathlib import Path
from typing import Optional, Dict
from .types import ProcessedQuery, QueryType

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds prompts for LLM based on Perplexity system prompt."""
    
    QUERY_TYPE_INSTRUCTIONS = {
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
    
    def __init__(self, system_prompt_path: Optional[str] = None):
        """
        Initialize prompt builder.
        
        Args:
            system_prompt_path: Path to SYSTEM_PROMPT.md file
        """
        if system_prompt_path is None:
            current_file = Path(__file__)
            system_prompt_path = current_file.parent.parent.parent / "SYSTEM_PROMPT.md"
        
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
        processed_query: ProcessedQuery,
        include_planning: bool = True
    ) -> str:
        """
        Build a complete prompt for LLM answer generation.
        
        Args:
            processed_query: ProcessedQuery object
            include_planning: Whether to include planning instructions
            
        Returns:
            Complete prompt string
        """
        system_prompt = self.load_system_prompt()
        search_results_text = self._format_search_results(processed_query.search_results)
        query_type_instructions = self.QUERY_TYPE_INSTRUCTIONS.get(
            processed_query.query_type,
            self.QUERY_TYPE_INSTRUCTIONS[QueryType.GENERAL]
        )
        
        prompt_parts = [
            system_prompt,
            "",
            "## Query",
            processed_query.original_query,
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
    
    def _format_search_results(self, search_results) -> str:
        """Format search results for the prompt."""
        if not search_results:
            return ""
        
        formatted = []
        for result in search_results:
            formatted.append(f"[{result.index}] {result.title}")
            formatted.append(f"URL: {result.url}")
            formatted.append(f"Snippet: {result.snippet}")
            if result.content:
                content = result.content[:500] + "..." if len(result.content) > 500 else result.content
                formatted.append(f"Content: {content}")
            formatted.append("")
        
        return "\n".join(formatted)




