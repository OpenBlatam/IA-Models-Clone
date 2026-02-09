"""
Constants - Constants for Perplexity system
===========================================

Constants used throughout the Perplexity system.
"""

# Default values
DEFAULT_CACHE_TTL = 3600  # 1 hour
DEFAULT_CACHE_MAX_SIZE = 1000
DEFAULT_MAX_SEARCH_RESULTS = 50
DEFAULT_MAX_ANSWER_LENGTH = 10000

# LLM defaults
DEFAULT_LLM_MODEL = "gpt-4"
DEFAULT_LLM_TEMPERATURE = 0.7
DEFAULT_LLM_TIMEOUT = 60

# Citation limits
MAX_CITATIONS_PER_SENTENCE = 3
MIN_CITATION_RELEVANCE_SCORE = 2

# Query type priorities (for detection)
QUERY_TYPE_PRIORITIES = [
    "url_lookup",
    "translation",
    "weather",
    "recent_news",
    "people",
    "coding",
    "cooking_recipes",
    "academic_research",
    "science_math",
    "creative_writing",
    "general"
]

# Validation rules
VALIDATION_RULES = {
    "no_leading_header": True,
    "no_ending_question": True,
    "citation_no_space": True,
    "citation_separate_brackets": True,
    "citation_max_per_sentence": True,
    "no_references_section": True,
    "latex_no_dollar": True,
    "latex_no_label": True,
    "no_emojis": True,
    "forbidden_phrases": True,
    "list_formatting": True,
    "query_type_specific": True
}

# Forbidden phrases
FORBIDDEN_PHRASES = [
    "It is important to",
    "It is inappropriate",
    "It is subjective",
    "based on search results",
    "based on browser history"
]

# LaTeX patterns
LATEX_INLINE_PATTERN = r'\\(?:\(|\[).*?\\(?:\)|\])'
LATEX_DOLLAR_PATTERN = r'\$[^$]+\$'
LATEX_DOUBLE_DOLLAR_PATTERN = r'\$\$[^$]+\$\$'

# Citation patterns
CITATION_PATTERN = r'\[\d+\]'
CITATION_WITH_SPACE_PATTERN = r'\s+\[\d+\]'
CITATION_MULTIPLE_PATTERN = r'\[\d+,\s*\d+'

# System prompt default location
DEFAULT_SYSTEM_PROMPT_NAME = "SYSTEM_PROMPT.md"

# Current date (from prompt)
CURRENT_DATE = "Tuesday, May 13, 2025, 4:31:29 AM UTC"




