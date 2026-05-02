"""
ReAct Constants and Patterns

Centralized constants, patterns, and default values for the ReAct agent.
This module eliminates magic strings and provides a single source of truth.
"""

from typing import List


# ============================================================================
# Default Values
# ============================================================================

class Defaults:
    """Default configuration values."""
    MODEL_OPENAI = "gpt-3.5-turbo"
    MODEL_ANTHROPIC = "claude-3-sonnet-20240229"
    MAX_TOKENS = 1024
    MAX_ITERATIONS = 10
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 0.5  # Base delay for exponential backoff


# ============================================================================
# Keywords and Patterns
# ============================================================================

class FinishKeywords:
    """Keywords that indicate task completion."""
    KEYWORDS: List[str] = ["finish", "complete", "done", "task complete"]


class SearchKeywords:
    """Keywords that indicate search operations."""
    KEYWORDS: List[str] = ["search", "find", "look", "query"]


class CalculationKeywords:
    """Keywords that indicate calculation operations."""
    KEYWORDS: List[str] = ["calculate", "compute", "math", "sum"]


class ReadKeywords:
    """Keywords that indicate read operations."""
    KEYWORDS: List[str] = ["read", "get", "fetch", "retrieve"]


class WriteKeywords:
    """Keywords that indicate write operations."""
    KEYWORDS: List[str] = ["write", "save", "store", "create"]


# ============================================================================
# Regex Patterns
# ============================================================================

class ReActPatterns:
    """Regex patterns for parsing actions and extracting information."""
    
    # Function call patterns (ordered by specificity)
    FUNCTION_CALL_PATTERNS: List[str] = [
        r'Action:\s*(\w+)\(([^)]*)\)',  # "Action: tool_name(params)"
        r'action:\s*(\w+)\(([^)]*)\)',  # "action: tool_name(params)" (lowercase)
        r'(\w+)\(([^)]*)\)',            # Direct function call
    ]
    
    # JSON action patterns
    JSON_ACTION_PATTERN: str = r'\{[^{}]*"tool"[^{}]*\}'
    
    # Natural language action patterns
    NATURAL_ACTION_PATTERNS: List[str] = [
        r'use\s+(\w+)\s+(?:with|to|for)',
        r'call\s+(\w+)',
        r'execute\s+(\w+)',
        r'run\s+(\w+)',
    ]
    
    # Parameter extraction patterns
    PARAMETER_EXTRACTION_PATTERNS: List[str] = [
        r'(\w+)\s*=\s*["\']([^"\']+)["\']',  # key="value" or key='value'
        r'(\w+)\s*:\s*["\']([^"\']+)["\']',  # key: "value"
        r'(\w+)\s*is\s+["\']([^"\']+)["\']',  # key is "value"
    ]
    
    # Query extraction patterns
    QUERY_EXTRACTION_PATTERNS: List[str] = [
        r'search\s+for\s+["\']([^"\']+)["\']',
        r'find\s+["\']([^"\']+)["\']',
        r'query\s*[:=]\s*["\']([^"\']+)["\']',
    ]
    
    # Mathematical expression patterns
    MATH_EXPRESSION_PATTERN: str = r'(\d+\s*[+\-*/]\s*\d+)'
    
    # Resource extraction patterns
    RESOURCE_EXTRACTION_PATTERNS: List[str] = [
        r'["\']([^"\']+)["\']',      # Quoted strings
        r'/([^\s]+)',                 # File paths
        r'http[s]?://([^\s]+)',       # URLs
    ]


# ============================================================================
# Error Messages
# ============================================================================

class ErrorMessages:
    """Standardized error messages."""
    
    TOOL_NOT_FOUND = "Tool '{tool_name}' not found in registry"
    EXECUTION_FAILED = "Tool execution failed after {attempts} attempts: {error}"
    NO_ACTION_FOUND = "No actionable step found in thought"
    LLM_CALL_FAILED = "LLM call failed, using fallback: {error}"
    INVALID_THOUGHT = "I need to think about the next step."


# ============================================================================
# Observation Templates
# ============================================================================

class ObservationTemplates:
    """Templates for formatting observations."""
    
    SUCCESS_TEMPLATE = "Action '{tool}' succeeded. Result: {result}"
    ERROR_TEMPLATE = "Action '{tool}' failed. Error: {error}. I should try a different approach."
    TASK_COMPLETE = "Task completed successfully."
    CONTEXT_PREFIX = "Context: {context}"



