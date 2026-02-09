"""
Matcher Utilities for Piel Mejorador AI SAM3
===========================================

Unified pattern matching utilities.
"""

import re
import logging
from typing import TypeVar, Callable, Any, Optional, List, Dict, Pattern
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Matcher(ABC):
    """Base matcher interface."""
    
    @abstractmethod
    def matches(self, value: str) -> bool:
        """Check if value matches pattern."""
        pass


class RegexMatcher(Matcher):
    """Regex pattern matcher."""
    
    def __init__(self, pattern: str, flags: int = 0):
        """
        Initialize regex matcher.
        
        Args:
            pattern: Regex pattern
            flags: Regex flags
        """
        self._pattern = re.compile(pattern, flags)
        self.pattern = pattern
    
    def matches(self, value: str) -> bool:
        """Check if value matches pattern."""
        return bool(self._pattern.search(value))
    
    def findall(self, value: str) -> List[str]:
        """
        Find all matches.
        
        Args:
            value: Value to search
            
        Returns:
            List of matches
        """
        return self._pattern.findall(value)
    
    def match(self, value: str) -> Optional[re.Match]:
        """
        Match pattern.
        
        Args:
            value: Value to match
            
        Returns:
            Match object or None
        """
        return self._pattern.match(value)


class WildcardMatcher(Matcher):
    """Wildcard pattern matcher."""
    
    def __init__(self, pattern: str):
        """
        Initialize wildcard matcher.
        
        Args:
            pattern: Wildcard pattern (* for any, ? for single char)
        """
        self.pattern = pattern
        # Convert wildcard to regex
        regex_pattern = pattern.replace('.', r'\.')
        regex_pattern = regex_pattern.replace('*', '.*')
        regex_pattern = regex_pattern.replace('?', '.')
        self._regex = re.compile(f'^{regex_pattern}$')
    
    def matches(self, value: str) -> bool:
        """Check if value matches pattern."""
        return bool(self._regex.match(value))


class PrefixMatcher(Matcher):
    """Prefix pattern matcher."""
    
    def __init__(self, prefix: str):
        """
        Initialize prefix matcher.
        
        Args:
            prefix: Prefix to match
        """
        self.prefix = prefix
    
    def matches(self, value: str) -> bool:
        """Check if value matches pattern."""
        return value.startswith(self.prefix)


class SuffixMatcher(Matcher):
    """Suffix pattern matcher."""
    
    def __init__(self, suffix: str):
        """
        Initialize suffix matcher.
        
        Args:
            suffix: Suffix to match
        """
        self.suffix = suffix
    
    def matches(self, value: str) -> bool:
        """Check if value matches pattern."""
        return value.endswith(self.suffix)


class ContainsMatcher(Matcher):
    """Contains pattern matcher."""
    
    def __init__(self, substring: str):
        """
        Initialize contains matcher.
        
        Args:
            substring: Substring to match
        """
        self.substring = substring
    
    def matches(self, value: str) -> bool:
        """Check if value matches pattern."""
        return self.substring in value


class FunctionMatcher(Matcher):
    """Function-based matcher."""
    
    def __init__(self, match_func: Callable[[str], bool], name: Optional[str] = None):
        """
        Initialize function matcher.
        
        Args:
            match_func: Matching function
            name: Optional matcher name
        """
        self._match_func = match_func
        self.name = name or match_func.__name__
    
    def matches(self, value: str) -> bool:
        """Check if value matches pattern."""
        return self._match_func(value)


class CompositeMatcher(Matcher):
    """Composite matcher that combines multiple matchers."""
    
    def __init__(self, *matchers: Matcher, require_all: bool = True):
        """
        Initialize composite matcher.
        
        Args:
            *matchers: Matchers to combine
            require_all: If True, all must match; if False, any must match
        """
        self._matchers = matchers
        self._require_all = require_all
    
    def matches(self, value: str) -> bool:
        """Check if value matches pattern."""
        results = [matcher.matches(value) for matcher in self._matchers]
        
        if self._require_all:
            return all(results)
        return any(results)


class MatcherUtils:
    """Unified matcher utilities."""
    
    @staticmethod
    def create_regex_matcher(pattern: str, flags: int = 0) -> RegexMatcher:
        """
        Create regex matcher.
        
        Args:
            pattern: Regex pattern
            flags: Regex flags
            
        Returns:
            RegexMatcher
        """
        return RegexMatcher(pattern, flags)
    
    @staticmethod
    def create_wildcard_matcher(pattern: str) -> WildcardMatcher:
        """
        Create wildcard matcher.
        
        Args:
            pattern: Wildcard pattern
            
        Returns:
            WildcardMatcher
        """
        return WildcardMatcher(pattern)
    
    @staticmethod
    def create_prefix_matcher(prefix: str) -> PrefixMatcher:
        """
        Create prefix matcher.
        
        Args:
            prefix: Prefix to match
            
        Returns:
            PrefixMatcher
        """
        return PrefixMatcher(prefix)
    
    @staticmethod
    def create_suffix_matcher(suffix: str) -> SuffixMatcher:
        """
        Create suffix matcher.
        
        Args:
            suffix: Suffix to match
            
        Returns:
            SuffixMatcher
        """
        return SuffixMatcher(suffix)
    
    @staticmethod
    def create_contains_matcher(substring: str) -> ContainsMatcher:
        """
        Create contains matcher.
        
        Args:
            substring: Substring to match
            
        Returns:
            ContainsMatcher
        """
        return ContainsMatcher(substring)
    
    @staticmethod
    def create_function_matcher(
        match_func: Callable[[str], bool],
        name: Optional[str] = None
    ) -> FunctionMatcher:
        """
        Create function matcher.
        
        Args:
            match_func: Matching function
            name: Optional matcher name
            
        Returns:
            FunctionMatcher
        """
        return FunctionMatcher(match_func, name)
    
    @staticmethod
    def create_composite_matcher(
        *matchers: Matcher,
        require_all: bool = True
    ) -> CompositeMatcher:
        """
        Create composite matcher.
        
        Args:
            *matchers: Matchers to combine
            require_all: If True, all must match
            
        Returns:
            CompositeMatcher
        """
        return CompositeMatcher(*matchers, require_all=require_all)


# Convenience functions
def create_regex_matcher(pattern: str, **kwargs) -> RegexMatcher:
    """Create regex matcher."""
    return MatcherUtils.create_regex_matcher(pattern, **kwargs)


def create_wildcard_matcher(pattern: str) -> WildcardMatcher:
    """Create wildcard matcher."""
    return MatcherUtils.create_wildcard_matcher(pattern)


def create_prefix_matcher(prefix: str) -> PrefixMatcher:
    """Create prefix matcher."""
    return MatcherUtils.create_prefix_matcher(prefix)




