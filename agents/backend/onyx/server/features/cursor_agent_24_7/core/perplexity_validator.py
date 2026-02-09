"""
Perplexity Response Validator - Validates responses against formatting rules
================================================================================

Validates that generated answers comply with all Perplexity formatting rules
and restrictions.
"""

import re
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    level: ValidationLevel
    rule: str
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


class PerplexityValidator:
    """Validates responses against Perplexity formatting rules."""
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
    
    def validate(self, answer: str, query_type: str = "general") -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate an answer against all Perplexity rules.
        
        Args:
            answer: The answer text to validate
            query_type: The detected query type
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        self.issues = []
        
        # Check all rules
        self._check_no_leading_header(answer)
        self._check_no_ending_question(answer)
        self._check_citation_format(answer)
        self._check_no_references_section(answer)
        self._check_latex_format(answer)
        self._check_no_emojis(answer)
        self._check_no_forbidden_phrases(answer)
        self._check_list_formatting(answer)
        self._check_query_type_specific(answer, query_type)
        
        is_valid = all(issue.level != ValidationLevel.ERROR for issue in self.issues)
        return is_valid, self.issues
    
    def _check_no_leading_header(self, answer: str) -> None:
        """Check that answer doesn't start with a header."""
        lines = answer.strip().split('\n')
        if lines and (lines[0].startswith('##') or lines[0].startswith('###')):
            self.issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                rule="no_leading_header",
                message="Answer starts with a header (## or ###)",
                location="Beginning of answer",
                suggestion="Remove the header and start with summary sentences"
            ))
    
    def _check_no_ending_question(self, answer: str) -> None:
        """Check that answer doesn't end with a question."""
        answer_clean = answer.strip()
        if answer_clean.endswith('?'):
            self.issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                rule="no_ending_question",
                message="Answer ends with a question mark",
                location="End of answer",
                suggestion="Replace question with a statement or summary"
            ))
    
    def _check_citation_format(self, answer: str) -> None:
        """Check citation formatting rules."""
        # Check for space before citation
        space_before_citation = re.search(r'\s+\[\d+\]', answer)
        if space_before_citation:
            self.issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                rule="citation_no_space",
                message="Citation has space before bracket",
                location=f"Position {space_before_citation.start()}",
                suggestion="Remove space: 'water12' not 'water 12'"
            ))
        
        # Check for multiple citations in one bracket
        multiple_in_bracket = re.search(r'\[\d+,\s*\d+', answer)
        if multiple_in_bracket:
            self.issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                rule="citation_separate_brackets",
                message="Multiple citations in single bracket",
                location=f"Position {multiple_in_bracket.start()}",
                suggestion="Use separate brackets: '[1][2]' not '[1,2]'"
            ))
        
        # Check for more than 3 citations per sentence
        sentences = re.split(r'[.!?]\s+', answer)
        for i, sentence in enumerate(sentences):
            citations = re.findall(r'\[\d+\]', sentence)
            if len(citations) > 3:
                self.issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    rule="citation_max_per_sentence",
                    message=f"Sentence has {len(citations)} citations (max 3)",
                    location=f"Sentence {i+1}",
                    suggestion="Reduce to maximum 3 most relevant citations"
                ))
    
    def _check_no_references_section(self, answer: str) -> None:
        """Check that there's no References section."""
        if re.search(r'\n##?\s*References?\s*\n', answer, re.IGNORECASE):
            self.issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                rule="no_references_section",
                message="Answer contains a References section",
                location="End of answer",
                suggestion="Remove References section - citations are inline"
            ))
        
        if re.search(r'\n##?\s*Sources?\s*\n', answer, re.IGNORECASE):
            self.issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                rule="no_sources_section",
                message="Answer contains a Sources section",
                location="End of answer",
                suggestion="Remove Sources section - citations are inline"
            ))
    
    def _check_latex_format(self, answer: str) -> None:
        """Check LaTeX formatting rules."""
        # Check for $ or $$ usage
        if re.search(r'\$[^$]+\$', answer):
            self.issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                rule="latex_no_dollar",
                message="Answer uses $ for LaTeX (should use \\( and \\[)",
                location="Various",
                suggestion="Convert $...$ to \\(...\\) and $$...$$ to \\[...\\]"
            ))
        
        # Check for \label usage
        if re.search(r'\\label\{', answer):
            self.issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                rule="latex_no_label",
                message="Answer uses \\label instruction",
                location="Various",
                suggestion="Remove all \\label instructions"
            ))
    
    def _check_no_emojis(self, answer: str) -> None:
        """Check for emoji usage."""
        # Basic emoji pattern (Unicode ranges)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "]+", flags=re.UNICODE
        )
        
        if emoji_pattern.search(answer):
            self.issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                rule="no_emojis",
                message="Answer contains emojis",
                location="Various",
                suggestion="Remove all emojis from the answer"
            ))
    
    def _check_no_forbidden_phrases(self, answer: str) -> None:
        """Check for forbidden phrases."""
        forbidden_phrases = [
            (r'\bIt is important to\b', "It is important to"),
            (r'\bIt is inappropriate\b', "It is inappropriate"),
            (r'\bIt is subjective\b', "It is subjective"),
            (r'\bbased on search results\b', "based on search results"),
            (r'\bbased on browser history\b', "based on browser history"),
        ]
        
        for pattern, phrase in forbidden_phrases:
            if re.search(pattern, answer, re.IGNORECASE):
                self.issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    rule="forbidden_phrase",
                    message=f"Answer contains forbidden phrase: '{phrase}'",
                    location="Various",
                    suggestion=f"Remove or rephrase to avoid '{phrase}'"
                ))
    
    def _check_list_formatting(self, answer: str) -> None:
        """Check list formatting rules."""
        # Check for single bullet lists
        lines = answer.split('\n')
        in_list = False
        list_items = []
        
        for line in lines:
            if re.match(r'^[\s]*[-*+]\s+', line):
                if not in_list:
                    in_list = True
                    list_items = []
                list_items.append(line)
            elif in_list and line.strip() == '':
                # End of list
                if len(list_items) == 1:
                    self.issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        rule="no_single_bullet",
                        message="List contains only one item",
                        location="Various",
                        suggestion="Either add more items or convert to paragraph"
                    ))
                in_list = False
                list_items = []
    
    def _check_query_type_specific(self, answer: str, query_type: str) -> None:
        """Check query-type specific rules."""
        if query_type == "url_lookup":
            # Should only have [1] citations
            citations = set(re.findall(r'\[(\d+)\]', answer))
            if len(citations) > 1 or (len(citations) == 1 and '1' not in citations):
                self.issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    rule="url_lookup_citation",
                    message="URL lookup should only cite [1]",
                    location="Various",
                    suggestion="Replace all citations with [1] only"
                ))
        
        elif query_type in ["translation", "creative_writing"]:
            # Should have no citations
            if re.search(r'\[\d+\]', answer):
                self.issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    rule="no_citations_for_type",
                    message=f"{query_type} queries should not have citations",
                    location="Various",
                    suggestion="Remove all citations from the answer"
                ))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        error_count = sum(1 for issue in self.issues if issue.level == ValidationLevel.ERROR)
        warning_count = sum(1 for issue in self.issues if issue.level == ValidationLevel.WARNING)
        info_count = sum(1 for issue in self.issues if issue.level == ValidationLevel.INFO)
        
        return {
            "valid": error_count == 0,
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "total_issues": len(self.issues),
            "issues": [
                {
                    "level": issue.level.value,
                    "rule": issue.rule,
                    "message": issue.message,
                    "location": issue.location,
                    "suggestion": issue.suggestion
                }
                for issue in self.issues
            ]
        }

