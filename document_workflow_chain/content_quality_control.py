"""
Content Quality Control - Advanced quality assurance system for document workflow chains
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import json
import uuid
from collections import defaultdict, Counter
import difflib

logger = logging.getLogger(__name__)

class QualityLevel(Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

class QualityCheckType(Enum):
    """Types of quality checks"""
    GRAMMAR = "grammar"
    SPELLING = "spelling"
    STYLE = "style"
    CONSISTENCY = "consistency"
    READABILITY = "readability"
    FACTUAL = "factual"
    PLAGIARISM = "plagiarism"
    SEO = "seo"
    STRUCTURE = "structure"
    TONE = "tone"
    LENGTH = "length"
    FORMATTING = "formatting"

@dataclass
class QualityIssue:
    """Represents a quality issue found in content"""
    issue_id: str
    check_type: QualityCheckType
    severity: QualityLevel
    message: str
    location: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None
    confidence: float = 1.0
    auto_fixable: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    report_id: str
    content_id: str
    overall_score: float
    quality_level: QualityLevel
    issues: List[QualityIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    processing_time: float = 0.0

@dataclass
class QualityRule:
    """Defines a quality control rule"""
    rule_id: str
    name: str
    description: str
    check_type: QualityCheckType
    severity: QualityLevel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

class ContentQualityController:
    """Advanced content quality control system"""
    
    def __init__(self):
        self.quality_rules: Dict[str, QualityRule] = {}
        self.quality_history: List[QualityReport] = []
        self.quality_metrics: Dict[str, Any] = defaultdict(list)
        self.auto_fix_enabled = True
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: 0.9,
            QualityLevel.GOOD: 0.7,
            QualityLevel.FAIR: 0.5,
            QualityLevel.POOR: 0.3,
            QualityLevel.CRITICAL: 0.0
        }
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info("ContentQualityController initialized")

    def _initialize_default_rules(self):
        """Initialize default quality control rules"""
        default_rules = [
            {
                "name": "Grammar Check",
                "description": "Check for grammatical errors",
                "check_type": QualityCheckType.GRAMMAR,
                "severity": QualityLevel.POOR,
                "config": {"min_confidence": 0.8}
            },
            {
                "name": "Spelling Check",
                "description": "Check for spelling errors",
                "check_type": QualityCheckType.SPELLING,
                "severity": QualityLevel.POOR,
                "config": {"min_confidence": 0.9}
            },
            {
                "name": "Readability Check",
                "description": "Ensure content is readable",
                "check_type": QualityCheckType.READABILITY,
                "severity": QualityLevel.FAIR,
                "config": {"max_flesch_score": 60, "min_flesch_score": 30}
            },
            {
                "name": "Length Check",
                "description": "Check content length appropriateness",
                "check_type": QualityCheckType.LENGTH,
                "severity": QualityLevel.FAIR,
                "config": {"min_words": 100, "max_words": 2000}
            },
            {
                "name": "Structure Check",
                "description": "Check content structure and organization",
                "check_type": QualityCheckType.STRUCTURE,
                "severity": QualityLevel.GOOD,
                "config": {"require_headings": True, "min_paragraphs": 3}
            },
            {
                "name": "SEO Check",
                "description": "Check SEO optimization",
                "check_type": QualityCheckType.SEO,
                "severity": QualityLevel.GOOD,
                "config": {"require_meta_description": True, "max_title_length": 60}
            }
        ]
        
        for rule_data in default_rules:
            rule = QualityRule(
                rule_id=str(uuid.uuid4()),
                **rule_data
            )
            self.quality_rules[rule.rule_id] = rule

    async def analyze_content_quality(
        self,
        content: str,
        content_id: str,
        content_type: str = "general",
        custom_rules: Optional[List[QualityRule]] = None
    ) -> QualityReport:
        """Perform comprehensive quality analysis on content"""
        start_time = datetime.utcnow()
        report_id = str(uuid.uuid4())
        
        # Get applicable rules
        rules = list(self.quality_rules.values())
        if custom_rules:
            rules.extend(custom_rules)
            
        # Filter rules by content type
        applicable_rules = [r for r in rules if r.enabled and self._is_rule_applicable(r, content_type)]
        
        # Perform quality checks
        issues = []
        for rule in applicable_rules:
            rule_issues = await self._check_rule(rule, content, content_id)
            issues.extend(rule_issues)
            
        # Calculate overall score
        overall_score = self._calculate_overall_score(issues)
        quality_level = self._determine_quality_level(overall_score)
        
        # Generate metrics
        metrics = self._generate_quality_metrics(content, issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, metrics)
        
        # Create report
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        report = QualityReport(
            report_id=report_id,
            content_id=content_id,
            overall_score=overall_score,
            quality_level=quality_level,
            issues=issues,
            metrics=metrics,
            recommendations=recommendations,
            processing_time=processing_time
        )
        
        # Store report
        self.quality_history.append(report)
        self._update_quality_metrics(report)
        
        logger.info(f"Quality analysis completed for content {content_id}: {quality_level.value} ({overall_score:.2f})")
        return report

    async def _check_rule(self, rule: QualityRule, content: str, content_id: str) -> List[QualityIssue]:
        """Check a specific quality rule against content"""
        issues = []
        
        try:
            if rule.check_type == QualityCheckType.GRAMMAR:
                issues = await self._check_grammar(rule, content)
            elif rule.check_type == QualityCheckType.SPELLING:
                issues = await self._check_spelling(rule, content)
            elif rule.check_type == QualityCheckType.READABILITY:
                issues = await self._check_readability(rule, content)
            elif rule.check_type == QualityCheckType.LENGTH:
                issues = await self._check_length(rule, content)
            elif rule.check_type == QualityCheckType.STRUCTURE:
                issues = await self._check_structure(rule, content)
            elif rule.check_type == QualityCheckType.SEO:
                issues = await self._check_seo(rule, content)
            elif rule.check_type == QualityCheckType.CONSISTENCY:
                issues = await self._check_consistency(rule, content)
            elif rule.check_type == QualityCheckType.TONE:
                issues = await self._check_tone(rule, content)
            elif rule.check_type == QualityCheckType.FORMATTING:
                issues = await self._check_formatting(rule, content)
                
        except Exception as e:
            logger.error(f"Error checking rule {rule.rule_id}: {e}")
            
        return issues

    async def _check_grammar(self, rule: QualityRule, content: str) -> List[QualityIssue]:
        """Check grammar issues"""
        issues = []
        
        # Basic grammar patterns (can be enhanced with NLP libraries)
        grammar_patterns = [
            (r'\b(a|an)\s+[aeiouAEIOU]', "Use 'an' before vowel sounds"),
            (r'\b(its|it\'s)\b', "Check 'its' vs 'it's' usage"),
            (r'\b(there|their|they\'re)\b', "Check 'there', 'their', 'they're' usage"),
            (r'\b(your|you\'re)\b', "Check 'your' vs 'you're' usage"),
            (r'\b(loose|lose)\b', "Check 'loose' vs 'lose' usage"),
        ]
        
        for pattern, suggestion in grammar_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                issue = QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    check_type=QualityCheckType.GRAMMAR,
                    severity=rule.severity,
                    message=f"Potential grammar issue: {match.group()}",
                    location={"start": match.start(), "end": match.end(), "text": match.group()},
                    suggestion=suggestion,
                    confidence=0.7,
                    auto_fixable=True
                )
                issues.append(issue)
                
        return issues

    async def _check_spelling(self, rule: QualityRule, content: str) -> List[QualityIssue]:
        """Check spelling issues"""
        issues = []
        
        # Common misspellings (can be enhanced with spell check libraries)
        common_misspellings = {
            "recieve": "receive",
            "seperate": "separate",
            "definately": "definitely",
            "occured": "occurred",
            "accomodate": "accommodate",
            "begining": "beginning",
            "neccessary": "necessary",
            "priviledge": "privilege",
            "independant": "independent",
            "occassion": "occasion"
        }
        
        words = re.findall(r'\b\w+\b', content.lower())
        for word in words:
            if word in common_misspellings:
                issue = QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    check_type=QualityCheckType.SPELLING,
                    severity=rule.severity,
                    message=f"Possible misspelling: '{word}'",
                    suggestion=f"Did you mean '{common_misspellings[word]}'?",
                    confidence=0.9,
                    auto_fixable=True
                )
                issues.append(issue)
                
        return issues

    async def _check_readability(self, rule: QualityRule, content: str) -> List[QualityIssue]:
        """Check readability metrics"""
        issues = []
        
        # Calculate Flesch Reading Ease Score
        sentences = re.split(r'[.!?]+', content)
        words = re.findall(r'\b\w+\b', content)
        syllables = sum(self._count_syllables(word) for word in words)
        
        if len(sentences) > 0 and len(words) > 0:
            flesch_score = 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
            
            min_score = rule.config.get("min_flesch_score", 30)
            max_score = rule.config.get("max_flesch_score", 60)
            
            if flesch_score < min_score:
                issue = QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    check_type=QualityCheckType.READABILITY,
                    severity=rule.severity,
                    message=f"Content may be too difficult to read (Flesch score: {flesch_score:.1f})",
                    suggestion="Consider using shorter sentences and simpler words",
                    confidence=0.8,
                    auto_fixable=False
                )
                issues.append(issue)
            elif flesch_score > max_score:
                issue = QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    check_type=QualityCheckType.READABILITY,
                    severity=QualityLevel.FAIR,
                    message=f"Content may be too simple (Flesch score: {flesch_score:.1f})",
                    suggestion="Consider using more sophisticated language",
                    confidence=0.6,
                    auto_fixable=False
                )
                issues.append(issue)
                
        return issues

    async def _check_length(self, rule: QualityRule, content: str) -> List[QualityIssue]:
        """Check content length"""
        issues = []
        
        words = re.findall(r'\b\w+\b', content)
        word_count = len(words)
        
        min_words = rule.config.get("min_words", 100)
        max_words = rule.config.get("max_words", 2000)
        
        if word_count < min_words:
            issue = QualityIssue(
                issue_id=str(uuid.uuid4()),
                check_type=QualityCheckType.LENGTH,
                severity=rule.severity,
                message=f"Content is too short ({word_count} words)",
                suggestion=f"Consider adding more content (minimum: {min_words} words)",
                confidence=1.0,
                auto_fixable=False
            )
            issues.append(issue)
        elif word_count > max_words:
            issue = QualityIssue(
                issue_id=str(uuid.uuid4()),
                check_type=QualityCheckType.LENGTH,
                severity=QualityLevel.FAIR,
                message=f"Content is too long ({word_count} words)",
                suggestion=f"Consider shortening content (maximum: {max_words} words)",
                confidence=1.0,
                auto_fixable=False
            )
            issues.append(issue)
            
        return issues

    async def _check_structure(self, rule: QualityRule, content: str) -> List[QualityIssue]:
        """Check content structure"""
        issues = []
        
        # Check for headings
        if rule.config.get("require_headings", True):
            headings = re.findall(r'^#+\s+.+$', content, re.MULTILINE)
            if not headings:
                issue = QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    check_type=QualityCheckType.STRUCTURE,
                    severity=rule.severity,
                    message="Content lacks proper headings",
                    suggestion="Add headings to improve structure and readability",
                    confidence=0.9,
                    auto_fixable=False
                )
                issues.append(issue)
                
        # Check for paragraphs
        min_paragraphs = rule.config.get("min_paragraphs", 3)
        paragraphs = content.split('\n\n')
        if len(paragraphs) < min_paragraphs:
            issue = QualityIssue(
                issue_id=str(uuid.uuid4()),
                check_type=QualityCheckType.STRUCTURE,
                severity=rule.severity,
                message=f"Content has too few paragraphs ({len(paragraphs)})",
                suggestion=f"Consider breaking content into more paragraphs (minimum: {min_paragraphs})",
                confidence=0.8,
                auto_fixable=False
            )
            issues.append(issue)
            
        return issues

    async def _check_seo(self, rule: QualityRule, content: str) -> List[QualityIssue]:
        """Check SEO optimization"""
        issues = []
        
        # Check for title
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1)
            max_title_length = rule.config.get("max_title_length", 60)
            if len(title) > max_title_length:
                issue = QualityIssue(
                    issue_id=str(uuid.uuid4()),
                    check_type=QualityCheckType.SEO,
                    severity=rule.severity,
                    message=f"Title is too long ({len(title)} characters)",
                    suggestion=f"Shorten title to {max_title_length} characters or less",
                    confidence=0.9,
                    auto_fixable=False
                )
                issues.append(issue)
        else:
            issue = QualityIssue(
                issue_id=str(uuid.uuid4()),
                check_type=QualityCheckType.SEO,
                severity=rule.severity,
                message="Content lacks a title",
                suggestion="Add a descriptive title at the beginning",
                confidence=1.0,
                auto_fixable=False
            )
            issues.append(issue)
            
        return issues

    async def _check_consistency(self, rule: QualityRule, content: str) -> List[QualityIssue]:
        """Check content consistency"""
        issues = []
        
        # Check for consistent terminology
        words = re.findall(r'\b\w+\b', content.lower())
        word_counts = Counter(words)
        
        # Find potential inconsistencies (words that appear in different forms)
        for word, count in word_counts.items():
            if count > 1:
                # Check for variations
                variations = [w for w in word_counts.keys() if self._are_similar_words(word, w)]
                if len(variations) > 1:
                    issue = QualityIssue(
                        issue_id=str(uuid.uuid4()),
                        check_type=QualityCheckType.CONSISTENCY,
                        severity=rule.severity,
                        message=f"Inconsistent terminology: {', '.join(variations)}",
                        suggestion="Use consistent terminology throughout the content",
                        confidence=0.7,
                        auto_fixable=False
                    )
                    issues.append(issue)
                    
        return issues

    async def _check_tone(self, rule: QualityRule, content: str) -> List[QualityIssue]:
        """Check content tone"""
        issues = []
        
        # Basic tone analysis (can be enhanced with sentiment analysis)
        tone_indicators = {
            "formal": ["therefore", "furthermore", "consequently", "moreover"],
            "informal": ["yeah", "gonna", "wanna", "cool", "awesome"],
            "technical": ["algorithm", "implementation", "optimization", "architecture"],
            "casual": ["hey", "hi", "thanks", "cheers"]
        }
        
        content_lower = content.lower()
        detected_tones = []
        
        for tone, indicators in tone_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                detected_tones.append(tone)
                
        if len(detected_tones) > 1:
            issue = QualityIssue(
                issue_id=str(uuid.uuid4()),
                check_type=QualityCheckType.TONE,
                severity=QualityLevel.FAIR,
                message=f"Mixed tone detected: {', '.join(detected_tones)}",
                suggestion="Maintain consistent tone throughout the content",
                confidence=0.6,
                auto_fixable=False
            )
            issues.append(issue)
            
        return issues

    async def _check_formatting(self, rule: QualityRule, content: str) -> List[QualityIssue]:
        """Check content formatting"""
        issues = []
        
        # Check for proper line breaks
        if '\r\n' in content and '\n' in content:
            issue = QualityIssue(
                issue_id=str(uuid.uuid4()),
                check_type=QualityCheckType.FORMATTING,
                severity=QualityLevel.FAIR,
                message="Mixed line ending formats detected",
                suggestion="Use consistent line endings throughout the content",
                confidence=0.8,
                auto_fixable=True
            )
            issues.append(issue)
            
        # Check for excessive whitespace
        if re.search(r'\n\s*\n\s*\n', content):
            issue = QualityIssue(
                issue_id=str(uuid.uuid4()),
                check_type=QualityCheckType.FORMATTING,
                severity=QualityLevel.FAIR,
                message="Excessive whitespace detected",
                suggestion="Remove extra blank lines",
                confidence=0.9,
                auto_fixable=True
            )
            issues.append(issue)
            
        return issues

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
            
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
            
        return max(1, syllable_count)

    def _are_similar_words(self, word1: str, word2: str) -> bool:
        """Check if two words are similar (for consistency checking)"""
        if word1 == word2:
            return False
            
        # Check for common variations
        variations = [
            (word1 + 's', word2),  # plural
            (word1, word2 + 's'),
            (word1 + 'ing', word2),  # gerund
            (word1, word2 + 'ing'),
            (word1 + 'ed', word2),  # past tense
            (word1, word2 + 'ed'),
        ]
        
        for var1, var2 in variations:
            if var1 == var2:
                return True
                
        # Check similarity using difflib
        similarity = difflib.SequenceMatcher(None, word1, word2).ratio()
        return similarity > 0.8

    def _is_rule_applicable(self, rule: QualityRule, content_type: str) -> bool:
        """Check if a rule is applicable to the content type"""
        applicable_types = rule.config.get("content_types", ["general"])
        return content_type in applicable_types or "general" in applicable_types

    def _calculate_overall_score(self, issues: List[QualityIssue]) -> float:
        """Calculate overall quality score based on issues"""
        if not issues:
            return 1.0
            
        # Weight issues by severity
        severity_weights = {
            QualityLevel.CRITICAL: 1.0,
            QualityLevel.POOR: 0.8,
            QualityLevel.FAIR: 0.6,
            QualityLevel.GOOD: 0.4,
            QualityLevel.EXCELLENT: 0.2
        }
        
        total_weight = sum(severity_weights[issue.severity] * issue.confidence for issue in issues)
        max_possible_weight = len(issues) * 1.0  # Maximum weight per issue
        
        if max_possible_weight == 0:
            return 1.0
            
        score = 1.0 - (total_weight / max_possible_weight)
        return max(0.0, min(1.0, score))

    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level based on score"""
        for level, threshold in self.quality_thresholds.items():
            if score >= threshold:
                return level
        return QualityLevel.CRITICAL

    def _generate_quality_metrics(self, content: str, issues: List[QualityIssue]) -> Dict[str, Any]:
        """Generate detailed quality metrics"""
        words = re.findall(r'\b\w+\b', content)
        sentences = re.split(r'[.!?]+', content)
        paragraphs = content.split('\n\n')
        
        # Count issues by type
        issues_by_type = defaultdict(int)
        for issue in issues:
            issues_by_type[issue.check_type.value] += 1
            
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "average_words_per_sentence": len(words) / len(sentences) if sentences else 0,
            "issues_by_type": dict(issues_by_type),
            "total_issues": len(issues),
            "auto_fixable_issues": len([i for i in issues if i.auto_fixable]),
            "critical_issues": len([i for i in issues if i.severity == QualityLevel.CRITICAL]),
            "readability_score": self._calculate_flesch_score(content)
        }

    def _calculate_flesch_score(self, content: str) -> float:
        """Calculate Flesch Reading Ease Score"""
        sentences = re.split(r'[.!?]+', content)
        words = re.findall(r'\b\w+\b', content)
        syllables = sum(self._count_syllables(word) for word in words)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
            
        return 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))

    def _generate_recommendations(self, issues: List[QualityIssue], metrics: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Issue-based recommendations
        if issues:
            critical_issues = [i for i in issues if i.severity == QualityLevel.CRITICAL]
            if critical_issues:
                recommendations.append("Address critical issues first to improve content quality")
                
            auto_fixable = [i for i in issues if i.auto_fixable]
            if auto_fixable:
                recommendations.append(f"Consider auto-fixing {len(auto_fixable)} issues that can be automatically corrected")
                
        # Metric-based recommendations
        if metrics.get("average_words_per_sentence", 0) > 20:
            recommendations.append("Consider shortening sentences for better readability")
            
        if metrics.get("readability_score", 0) < 30:
            recommendations.append("Content may be too difficult to read - consider simplifying language")
        elif metrics.get("readability_score", 0) > 70:
            recommendations.append("Content may be too simple - consider using more sophisticated language")
            
        if metrics.get("paragraph_count", 0) < 3:
            recommendations.append("Consider breaking content into more paragraphs for better structure")
            
        return recommendations

    def _update_quality_metrics(self, report: QualityReport):
        """Update overall quality metrics"""
        self.quality_metrics["overall_scores"].append(report.overall_score)
        self.quality_metrics["processing_times"].append(report.processing_time)
        self.quality_metrics["issue_counts"].append(len(report.issues))
        
        # Keep only last 1000 reports for performance
        for key in self.quality_metrics:
            if len(self.quality_metrics[key]) > 1000:
                self.quality_metrics[key] = self.quality_metrics[key][-1000:]

    async def auto_fix_content(self, content: str, issues: List[QualityIssue]) -> Tuple[str, List[QualityIssue]]:
        """Automatically fix content issues where possible"""
        if not self.auto_fix_enabled:
            return content, issues
            
        fixed_content = content
        fixed_issues = []
        remaining_issues = []
        
        for issue in issues:
            if issue.auto_fixable:
                try:
                    fixed_content = await self._apply_fix(fixed_content, issue)
                    fixed_issues.append(issue)
                except Exception as e:
                    logger.error(f"Failed to auto-fix issue {issue.issue_id}: {e}")
                    remaining_issues.append(issue)
            else:
                remaining_issues.append(issue)
                
        return fixed_content, remaining_issues

    async def _apply_fix(self, content: str, issue: QualityIssue) -> str:
        """Apply a specific fix to content"""
        if issue.check_type == QualityCheckType.GRAMMAR:
            return self._fix_grammar(content, issue)
        elif issue.check_type == QualityCheckType.SPELLING:
            return self._fix_spelling(content, issue)
        elif issue.check_type == QualityCheckType.FORMATTING:
            return self._fix_formatting(content, issue)
        else:
            return content

    def _fix_grammar(self, content: str, issue: QualityIssue) -> str:
        """Fix grammar issues"""
        # Basic grammar fixes
        fixes = {
            r'\ba\s+([aeiouAEIOU])': r'an \1',
            r'\b(its)\b(?!\s*[a-z])': r"it's",
        }
        
        for pattern, replacement in fixes.items():
            content = re.sub(pattern, replacement, content)
            
        return content

    def _fix_spelling(self, content: str, issue: QualityIssue) -> str:
        """Fix spelling issues"""
        # Apply spelling corrections
        if issue.suggestion and "Did you mean" in issue.suggestion:
            # Extract the correct spelling from suggestion
            correct_word = issue.suggestion.split("'")[1]
            incorrect_word = issue.message.split("'")[1]
            content = re.sub(r'\b' + re.escape(incorrect_word) + r'\b', correct_word, content, flags=re.IGNORECASE)
            
        return content

    def _fix_formatting(self, content: str, issue: QualityIssue) -> str:
        """Fix formatting issues"""
        # Fix line endings
        content = content.replace('\r\n', '\n')
        
        # Fix excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        return content

    async def get_quality_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get quality trends over time"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_reports = [r for r in self.quality_history if r.created_at >= cutoff_date]
        
        if not recent_reports:
            return {"message": "No quality data available for the specified period"}
            
        # Calculate trends
        scores = [r.overall_score for r in recent_reports]
        issue_counts = [len(r.issues) for r in recent_reports]
        
        return {
            "period_days": days,
            "total_reports": len(recent_reports),
            "average_score": sum(scores) / len(scores),
            "score_trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "declining",
            "average_issues": sum(issue_counts) / len(issue_counts),
            "issue_trend": "decreasing" if len(issue_counts) > 1 and issue_counts[-1] < issue_counts[0] else "increasing",
            "quality_distribution": {
                level.value: len([r for r in recent_reports if r.quality_level == level])
                for level in QualityLevel
            }
        }

    async def create_custom_rule(
        self,
        name: str,
        description: str,
        check_type: QualityCheckType,
        severity: QualityLevel,
        config: Dict[str, Any]
    ) -> str:
        """Create a custom quality rule"""
        rule = QualityRule(
            rule_id=str(uuid.uuid4()),
            name=name,
            description=description,
            check_type=check_type,
            severity=severity,
            config=config
        )
        
        self.quality_rules[rule.rule_id] = rule
        logger.info(f"Created custom quality rule: {name}")
        return rule.rule_id

    async def update_rule(self, rule_id: str, **kwargs) -> bool:
        """Update a quality rule"""
        if rule_id not in self.quality_rules:
            return False
            
        rule = self.quality_rules[rule_id]
        for key, value in kwargs.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
                
        logger.info(f"Updated quality rule: {rule_id}")
        return True

    async def delete_rule(self, rule_id: str) -> bool:
        """Delete a quality rule"""
        if rule_id not in self.quality_rules:
            return False
            
        del self.quality_rules[rule_id]
        logger.info(f"Deleted quality rule: {rule_id}")
        return True

    async def get_rule(self, rule_id: str) -> Optional[QualityRule]:
        """Get a quality rule"""
        return self.quality_rules.get(rule_id)

    async def list_rules(self, enabled_only: bool = False) -> List[QualityRule]:
        """List all quality rules"""
        rules = list(self.quality_rules.values())
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        return rules

# Global quality controller instance
content_quality_controller = ContentQualityController()

# Convenience functions
async def analyze_content_quality(
    content: str,
    content_id: str,
    content_type: str = "general"
) -> QualityReport:
    """Analyze content quality"""
    return await content_quality_controller.analyze_content_quality(
        content, content_id, content_type
    )

async def auto_fix_content_issues(content: str, issues: List[QualityIssue]) -> Tuple[str, List[QualityIssue]]:
    """Auto-fix content issues"""
    return await content_quality_controller.auto_fix_content(content, issues)

async def get_quality_trends(days: int = 30) -> Dict[str, Any]:
    """Get quality trends"""
    return await content_quality_controller.get_quality_trends(days)