"""
AI History Comparison System - Content Quality Assurance

This module provides advanced content quality assurance, validation,
and compliance checking capabilities.
"""

import logging
import re
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
from collections import defaultdict, Counter
import statistics

# Advanced NLP libraries
try:
    import spacy
    from spacy import displacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade, gunning_fog
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

from .ai_history_analyzer import AIHistoryAnalyzer, HistoryEntry

logger = logging.getLogger(__name__)

class QualityStandard(Enum):
    """Content quality standards"""
    ACADEMIC = "academic"
    BUSINESS = "business"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    JOURNALISTIC = "journalistic"
    MARKETING = "marketing"
    LEGAL = "legal"
    MEDICAL = "medical"

class ComplianceType(Enum):
    """Types of compliance checks"""
    ACCESSIBILITY = "accessibility"
    SEO = "seo"
    BRAND_GUIDELINES = "brand_guidelines"
    LEGAL_COMPLIANCE = "legal_compliance"
    FACT_CHECKING = "fact_checking"
    BIAS_DETECTION = "bias_detection"
    TONE_CONSISTENCY = "tone_consistency"

class QualityIssue(Enum):
    """Types of quality issues"""
    READABILITY_ISSUE = "readability_issue"
    GRAMMAR_ERROR = "grammar_error"
    SPELLING_ERROR = "spelling_error"
    TONE_INCONSISTENCY = "tone_inconsistency"
    BIAS_DETECTED = "bias_detected"
    FACTUAL_ERROR = "factual_error"
    ACCESSIBILITY_ISSUE = "accessibility_issue"
    SEO_ISSUE = "seo_issue"
    BRAND_VIOLATION = "brand_violation"
    LEGAL_ISSUE = "legal_issue"

@dataclass
class QualityCheck:
    """Result of a quality check"""
    check_type: str
    passed: bool
    score: float
    issues: List[str]
    recommendations: List[str]
    confidence: float
    checked_at: datetime

@dataclass
class ComplianceCheck:
    """Result of a compliance check"""
    compliance_type: ComplianceType
    passed: bool
    violations: List[str]
    recommendations: List[str]
    severity: str  # "low", "medium", "high", "critical"
    confidence: float
    checked_at: datetime

@dataclass
class QualityReport:
    """Comprehensive quality report"""
    overall_score: float
    quality_checks: List[QualityCheck]
    compliance_checks: List[ComplianceCheck]
    issues_summary: Dict[str, int]
    recommendations: List[str]
    quality_grade: str  # "A", "B", "C", "D", "F"
    report_timestamp: datetime

class ContentQualityAssurance:
    """
    Advanced content quality assurance and validation system
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize content quality assurance system"""
        self.config = config or {}
        self.nlp = None
        self.quality_standards = {}
        self.compliance_rules = {}
        
        # Initialize spaCy if available
        if HAS_SPACY:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded for quality assurance")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Initialize quality standards
        self._initialize_quality_standards()
        
        # Initialize compliance rules
        self._initialize_compliance_rules()
        
        logger.info("Content Quality Assurance system initialized")

    def comprehensive_quality_check(self, content: str, 
                                  quality_standard: QualityStandard = QualityStandard.BUSINESS,
                                  compliance_types: List[ComplianceType] = None) -> QualityReport:
        """Perform comprehensive quality and compliance check"""
        try:
            if compliance_types is None:
                compliance_types = list(ComplianceType)
            
            # Perform quality checks
            quality_checks = []
            
            # Readability check
            readability_check = self._check_readability(content, quality_standard)
            quality_checks.append(readability_check)
            
            # Grammar and spelling check
            grammar_check = self._check_grammar_and_spelling(content)
            quality_checks.append(grammar_check)
            
            # Tone consistency check
            tone_check = self._check_tone_consistency(content, quality_standard)
            quality_checks.append(tone_check)
            
            # Bias detection
            bias_check = self._check_bias(content)
            quality_checks.append(bias_check)
            
            # Fact checking (basic)
            fact_check = self._check_facts(content)
            quality_checks.append(fact_check)
            
            # Perform compliance checks
            compliance_checks = []
            for compliance_type in compliance_types:
                try:
                    compliance_check = self._perform_compliance_check(content, compliance_type)
                    compliance_checks.append(compliance_check)
                except Exception as e:
                    logger.warning(f"Compliance check failed for {compliance_type.value}: {e}")
                    continue
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(quality_checks, compliance_checks)
            
            # Generate issues summary
            issues_summary = self._generate_issues_summary(quality_checks, compliance_checks)
            
            # Generate recommendations
            recommendations = self._generate_quality_recommendations(quality_checks, compliance_checks)
            
            # Determine quality grade
            quality_grade = self._determine_quality_grade(overall_score)
            
            # Create quality report
            report = QualityReport(
                overall_score=overall_score,
                quality_checks=quality_checks,
                compliance_checks=compliance_checks,
                issues_summary=issues_summary,
                recommendations=recommendations,
                quality_grade=quality_grade,
                report_timestamp=datetime.now()
            )
            
            logger.info(f"Quality assurance check completed. Overall score: {overall_score:.2f}")
            return report
            
        except Exception as e:
            logger.error(f"Error in comprehensive quality check: {e}")
            return QualityReport(
                overall_score=0.0,
                quality_checks=[],
                compliance_checks=[],
                issues_summary={},
                recommendations=[f"Quality check failed: {str(e)}"],
                quality_grade="F",
                report_timestamp=datetime.now()
            )

    def check_content_standards(self, content: str, 
                              standard: QualityStandard) -> Dict[str, Any]:
        """Check content against specific quality standards"""
        try:
            standard_config = self.quality_standards.get(standard, {})
            
            # Readability check
            readability_score = self._calculate_readability_score(content)
            readability_passed = self._check_readability_standard(readability_score, standard)
            
            # Word count check
            word_count = len(content.split())
            word_count_passed = self._check_word_count_standard(word_count, standard)
            
            # Sentence structure check
            sentence_structure_passed = self._check_sentence_structure(content, standard)
            
            # Vocabulary check
            vocabulary_passed = self._check_vocabulary_standard(content, standard)
            
            # Overall compliance
            checks_passed = sum([
                readability_passed,
                word_count_passed,
                sentence_structure_passed,
                vocabulary_passed
            ])
            
            compliance_percentage = (checks_passed / 4) * 100
            
            return {
                "standard": standard.value,
                "compliance_percentage": compliance_percentage,
                "checks": {
                    "readability": {
                        "passed": readability_passed,
                        "score": readability_score,
                        "target": standard_config.get("readability_target", 60)
                    },
                    "word_count": {
                        "passed": word_count_passed,
                        "count": word_count,
                        "target_range": standard_config.get("word_count_range", [100, 1000])
                    },
                    "sentence_structure": {
                        "passed": sentence_structure_passed
                    },
                    "vocabulary": {
                        "passed": vocabulary_passed
                    }
                },
                "recommendations": self._generate_standard_recommendations(
                    standard, readability_passed, word_count_passed, 
                    sentence_structure_passed, vocabulary_passed
                )
            }
            
        except Exception as e:
            logger.error(f"Error checking content standards: {e}")
            return {"error": str(e)}

    def validate_content_compliance(self, content: str, 
                                  compliance_types: List[ComplianceType]) -> Dict[str, Any]:
        """Validate content compliance against specified types"""
        try:
            compliance_results = {}
            
            for compliance_type in compliance_types:
                try:
                    check_result = self._perform_compliance_check(content, compliance_type)
                    compliance_results[compliance_type.value] = {
                        "passed": check_result.passed,
                        "violations": check_result.violations,
                        "recommendations": check_result.recommendations,
                        "severity": check_result.severity,
                        "confidence": check_result.confidence
                    }
                except Exception as e:
                    logger.warning(f"Compliance check failed for {compliance_type.value}: {e}")
                    compliance_results[compliance_type.value] = {
                        "error": str(e),
                        "passed": False
                    }
            
            # Calculate overall compliance score
            passed_checks = sum(1 for result in compliance_results.values() 
                              if result.get("passed", False))
            total_checks = len(compliance_results)
            compliance_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
            
            return {
                "overall_compliance_score": compliance_score,
                "compliance_results": compliance_results,
                "summary": {
                    "total_checks": total_checks,
                    "passed_checks": passed_checks,
                    "failed_checks": total_checks - passed_checks
                }
            }
            
        except Exception as e:
            logger.error(f"Error validating content compliance: {e}")
            return {"error": str(e)}

    def generate_quality_improvement_plan(self, content: str, 
                                        current_quality_score: float,
                                        target_quality_score: float = 80.0) -> Dict[str, Any]:
        """Generate a plan to improve content quality"""
        try:
            # Analyze current content
            quality_report = self.comprehensive_quality_check(content)
            
            # Identify improvement areas
            improvement_areas = []
            priority_actions = []
            
            # Check readability
            if quality_report.overall_score < target_quality_score:
                readability_issues = [check for check in quality_report.quality_checks 
                                    if "readability" in check.check_type and not check.passed]
                if readability_issues:
                    improvement_areas.append("readability")
                    priority_actions.append({
                        "action": "improve_readability",
                        "priority": "high",
                        "description": "Improve content readability",
                        "specific_recommendations": readability_issues[0].recommendations
                    })
            
            # Check grammar and spelling
            grammar_issues = [check for check in quality_report.quality_checks 
                            if "grammar" in check.check_type and not check.passed]
            if grammar_issues:
                improvement_areas.append("grammar_spelling")
                priority_actions.append({
                    "action": "fix_grammar_spelling",
                    "priority": "high",
                    "description": "Fix grammar and spelling errors",
                    "specific_recommendations": grammar_issues[0].recommendations
                })
            
            # Check compliance issues
            critical_compliance_issues = [check for check in quality_report.compliance_checks 
                                        if check.severity in ["high", "critical"] and not check.passed]
            if critical_compliance_issues:
                improvement_areas.append("compliance")
                priority_actions.append({
                    "action": "address_compliance_issues",
                    "priority": "critical",
                    "description": "Address critical compliance issues",
                    "specific_recommendations": critical_compliance_issues[0].recommendations
                })
            
            # Generate improvement timeline
            improvement_timeline = self._generate_improvement_timeline(priority_actions)
            
            # Calculate expected improvement
            expected_improvement = self._calculate_expected_improvement(
                current_quality_score, target_quality_score, improvement_areas
            )
            
            return {
                "current_quality_score": current_quality_score,
                "target_quality_score": target_quality_score,
                "improvement_areas": improvement_areas,
                "priority_actions": priority_actions,
                "improvement_timeline": improvement_timeline,
                "expected_improvement": expected_improvement,
                "estimated_effort": self._estimate_improvement_effort(priority_actions),
                "success_metrics": self._define_success_metrics(target_quality_score)
            }
            
        except Exception as e:
            logger.error(f"Error generating improvement plan: {e}")
            return {"error": str(e)}

    def _initialize_quality_standards(self):
        """Initialize quality standards for different content types"""
        self.quality_standards = {
            QualityStandard.ACADEMIC: {
                "readability_target": 50,
                "word_count_range": [2000, 8000],
                "sentence_length_max": 25,
                "vocabulary_complexity": "high",
                "tone": "formal",
                "bias_tolerance": "low"
            },
            QualityStandard.BUSINESS: {
                "readability_target": 60,
                "word_count_range": [500, 2000],
                "sentence_length_max": 20,
                "vocabulary_complexity": "medium",
                "tone": "professional",
                "bias_tolerance": "medium"
            },
            QualityStandard.TECHNICAL: {
                "readability_target": 55,
                "word_count_range": [1000, 5000],
                "sentence_length_max": 30,
                "vocabulary_complexity": "high",
                "tone": "technical",
                "bias_tolerance": "low"
            },
            QualityStandard.CREATIVE: {
                "readability_target": 70,
                "word_count_range": [300, 3000],
                "sentence_length_max": 25,
                "vocabulary_complexity": "medium",
                "tone": "engaging",
                "bias_tolerance": "medium"
            },
            QualityStandard.JOURNALISTIC: {
                "readability_target": 65,
                "word_count_range": [400, 1500],
                "sentence_length_max": 20,
                "vocabulary_complexity": "medium",
                "tone": "neutral",
                "bias_tolerance": "low"
            },
            QualityStandard.MARKETING: {
                "readability_target": 75,
                "word_count_range": [200, 1000],
                "sentence_length_max": 15,
                "vocabulary_complexity": "low",
                "tone": "persuasive",
                "bias_tolerance": "medium"
            }
        }

    def _initialize_compliance_rules(self):
        """Initialize compliance rules for different types"""
        self.compliance_rules = {
            ComplianceType.ACCESSIBILITY: {
                "alt_text_required": True,
                "heading_structure": True,
                "color_contrast": True,
                "font_size_min": 12
            },
            ComplianceType.SEO: {
                "keyword_density_max": 0.03,
                "meta_description_length": [120, 160],
                "heading_hierarchy": True,
                "internal_links": True
            },
            ComplianceType.BRAND_GUIDELINES: {
                "tone_consistency": True,
                "brand_voice": True,
                "terminology_consistency": True
            },
            ComplianceType.LEGAL_COMPLIANCE: {
                "disclaimer_required": False,
                "copyright_compliance": True,
                "privacy_mention": False
            },
            ComplianceType.BIAS_DETECTION: {
                "gender_bias_check": True,
                "racial_bias_check": True,
                "age_bias_check": True,
                "cultural_bias_check": True
            }
        }

    def _check_readability(self, content: str, standard: QualityStandard) -> QualityCheck:
        """Check content readability"""
        try:
            readability_score = self._calculate_readability_score(content)
            standard_config = self.quality_standards.get(standard, {})
            target_score = standard_config.get("readability_target", 60)
            
            passed = readability_score >= target_score
            issues = []
            recommendations = []
            
            if not passed:
                issues.append(f"Readability score {readability_score:.1f} below target {target_score}")
                if readability_score < target_score - 10:
                    recommendations.append("Simplify sentence structure and vocabulary")
                else:
                    recommendations.append("Minor readability improvements needed")
            
            return QualityCheck(
                check_type="readability",
                passed=passed,
                score=readability_score,
                issues=issues,
                recommendations=recommendations,
                confidence=0.9,
                checked_at=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Readability check failed: {e}")
            return QualityCheck(
                check_type="readability",
                passed=False,
                score=0.0,
                issues=[f"Readability check failed: {str(e)}"],
                recommendations=["Manual readability review required"],
                confidence=0.0,
                checked_at=datetime.now()
            )

    def _check_grammar_and_spelling(self, content: str) -> QualityCheck:
        """Check grammar and spelling"""
        try:
            issues = []
            recommendations = []
            
            # Basic spelling check (simplified)
            words = content.split()
            common_misspellings = {
                "teh": "the", "adn": "and", "recieve": "receive",
                "seperate": "separate", "occured": "occurred"
            }
            
            spelling_errors = []
            for word in words:
                clean_word = re.sub(r'[^\w]', '', word.lower())
                if clean_word in common_misspellings:
                    spelling_errors.append(f"'{word}' should be '{common_misspellings[clean_word]}'")
            
            if spelling_errors:
                issues.extend(spelling_errors)
                recommendations.append("Review and correct spelling errors")
            
            # Basic grammar check (simplified)
            grammar_issues = []
            
            # Check for common grammar mistakes
            if "its" in content and "it's" not in content:
                # This is a very basic check - in reality, you'd use a proper grammar checker
                pass
            
            # Check sentence structure
            sentences = content.split('.')
            for sentence in sentences:
                if len(sentence.strip()) > 0:
                    words_in_sentence = len(sentence.split())
                    if words_in_sentence > 50:
                        grammar_issues.append("Very long sentence detected")
            
            if grammar_issues:
                issues.extend(grammar_issues)
                recommendations.append("Review sentence structure and grammar")
            
            passed = len(issues) == 0
            score = max(0, 100 - len(issues) * 10)  # Simple scoring
            
            return QualityCheck(
                check_type="grammar_spelling",
                passed=passed,
                score=score,
                issues=issues,
                recommendations=recommendations,
                confidence=0.7,
                checked_at=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Grammar and spelling check failed: {e}")
            return QualityCheck(
                check_type="grammar_spelling",
                passed=False,
                score=0.0,
                issues=[f"Grammar check failed: {str(e)}"],
                recommendations=["Manual grammar review required"],
                confidence=0.0,
                checked_at=datetime.now()
            )

    def _check_tone_consistency(self, content: str, standard: QualityStandard) -> QualityCheck:
        """Check tone consistency"""
        try:
            standard_config = self.quality_standards.get(standard, {})
            expected_tone = standard_config.get("tone", "neutral")
            
            # Analyze tone indicators (simplified)
            formal_words = ['therefore', 'however', 'furthermore', 'moreover']
            informal_words = ['yeah', 'okay', 'cool', 'awesome', 'gonna']
            
            formal_count = sum(1 for word in formal_words if word in content.lower())
            informal_count = sum(1 for word in informal_words if word in content.lower())
            
            issues = []
            recommendations = []
            
            if expected_tone == "formal" and informal_count > 0:
                issues.append("Informal language detected in formal content")
                recommendations.append("Use more formal language")
            elif expected_tone == "informal" and formal_count > informal_count * 2:
                issues.append("Overly formal language for informal content")
                recommendations.append("Use more conversational language")
            
            passed = len(issues) == 0
            score = 80 if passed else 60
            
            return QualityCheck(
                check_type="tone_consistency",
                passed=passed,
                score=score,
                issues=issues,
                recommendations=recommendations,
                confidence=0.6,
                checked_at=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Tone consistency check failed: {e}")
            return QualityCheck(
                check_type="tone_consistency",
                passed=False,
                score=0.0,
                issues=[f"Tone check failed: {str(e)}"],
                recommendations=["Manual tone review required"],
                confidence=0.0,
                checked_at=datetime.now()
            )

    def _check_bias(self, content: str) -> QualityCheck:
        """Check for bias in content"""
        try:
            issues = []
            recommendations = []
            
            # Gender bias check (simplified)
            gender_biased_terms = {
                'chairman': 'chairperson',
                'policeman': 'police officer',
                'fireman': 'firefighter',
                'mailman': 'mail carrier'
            }
            
            for biased_term, neutral_term in gender_biased_terms.items():
                if biased_term in content.lower():
                    issues.append(f"Gender-biased term '{biased_term}' found")
                    recommendations.append(f"Consider using '{neutral_term}' instead")
            
            # Racial bias check (simplified)
            racial_terms = ['colored', 'oriental', 'exotic']
            for term in racial_terms:
                if term in content.lower():
                    issues.append(f"Potentially problematic term '{term}' found")
                    recommendations.append("Review for appropriate terminology")
            
            passed = len(issues) == 0
            score = 90 if passed else max(0, 90 - len(issues) * 15)
            
            return QualityCheck(
                check_type="bias_detection",
                passed=passed,
                score=score,
                issues=issues,
                recommendations=recommendations,
                confidence=0.8,
                checked_at=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Bias check failed: {e}")
            return QualityCheck(
                check_type="bias_detection",
                passed=False,
                score=0.0,
                issues=[f"Bias check failed: {str(e)}"],
                recommendations=["Manual bias review required"],
                confidence=0.0,
                checked_at=datetime.now()
            )

    def _check_facts(self, content: str) -> QualityCheck:
        """Basic fact checking (simplified)"""
        try:
            issues = []
            recommendations = []
            
            # Check for common factual errors (simplified)
            factual_errors = {
                'the earth is flat': 'The Earth is approximately spherical',
                'climate change is not real': 'Climate change is scientifically established',
                'vaccines cause autism': 'Vaccines do not cause autism'
            }
            
            content_lower = content.lower()
            for error, correction in factual_errors.items():
                if error in content_lower:
                    issues.append(f"Factual error detected: '{error}'")
                    recommendations.append(f"Consider: '{correction}'")
            
            # Check for unsupported claims
            unsupported_indicators = ['everyone knows', 'it is a fact that', 'scientists agree']
            for indicator in unsupported_indicators:
                if indicator in content_lower:
                    issues.append(f"Unsupported claim indicator: '{indicator}'")
                    recommendations.append("Provide evidence or sources for claims")
            
            passed = len(issues) == 0
            score = 85 if passed else max(0, 85 - len(issues) * 20)
            
            return QualityCheck(
                check_type="fact_checking",
                passed=passed,
                score=score,
                issues=issues,
                recommendations=recommendations,
                confidence=0.7,
                checked_at=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Fact check failed: {e}")
            return QualityCheck(
                check_type="fact_checking",
                passed=False,
                score=0.0,
                issues=[f"Fact check failed: {str(e)}"],
                recommendations=["Manual fact checking required"],
                confidence=0.0,
                checked_at=datetime.now()
            )

    def _perform_compliance_check(self, content: str, compliance_type: ComplianceType) -> ComplianceCheck:
        """Perform specific compliance check"""
        try:
            rules = self.compliance_rules.get(compliance_type, {})
            violations = []
            recommendations = []
            severity = "low"
            
            if compliance_type == ComplianceType.ACCESSIBILITY:
                # Check for accessibility issues
                if not re.search(r'alt\s*=', content, re.IGNORECASE):
                    violations.append("Missing alt text for images")
                    recommendations.append("Add descriptive alt text to all images")
                
                if not re.search(r'<h[1-6]', content, re.IGNORECASE):
                    violations.append("Missing heading structure")
                    recommendations.append("Use proper heading hierarchy (H1, H2, etc.)")
            
            elif compliance_type == ComplianceType.SEO:
                # Check for SEO issues
                if len(content) < 300:
                    violations.append("Content too short for SEO")
                    recommendations.append("Increase content length to at least 300 words")
                
                # Check for keyword density (simplified)
                words = content.lower().split()
                if len(words) > 0:
                    word_freq = Counter(words)
                    max_freq = max(word_freq.values())
                    if max_freq / len(words) > 0.03:
                        violations.append("High keyword density detected")
                        recommendations.append("Reduce keyword density to under 3%")
            
            elif compliance_type == ComplianceType.BIAS_DETECTION:
                # Use the bias check from quality checks
                bias_check = self._check_bias(content)
                if not bias_check.passed:
                    violations.extend(bias_check.issues)
                    recommendations.extend(bias_check.recommendations)
                    severity = "medium"
            
            # Determine severity
            if len(violations) > 3:
                severity = "high"
            elif len(violations) > 1:
                severity = "medium"
            
            passed = len(violations) == 0
            confidence = 0.8 if compliance_type in [ComplianceType.ACCESSIBILITY, ComplianceType.SEO] else 0.6
            
            return ComplianceCheck(
                compliance_type=compliance_type,
                passed=passed,
                violations=violations,
                recommendations=recommendations,
                severity=severity,
                confidence=confidence,
                checked_at=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Compliance check failed for {compliance_type.value}: {e}")
            return ComplianceCheck(
                compliance_type=compliance_type,
                passed=False,
                violations=[f"Check failed: {str(e)}"],
                recommendations=["Manual compliance review required"],
                severity="high",
                confidence=0.0,
                checked_at=datetime.now()
            )

    def _calculate_readability_score(self, content: str) -> float:
        """Calculate readability score"""
        try:
            if HAS_TEXTSTAT:
                # Use textstat library for accurate readability
                return flesch_reading_ease(content)
            else:
                # Simplified readability calculation
                sentences = content.split('.')
                words = content.split()
                
                if len(sentences) == 0 or len(words) == 0:
                    return 0.0
                
                avg_sentence_length = len(words) / len(sentences)
                avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)
                
                # Simplified Flesch Reading Ease formula
                score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
                return max(0, min(100, score))
                
        except Exception as e:
            logger.warning(f"Readability calculation failed: {e}")
            return 50.0  # Default middle score

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

    def _calculate_overall_score(self, quality_checks: List[QualityCheck], 
                               compliance_checks: List[ComplianceCheck]) -> float:
        """Calculate overall quality score"""
        try:
            if not quality_checks and not compliance_checks:
                return 0.0
            
            # Weight quality checks more heavily
            quality_score = 0.0
            if quality_checks:
                quality_score = sum(check.score for check in quality_checks) / len(quality_checks)
            
            compliance_score = 0.0
            if compliance_checks:
                passed_checks = sum(1 for check in compliance_checks if check.passed)
                compliance_score = (passed_checks / len(compliance_checks)) * 100
            
            # Weighted average (70% quality, 30% compliance)
            overall_score = (quality_score * 0.7) + (compliance_score * 0.3)
            return max(0, min(100, overall_score))
            
        except Exception as e:
            logger.warning(f"Overall score calculation failed: {e}")
            return 0.0

    def _generate_issues_summary(self, quality_checks: List[QualityCheck], 
                               compliance_checks: List[ComplianceCheck]) -> Dict[str, int]:
        """Generate summary of issues found"""
        issues_summary = defaultdict(int)
        
        for check in quality_checks:
            if not check.passed:
                issues_summary[check.check_type] += len(check.issues)
        
        for check in compliance_checks:
            if not check.passed:
                issues_summary[f"compliance_{check.compliance_type.value}"] += len(check.violations)
        
        return dict(issues_summary)

    def _generate_quality_recommendations(self, quality_checks: List[QualityCheck], 
                                        compliance_checks: List[ComplianceCheck]) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        # Collect recommendations from quality checks
        for check in quality_checks:
            if not check.passed:
                recommendations.extend(check.recommendations)
        
        # Collect recommendations from compliance checks
        for check in compliance_checks:
            if not check.passed:
                recommendations.extend(check.recommendations)
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        return unique_recommendations

    def _determine_quality_grade(self, score: float) -> str:
        """Determine quality grade based on score"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _check_readability_standard(self, score: float, standard: QualityStandard) -> bool:
        """Check if readability meets standard"""
        standard_config = self.quality_standards.get(standard, {})
        target = standard_config.get("readability_target", 60)
        return score >= target

    def _check_word_count_standard(self, word_count: int, standard: QualityStandard) -> bool:
        """Check if word count meets standard"""
        standard_config = self.quality_standards.get(standard, {})
        word_range = standard_config.get("word_count_range", [100, 1000])
        return word_range[0] <= word_count <= word_range[1]

    def _check_sentence_structure(self, content: str, standard: QualityStandard) -> bool:
        """Check sentence structure"""
        try:
            sentences = content.split('.')
            standard_config = self.quality_standards.get(standard, {})
            max_length = standard_config.get("sentence_length_max", 25)
            
            for sentence in sentences:
                words = sentence.split()
                if len(words) > max_length:
                    return False
            return True
        except:
            return False

    def _check_vocabulary_standard(self, content: str, standard: QualityStandard) -> bool:
        """Check vocabulary complexity"""
        try:
            words = content.split()
            if len(words) == 0:
                return False
            
            # Simple vocabulary complexity check
            complex_words = [word for word in words if len(word) > 6]
            complexity_ratio = len(complex_words) / len(words)
            
            standard_config = self.quality_standards.get(standard, {})
            expected_complexity = standard_config.get("vocabulary_complexity", "medium")
            
            if expected_complexity == "high":
                return complexity_ratio >= 0.3
            elif expected_complexity == "low":
                return complexity_ratio <= 0.2
            else:  # medium
                return 0.1 <= complexity_ratio <= 0.4
        except:
            return False

    def _generate_standard_recommendations(self, standard: QualityStandard, 
                                         readability_passed: bool, word_count_passed: bool,
                                         sentence_structure_passed: bool, 
                                         vocabulary_passed: bool) -> List[str]:
        """Generate recommendations for standard compliance"""
        recommendations = []
        
        if not readability_passed:
            recommendations.append("Improve readability to meet standard requirements")
        
        if not word_count_passed:
            recommendations.append("Adjust content length to meet word count requirements")
        
        if not sentence_structure_passed:
            recommendations.append("Simplify sentence structure")
        
        if not vocabulary_passed:
            recommendations.append("Adjust vocabulary complexity to match standard")
        
        return recommendations

    def _generate_improvement_timeline(self, priority_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate improvement timeline"""
        timeline = []
        
        for i, action in enumerate(priority_actions):
            timeline.append({
                "phase": i + 1,
                "action": action["action"],
                "priority": action["priority"],
                "estimated_duration": "1-2 days" if action["priority"] == "high" else "3-5 days",
                "description": action["description"]
            })
        
        return timeline

    def _calculate_expected_improvement(self, current_score: float, target_score: float, 
                                      improvement_areas: List[str]) -> Dict[str, Any]:
        """Calculate expected improvement"""
        score_gap = target_score - current_score
        improvement_per_area = score_gap / len(improvement_areas) if improvement_areas else 0
        
        return {
            "current_score": current_score,
            "target_score": target_score,
            "score_gap": score_gap,
            "improvement_per_area": improvement_per_area,
            "estimated_final_score": min(100, current_score + (improvement_per_area * len(improvement_areas)))
        }

    def _estimate_improvement_effort(self, priority_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate improvement effort"""
        high_priority = sum(1 for action in priority_actions if action["priority"] == "high")
        critical_priority = sum(1 for action in priority_actions if action["priority"] == "critical")
        
        total_effort_hours = (critical_priority * 8) + (high_priority * 4)
        
        return {
            "total_effort_hours": total_effort_hours,
            "critical_actions": critical_priority,
            "high_priority_actions": high_priority,
            "estimated_completion": f"{total_effort_hours // 8} days" if total_effort_hours > 8 else "1 day"
        }

    def _define_success_metrics(self, target_score: float) -> List[Dict[str, Any]]:
        """Define success metrics"""
        return [
            {
                "metric": "overall_quality_score",
                "target": target_score,
                "measurement": "automated_analysis"
            },
            {
                "metric": "compliance_rate",
                "target": 95,
                "measurement": "compliance_checks"
            },
            {
                "metric": "readability_score",
                "target": 70,
                "measurement": "readability_analysis"
            }
        ]


# Global quality assurance instance
quality_assurance = ContentQualityAssurance()

# Convenience functions
def comprehensive_quality_check(content: str, quality_standard: QualityStandard = QualityStandard.BUSINESS) -> QualityReport:
    """Perform comprehensive quality check"""
    return quality_assurance.comprehensive_quality_check(content, quality_standard)

def check_content_standards(content: str, standard: QualityStandard) -> Dict[str, Any]:
    """Check content against standards"""
    return quality_assurance.check_content_standards(content, standard)

def validate_content_compliance(content: str, compliance_types: List[ComplianceType]) -> Dict[str, Any]:
    """Validate content compliance"""
    return quality_assurance.validate_content_compliance(content, compliance_types)

def generate_quality_improvement_plan(content: str, current_score: float, target_score: float = 80.0) -> Dict[str, Any]:
    """Generate quality improvement plan"""
    return quality_assurance.generate_quality_improvement_plan(content, current_score, target_score)



























