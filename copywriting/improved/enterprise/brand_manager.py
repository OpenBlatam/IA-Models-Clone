"""
Enterprise Brand Manager
========================

Advanced brand management and consistency enforcement for enterprise content.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import UUID, uuid4
import re
import json

from ..schemas import CopywritingRequest, CopywritingVariant
from ..exceptions import BrandViolationError, ValidationError
from ..utils import extract_keywords, calculate_readability_score

logger = logging.getLogger(__name__)


class BrandTone(str, Enum):
    """Brand tone options"""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"
    INSPIRATIONAL = "inspirational"
    URGENT = "urgent"
    CASUAL = "casual"


class BrandStyle(str, Enum):
    """Brand style options"""
    FORMAL = "formal"
    INFORMAL = "informal"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    MINIMALIST = "minimalist"
    DETAILED = "detailed"


class BrandVoice(str, Enum):
    """Brand voice characteristics"""
    CONFIDENT = "confident"
    HUMBLE = "humble"
    INNOVATIVE = "innovative"
    TRADITIONAL = "traditional"
    PLAYFUL = "playful"
    SERIOUS = "serious"


@dataclass
class BrandGuidelines:
    """Brand guidelines configuration"""
    id: UUID
    brand_name: str
    primary_tone: BrandTone
    secondary_tones: List[BrandTone]
    style: BrandStyle
    voice_characteristics: List[BrandVoice]
    
    # Content guidelines
    preferred_words: List[str]
    forbidden_words: List[str]
    industry_terms: List[str]
    brand_terms: List[str]
    
    # Formatting guidelines
    max_sentence_length: int = 25
    min_sentence_length: int = 5
    target_readability_score: float = 0.7
    preferred_word_count_range: Tuple[int, int] = (100, 500)
    
    # Brand-specific rules
    must_include_elements: List[str] = None
    must_avoid_elements: List[str] = None
    call_to_action_style: str = "direct"
    
    # Visual guidelines
    preferred_emojis: List[str] = None
    forbidden_emojis: List[str] = None
    
    # Compliance
    legal_requirements: List[str] = None
    industry_regulations: List[str] = None
    
    created_at: datetime = None
    updated_at: datetime = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.must_include_elements is None:
            self.must_include_elements = []
        if self.must_avoid_elements is None:
            self.must_avoid_elements = []
        if self.preferred_emojis is None:
            self.preferred_emojis = []
        if self.forbidden_emojis is None:
            self.forbidden_emojis = []
        if self.legal_requirements is None:
            self.legal_requirements = []
        if self.industry_regulations is None:
            self.industry_regulations = []


@dataclass
class BrandViolation:
    """Brand violation record"""
    id: UUID
    content_id: UUID
    violation_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    suggested_fix: str
    line_number: Optional[int] = None
    word_position: Optional[int] = None
    detected_at: datetime = None
    
    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.utcnow()


@dataclass
class BrandComplianceReport:
    """Brand compliance report"""
    content_id: UUID
    brand_guidelines_id: UUID
    overall_score: float  # 0-1
    compliance_percentage: float
    violations: List[BrandViolation]
    recommendations: List[str]
    generated_at: datetime = None
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.utcnow()


class BrandManager:
    """Enterprise brand management and consistency enforcement"""
    
    def __init__(self):
        self.brand_guidelines: Dict[UUID, BrandGuidelines] = {}
        self.violations: Dict[UUID, List[BrandViolation]] = {}
        self.compliance_reports: Dict[UUID, BrandComplianceReport] = {}
        self._initialize_default_brands()
    
    def _initialize_default_brands(self):
        """Initialize default brand guidelines"""
        
        # Tech Startup Brand
        tech_startup_guidelines = BrandGuidelines(
            id=uuid4(),
            brand_name="TechStartup",
            primary_tone=BrandTone.INNOVATIVE,
            secondary_tones=[BrandTone.FRIENDLY, BrandTone.CONFIDENT],
            style=BrandStyle.INFORMAL,
            voice_characteristics=[BrandVoice.INNOVATIVE, BrandVoice.CONFIDENT],
            preferred_words=["innovative", "cutting-edge", "revolutionary", "disruptive", "scalable"],
            forbidden_words=["old", "outdated", "traditional", "slow", "expensive"],
            industry_terms=["SaaS", "API", "cloud", "AI", "machine learning", "scalability"],
            brand_terms=["TechStartup", "our platform", "our solution"],
            max_sentence_length=20,
            target_readability_score=0.8,
            preferred_word_count_range=(150, 400),
            must_include_elements=["innovation", "technology"],
            call_to_action_style="energetic",
            preferred_emojis=["🚀", "💡", "⚡", "🎯"],
            forbidden_emojis=["😢", "😞", "💸", "❌"]
        )
        
        self.brand_guidelines[tech_startup_guidelines.id] = tech_startup_guidelines
        
        # Financial Services Brand
        financial_guidelines = BrandGuidelines(
            id=uuid4(),
            brand_name="SecureFinance",
            primary_tone=BrandTone.PROFESSIONAL,
            secondary_tones=[BrandTone.AUTHORITATIVE, BrandTone.CONFIDENT],
            style=BrandStyle.FORMAL,
            voice_characteristics=[BrandVoice.CONFIDENT, BrandVoice.TRADITIONAL],
            preferred_words=["secure", "reliable", "trusted", "established", "proven"],
            forbidden_words=["risky", "uncertain", "volatile", "speculative"],
            industry_terms=["investment", "portfolio", "returns", "risk management", "compliance"],
            brand_terms=["SecureFinance", "our services", "your investments"],
            max_sentence_length=30,
            target_readability_score=0.6,
            preferred_word_count_range=(200, 600),
            must_include_elements=["security", "compliance"],
            call_to_action_style="professional",
            legal_requirements=["SEC compliance", "FDIC insurance", "privacy policy"],
            industry_regulations=["SOX", "Basel III", "MiFID II"]
        )
        
        self.brand_guidelines[financial_guidelines.id] = financial_guidelines
        
        # Healthcare Brand
        healthcare_guidelines = BrandGuidelines(
            id=uuid4(),
            brand_name="HealthCare Plus",
            primary_tone=BrandTone.PROFESSIONAL,
            secondary_tones=[BrandTone.FRIENDLY, BrandTone.INSPIRATIONAL],
            style=BrandStyle.FORMAL,
            voice_characteristics=[BrandVoice.CONFIDENT, BrandVoice.HUMBLE],
            preferred_words=["care", "health", "wellness", "support", "healing"],
            forbidden_words=["cure", "guarantee", "miracle", "instant"],
            industry_terms=["treatment", "diagnosis", "therapy", "wellness", "prevention"],
            brand_terms=["HealthCare Plus", "our care team", "your health"],
            max_sentence_length=25,
            target_readability_score=0.7,
            preferred_word_count_range=(150, 500),
            must_include_elements=["health", "care"],
            call_to_action_style="caring",
            legal_requirements=["HIPAA compliance", "medical disclaimer"],
            industry_regulations=["FDA guidelines", "medical ethics"]
        )
        
        self.brand_guidelines[healthcare_guidelines.id] = healthcare_guidelines
    
    async def validate_content(
        self,
        content: str,
        brand_guidelines_id: UUID,
        content_id: Optional[UUID] = None
    ) -> BrandComplianceReport:
        """Validate content against brand guidelines"""
        
        if brand_guidelines_id not in self.brand_guidelines:
            raise BrandViolationError(f"Brand guidelines {brand_guidelines_id} not found")
        
        guidelines = self.brand_guidelines[brand_guidelines_id]
        violations = []
        recommendations = []
        
        # Check tone and style
        tone_violations = await self._check_tone_compliance(content, guidelines)
        violations.extend(tone_violations)
        
        # Check word usage
        word_violations = await self._check_word_usage(content, guidelines)
        violations.extend(word_violations)
        
        # Check formatting
        formatting_violations = await self._check_formatting(content, guidelines)
        violations.extend(formatting_violations)
        
        # Check required elements
        element_violations = await self._check_required_elements(content, guidelines)
        violations.extend(element_violations)
        
        # Check readability
        readability_violations = await self._check_readability(content, guidelines)
        violations.extend(readability_violations)
        
        # Check emoji usage
        emoji_violations = await self._check_emoji_usage(content, guidelines)
        violations.extend(emoji_violations)
        
        # Check legal compliance
        legal_violations = await self._check_legal_compliance(content, guidelines)
        violations.extend(legal_violations)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(content, guidelines, violations)
        
        # Calculate compliance score
        total_checks = 7  # Number of check categories
        passed_checks = total_checks - len([v for v in violations if v.severity in ["high", "critical"]])
        compliance_percentage = (passed_checks / total_checks) * 100
        
        # Calculate overall score (weighted by severity)
        severity_weights = {"low": 0.1, "medium": 0.3, "high": 0.6, "critical": 1.0}
        total_penalty = sum(severity_weights.get(v.severity, 0.5) for v in violations)
        overall_score = max(0, 1 - (total_penalty / len(violations)) if violations else 1)
        
        report = BrandComplianceReport(
            content_id=content_id or uuid4(),
            brand_guidelines_id=brand_guidelines_id,
            overall_score=overall_score,
            compliance_percentage=compliance_percentage,
            violations=violations,
            recommendations=recommendations
        )
        
        # Store violations and report
        if content_id:
            self.violations[content_id] = violations
            self.compliance_reports[content_id] = report
        
        logger.info(f"Brand validation completed for content {content_id}: {compliance_percentage:.1f}% compliant")
        
        return report
    
    async def _check_tone_compliance(self, content: str, guidelines: BrandGuidelines) -> List[BrandViolation]:
        """Check tone compliance"""
        violations = []
        
        # Simple tone analysis (would be more sophisticated in production)
        content_lower = content.lower()
        
        # Check for forbidden tone indicators
        if guidelines.primary_tone == BrandTone.PROFESSIONAL:
            casual_indicators = ["hey", "awesome", "cool", "yeah", "gonna"]
            for indicator in casual_indicators:
                if indicator in content_lower:
                    violations.append(BrandViolation(
                        id=uuid4(),
                        content_id=uuid4(),
                        violation_type="tone",
                        severity="medium",
                        description=f"Casual language '{indicator}' conflicts with professional tone",
                        suggested_fix=f"Replace '{indicator}' with more professional language"
                    ))
        
        elif guidelines.primary_tone == BrandTone.CASUAL:
            formal_indicators = ["therefore", "furthermore", "consequently", "moreover"]
            for indicator in formal_indicators:
                if indicator in content_lower:
                    violations.append(BrandViolation(
                        id=uuid4(),
                        content_id=uuid4(),
                        violation_type="tone",
                        severity="low",
                        description=f"Formal language '{indicator}' conflicts with casual tone",
                        suggested_fix=f"Replace '{indicator}' with more casual language"
                    ))
        
        return violations
    
    async def _check_word_usage(self, content: str, guidelines: BrandGuidelines) -> List[BrandViolation]:
        """Check word usage compliance"""
        violations = []
        content_lower = content.lower()
        
        # Check for forbidden words
        for forbidden_word in guidelines.forbidden_words:
            if forbidden_word.lower() in content_lower:
                violations.append(BrandViolation(
                    id=uuid4(),
                    content_id=uuid4(),
                    violation_type="word_usage",
                    severity="high",
                    description=f"Forbidden word '{forbidden_word}' found in content",
                    suggested_fix=f"Remove or replace '{forbidden_word}' with approved alternative"
                ))
        
        # Check for preferred words (encourage usage)
        preferred_word_count = sum(1 for word in guidelines.preferred_words if word.lower() in content_lower)
        if preferred_word_count == 0 and len(guidelines.preferred_words) > 0:
            violations.append(BrandViolation(
                id=uuid4(),
                content_id=uuid4(),
                violation_type="word_usage",
                severity="low",
                description="No preferred brand words found in content",
                suggested_fix=f"Consider including some of these words: {', '.join(guidelines.preferred_words[:3])}"
            ))
        
        return violations
    
    async def _check_formatting(self, content: str, guidelines: BrandGuidelines) -> List[BrandViolation]:
        """Check formatting compliance"""
        violations = []
        
        # Check sentence length
        sentences = re.split(r'[.!?]+', content)
        for i, sentence in enumerate(sentences):
            word_count = len(sentence.split())
            if word_count > guidelines.max_sentence_length:
                violations.append(BrandViolation(
                    id=uuid4(),
                    content_id=uuid4(),
                    violation_type="formatting",
                    severity="medium",
                    description=f"Sentence {i+1} is too long ({word_count} words)",
                    suggested_fix=f"Split into shorter sentences (max {guidelines.max_sentence_length} words)",
                    line_number=i+1
                ))
            elif word_count < guidelines.min_sentence_length:
                violations.append(BrandViolation(
                    id=uuid4(),
                    content_id=uuid4(),
                    violation_type="formatting",
                    severity="low",
                    description=f"Sentence {i+1} is too short ({word_count} words)",
                    suggested_fix=f"Expand sentence (min {guidelines.min_sentence_length} words)",
                    line_number=i+1
                ))
        
        # Check word count
        total_words = len(content.split())
        min_words, max_words = guidelines.preferred_word_count_range
        if total_words < min_words:
            violations.append(BrandViolation(
                id=uuid4(),
                content_id=uuid4(),
                violation_type="formatting",
                severity="medium",
                description=f"Content is too short ({total_words} words)",
                suggested_fix=f"Expand content to at least {min_words} words"
            ))
        elif total_words > max_words:
            violations.append(BrandViolation(
                id=uuid4(),
                content_id=uuid4(),
                violation_type="formatting",
                severity="medium",
                description=f"Content is too long ({total_words} words)",
                suggested_fix=f"Reduce content to maximum {max_words} words"
            ))
        
        return violations
    
    async def _check_required_elements(self, content: str, guidelines: BrandGuidelines) -> List[BrandViolation]:
        """Check required elements"""
        violations = []
        content_lower = content.lower()
        
        # Check must-include elements
        for element in guidelines.must_include_elements:
            if element.lower() not in content_lower:
                violations.append(BrandViolation(
                    id=uuid4(),
                    content_id=uuid4(),
                    violation_type="required_elements",
                    severity="high",
                    description=f"Required element '{element}' not found",
                    suggested_fix=f"Include '{element}' in the content"
                ))
        
        # Check must-avoid elements
        for element in guidelines.must_avoid_elements:
            if element.lower() in content_lower:
                violations.append(BrandViolation(
                    id=uuid4(),
                    content_id=uuid4(),
                    violation_type="required_elements",
                    severity="high",
                    description=f"Forbidden element '{element}' found",
                    suggested_fix=f"Remove '{element}' from the content"
                ))
        
        return violations
    
    async def _check_readability(self, content: str, guidelines: BrandGuidelines) -> List[BrandViolation]:
        """Check readability compliance"""
        violations = []
        
        readability_score = calculate_readability_score(content)
        target_score = guidelines.target_readability_score
        
        if readability_score < target_score * 0.8:  # 20% tolerance
            violations.append(BrandViolation(
                id=uuid4(),
                content_id=uuid4(),
                violation_type="readability",
                severity="medium",
                description=f"Readability score {readability_score:.2f} is below target {target_score:.2f}",
                suggested_fix="Simplify sentence structure and use shorter words"
            ))
        elif readability_score > target_score * 1.2:  # 20% tolerance
            violations.append(BrandViolation(
                id=uuid4(),
                content_id=uuid4(),
                violation_type="readability",
                severity="low",
                description=f"Readability score {readability_score:.2f} is above target {target_score:.2f}",
                suggested_fix="Consider using more sophisticated language"
            ))
        
        return violations
    
    async def _check_emoji_usage(self, content: str, guidelines: BrandGuidelines) -> List[BrandViolation]:
        """Check emoji usage compliance"""
        violations = []
        
        # Extract emojis from content
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U0001F018-\U0001F0F5\U0001F200-\U0001F2FF]')
        emojis = emoji_pattern.findall(content)
        
        # Check for forbidden emojis
        for emoji in emojis:
            if emoji in guidelines.forbidden_emojis:
                violations.append(BrandViolation(
                    id=uuid4(),
                    content_id=uuid4(),
                    violation_type="emoji_usage",
                    severity="medium",
                    description=f"Forbidden emoji '{emoji}' found",
                    suggested_fix=f"Remove emoji '{emoji}' or replace with approved emoji"
                ))
        
        # Check if preferred emojis are used (encouragement, not violation)
        if guidelines.preferred_emojis and not any(emoji in emojis for emoji in guidelines.preferred_emojis):
            violations.append(BrandViolation(
                id=uuid4(),
                content_id=uuid4(),
                violation_type="emoji_usage",
                severity="low",
                description="No preferred brand emojis found",
                suggested_fix=f"Consider using these emojis: {', '.join(guidelines.preferred_emojis[:3])}"
            ))
        
        return violations
    
    async def _check_legal_compliance(self, content: str, guidelines: BrandGuidelines) -> List[BrandViolation]:
        """Check legal compliance"""
        violations = []
        
        # Check for required legal disclaimers
        for requirement in guidelines.legal_requirements:
            if requirement.lower() not in content.lower():
                violations.append(BrandViolation(
                    id=uuid4(),
                    content_id=uuid4(),
                    violation_type="legal_compliance",
                    severity="critical",
                    description=f"Legal requirement '{requirement}' not found",
                    suggested_fix=f"Include '{requirement}' disclaimer"
                ))
        
        return violations
    
    async def _generate_recommendations(
        self,
        content: str,
        guidelines: BrandGuidelines,
        violations: List[BrandViolation]
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # General recommendations based on violations
        if any(v.violation_type == "tone" for v in violations):
            recommendations.append(f"Adjust tone to be more {guidelines.primary_tone.value}")
        
        if any(v.violation_type == "readability" for v in violations):
            recommendations.append("Improve readability by using shorter sentences and simpler words")
        
        if any(v.violation_type == "word_usage" for v in violations):
            recommendations.append("Use more brand-preferred words and avoid forbidden terms")
        
        # Specific recommendations
        if len(violations) == 0:
            recommendations.append("Content fully complies with brand guidelines")
        elif len([v for v in violations if v.severity == "critical"]) > 0:
            recommendations.append("Address critical violations immediately")
        elif len([v for v in violations if v.severity == "high"]) > 0:
            recommendations.append("Address high-priority violations before publishing")
        else:
            recommendations.append("Content is mostly compliant with minor improvements needed")
        
        return recommendations
    
    async def create_brand_guidelines(
        self,
        brand_name: str,
        primary_tone: BrandTone,
        style: BrandStyle,
        voice_characteristics: List[BrandVoice],
        **kwargs
    ) -> BrandGuidelines:
        """Create new brand guidelines"""
        
        guidelines = BrandGuidelines(
            id=uuid4(),
            brand_name=brand_name,
            primary_tone=primary_tone,
            secondary_tones=kwargs.get('secondary_tones', []),
            style=style,
            voice_characteristics=voice_characteristics,
            preferred_words=kwargs.get('preferred_words', []),
            forbidden_words=kwargs.get('forbidden_words', []),
            industry_terms=kwargs.get('industry_terms', []),
            brand_terms=kwargs.get('brand_terms', []),
            max_sentence_length=kwargs.get('max_sentence_length', 25),
            min_sentence_length=kwargs.get('min_sentence_length', 5),
            target_readability_score=kwargs.get('target_readability_score', 0.7),
            preferred_word_count_range=kwargs.get('preferred_word_count_range', (100, 500)),
            must_include_elements=kwargs.get('must_include_elements', []),
            must_avoid_elements=kwargs.get('must_avoid_elements', []),
            call_to_action_style=kwargs.get('call_to_action_style', 'direct'),
            preferred_emojis=kwargs.get('preferred_emojis', []),
            forbidden_emojis=kwargs.get('forbidden_emojis', []),
            legal_requirements=kwargs.get('legal_requirements', []),
            industry_regulations=kwargs.get('industry_regulations', [])
        )
        
        self.brand_guidelines[guidelines.id] = guidelines
        
        logger.info(f"Created brand guidelines for {brand_name}")
        
        return guidelines
    
    async def get_brand_analytics(self) -> Dict[str, Any]:
        """Get brand compliance analytics"""
        
        total_reports = len(self.compliance_reports)
        if total_reports == 0:
            return {"message": "No compliance reports available"}
        
        # Calculate average compliance
        avg_compliance = sum(report.compliance_percentage for report in self.compliance_reports.values()) / total_reports
        
        # Count violations by type
        violation_types = {}
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for violations in self.violations.values():
            for violation in violations:
                violation_types[violation.violation_type] = violation_types.get(violation.violation_type, 0) + 1
                severity_counts[violation.severity] += 1
        
        # Most common violations
        most_common_violations = sorted(violation_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "summary": {
                "total_reports": total_reports,
                "average_compliance_percentage": avg_compliance,
                "total_violations": sum(len(v) for v in self.violations.values()),
                "brand_guidelines_count": len(self.brand_guidelines)
            },
            "violation_analysis": {
                "by_type": violation_types,
                "by_severity": severity_counts,
                "most_common": most_common_violations
            },
            "brand_guidelines": [
                {
                    "id": str(guidelines.id),
                    "brand_name": guidelines.brand_name,
                    "primary_tone": guidelines.primary_tone.value,
                    "style": guidelines.style.value
                }
                for guidelines in self.brand_guidelines.values()
            ]
        }


# Global brand manager instance
brand_manager = BrandManager()






























