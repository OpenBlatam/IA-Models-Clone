"""
Quality management system for Export IA.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .models import QualityMetrics, ExportConfig, QualityLevel
from .config import ConfigManager

logger = logging.getLogger(__name__)


class QualityManager:
    """Manages quality assurance and validation for exported documents."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.quality_rules = self._initialize_quality_rules()
    
    def _initialize_quality_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize quality rules for different aspects."""
        return {
            "formatting": {
                "min_font_size": 10,
                "max_font_size": 18,
                "required_margins": True,
                "consistent_spacing": True,
                "proper_alignment": True
            },
            "content": {
                "min_title_length": 5,
                "max_title_length": 100,
                "min_section_length": 10,
                "required_sections": True,
                "proper_structure": True
            },
            "accessibility": {
                "alt_text_required": True,
                "heading_structure": True,
                "color_contrast": True,
                "readable_fonts": True,
                "logical_order": True
            },
            "professional": {
                "consistent_styling": True,
                "proper_branding": True,
                "error_free": True,
                "complete_sections": True,
                "professional_language": True
            }
        }
    
    async def process_content_for_quality(
        self, 
        content: Dict[str, Any], 
        config: ExportConfig
    ) -> Dict[str, Any]:
        """Process content to ensure professional quality."""
        processed = content.copy()
        
        # Apply quality enhancements based on quality level
        quality_config = self.config_manager.get_quality_config(config.quality_level)
        
        # Enhance structure
        if "structure" not in processed:
            processed["structure"] = self._generate_document_structure(config.document_type)
        
        # Enhance formatting
        processed["formatting"] = self._apply_professional_formatting(processed, quality_config)
        
        # Add branding if configured
        if config.branding and quality_config.custom_branding:
            processed["branding"] = config.branding
        
        # Enhance accessibility
        if quality_config.accessibility_features:
            processed["accessibility"] = self._add_accessibility_features(processed)
        
        # Validate content
        validation_results = await self._validate_content(processed, config)
        processed["validation"] = validation_results
        
        return processed
    
    def _generate_document_structure(self, document_type) -> Dict[str, Any]:
        """Generate professional document structure."""
        template = self.config_manager.get_template(document_type)
        return {
            "sections": template.get("sections", []),
            "hierarchy": template.get("heading_styles", []),
            "style_mapping": template
        }
    
    def _apply_professional_formatting(
        self, 
        content: Dict[str, Any], 
        quality_config
    ) -> Dict[str, Any]:
        """Apply professional formatting to content."""
        return {
            "typography": {
                "font_family": quality_config.font_family,
                "font_size": quality_config.font_size,
                "line_spacing": quality_config.line_spacing
            },
            "layout": {
                "margins": quality_config.margins,
                "colors": quality_config.colors
            },
            "features": {
                "header_footer": quality_config.header_footer,
                "page_numbers": quality_config.page_numbers,
                "table_styling": quality_config.table_styling
            }
        }
    
    def _add_accessibility_features(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Add accessibility features to content."""
        return {
            "alt_text": True,
            "heading_structure": True,
            "color_contrast": True,
            "readable_fonts": True,
            "logical_reading_order": True
        }
    
    async def _validate_content(
        self, 
        content: Dict[str, Any], 
        config: ExportConfig
    ) -> Dict[str, Any]:
        """Validate content against quality rules."""
        validation_results = {
            "formatting": self._validate_formatting(content, config),
            "content": self._validate_content_structure(content, config),
            "accessibility": self._validate_accessibility(content, config),
            "professional": self._validate_professional_standards(content, config)
        }
        
        return validation_results
    
    def _validate_formatting(self, content: Dict[str, Any], config: ExportConfig) -> Dict[str, Any]:
        """Validate formatting aspects."""
        issues = []
        score = 1.0
        
        # Check title formatting
        if "title" in content:
            title = content["title"]
            if len(title) < self.quality_rules["formatting"]["min_font_size"]:
                issues.append("Title too short")
                score -= 0.1
            elif len(title) > self.quality_rules["formatting"]["max_font_size"]:
                issues.append("Title too long")
                score -= 0.1
        
        # Check section formatting
        if "sections" in content:
            for section in content["sections"]:
                if "heading" in section and len(section["heading"]) < 5:
                    issues.append("Section heading too short")
                    score -= 0.05
        
        return {
            "score": max(score, 0.0),
            "issues": issues,
            "passed": len(issues) == 0
        }
    
    def _validate_content_structure(self, content: Dict[str, Any], config: ExportConfig) -> Dict[str, Any]:
        """Validate content structure."""
        issues = []
        score = 1.0
        
        # Check for required sections
        template = self.config_manager.get_template(config.document_type)
        required_sections = template.get("sections", [])
        
        if "sections" in content:
            content_sections = [s.get("heading", "") for s in content["sections"]]
            missing_sections = [s for s in required_sections if s not in content_sections]
            
            if missing_sections:
                issues.append(f"Missing sections: {', '.join(missing_sections)}")
                score -= 0.1 * len(missing_sections)
        
        # Check section content
        if "sections" in content:
            for section in content["sections"]:
                if "content" in section and len(section["content"]) < 10:
                    issues.append(f"Section '{section.get('heading', 'Unknown')}' content too short")
                    score -= 0.05
        
        return {
            "score": max(score, 0.0),
            "issues": issues,
            "passed": len(issues) == 0
        }
    
    def _validate_accessibility(self, content: Dict[str, Any], config: ExportConfig) -> Dict[str, Any]:
        """Validate accessibility features."""
        issues = []
        score = 1.0
        
        # Check heading structure
        if "sections" in content:
            headings = [s.get("heading", "") for s in content["sections"]]
            if not headings:
                issues.append("No headings found")
                score -= 0.2
        
        # Check for alt text (if images present)
        if "images" in content:
            for img in content["images"]:
                if not img.get("alt_text"):
                    issues.append("Image missing alt text")
                    score -= 0.1
        
        return {
            "score": max(score, 0.0),
            "issues": issues,
            "passed": len(issues) == 0
        }
    
    def _validate_professional_standards(self, content: Dict[str, Any], config: ExportConfig) -> Dict[str, Any]:
        """Validate professional standards."""
        issues = []
        score = 1.0
        
        # Check for professional language
        if "title" in content:
            title = content["title"].lower()
            unprofessional_words = ["awesome", "cool", "amazing", "fantastic"]
            if any(word in title for word in unprofessional_words):
                issues.append("Title contains unprofessional language")
                score -= 0.1
        
        # Check for complete content
        if "sections" in content:
            empty_sections = [s for s in content["sections"] if not s.get("content", "").strip()]
            if empty_sections:
                issues.append(f"Empty sections: {len(empty_sections)}")
                score -= 0.1 * len(empty_sections)
        
        return {
            "score": max(score, 0.0),
            "issues": issues,
            "passed": len(issues) == 0
        }
    
    async def calculate_quality_score(
        self, 
        result: Dict[str, Any], 
        config: ExportConfig
    ) -> float:
        """Calculate overall quality score for exported document."""
        score = 0.0
        
        # Base score for successful export
        score += 0.3
        
        # Quality level bonus
        quality_bonus = {
            QualityLevel.BASIC: 0.1,
            QualityLevel.STANDARD: 0.2,
            QualityLevel.PROFESSIONAL: 0.3,
            QualityLevel.PREMIUM: 0.4,
            QualityLevel.ENTERPRISE: 0.5
        }
        score += quality_bonus.get(config.quality_level, 0.2)
        
        # Format-specific quality checks
        if config.format.value == "pdf":
            score += 0.2  # PDF is generally high quality
        elif config.format.value in ["docx", "html"]:
            score += 0.15
        
        # Professional features bonus
        quality_config = self.config_manager.get_quality_config(config.quality_level)
        if quality_config.header_footer:
            score += 0.1
        if quality_config.table_styling:
            score += 0.1
        if quality_config.custom_branding:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def get_quality_metrics(
        self, 
        content: Dict[str, Any], 
        config: ExportConfig
    ) -> QualityMetrics:
        """Get detailed quality metrics for content."""
        # Validate all aspects
        formatting_validation = self._validate_formatting(content, config)
        content_validation = self._validate_content_structure(content, config)
        accessibility_validation = self._validate_accessibility(content, config)
        professional_validation = self._validate_professional_standards(content, config)
        
        # Calculate overall score
        overall_score = (
            formatting_validation["score"] * 0.25 +
            content_validation["score"] * 0.25 +
            accessibility_validation["score"] * 0.25 +
            professional_validation["score"] * 0.25
        )
        
        # Collect all issues and suggestions
        all_issues = []
        all_issues.extend(formatting_validation["issues"])
        all_issues.extend(content_validation["issues"])
        all_issues.extend(accessibility_validation["issues"])
        all_issues.extend(professional_validation["issues"])
        
        suggestions = self._generate_suggestions(all_issues)
        
        return QualityMetrics(
            overall_score=overall_score,
            formatting_score=formatting_validation["score"],
            content_score=content_validation["score"],
            accessibility_score=accessibility_validation["score"],
            professional_score=professional_validation["score"],
            issues=all_issues,
            suggestions=suggestions
        )
    
    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        """Generate improvement suggestions based on issues."""
        suggestions = []
        
        for issue in issues:
            if "too short" in issue.lower():
                suggestions.append("Consider expanding content for better detail")
            elif "missing" in issue.lower():
                suggestions.append("Add missing elements to improve completeness")
            elif "unprofessional" in issue.lower():
                suggestions.append("Use more professional language")
            elif "alt text" in issue.lower():
                suggestions.append("Add descriptive alt text for images")
            elif "heading" in issue.lower():
                suggestions.append("Improve heading structure and hierarchy")
        
        return suggestions




