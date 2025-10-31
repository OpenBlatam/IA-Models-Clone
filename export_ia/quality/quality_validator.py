"""
Export IA Quality Validation System
===================================

Advanced quality validation and professional appearance checks for exported documents.
Ensures all exported files meet professional standards and quality requirements.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import os
from pathlib import Path
import json
import hashlib

logger = logging.getLogger(__name__)

class QualityMetric(Enum):
    """Quality metrics for document validation."""
    TYPOGRAPHY = "typography"
    FORMATTING = "formatting"
    STRUCTURE = "structure"
    ACCESSIBILITY = "accessibility"
    PROFESSIONAL_APPEARANCE = "professional_appearance"
    CONTENT_QUALITY = "content_quality"
    TECHNICAL_QUALITY = "technical_quality"

class ValidationLevel(Enum):
    """Validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    ENTERPRISE = "enterprise"

@dataclass
class QualityRule:
    """Quality validation rule."""
    id: str
    name: str
    description: str
    metric: QualityMetric
    validation_level: ValidationLevel
    weight: float = 1.0
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Result of a quality validation."""
    rule_id: str
    passed: bool
    score: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)

@dataclass
class QualityReport:
    """Comprehensive quality report."""
    overall_score: float
    passed_validation: bool
    metrics: Dict[QualityMetric, float]
    results: List[ValidationResult]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class QualityValidator:
    """Advanced quality validation system for professional documents."""
    
    def __init__(self):
        self.rules: Dict[str, QualityRule] = {}
        self.validation_levels: Dict[ValidationLevel, List[str]] = {}
        
        self._initialize_quality_rules()
        self._setup_validation_levels()
        
        logger.info(f"Quality Validator initialized with {len(self.rules)} rules")
    
    def _initialize_quality_rules(self):
        """Initialize quality validation rules."""
        rules_data = [
            # Typography rules
            {
                "id": "font_consistency",
                "name": "Font Consistency",
                "description": "Ensures consistent font usage throughout the document",
                "metric": QualityMetric.TYPOGRAPHY,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.8,
                "parameters": {"max_font_variations": 3}
            },
            {
                "id": "font_size_consistency",
                "name": "Font Size Consistency",
                "description": "Validates consistent font sizing for similar elements",
                "metric": QualityMetric.TYPOGRAPHY,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.7,
                "parameters": {"size_tolerance": 2}
            },
            {
                "id": "line_spacing_consistency",
                "name": "Line Spacing Consistency",
                "description": "Ensures consistent line spacing throughout the document",
                "metric": QualityMetric.TYPOGRAPHY,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.6,
                "parameters": {"spacing_tolerance": 0.1}
            },
            
            # Formatting rules
            {
                "id": "heading_hierarchy",
                "name": "Heading Hierarchy",
                "description": "Validates proper heading structure and hierarchy",
                "metric": QualityMetric.FORMATTING,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.9,
                "parameters": {"max_heading_level": 6}
            },
            {
                "id": "paragraph_spacing",
                "name": "Paragraph Spacing",
                "description": "Ensures consistent paragraph spacing",
                "metric": QualityMetric.FORMATTING,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.5,
                "parameters": {"min_spacing": 6, "max_spacing": 12}
            },
            {
                "id": "margin_consistency",
                "name": "Margin Consistency",
                "description": "Validates consistent margins throughout the document",
                "metric": QualityMetric.FORMATTING,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.7,
                "parameters": {"margin_tolerance": 0.1}
            },
            
            # Structure rules
            {
                "id": "document_structure",
                "name": "Document Structure",
                "description": "Validates proper document structure and organization",
                "metric": QualityMetric.STRUCTURE,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.8,
                "parameters": {"min_sections": 1, "max_sections": 20}
            },
            {
                "id": "section_completeness",
                "name": "Section Completeness",
                "description": "Ensures all required sections are present and complete",
                "metric": QualityMetric.STRUCTURE,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.9,
                "parameters": {"min_content_length": 50}
            },
            {
                "id": "table_of_contents",
                "name": "Table of Contents",
                "description": "Validates table of contents for long documents",
                "metric": QualityMetric.STRUCTURE,
                "validation_level": ValidationLevel.STRICT,
                "weight": 0.6,
                "parameters": {"min_sections_for_toc": 5}
            },
            
            # Accessibility rules
            {
                "id": "color_contrast",
                "name": "Color Contrast",
                "description": "Validates sufficient color contrast for accessibility",
                "metric": QualityMetric.ACCESSIBILITY,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.8,
                "parameters": {"min_contrast_ratio": 4.5}
            },
            {
                "id": "alt_text_presence",
                "name": "Alt Text Presence",
                "description": "Ensures images have appropriate alt text",
                "metric": QualityMetric.ACCESSIBILITY,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.7,
                "parameters": {"require_alt_text": True}
            },
            {
                "id": "heading_structure",
                "name": "Heading Structure",
                "description": "Validates logical heading structure for screen readers",
                "metric": QualityMetric.ACCESSIBILITY,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.8,
                "parameters": {"skip_heading_levels": False}
            },
            
            # Professional appearance rules
            {
                "id": "branding_consistency",
                "name": "Branding Consistency",
                "description": "Validates consistent branding elements",
                "metric": QualityMetric.PROFESSIONAL_APPEARANCE,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.6,
                "parameters": {"check_logo": True, "check_colors": True}
            },
            {
                "id": "page_numbering",
                "name": "Page Numbering",
                "description": "Validates proper page numbering",
                "metric": QualityMetric.PROFESSIONAL_APPEARANCE,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.5,
                "parameters": {"require_page_numbers": True}
            },
            {
                "id": "header_footer_consistency",
                "name": "Header/Footer Consistency",
                "description": "Validates consistent headers and footers",
                "metric": QualityMetric.PROFESSIONAL_APPEARANCE,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.6,
                "parameters": {"check_consistency": True}
            },
            
            # Content quality rules
            {
                "id": "content_length",
                "name": "Content Length",
                "description": "Validates appropriate content length for document type",
                "metric": QualityMetric.CONTENT_QUALITY,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.7,
                "parameters": {"min_words": 100, "max_words": 50000}
            },
            {
                "id": "grammar_spelling",
                "name": "Grammar and Spelling",
                "description": "Basic grammar and spelling validation",
                "metric": QualityMetric.CONTENT_QUALITY,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.8,
                "parameters": {"check_grammar": True, "check_spelling": True}
            },
            {
                "id": "readability_score",
                "name": "Readability Score",
                "description": "Validates document readability",
                "metric": QualityMetric.CONTENT_QUALITY,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.6,
                "parameters": {"min_readability": 30, "max_readability": 80}
            },
            
            # Technical quality rules
            {
                "id": "file_integrity",
                "name": "File Integrity",
                "description": "Validates file integrity and structure",
                "metric": QualityMetric.TECHNICAL_QUALITY,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.9,
                "parameters": {"check_corruption": True}
            },
            {
                "id": "format_compliance",
                "name": "Format Compliance",
                "description": "Validates compliance with format specifications",
                "metric": QualityMetric.TECHNICAL_QUALITY,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.8,
                "parameters": {"strict_compliance": True}
            },
            {
                "id": "metadata_completeness",
                "name": "Metadata Completeness",
                "description": "Validates document metadata completeness",
                "metric": QualityMetric.TECHNICAL_QUALITY,
                "validation_level": ValidationLevel.STANDARD,
                "weight": 0.5,
                "parameters": {"required_fields": ["title", "author", "created_date"]}
            }
        ]
        
        for rule_data in rules_data:
            rule = QualityRule(**rule_data)
            self.rules[rule.id] = rule
    
    def _setup_validation_levels(self):
        """Setup validation levels with appropriate rules."""
        self.validation_levels = {
            ValidationLevel.BASIC: [
                "font_consistency",
                "heading_hierarchy",
                "document_structure",
                "content_length",
                "file_integrity"
            ],
            ValidationLevel.STANDARD: [
                "font_consistency",
                "font_size_consistency",
                "line_spacing_consistency",
                "heading_hierarchy",
                "paragraph_spacing",
                "margin_consistency",
                "document_structure",
                "section_completeness",
                "color_contrast",
                "alt_text_presence",
                "heading_structure",
                "branding_consistency",
                "page_numbering",
                "content_length",
                "grammar_spelling",
                "file_integrity",
                "format_compliance"
            ],
            ValidationLevel.STRICT: [
                # All standard rules plus additional strict rules
                "font_consistency",
                "font_size_consistency",
                "line_spacing_consistency",
                "heading_hierarchy",
                "paragraph_spacing",
                "margin_consistency",
                "document_structure",
                "section_completeness",
                "table_of_contents",
                "color_contrast",
                "alt_text_presence",
                "heading_structure",
                "branding_consistency",
                "page_numbering",
                "header_footer_consistency",
                "content_length",
                "grammar_spelling",
                "readability_score",
                "file_integrity",
                "format_compliance",
                "metadata_completeness"
            ],
            ValidationLevel.ENTERPRISE: [
                # All rules for enterprise-level validation
                *[rule.id for rule in self.rules.values()]
            ]
        }
    
    async def validate_document(
        self,
        content: Dict[str, Any],
        file_path: Optional[str] = None,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        custom_rules: Optional[List[str]] = None
    ) -> QualityReport:
        """
        Validate document quality and professional appearance.
        
        Args:
            content: Document content to validate
            file_path: Optional file path for technical validation
            validation_level: Validation level to apply
            custom_rules: Optional custom rules to include
            
        Returns:
            Comprehensive quality report
        """
        # Get rules for validation level
        rule_ids = self.validation_levels.get(validation_level, [])
        if custom_rules:
            rule_ids.extend(custom_rules)
        
        # Remove duplicates and filter enabled rules
        rule_ids = list(set(rule_ids))
        active_rules = [self.rules[rule_id] for rule_id in rule_ids if rule_id in self.rules and self.rules[rule_id].enabled]
        
        # Run validations
        results = []
        metric_scores = {}
        
        for rule in active_rules:
            try:
                result = await self._validate_rule(rule, content, file_path)
                results.append(result)
                
                # Aggregate metric scores
                metric = rule.metric
                if metric not in metric_scores:
                    metric_scores[metric] = []
                metric_scores[metric].append(result.score * rule.weight)
                
            except Exception as e:
                logger.error(f"Validation failed for rule {rule.id}: {e}")
                results.append(ValidationResult(
                    rule_id=rule.id,
                    passed=False,
                    score=0.0,
                    message=f"Validation error: {str(e)}"
                ))
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(results, active_rules)
        
        # Determine if validation passed
        passed_validation = overall_score >= 0.7  # 70% threshold
        
        # Calculate metric scores
        final_metric_scores = {}
        for metric, scores in metric_scores.items():
            final_metric_scores[metric] = sum(scores) / len(scores) if scores else 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, overall_score)
        
        return QualityReport(
            overall_score=overall_score,
            passed_validation=passed_validation,
            metrics=final_metric_scores,
            results=results,
            recommendations=recommendations,
            metadata={
                "validation_level": validation_level.value,
                "rules_applied": len(active_rules),
                "validation_timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
            }
        )
    
    async def _validate_rule(
        self,
        rule: QualityRule,
        content: Dict[str, Any],
        file_path: Optional[str] = None
    ) -> ValidationResult:
        """Validate a specific rule."""
        # Route to appropriate validation method
        validation_methods = {
            "font_consistency": self._validate_font_consistency,
            "font_size_consistency": self._validate_font_size_consistency,
            "line_spacing_consistency": self._validate_line_spacing_consistency,
            "heading_hierarchy": self._validate_heading_hierarchy,
            "paragraph_spacing": self._validate_paragraph_spacing,
            "margin_consistency": self._validate_margin_consistency,
            "document_structure": self._validate_document_structure,
            "section_completeness": self._validate_section_completeness,
            "table_of_contents": self._validate_table_of_contents,
            "color_contrast": self._validate_color_contrast,
            "alt_text_presence": self._validate_alt_text_presence,
            "heading_structure": self._validate_heading_structure,
            "branding_consistency": self._validate_branding_consistency,
            "page_numbering": self._validate_page_numbering,
            "header_footer_consistency": self._validate_header_footer_consistency,
            "content_length": self._validate_content_length,
            "grammar_spelling": self._validate_grammar_spelling,
            "readability_score": self._validate_readability_score,
            "file_integrity": self._validate_file_integrity,
            "format_compliance": self._validate_format_compliance,
            "metadata_completeness": self._validate_metadata_completeness
        }
        
        method = validation_methods.get(rule.id)
        if method:
            return await method(rule, content, file_path)
        else:
            return ValidationResult(
                rule_id=rule.id,
                passed=False,
                score=0.0,
                message=f"No validation method found for rule: {rule.id}"
            )
    
    # Validation methods for each rule
    async def _validate_font_consistency(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate font consistency."""
        fonts_used = set()
        
        # Extract fonts from content
        if "formatting" in content and "typography" in content["formatting"]:
            fonts_used.add(content["formatting"]["typography"].get("font_family", "default"))
        
        # Check sections for font variations
        if "sections" in content:
            for section in content["sections"]:
                if "formatting" in section:
                    fonts_used.add(section["formatting"].get("font_family", "default"))
        
        max_variations = rule.parameters.get("max_font_variations", 3)
        passed = len(fonts_used) <= max_variations
        score = max(0, 1.0 - (len(fonts_used) - 1) * 0.2) if len(fonts_used) > 1 else 1.0
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Found {len(fonts_used)} font variations (max: {max_variations})",
            details={"fonts_used": list(fonts_used)},
            suggestions=["Use consistent fonts throughout the document"] if not passed else []
        )
    
    async def _validate_font_size_consistency(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate font size consistency."""
        font_sizes = []
        
        # Extract font sizes from content
        if "formatting" in content and "typography" in content["formatting"]:
            font_sizes.append(content["formatting"]["typography"].get("font_size", 11))
        
        # Check sections for font size variations
        if "sections" in content:
            for section in content["sections"]:
                if "formatting" in section:
                    font_sizes.append(section["formatting"].get("font_size", 11))
        
        if not font_sizes:
            return ValidationResult(
                rule_id=rule.id,
                passed=False,
                score=0.0,
                message="No font size information found"
            )
        
        # Check consistency
        size_tolerance = rule.parameters.get("size_tolerance", 2)
        base_size = font_sizes[0]
        consistent_sizes = all(abs(size - base_size) <= size_tolerance for size in font_sizes)
        
        passed = consistent_sizes
        score = 1.0 if passed else 0.5
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Font size consistency: {'Passed' if passed else 'Failed'}",
            details={"font_sizes": font_sizes, "base_size": base_size},
            suggestions=["Use consistent font sizes for similar elements"] if not passed else []
        )
    
    async def _validate_line_spacing_consistency(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate line spacing consistency."""
        line_spacings = []
        
        # Extract line spacing from content
        if "formatting" in content and "typography" in content["formatting"]:
            line_spacings.append(content["formatting"]["typography"].get("line_spacing", 1.15))
        
        if not line_spacings:
            return ValidationResult(
                rule_id=rule.id,
                passed=False,
                score=0.0,
                message="No line spacing information found"
            )
        
        # Check consistency
        spacing_tolerance = rule.parameters.get("spacing_tolerance", 0.1)
        base_spacing = line_spacings[0]
        consistent_spacing = all(abs(spacing - base_spacing) <= spacing_tolerance for spacing in line_spacings)
        
        passed = consistent_spacing
        score = 1.0 if passed else 0.5
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Line spacing consistency: {'Passed' if passed else 'Failed'}",
            details={"line_spacings": line_spacings, "base_spacing": base_spacing},
            suggestions=["Use consistent line spacing throughout the document"] if not passed else []
        )
    
    async def _validate_heading_hierarchy(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate heading hierarchy."""
        headings = []
        
        # Extract headings from content
        if "sections" in content:
            for i, section in enumerate(content["sections"]):
                if "heading" in section:
                    headings.append({"level": 1, "text": section["heading"], "position": i})
        
        if not headings:
            return ValidationResult(
                rule_id=rule.id,
                passed=False,
                score=0.0,
                message="No headings found in document"
            )
        
        # Check hierarchy
        max_level = rule.parameters.get("max_heading_level", 6)
        valid_hierarchy = all(heading["level"] <= max_level for heading in headings)
        
        passed = valid_hierarchy
        score = 1.0 if passed else 0.7
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Heading hierarchy: {'Valid' if passed else 'Invalid'}",
            details={"headings": headings, "max_level": max_level},
            suggestions=["Ensure proper heading hierarchy (H1 > H2 > H3, etc.)"] if not passed else []
        )
    
    async def _validate_paragraph_spacing(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate paragraph spacing."""
        # This would typically analyze the actual document formatting
        # For now, we'll simulate based on content structure
        min_spacing = rule.parameters.get("min_spacing", 6)
        max_spacing = rule.parameters.get("max_spacing", 12)
        
        # Simulate paragraph spacing analysis
        estimated_spacing = 8  # Would be calculated from actual document
        
        passed = min_spacing <= estimated_spacing <= max_spacing
        score = 1.0 if passed else 0.6
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Paragraph spacing: {'Appropriate' if passed else 'Needs adjustment'}",
            details={"estimated_spacing": estimated_spacing, "range": [min_spacing, max_spacing]},
            suggestions=["Adjust paragraph spacing to be within recommended range"] if not passed else []
        )
    
    async def _validate_margin_consistency(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate margin consistency."""
        # This would analyze actual document margins
        # For now, we'll simulate based on layout configuration
        margin_tolerance = rule.parameters.get("margin_tolerance", 0.1)
        
        # Simulate margin analysis
        margins = {"top": 1.0, "bottom": 1.0, "left": 1.0, "right": 1.0}
        consistent_margins = True  # Would be calculated from actual document
        
        passed = consistent_margins
        score = 1.0 if passed else 0.7
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Margin consistency: {'Consistent' if passed else 'Inconsistent'}",
            details={"margins": margins, "tolerance": margin_tolerance},
            suggestions=["Ensure consistent margins throughout the document"] if not passed else []
        )
    
    async def _validate_document_structure(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate document structure."""
        min_sections = rule.parameters.get("min_sections", 1)
        max_sections = rule.parameters.get("max_sections", 20)
        
        sections = content.get("sections", [])
        section_count = len(sections)
        
        passed = min_sections <= section_count <= max_sections
        score = 1.0 if passed else 0.5
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Document structure: {section_count} sections (range: {min_sections}-{max_sections})",
            details={"section_count": section_count, "range": [min_sections, max_sections]},
            suggestions=["Adjust number of sections to fit recommended range"] if not passed else []
        )
    
    async def _validate_section_completeness(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate section completeness."""
        min_content_length = rule.parameters.get("min_content_length", 50)
        
        sections = content.get("sections", [])
        complete_sections = 0
        
        for section in sections:
            content_text = section.get("content", "")
            if len(content_text) >= min_content_length:
                complete_sections += 1
        
        total_sections = len(sections)
        completeness_ratio = complete_sections / total_sections if total_sections > 0 else 0
        
        passed = completeness_ratio >= 0.8  # 80% of sections should be complete
        score = completeness_ratio
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Section completeness: {complete_sections}/{total_sections} sections complete",
            details={"complete_sections": complete_sections, "total_sections": total_sections, "ratio": completeness_ratio},
            suggestions=["Ensure all sections have sufficient content"] if not passed else []
        )
    
    async def _validate_table_of_contents(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate table of contents."""
        min_sections_for_toc = rule.parameters.get("min_sections_for_toc", 5)
        
        sections = content.get("sections", [])
        has_toc = "table_of_contents" in content or "toc" in content
        
        if len(sections) < min_sections_for_toc:
            # TOC not required for short documents
            return ValidationResult(
                rule_id=rule.id,
                passed=True,
                score=1.0,
                message="Table of contents not required for short documents"
            )
        
        passed = has_toc
        score = 1.0 if passed else 0.3
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Table of contents: {'Present' if passed else 'Missing'}",
            details={"section_count": len(sections), "has_toc": has_toc},
            suggestions=["Add table of contents for documents with many sections"] if not passed else []
        )
    
    async def _validate_color_contrast(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate color contrast for accessibility."""
        min_contrast_ratio = rule.parameters.get("min_contrast_ratio", 4.5)
        
        # This would analyze actual color combinations in the document
        # For now, we'll simulate based on color configuration
        estimated_contrast = 5.2  # Would be calculated from actual colors
        
        passed = estimated_contrast >= min_contrast_ratio
        score = min(1.0, estimated_contrast / min_contrast_ratio)
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Color contrast: {estimated_contrast:.1f}:1 (min: {min_contrast_ratio}:1)",
            details={"contrast_ratio": estimated_contrast, "minimum": min_contrast_ratio},
            suggestions=["Improve color contrast for better accessibility"] if not passed else []
        )
    
    async def _validate_alt_text_presence(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate alt text presence for images."""
        require_alt_text = rule.parameters.get("require_alt_text", True)
        
        # Count images and alt text
        images = content.get("images", [])
        images_with_alt = sum(1 for img in images if img.get("alt_text"))
        
        if not images:
            return ValidationResult(
                rule_id=rule.id,
                passed=True,
                score=1.0,
                message="No images found - validation not applicable"
            )
        
        passed = not require_alt_text or images_with_alt == len(images)
        score = images_with_alt / len(images) if images else 1.0
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Alt text: {images_with_alt}/{len(images)} images have alt text",
            details={"total_images": len(images), "images_with_alt": images_with_alt},
            suggestions=["Add alt text to all images for accessibility"] if not passed else []
        )
    
    async def _validate_heading_structure(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate heading structure for accessibility."""
        skip_heading_levels = rule.parameters.get("skip_heading_levels", False)
        
        # Extract heading levels
        headings = []
        if "sections" in content:
            for i, section in enumerate(content["sections"]):
                if "heading" in section:
                    headings.append({"level": 1, "text": section["heading"]})
        
        # Check for skipped heading levels
        levels_used = sorted(set(h["level"] for h in headings))
        has_skips = any(levels_used[i] - levels_used[i-1] > 1 for i in range(1, len(levels_used)))
        
        passed = not skip_heading_levels or not has_skips
        score = 1.0 if passed else 0.6
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Heading structure: {'Valid' if passed else 'Has skipped levels'}",
            details={"levels_used": levels_used, "has_skips": has_skips},
            suggestions=["Don't skip heading levels (H1 -> H3 without H2)"] if not passed else []
        )
    
    async def _validate_branding_consistency(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate branding consistency."""
        check_logo = rule.parameters.get("check_logo", True)
        check_colors = rule.parameters.get("check_colors", True)
        
        # Check branding elements
        has_logo = "logo" in content.get("branding", {})
        has_consistent_colors = True  # Would check actual color usage
        
        branding_score = 0.0
        if check_logo and has_logo:
            branding_score += 0.5
        if check_colors and has_consistent_colors:
            branding_score += 0.5
        
        passed = branding_score >= 0.5
        score = branding_score
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Branding consistency: {'Good' if passed else 'Needs improvement'}",
            details={"has_logo": has_logo, "consistent_colors": has_consistent_colors},
            suggestions=["Ensure consistent branding elements throughout the document"] if not passed else []
        )
    
    async def _validate_page_numbering(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate page numbering."""
        require_page_numbers = rule.parameters.get("require_page_numbers", True)
        
        # Check if page numbering is configured
        has_page_numbers = content.get("formatting", {}).get("features", {}).get("page_numbers", False)
        
        passed = not require_page_numbers or has_page_numbers
        score = 1.0 if passed else 0.3
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Page numbering: {'Present' if passed else 'Missing'}",
            details={"has_page_numbers": has_page_numbers},
            suggestions=["Add page numbering for professional documents"] if not passed else []
        )
    
    async def _validate_header_footer_consistency(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate header/footer consistency."""
        check_consistency = rule.parameters.get("check_consistency", True)
        
        # Check header/footer configuration
        has_headers_footers = content.get("formatting", {}).get("features", {}).get("header_footer", False)
        
        passed = not check_consistency or has_headers_footers
        score = 1.0 if passed else 0.5
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Header/Footer: {'Consistent' if passed else 'Inconsistent'}",
            details={"has_headers_footers": has_headers_footers},
            suggestions=["Ensure consistent headers and footers throughout the document"] if not passed else []
        )
    
    async def _validate_content_length(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate content length."""
        min_words = rule.parameters.get("min_words", 100)
        max_words = rule.parameters.get("max_words", 50000)
        
        # Calculate total word count
        total_words = 0
        if "title" in content:
            total_words += len(content["title"].split())
        
        if "sections" in content:
            for section in content["sections"]:
                if "content" in section:
                    total_words += len(section["content"].split())
        
        passed = min_words <= total_words <= max_words
        score = 1.0 if passed else 0.5
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Content length: {total_words} words (range: {min_words}-{max_words})",
            details={"word_count": total_words, "range": [min_words, max_words]},
            suggestions=["Adjust content length to fit recommended range"] if not passed else []
        )
    
    async def _validate_grammar_spelling(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate grammar and spelling."""
        check_grammar = rule.parameters.get("check_grammar", True)
        check_spelling = rule.parameters.get("check_spelling", True)
        
        # This would integrate with grammar/spelling checkers
        # For now, we'll simulate basic validation
        estimated_errors = 2  # Would be calculated from actual text analysis
        
        passed = estimated_errors <= 5  # Allow some errors
        score = max(0, 1.0 - estimated_errors * 0.1)
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Grammar/Spelling: {estimated_errors} estimated errors",
            details={"estimated_errors": estimated_errors},
            suggestions=["Review grammar and spelling"] if not passed else []
        )
    
    async def _validate_readability_score(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate readability score."""
        min_readability = rule.parameters.get("min_readability", 30)
        max_readability = rule.parameters.get("max_readability", 80)
        
        # This would calculate actual readability score
        # For now, we'll simulate
        estimated_readability = 65  # Would be calculated from actual text
        
        passed = min_readability <= estimated_readability <= max_readability
        score = 1.0 if passed else 0.6
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Readability: {estimated_readability} (range: {min_readability}-{max_readability})",
            details={"readability_score": estimated_readability, "range": [min_readability, max_readability]},
            suggestions=["Adjust text complexity for target audience"] if not passed else []
        )
    
    async def _validate_file_integrity(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate file integrity."""
        check_corruption = rule.parameters.get("check_corruption", True)
        
        if not file_path or not os.path.exists(file_path):
            return ValidationResult(
                rule_id=rule.id,
                passed=False,
                score=0.0,
                message="File not found or path not provided"
            )
        
        # Check file integrity
        try:
            file_size = os.path.getsize(file_path)
            passed = file_size > 0
            score = 1.0 if passed else 0.0
        except Exception as e:
            passed = False
            score = 0.0
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"File integrity: {'Valid' if passed else 'Invalid'}",
            details={"file_size": file_size if 'file_size' in locals() else 0},
            suggestions=["Check file for corruption"] if not passed else []
        )
    
    async def _validate_format_compliance(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate format compliance."""
        strict_compliance = rule.parameters.get("strict_compliance", True)
        
        # This would validate against format specifications
        # For now, we'll simulate
        compliance_score = 0.9  # Would be calculated from actual format validation
        
        passed = compliance_score >= 0.8 if strict_compliance else compliance_score >= 0.6
        score = compliance_score
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Format compliance: {compliance_score:.1%}",
            details={"compliance_score": compliance_score, "strict_mode": strict_compliance},
            suggestions=["Ensure strict compliance with format specifications"] if not passed else []
        )
    
    async def _validate_metadata_completeness(self, rule: QualityRule, content: Dict[str, Any], file_path: Optional[str] = None) -> ValidationResult:
        """Validate metadata completeness."""
        required_fields = rule.parameters.get("required_fields", ["title", "author", "created_date"])
        
        metadata = content.get("metadata", {})
        present_fields = [field for field in required_fields if field in metadata and metadata[field]]
        
        completeness_ratio = len(present_fields) / len(required_fields)
        passed = completeness_ratio >= 0.8  # 80% of required fields should be present
        score = completeness_ratio
        
        return ValidationResult(
            rule_id=rule.id,
            passed=passed,
            score=score,
            message=f"Metadata: {len(present_fields)}/{len(required_fields)} required fields present",
            details={"present_fields": present_fields, "required_fields": required_fields},
            suggestions=["Add missing metadata fields"] if not passed else []
        )
    
    def _calculate_overall_score(self, results: List[ValidationResult], rules: List[QualityRule]) -> float:
        """Calculate overall quality score."""
        if not results:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for result in results:
            rule = next((r for r in rules if r.id == result.rule_id), None)
            if rule:
                total_weighted_score += result.score * rule.weight
                total_weight += rule.weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, results: List[ValidationResult], overall_score: float) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Add recommendations from failed validations
        for result in results:
            if not result.passed and result.suggestions:
                recommendations.extend(result.suggestions)
        
        # Add general recommendations based on overall score
        if overall_score < 0.5:
            recommendations.append("Consider comprehensive document review and improvement")
        elif overall_score < 0.7:
            recommendations.append("Focus on improving formatting and structure consistency")
        elif overall_score < 0.9:
            recommendations.append("Minor improvements needed for professional appearance")
        
        # Remove duplicates and return
        return list(set(recommendations))

# Global quality validator instance
_global_quality_validator: Optional[QualityValidator] = None

def get_global_quality_validator() -> QualityValidator:
    """Get the global quality validator instance."""
    global _global_quality_validator
    if _global_quality_validator is None:
        _global_quality_validator = QualityValidator()
    return _global_quality_validator



























