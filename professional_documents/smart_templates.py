"""
Smart Templates Service
======================

Intelligent templates that adapt to content and context.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from uuid import uuid4
import json
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class TemplateType(str, Enum):
    """Template type."""
    BUSINESS = "business"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    LEGAL = "legal"
    CREATIVE = "creative"
    PERSONAL = "personal"
    ADAPTIVE = "adaptive"


class ContentCategory(str, Enum):
    """Content category."""
    PROPOSAL = "proposal"
    REPORT = "report"
    PRESENTATION = "presentation"
    MANUAL = "manual"
    CONTRACT = "contract"
    LETTER = "letter"
    ARTICLE = "article"
    BLOG = "blog"
    EMAIL = "email"
    MEMO = "memo"


@dataclass
class SmartTemplate:
    """Smart template."""
    template_id: str
    name: str
    description: str
    template_type: TemplateType
    content_category: ContentCategory
    base_template: str
    adaptive_rules: List[Dict[str, Any]]
    context_requirements: Dict[str, Any]
    style_guidelines: Dict[str, Any]
    created_at: datetime
    version: str
    is_active: bool = True


@dataclass
class TemplateMatch:
    """Template match result."""
    template: SmartTemplate
    match_score: float
    confidence: float
    reasoning: str
    adaptations: List[Dict[str, Any]]


@dataclass
class ContentAnalysis:
    """Content analysis for template selection."""
    content_type: str
    keywords: List[str]
    entities: List[str]
    tone: str
    complexity: str
    length_category: str
    structure_hints: List[str]
    domain: str


class SmartTemplatesService:
    """Smart templates service."""
    
    def __init__(self):
        self.templates: Dict[str, SmartTemplate] = {}
        self.template_rules: Dict[str, List[Dict[str, Any]]] = {}
        self.content_patterns: Dict[str, List[str]] = {}
        self._initialize_default_templates()
        self._initialize_patterns()
    
    def _initialize_default_templates(self):
        """Initialize default smart templates."""
        
        # Business Proposal Template
        business_proposal = SmartTemplate(
            template_id="business_proposal_v1",
            name="Business Proposal",
            description="Adaptive business proposal template",
            template_type=TemplateType.BUSINESS,
            content_category=ContentCategory.PROPOSAL,
            base_template=self._get_business_proposal_template(),
            adaptive_rules=[
                {
                    "condition": "content_contains:['budget', 'cost', 'pricing']",
                    "action": "add_financial_section",
                    "priority": "high"
                },
                {
                    "condition": "content_contains:['timeline', 'schedule', 'deadline']",
                    "action": "add_timeline_section",
                    "priority": "medium"
                },
                {
                    "condition": "content_contains:['team', 'staff', 'personnel']",
                    "action": "add_team_section",
                    "priority": "medium"
                }
            ],
            context_requirements={
                "min_length": 500,
                "required_sections": ["executive_summary", "problem_statement", "solution"],
                "optional_sections": ["budget", "timeline", "team", "risks"]
            },
            style_guidelines={
                "tone": "professional",
                "formality": "high",
                "structure": "formal",
                "visual_elements": ["charts", "tables", "diagrams"]
            },
            created_at=datetime.now(),
            version="1.0"
        )
        
        # Technical Report Template
        technical_report = SmartTemplate(
            template_id="technical_report_v1",
            name="Technical Report",
            description="Adaptive technical report template",
            template_type=TemplateType.TECHNICAL,
            content_category=ContentCategory.REPORT,
            base_template=self._get_technical_report_template(),
            adaptive_rules=[
                {
                    "condition": "content_contains:['code', 'programming', 'software']",
                    "action": "add_code_blocks",
                    "priority": "high"
                },
                {
                    "condition": "content_contains:['data', 'analysis', 'results']",
                    "action": "add_data_visualization",
                    "priority": "high"
                },
                {
                    "condition": "content_contains:['methodology', 'approach', 'process']",
                    "action": "add_methodology_section",
                    "priority": "medium"
                }
            ],
            context_requirements={
                "min_length": 1000,
                "required_sections": ["abstract", "introduction", "methodology", "results", "conclusion"],
                "optional_sections": ["appendix", "references", "glossary"]
            },
            style_guidelines={
                "tone": "technical",
                "formality": "high",
                "structure": "academic",
                "visual_elements": ["charts", "graphs", "code_blocks", "diagrams"]
            },
            created_at=datetime.now(),
            version="1.0"
        )
        
        # Academic Paper Template
        academic_paper = SmartTemplate(
            template_id="academic_paper_v1",
            name="Academic Paper",
            description="Adaptive academic paper template",
            template_type=TemplateType.ACADEMIC,
            content_category=ContentCategory.ARTICLE,
            base_template=self._get_academic_paper_template(),
            adaptive_rules=[
                {
                    "condition": "content_contains:['research', 'study', 'experiment']",
                    "action": "add_research_methodology",
                    "priority": "high"
                },
                {
                    "condition": "content_contains:['literature', 'references', 'citations']",
                    "action": "add_literature_review",
                    "priority": "high"
                },
                {
                    "condition": "content_contains:['hypothesis', 'theory', 'model']",
                    "action": "add_theoretical_framework",
                    "priority": "medium"
                }
            ],
            context_requirements={
                "min_length": 2000,
                "required_sections": ["abstract", "introduction", "literature_review", "methodology", "results", "discussion", "conclusion", "references"],
                "optional_sections": ["acknowledgments", "appendix"]
            },
            style_guidelines={
                "tone": "academic",
                "formality": "very_high",
                "structure": "formal_academic",
                "visual_elements": ["tables", "figures", "equations"]
            },
            created_at=datetime.now(),
            version="1.0"
        )
        
        # Store templates
        self.templates[business_proposal.template_id] = business_proposal
        self.templates[technical_report.template_id] = technical_report
        self.templates[academic_paper.template_id] = academic_paper
    
    def _initialize_patterns(self):
        """Initialize content patterns for template matching."""
        
        self.content_patterns = {
            "business": [
                "proposal", "business plan", "strategy", "marketing", "sales",
                "revenue", "profit", "customer", "client", "partnership"
            ],
            "technical": [
                "software", "programming", "code", "algorithm", "system",
                "database", "API", "framework", "architecture", "development"
            ],
            "academic": [
                "research", "study", "analysis", "hypothesis", "methodology",
                "literature", "theory", "experiment", "data", "conclusion"
            ],
            "legal": [
                "contract", "agreement", "terms", "conditions", "liability",
                "compliance", "regulation", "law", "legal", "rights"
            ],
            "creative": [
                "story", "narrative", "character", "plot", "theme",
                "creative", "artistic", "design", "visual", "aesthetic"
            ]
        }
    
    async def find_best_template(
        self,
        content: str,
        context: Dict[str, Any] = None,
        preferences: Dict[str, Any] = None
    ) -> TemplateMatch:
        """Find the best template for given content."""
        
        try:
            # Analyze content
            content_analysis = await self._analyze_content(content)
            
            # Score all templates
            template_scores = []
            for template in self.templates.values():
                if not template.is_active:
                    continue
                
                score = await self._score_template(template, content_analysis, context, preferences)
                template_scores.append((template, score))
            
            # Sort by score
            template_scores.sort(key=lambda x: x[1], reverse=True)
            
            if not template_scores:
                raise ValueError("No suitable templates found")
            
            best_template, best_score = template_scores[0]
            
            # Generate adaptations
            adaptations = await self._generate_adaptations(best_template, content_analysis, context)
            
            # Generate reasoning
            reasoning = await self._generate_reasoning(best_template, content_analysis, best_score)
            
            return TemplateMatch(
                template=best_template,
                match_score=best_score,
                confidence=min(1.0, best_score / 100),
                reasoning=reasoning,
                adaptations=adaptations
            )
            
        except Exception as e:
            logger.error(f"Error finding best template: {str(e)}")
            raise
    
    async def _analyze_content(self, content: str) -> ContentAnalysis:
        """Analyze content to determine characteristics."""
        
        # Extract keywords
        keywords = self._extract_keywords(content)
        
        # Determine content type
        content_type = self._determine_content_type(content, keywords)
        
        # Extract entities
        entities = self._extract_entities(content)
        
        # Determine tone
        tone = self._determine_tone(content)
        
        # Determine complexity
        complexity = self._determine_complexity(content)
        
        # Determine length category
        length_category = self._determine_length_category(content)
        
        # Extract structure hints
        structure_hints = self._extract_structure_hints(content)
        
        # Determine domain
        domain = self._determine_domain(keywords)
        
        return ContentAnalysis(
            content_type=content_type,
            keywords=keywords,
            entities=entities,
            tone=tone,
            complexity=complexity,
            length_category=length_category,
            structure_hints=structure_hints,
            domain=domain
        )
    
    async def _score_template(
        self,
        template: SmartTemplate,
        content_analysis: ContentAnalysis,
        context: Dict[str, Any],
        preferences: Dict[str, Any]
    ) -> float:
        """Score a template against content analysis."""
        
        score = 0.0
        
        # Domain matching (40% weight)
        domain_score = self._calculate_domain_score(template, content_analysis.domain)
        score += domain_score * 0.4
        
        # Content type matching (30% weight)
        type_score = self._calculate_type_score(template, content_analysis.content_type)
        score += type_score * 0.3
        
        # Style matching (20% weight)
        style_score = self._calculate_style_score(template, content_analysis)
        score += style_score * 0.2
        
        # Context matching (10% weight)
        context_score = self._calculate_context_score(template, context)
        score += context_score * 0.1
        
        return min(100.0, score)
    
    def _calculate_domain_score(self, template: SmartTemplate, domain: str) -> float:
        """Calculate domain matching score."""
        
        domain_mapping = {
            "business": TemplateType.BUSINESS,
            "technical": TemplateType.TECHNICAL,
            "academic": TemplateType.ACADEMIC,
            "legal": TemplateType.LEGAL,
            "creative": TemplateType.CREATIVE
        }
        
        expected_type = domain_mapping.get(domain)
        if expected_type and template.template_type == expected_type:
            return 100.0
        elif expected_type and template.template_type == TemplateType.ADAPTIVE:
            return 80.0
        else:
            return 20.0
    
    def _calculate_type_score(self, template: SmartTemplate, content_type: str) -> float:
        """Calculate content type matching score."""
        
        type_mapping = {
            "proposal": ContentCategory.PROPOSAL,
            "report": ContentCategory.REPORT,
            "presentation": ContentCategory.PRESENTATION,
            "manual": ContentCategory.MANUAL,
            "contract": ContentCategory.CONTRACT,
            "letter": ContentCategory.LETTER,
            "article": ContentCategory.ARTICLE,
            "blog": ContentCategory.BLOG,
            "email": ContentCategory.EMAIL,
            "memo": ContentCategory.MEMO
        }
        
        expected_category = type_mapping.get(content_type)
        if expected_category and template.content_category == expected_category:
            return 100.0
        else:
            return 50.0
    
    def _calculate_style_score(self, template: SmartTemplate, content_analysis: ContentAnalysis) -> float:
        """Calculate style matching score."""
        
        score = 0.0
        
        # Tone matching
        template_tone = template.style_guidelines.get("tone", "neutral")
        if template_tone == content_analysis.tone:
            score += 50.0
        elif template_tone == "professional" and content_analysis.tone in ["formal", "academic"]:
            score += 40.0
        else:
            score += 20.0
        
        # Complexity matching
        template_complexity = template.style_guidelines.get("complexity", "medium")
        if template_complexity == content_analysis.complexity:
            score += 30.0
        else:
            score += 15.0
        
        # Length matching
        if template.context_requirements.get("min_length", 0) <= len(content_analysis.keywords) * 10:
            score += 20.0
        else:
            score += 5.0
        
        return score
    
    def _calculate_context_score(self, template: SmartTemplate, context: Dict[str, Any]) -> float:
        """Calculate context matching score."""
        
        if not context:
            return 50.0
        
        score = 50.0
        
        # Check if context requirements are met
        requirements = template.context_requirements
        for key, value in requirements.items():
            if key in context and context[key] == value:
                score += 10.0
        
        return min(100.0, score)
    
    async def _generate_adaptations(
        self,
        template: SmartTemplate,
        content_analysis: ContentAnalysis,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate template adaptations."""
        
        adaptations = []
        
        # Apply adaptive rules
        for rule in template.adaptive_rules:
            if await self._evaluate_rule_condition(rule["condition"], content_analysis, context):
                adaptation = {
                    "action": rule["action"],
                    "priority": rule["priority"],
                    "description": self._get_adaptation_description(rule["action"]),
                    "parameters": self._get_adaptation_parameters(rule["action"], content_analysis)
                }
                adaptations.append(adaptation)
        
        # Sort by priority
        priority_order = {"high": 3, "medium": 2, "low": 1}
        adaptations.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)
        
        return adaptations
    
    async def _evaluate_rule_condition(
        self,
        condition: str,
        content_analysis: ContentAnalysis,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate a rule condition."""
        
        if condition.startswith("content_contains:"):
            keywords = condition.split(":")[1].strip("[]").split(",")
            keywords = [kw.strip().strip("'\"") for kw in keywords]
            return any(keyword in content_analysis.keywords for keyword in keywords)
        
        elif condition.startswith("context_has:"):
            key = condition.split(":")[1].strip()
            return key in context and context[key] is not None
        
        elif condition.startswith("length_"):
            length_condition = condition.split("_", 1)[1]
            content_length = len(content_analysis.keywords) * 10  # Rough estimate
            if ">" in length_condition:
                min_length = int(length_condition.split(">")[1])
                return content_length > min_length
            elif "<" in length_condition:
                max_length = int(length_condition.split("<")[1])
                return content_length < max_length
        
        return False
    
    def _get_adaptation_description(self, action: str) -> str:
        """Get adaptation description."""
        
        descriptions = {
            "add_financial_section": "Add financial analysis and budget section",
            "add_timeline_section": "Add project timeline and milestones section",
            "add_team_section": "Add team and personnel information section",
            "add_code_blocks": "Add code formatting and syntax highlighting",
            "add_data_visualization": "Add charts and data visualization elements",
            "add_methodology_section": "Add detailed methodology and approach section",
            "add_research_methodology": "Add research methodology and data collection section",
            "add_literature_review": "Add literature review and references section",
            "add_theoretical_framework": "Add theoretical framework and hypothesis section"
        }
        
        return descriptions.get(action, f"Apply {action} adaptation")
    
    def _get_adaptation_parameters(self, action: str, content_analysis: ContentAnalysis) -> Dict[str, Any]:
        """Get adaptation parameters."""
        
        parameters = {
            "content_analysis": {
                "keywords": content_analysis.keywords[:10],  # Limit for display
                "tone": content_analysis.tone,
                "complexity": content_analysis.complexity
            }
        }
        
        if action in ["add_financial_section", "add_timeline_section"]:
            parameters["suggested_placement"] = "after_solution_section"
        elif action in ["add_code_blocks", "add_data_visualization"]:
            parameters["suggested_placement"] = "inline_with_content"
        elif action in ["add_methodology_section", "add_research_methodology"]:
            parameters["suggested_placement"] = "before_results_section"
        
        return parameters
    
    async def _generate_reasoning(
        self,
        template: SmartTemplate,
        content_analysis: ContentAnalysis,
        score: float
    ) -> str:
        """Generate reasoning for template selection."""
        
        reasons = []
        
        # Domain match
        if template.template_type.value == content_analysis.domain:
            reasons.append(f"Perfect domain match ({template.template_type.value})")
        elif template.template_type == TemplateType.ADAPTIVE:
            reasons.append("Adaptive template suitable for multiple domains")
        
        # Content type match
        reasons.append(f"Designed for {template.content_category.value} content")
        
        # Style match
        template_tone = template.style_guidelines.get("tone", "neutral")
        if template_tone == content_analysis.tone:
            reasons.append(f"Tone match ({template_tone})")
        
        # Score
        if score >= 80:
            reasons.append("High compatibility score")
        elif score >= 60:
            reasons.append("Good compatibility score")
        else:
            reasons.append("Moderate compatibility score")
        
        return "; ".join(reasons)
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content."""
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Remove common stop words
        stop_words = {
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before",
            "after", "above", "below", "between", "among", "this", "that", "these",
            "those", "i", "you", "he", "she", "it", "we", "they", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "do",
            "does", "did", "will", "would", "could", "should", "may", "might",
            "must", "can", "shall"
        }
        
        keywords = [word for word in words if word not in stop_words]
        
        # Count frequency and return top keywords
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(20)]
    
    def _determine_content_type(self, content: str, keywords: List[str]) -> str:
        """Determine content type."""
        
        type_indicators = {
            "proposal": ["proposal", "propose", "suggest", "recommend", "plan"],
            "report": ["report", "analysis", "findings", "results", "conclusion"],
            "presentation": ["presentation", "slides", "overview", "summary"],
            "manual": ["manual", "guide", "instructions", "how to", "steps"],
            "contract": ["contract", "agreement", "terms", "conditions", "legal"],
            "letter": ["dear", "sincerely", "yours", "regards", "letter"],
            "article": ["article", "research", "study", "analysis", "paper"],
            "blog": ["blog", "post", "opinion", "thoughts", "experience"],
            "email": ["email", "message", "subject", "recipient"],
            "memo": ["memo", "memorandum", "internal", "staff", "team"]
        }
        
        for content_type, indicators in type_indicators.items():
            if any(indicator in content.lower() for indicator in indicators):
                return content_type
        
        return "general"
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities from content."""
        
        # Simple entity extraction (in production, use NER)
        entities = []
        
        # Extract capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', content)
        entities.extend(capitalized_words[:10])  # Limit to 10
        
        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        entities.extend(emails)
        
        # Extract URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        entities.extend(urls)
        
        return list(set(entities))  # Remove duplicates
    
    def _determine_tone(self, content: str) -> str:
        """Determine content tone."""
        
        # Simple tone detection based on keywords
        formal_words = ["therefore", "however", "furthermore", "moreover", "consequently"]
        casual_words = ["hey", "awesome", "cool", "great", "amazing"]
        technical_words = ["algorithm", "methodology", "analysis", "framework", "implementation"]
        
        content_lower = content.lower()
        
        formal_count = sum(1 for word in formal_words if word in content_lower)
        casual_count = sum(1 for word in casual_words if word in content_lower)
        technical_count = sum(1 for word in technical_words if word in content_lower)
        
        if technical_count > formal_count and technical_count > casual_count:
            return "technical"
        elif formal_count > casual_count:
            return "formal"
        elif casual_count > formal_count:
            return "casual"
        else:
            return "neutral"
    
    def _determine_complexity(self, content: str) -> str:
        """Determine content complexity."""
        
        # Simple complexity analysis
        sentences = re.split(r'[.!?]+', content)
        words = content.split()
        
        if not sentences or not words:
            return "low"
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        if avg_sentence_length > 20 or avg_word_length > 6:
            return "high"
        elif avg_sentence_length > 15 or avg_word_length > 5:
            return "medium"
        else:
            return "low"
    
    def _determine_length_category(self, content: str) -> str:
        """Determine content length category."""
        
        word_count = len(content.split())
        
        if word_count < 100:
            return "short"
        elif word_count < 500:
            return "medium"
        elif word_count < 2000:
            return "long"
        else:
            return "very_long"
    
    def _extract_structure_hints(self, content: str) -> List[str]:
        """Extract structure hints from content."""
        
        hints = []
        
        # Check for headings
        if re.search(r'^#+\s', content, re.MULTILINE):
            hints.append("has_headings")
        
        # Check for lists
        if re.search(r'^\s*[-*+]\s', content, re.MULTILINE):
            hints.append("has_lists")
        
        # Check for numbered lists
        if re.search(r'^\s*\d+\.\s', content, re.MULTILINE):
            hints.append("has_numbered_lists")
        
        # Check for links
        if re.search(r'\[([^\]]+)\]\(([^)]+)\)', content):
            hints.append("has_links")
        
        # Check for tables
        if '|' in content and content.count('|') > 10:
            hints.append("has_tables")
        
        return hints
    
    def _determine_domain(self, keywords: List[str]) -> str:
        """Determine content domain."""
        
        domain_scores = {}
        
        for domain, domain_keywords in self.content_patterns.items():
            score = sum(1 for keyword in keywords if keyword in domain_keywords)
            domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        else:
            return "general"
    
    def _get_business_proposal_template(self) -> str:
        """Get business proposal template."""
        
        return """
# {title}

## Executive Summary
{executive_summary}

## Problem Statement
{problem_statement}

## Proposed Solution
{solution}

## Implementation Plan
{implementation_plan}

## Budget and Resources
{budget}

## Timeline
{timeline}

## Team and Expertise
{team}

## Risk Assessment
{risks}

## Conclusion
{conclusion}

---
*Prepared by: {author}*  
*Date: {date}*
"""
    
    def _get_technical_report_template(self) -> str:
        """Get technical report template."""
        
        return """
# {title}

## Abstract
{abstract}

## 1. Introduction
{introduction}

## 2. Methodology
{methodology}

## 3. Results
{results}

## 4. Analysis
{analysis}

## 5. Discussion
{discussion}

## 6. Conclusion
{conclusion}

## References
{references}

## Appendix
{appendix}

---
*Author: {author}*  
*Date: {date}*  
*Version: {version}*
"""
    
    def _get_academic_paper_template(self) -> str:
        """Get academic paper template."""
        
        return """
# {title}

**Authors:** {authors}  
**Institution:** {institution}  
**Date:** {date}

## Abstract
{abstract}

## Keywords
{keywords}

## 1. Introduction
{introduction}

## 2. Literature Review
{literature_review}

## 3. Theoretical Framework
{theoretical_framework}

## 4. Methodology
{methodology}

## 5. Results
{results}

## 6. Discussion
{discussion}

## 7. Conclusion
{conclusion}

## 8. References
{references}

## Acknowledgments
{acknowledgments}

---
*Corresponding author: {corresponding_author}*  
*Email: {email}*
"""
    
    async def apply_template(
        self,
        template: SmartTemplate,
        content: str,
        adaptations: List[Dict[str, Any]] = None,
        context: Dict[str, Any] = None
    ) -> str:
        """Apply template to content with adaptations."""
        
        try:
            # Start with base template
            result = template.base_template
            
            # Apply adaptations
            if adaptations:
                for adaptation in adaptations:
                    result = await self._apply_adaptation(result, adaptation, content, context)
            
            # Fill in context variables
            if context:
                for key, value in context.items():
                    result = result.replace(f"{{{key}}}", str(value))
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying template: {str(e)}")
            raise
    
    async def _apply_adaptation(
        self,
        template: str,
        adaptation: Dict[str, Any],
        content: str,
        context: Dict[str, Any]
    ) -> str:
        """Apply a specific adaptation to template."""
        
        action = adaptation["action"]
        
        if action == "add_financial_section":
            financial_section = """
## Budget and Financial Analysis
{budget_analysis}

### Cost Breakdown
{cost_breakdown}

### Return on Investment
{roi_analysis}
"""
            return template.replace("{budget}", financial_section)
        
        elif action == "add_timeline_section":
            timeline_section = """
## Project Timeline
{timeline}

### Key Milestones
{milestones}

### Deliverables
{deliverables}
"""
            return template.replace("{timeline}", timeline_section)
        
        elif action == "add_code_blocks":
            # Add code formatting instructions
            return template.replace("{code}", "```\n{code}\n```")
        
        elif action == "add_data_visualization":
            # Add data visualization placeholders
            viz_section = """
## Data Visualization
{charts}

### Key Metrics
{metrics}
"""
            return template.replace("{results}", viz_section + "\n{results}")
        
        return template
    
    async def get_template_suggestions(
        self,
        content_preview: str,
        user_preferences: Dict[str, Any] = None
    ) -> List[TemplateMatch]:
        """Get template suggestions based on content preview."""
        
        try:
            # Analyze preview
            content_analysis = await self._analyze_content(content_preview)
            
            # Get all template matches
            suggestions = []
            for template in self.templates.values():
                if not template.is_active:
                    continue
                
                score = await self._score_template(template, content_analysis, {}, user_preferences)
                if score > 30:  # Minimum threshold
                    adaptations = await self._generate_adaptations(template, content_analysis, {})
                    reasoning = await self._generate_reasoning(template, content_analysis, score)
                    
                    suggestions.append(TemplateMatch(
                        template=template,
                        match_score=score,
                        confidence=min(1.0, score / 100),
                        reasoning=reasoning,
                        adaptations=adaptations
                    ))
            
            # Sort by score
            suggestions.sort(key=lambda x: x.match_score, reverse=True)
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            logger.error(f"Error getting template suggestions: {str(e)}")
            return []



























