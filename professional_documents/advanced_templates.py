"""
Advanced Professional Document Templates
========================================

Enhanced templates with advanced features, visual elements, and complex structures.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

from .models import DocumentTemplate, DocumentType, DocumentStyle
from .templates import ProfessionalTemplates


class TemplateComplexity(str, Enum):
    """Template complexity levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"


class VisualElement(str, Enum):
    """Visual elements that can be included in templates."""
    CHARTS = "charts"
    GRAPHS = "graphs"
    TABLES = "tables"
    IMAGES = "images"
    INFOGRAPHICS = "infographics"
    DIAGRAMS = "diagrams"
    TIMELINES = "timelines"
    FLOWCHARTS = "flowcharts"
    DASHBOARDS = "dashboards"


class AdvancedProfessionalTemplates:
    """Collection of advanced professional document templates with enhanced features."""
    
    @staticmethod
    def get_enterprise_business_report_template() -> DocumentTemplate:
        """Enterprise-grade business report with advanced analytics and visualizations."""
        return DocumentTemplate(
            name="Enterprise Business Report",
            description="Comprehensive enterprise business report with advanced analytics, data visualizations, and executive insights",
            document_type=DocumentType.REPORT,
            sections=[
                "Executive Dashboard",
                "Strategic Overview",
                "Market Analysis & Trends",
                "Financial Performance",
                "Operational Metrics",
                "Competitive Landscape",
                "Risk Assessment",
                "Strategic Recommendations",
                "Implementation Roadmap",
                "Success Metrics & KPIs",
                "Appendices & Data Sources"
            ],
            style=DocumentStyle(
                font_family="Calibri",
                font_size=11,
                line_spacing=1.15,
                header_color="#1f4e79",
                body_color="#2f2f2f",
                accent_color="#4472c4",
                background_color="#ffffff",
                include_page_numbers=True,
                include_watermark=True,
                watermark_text="CONFIDENTIAL"
            ),
            metadata={
                "complexity": TemplateComplexity.ENTERPRISE,
                "target_audience": "executives",
                "formality": "very_high",
                "typical_length": "50-100 pages",
                "visual_elements": [
                    VisualElement.CHARTS,
                    VisualElement.GRAPHS,
                    VisualElement.DASHBOARDS,
                    VisualElement.INFOGRAPHICS
                ],
                "interactive_features": True,
                "data_visualization": True,
                "executive_summary": True,
                "appendix_support": True
            }
        )
    
    @staticmethod
    def get_strategic_proposal_template() -> DocumentTemplate:
        """Strategic proposal template with advanced features and visual elements."""
        return DocumentTemplate(
            name="Strategic Business Proposal",
            description="Advanced strategic proposal with ROI analysis, implementation timelines, and visual project plans",
            document_type=DocumentType.PROPOSAL,
            sections=[
                "Executive Summary",
                "Strategic Context",
                "Problem Analysis",
                "Proposed Solution Architecture",
                "Implementation Methodology",
                "Project Timeline & Milestones",
                "Resource Requirements",
                "Financial Analysis & ROI",
                "Risk Management Plan",
                "Success Metrics & KPIs",
                "Next Steps & Call to Action"
            ],
            style=DocumentStyle(
                font_family="Arial",
                font_size=11,
                line_spacing=1.2,
                header_color="#2c3e50",
                body_color="#34495e",
                accent_color="#3498db",
                background_color="#ffffff",
                include_page_numbers=True
            ),
            metadata={
                "complexity": TemplateComplexity.ADVANCED,
                "target_audience": "decision_makers",
                "formality": "high",
                "typical_length": "20-40 pages",
                "visual_elements": [
                    VisualElement.TIMELINES,
                    VisualElement.FLOWCHARTS,
                    VisualElement.TABLES,
                    VisualElement.DIAGRAMS
                ],
                "roi_analysis": True,
                "project_planning": True,
                "risk_assessment": True,
                "implementation_roadmap": True
            }
        )
    
    @staticmethod
    def get_technical_architecture_document_template() -> DocumentTemplate:
        """Advanced technical architecture document with diagrams and specifications."""
        return DocumentTemplate(
            name="Technical Architecture Document",
            description="Comprehensive technical architecture document with system diagrams, API specifications, and implementation details",
            document_type=DocumentType.TECHNICAL_DOCUMENT,
            sections=[
                "Architecture Overview",
                "System Architecture Diagrams",
                "Component Specifications",
                "API Documentation",
                "Database Design",
                "Security Architecture",
                "Performance Requirements",
                "Scalability Considerations",
                "Integration Points",
                "Deployment Architecture",
                "Monitoring & Observability",
                "Code Examples & Samples",
                "Testing Strategy",
                "Documentation & References"
            ],
            style=DocumentStyle(
                font_family="Consolas",
                font_size=10,
                line_spacing=1.3,
                header_color="#2c3e50",
                body_color="#2c3e50",
                accent_color="#e74c3c",
                background_color="#ffffff",
                include_page_numbers=True
            ),
            metadata={
                "complexity": TemplateComplexity.ADVANCED,
                "target_audience": "developers",
                "formality": "medium",
                "typical_length": "30-80 pages",
                "visual_elements": [
                    VisualElement.DIAGRAMS,
                    VisualElement.FLOWCHARTS,
                    VisualElement.TABLES,
                    VisualElement.INFOGRAPHICS
                ],
                "code_examples": True,
                "api_documentation": True,
                "architecture_diagrams": True,
                "implementation_guides": True
            }
        )
    
    @staticmethod
    def get_research_whitepaper_template() -> DocumentTemplate:
        """Advanced research whitepaper with data analysis and visualizations."""
        return DocumentTemplate(
            name="Research Whitepaper",
            description="Comprehensive research whitepaper with statistical analysis, data visualizations, and industry insights",
            document_type=DocumentType.WHITEPAPER,
            sections=[
                "Abstract & Key Findings",
                "Research Methodology",
                "Market Analysis",
                "Data Analysis & Statistics",
                "Trend Analysis",
                "Case Studies",
                "Industry Benchmarks",
                "Predictive Modeling",
                "Strategic Implications",
                "Recommendations",
                "Future Research Directions",
                "References & Citations"
            ],
            style=DocumentStyle(
                font_family="Georgia",
                font_size=11,
                line_spacing=1.4,
                header_color="#1a365d",
                body_color="#2d3748",
                accent_color="#3182ce",
                background_color="#ffffff",
                include_page_numbers=True
            ),
            metadata={
                "complexity": TemplateComplexity.ADVANCED,
                "target_audience": "researchers",
                "formality": "high",
                "typical_length": "25-60 pages",
                "visual_elements": [
                    VisualElement.GRAPHS,
                    VisualElement.CHARTS,
                    VisualElement.TABLES,
                    VisualElement.INFOGRAPHICS
                ],
                "statistical_analysis": True,
                "data_visualization": True,
                "case_studies": True,
                "predictive_modeling": True,
                "citations": True
            }
        )
    
    @staticmethod
    def get_comprehensive_manual_template() -> DocumentTemplate:
        """Comprehensive user manual with interactive elements and visual guides."""
        return DocumentTemplate(
            name="Comprehensive User Manual",
            description="Advanced user manual with step-by-step guides, visual tutorials, and interactive elements",
            document_type=DocumentType.MANUAL,
            sections=[
                "Quick Start Guide",
                "System Overview",
                "Installation & Setup",
                "Basic Operations",
                "Advanced Features",
                "Configuration Options",
                "Integration Guide",
                "Troubleshooting",
                "FAQ & Common Issues",
                "Best Practices",
                "Video Tutorials",
                "Support Resources",
                "Glossary & Index"
            ],
            style=DocumentStyle(
                font_family="Arial",
                font_size=10,
                line_spacing=1.2,
                header_color="#2c3e50",
                body_color="#34495e",
                accent_color="#3498db",
                background_color="#ffffff",
                include_page_numbers=True
            ),
            metadata={
                "complexity": TemplateComplexity.INTERMEDIATE,
                "target_audience": "end_users",
                "formality": "medium",
                "typical_length": "40-100 pages",
                "visual_elements": [
                    VisualElement.IMAGES,
                    VisualElement.DIAGRAMS,
                    VisualElement.FLOWCHARTS,
                    VisualElement.TABLES
                ],
                "step_by_step_guides": True,
                "visual_tutorials": True,
                "interactive_elements": True,
                "video_support": True,
                "searchable_index": True
            }
        )
    
    @staticmethod
    def get_investor_pitch_deck_template() -> DocumentTemplate:
        """Advanced investor pitch deck with financial models and market analysis."""
        return DocumentTemplate(
            name="Investor Pitch Deck",
            description="Professional investor pitch deck with financial projections, market analysis, and growth strategies",
            document_type=DocumentType.PRESENTATION,
            sections=[
                "Title Slide",
                "Problem Statement",
                "Solution Overview",
                "Market Opportunity",
                "Business Model",
                "Product Demo",
                "Traction & Metrics",
                "Financial Projections",
                "Funding Requirements",
                "Use of Funds",
                "Team & Advisors",
                "Competitive Analysis",
                "Go-to-Market Strategy",
                "Risk Factors",
                "Next Steps"
            ],
            style=DocumentStyle(
                font_family="Arial",
                font_size=14,
                line_spacing=1.2,
                header_color="#2c3e50",
                body_color="#34495e",
                accent_color="#e74c3c",
                background_color="#ffffff",
                include_page_numbers=False
            ),
            metadata={
                "complexity": TemplateComplexity.ADVANCED,
                "target_audience": "investors",
                "formality": "high",
                "typical_length": "15-25 slides",
                "visual_elements": [
                    VisualElement.CHARTS,
                    VisualElement.GRAPHS,
                    VisualElement.INFOGRAPHICS,
                    VisualElement.DASHBOARDS
                ],
                "financial_models": True,
                "market_analysis": True,
                "growth_projections": True,
                "competitive_analysis": True,
                "pitch_optimized": True
            }
        )
    
    @staticmethod
    def get_compliance_document_template() -> DocumentTemplate:
        """Compliance document template with regulatory requirements and audit trails."""
        return DocumentTemplate(
            name="Compliance Document",
            description="Comprehensive compliance document with regulatory requirements, audit trails, and risk assessments",
            document_type=DocumentType.REPORT,
            sections=[
                "Compliance Overview",
                "Regulatory Framework",
                "Current State Assessment",
                "Gap Analysis",
                "Risk Assessment",
                "Compliance Requirements",
                "Implementation Plan",
                "Monitoring & Reporting",
                "Audit Trail",
                "Training Requirements",
                "Documentation Standards",
                "Review & Approval Process",
                "Appendices & References"
            ],
            style=DocumentStyle(
                font_family="Times New Roman",
                font_size=11,
                line_spacing=1.15,
                header_color="#1f4e79",
                body_color="#2f2f2f",
                accent_color="#4472c4",
                background_color="#ffffff",
                include_page_numbers=True,
                include_watermark=True,
                watermark_text="COMPLIANCE DOCUMENT"
            ),
            metadata={
                "complexity": TemplateComplexity.ENTERPRISE,
                "target_audience": "compliance_officers",
                "formality": "very_high",
                "typical_length": "30-70 pages",
                "visual_elements": [
                    VisualElement.TABLES,
                    VisualElement.FLOWCHARTS,
                    VisualElement.DIAGRAMS
                ],
                "regulatory_compliance": True,
                "audit_trail": True,
                "risk_assessment": True,
                "documentation_standards": True,
                "approval_workflow": True
            }
        )
    
    @staticmethod
    def get_marketing_campaign_template() -> DocumentTemplate:
        """Advanced marketing campaign template with analytics and performance metrics."""
        return DocumentTemplate(
            name="Marketing Campaign Strategy",
            description="Comprehensive marketing campaign strategy with analytics, performance metrics, and ROI analysis",
            document_type=DocumentType.REPORT,
            sections=[
                "Campaign Overview",
                "Target Audience Analysis",
                "Market Research",
                "Campaign Objectives",
                "Creative Strategy",
                "Channel Strategy",
                "Content Calendar",
                "Budget Allocation",
                "Performance Metrics",
                "ROI Projections",
                "Risk Mitigation",
                "Success Criteria",
                "Campaign Timeline",
                "Post-Campaign Analysis"
            ],
            style=DocumentStyle(
                font_family="Arial",
                font_size=11,
                line_spacing=1.2,
                header_color="#8e44ad",
                body_color="#2c3e50",
                accent_color="#e74c3c",
                background_color="#ffffff",
                include_page_numbers=True
            ),
            metadata={
                "complexity": TemplateComplexity.ADVANCED,
                "target_audience": "marketing_teams",
                "formality": "medium",
                "typical_length": "25-50 pages",
                "visual_elements": [
                    VisualElement.CHARTS,
                    VisualElement.GRAPHS,
                    VisualElement.INFOGRAPHICS,
                    VisualElement.TIMELINES
                ],
                "analytics_integration": True,
                "performance_metrics": True,
                "roi_analysis": True,
                "campaign_optimization": True,
                "a_b_testing": True
            }
        )


class AdvancedTemplateManager:
    """Advanced manager for document templates with enhanced features."""
    
    def __init__(self):
        self._templates: Dict[str, DocumentTemplate] = {}
        self._advanced_templates: Dict[str, DocumentTemplate] = {}
        self._load_all_templates()
    
    def _load_all_templates(self):
        """Load all templates including advanced ones."""
        # Load basic templates
        basic_templates = [
            ProfessionalTemplates.get_business_report_template(),
            ProfessionalTemplates.get_proposal_template(),
            ProfessionalTemplates.get_technical_document_template(),
            ProfessionalTemplates.get_academic_paper_template(),
            ProfessionalTemplates.get_whitepaper_template(),
            ProfessionalTemplates.get_manual_template(),
            ProfessionalTemplates.get_business_plan_template(),
            ProfessionalTemplates.get_newsletter_template(),
            ProfessionalTemplates.get_brochure_template(),
            ProfessionalTemplates.get_guide_template(),
            ProfessionalTemplates.get_catalog_template(),
            ProfessionalTemplates.get_presentation_template()
        ]
        
        for template in basic_templates:
            self._templates[template.id] = template
        
        # Load advanced templates
        advanced_templates = [
            AdvancedProfessionalTemplates.get_enterprise_business_report_template(),
            AdvancedProfessionalTemplates.get_strategic_proposal_template(),
            AdvancedProfessionalTemplates.get_technical_architecture_document_template(),
            AdvancedProfessionalTemplates.get_research_whitepaper_template(),
            AdvancedProfessionalTemplates.get_comprehensive_manual_template(),
            AdvancedProfessionalTemplates.get_investor_pitch_deck_template(),
            AdvancedProfessionalTemplates.get_compliance_document_template(),
            AdvancedProfessionalTemplates.get_marketing_campaign_template()
        ]
        
        for template in advanced_templates:
            self._advanced_templates[template.id] = template
    
    def get_template(self, template_id: str) -> DocumentTemplate:
        """Get a template by ID."""
        if template_id in self._templates:
            return self._templates[template_id]
        elif template_id in self._advanced_templates:
            return self._advanced_templates[template_id]
        else:
            raise ValueError(f"Template with ID {template_id} not found")
    
    def get_templates_by_type(self, document_type: DocumentType) -> List[DocumentTemplate]:
        """Get all templates for a specific document type."""
        templates = []
        
        # Add basic templates
        for template in self._templates.values():
            if template.document_type == document_type:
                templates.append(template)
        
        # Add advanced templates
        for template in self._advanced_templates.values():
            if template.document_type == document_type:
                templates.append(template)
        
        return templates
    
    def get_advanced_templates(self) -> List[DocumentTemplate]:
        """Get all advanced templates."""
        return list(self._advanced_templates.values())
    
    def get_templates_by_complexity(self, complexity: TemplateComplexity) -> List[DocumentTemplate]:
        """Get templates by complexity level."""
        templates = []
        
        for template in self._advanced_templates.values():
            if template.metadata.get("complexity") == complexity:
                templates.append(template)
        
        return templates
    
    def get_templates_with_visual_elements(self, visual_element: VisualElement) -> List[DocumentTemplate]:
        """Get templates that include specific visual elements."""
        templates = []
        
        for template in self._advanced_templates.values():
            visual_elements = template.metadata.get("visual_elements", [])
            if visual_element in visual_elements:
                templates.append(template)
        
        return templates
    
    def get_all_templates(self) -> List[DocumentTemplate]:
        """Get all available templates."""
        all_templates = list(self._templates.values()) + list(self._advanced_templates.values())
        return all_templates
    
    def get_default_template(self, document_type: DocumentType) -> DocumentTemplate:
        """Get the default template for a document type."""
        templates = self.get_templates_by_type(document_type)
        if not templates:
            raise ValueError(f"No templates found for document type {document_type}")
        
        # Prefer basic templates as default
        basic_templates = [t for t in templates if t.id in self._templates]
        if basic_templates:
            return basic_templates[0]
        else:
            return templates[0]
    
    def get_advanced_template(self, document_type: DocumentType) -> Optional[DocumentTemplate]:
        """Get an advanced template for a document type."""
        advanced_templates = [t for t in self._advanced_templates.values() if t.document_type == document_type]
        return advanced_templates[0] if advanced_templates else None
    
    def add_custom_template(self, template: DocumentTemplate):
        """Add a custom template."""
        if template.metadata.get("complexity") in [TemplateComplexity.ADVANCED, TemplateComplexity.ENTERPRISE]:
            self._advanced_templates[template.id] = template
        else:
            self._templates[template.id] = template
    
    def remove_template(self, template_id: str):
        """Remove a template."""
        if template_id in self._templates:
            del self._templates[template_id]
        elif template_id in self._advanced_templates:
            del self._advanced_templates[template_id]
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """Get statistics about available templates."""
        total_templates = len(self._templates) + len(self._advanced_templates)
        basic_templates = len(self._templates)
        advanced_templates = len(self._advanced_templates)
        
        # Count by complexity
        complexity_counts = {}
        for template in self._advanced_templates.values():
            complexity = template.metadata.get("complexity", TemplateComplexity.BASIC)
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        # Count by document type
        type_counts = {}
        for template in self.get_all_templates():
            doc_type = template.document_type.value
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            "total_templates": total_templates,
            "basic_templates": basic_templates,
            "advanced_templates": advanced_templates,
            "complexity_distribution": complexity_counts,
            "type_distribution": type_counts
        }


# Global advanced template manager instance
advanced_template_manager = AdvancedTemplateManager()




























