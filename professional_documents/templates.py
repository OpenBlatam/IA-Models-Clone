"""
Professional Document Templates
===============================

Pre-defined templates for different types of professional documents.
Each template includes structure, styling, and content guidelines.
"""

from typing import Dict, List, Any
from .models import DocumentTemplate, DocumentType, DocumentStyle


class ProfessionalTemplates:
    """Collection of professional document templates."""
    
    @staticmethod
    def get_business_report_template() -> DocumentTemplate:
        """Business report template with executive summary, analysis, and recommendations."""
        return DocumentTemplate(
            name="Business Report",
            description="Comprehensive business report with executive summary, analysis, and recommendations",
            document_type=DocumentType.REPORT,
            sections=[
                "Executive Summary",
                "Introduction",
                "Methodology",
                "Findings and Analysis",
                "Recommendations",
                "Implementation Plan",
                "Conclusion",
                "Appendices"
            ],
            style=DocumentStyle(
                font_family="Calibri",
                font_size=11,
                line_spacing=1.15,
                header_color="#1f4e79",
                body_color="#2f2f2f",
                accent_color="#4472c4",
                include_page_numbers=True
            ),
            metadata={
                "target_audience": "executives",
                "formality": "high",
                "typical_length": "10-50 pages"
            }
        )
    
    @staticmethod
    def get_proposal_template() -> DocumentTemplate:
        """Professional proposal template for business opportunities."""
        return DocumentTemplate(
            name="Business Proposal",
            description="Professional business proposal with problem statement, solution, and pricing",
            document_type=DocumentType.PROPOSAL,
            sections=[
                "Cover Letter",
                "Executive Summary",
                "Problem Statement",
                "Proposed Solution",
                "Methodology",
                "Timeline",
                "Budget and Pricing",
                "Team Qualifications",
                "Terms and Conditions",
                "Next Steps"
            ],
            style=DocumentStyle(
                font_family="Arial",
                font_size=11,
                line_spacing=1.2,
                header_color="#2c3e50",
                body_color="#34495e",
                accent_color="#3498db",
                include_page_numbers=True
            ),
            metadata={
                "target_audience": "clients",
                "formality": "high",
                "typical_length": "5-20 pages"
            }
        )
    
    @staticmethod
    def get_technical_document_template() -> DocumentTemplate:
        """Technical documentation template with clear structure and code examples."""
        return DocumentTemplate(
            name="Technical Documentation",
            description="Technical documentation with clear structure, code examples, and diagrams",
            document_type=DocumentType.TECHNICAL_DOCUMENT,
            sections=[
                "Overview",
                "System Architecture",
                "Installation Guide",
                "Configuration",
                "API Reference",
                "Code Examples",
                "Troubleshooting",
                "FAQ",
                "References"
            ],
            style=DocumentStyle(
                font_family="Consolas",
                font_size=10,
                line_spacing=1.3,
                header_color="#2c3e50",
                body_color="#2c3e50",
                accent_color="#e74c3c",
                include_page_numbers=True
            ),
            metadata={
                "target_audience": "developers",
                "formality": "medium",
                "typical_length": "20-100 pages"
            }
        )
    
    @staticmethod
    def get_academic_paper_template() -> DocumentTemplate:
        """Academic paper template following standard academic formatting."""
        return DocumentTemplate(
            name="Academic Paper",
            description="Academic paper template with proper citations and academic formatting",
            document_type=DocumentType.ACADEMIC_PAPER,
            sections=[
                "Abstract",
                "Introduction",
                "Literature Review",
                "Methodology",
                "Results",
                "Discussion",
                "Conclusion",
                "References",
                "Appendices"
            ],
            style=DocumentStyle(
                font_family="Times New Roman",
                font_size=12,
                line_spacing=2.0,
                margin_top=1.0,
                margin_bottom=1.0,
                margin_left=1.0,
                margin_right=1.0,
                header_color="#000000",
                body_color="#000000",
                accent_color="#000000",
                include_page_numbers=True
            ),
            metadata={
                "target_audience": "academics",
                "formality": "very_high",
                "typical_length": "10-30 pages"
            }
        )
    
    @staticmethod
    def get_whitepaper_template() -> DocumentTemplate:
        """Whitepaper template for thought leadership and industry insights."""
        return DocumentTemplate(
            name="Whitepaper",
            description="Professional whitepaper for thought leadership and industry insights",
            document_type=DocumentType.WHITEPAPER,
            sections=[
                "Executive Summary",
                "Introduction",
                "Market Analysis",
                "Problem Definition",
                "Solution Overview",
                "Case Studies",
                "Industry Trends",
                "Future Outlook",
                "Conclusion",
                "About the Author"
            ],
            style=DocumentStyle(
                font_family="Georgia",
                font_size=11,
                line_spacing=1.4,
                header_color="#1a365d",
                body_color="#2d3748",
                accent_color="#3182ce",
                include_page_numbers=True
            ),
            metadata={
                "target_audience": "industry_professionals",
                "formality": "high",
                "typical_length": "15-40 pages"
            }
        )
    
    @staticmethod
    def get_manual_template() -> DocumentTemplate:
        """User manual template with step-by-step instructions."""
        return DocumentTemplate(
            name="User Manual",
            description="Comprehensive user manual with step-by-step instructions and troubleshooting",
            document_type=DocumentType.MANUAL,
            sections=[
                "Getting Started",
                "Installation",
                "Basic Operations",
                "Advanced Features",
                "Configuration Options",
                "Troubleshooting",
                "FAQ",
                "Support Information",
                "Index"
            ],
            style=DocumentStyle(
                font_family="Arial",
                font_size=10,
                line_spacing=1.2,
                header_color="#2c3e50",
                body_color="#34495e",
                accent_color="#3498db",
                include_page_numbers=True
            ),
            metadata={
                "target_audience": "end_users",
                "formality": "medium",
                "typical_length": "20-80 pages"
            }
        )
    
    @staticmethod
    def get_business_plan_template() -> DocumentTemplate:
        """Business plan template for startups and business development."""
        return DocumentTemplate(
            name="Business Plan",
            description="Comprehensive business plan for startups and business development",
            document_type=DocumentType.BUSINESS_PLAN,
            sections=[
                "Executive Summary",
                "Company Description",
                "Market Analysis",
                "Organization and Management",
                "Service or Product Line",
                "Marketing and Sales",
                "Financial Projections",
                "Funding Request",
                "Appendix"
            ],
            style=DocumentStyle(
                font_family="Calibri",
                font_size=11,
                line_spacing=1.15,
                header_color="#1f4e79",
                body_color="#2f2f2f",
                accent_color="#4472c4",
                include_page_numbers=True
            ),
            metadata={
                "target_audience": "investors",
                "formality": "very_high",
                "typical_length": "20-50 pages"
            }
        )
    
    @staticmethod
    def get_newsletter_template() -> DocumentTemplate:
        """Newsletter template for regular communications."""
        return DocumentTemplate(
            name="Newsletter",
            description="Professional newsletter template for regular communications",
            document_type=DocumentType.NEWSLETTER,
            sections=[
                "Header and Masthead",
                "Letter from the Editor",
                "Feature Articles",
                "Company News",
                "Industry Updates",
                "Upcoming Events",
                "Contact Information"
            ],
            style=DocumentStyle(
                font_family="Arial",
                font_size=10,
                line_spacing=1.3,
                header_color="#2c3e50",
                body_color="#34495e",
                accent_color="#3498db",
                include_page_numbers=False
            ),
            metadata={
                "target_audience": "customers",
                "formality": "medium",
                "typical_length": "2-8 pages"
            }
        )
    
    @staticmethod
    def get_brochure_template() -> DocumentTemplate:
        """Marketing brochure template for promotional materials."""
        return DocumentTemplate(
            name="Marketing Brochure",
            description="Professional marketing brochure template for promotional materials",
            document_type=DocumentType.BROCHURE,
            sections=[
                "Cover Page",
                "Company Overview",
                "Products/Services",
                "Key Benefits",
                "Customer Testimonials",
                "Contact Information",
                "Call to Action"
            ],
            style=DocumentStyle(
                font_family="Arial",
                font_size=10,
                line_spacing=1.2,
                header_color="#2c3e50",
                body_color="#34495e",
                accent_color="#e74c3c",
                include_page_numbers=False
            ),
            metadata={
                "target_audience": "prospects",
                "formality": "medium",
                "typical_length": "2-6 pages"
            }
        )
    
    @staticmethod
    def get_guide_template() -> DocumentTemplate:
        """How-to guide template with clear instructions."""
        return DocumentTemplate(
            name="How-to Guide",
            description="Step-by-step how-to guide template with clear instructions",
            document_type=DocumentType.GUIDE,
            sections=[
                "Introduction",
                "Prerequisites",
                "Step-by-Step Instructions",
                "Tips and Best Practices",
                "Common Mistakes",
                "Troubleshooting",
                "Additional Resources",
                "Conclusion"
            ],
            style=DocumentStyle(
                font_family="Arial",
                font_size=11,
                line_spacing=1.3,
                header_color="#2c3e50",
                body_color="#34495e",
                accent_color="#27ae60",
                include_page_numbers=True
            ),
            metadata={
                "target_audience": "general_users",
                "formality": "medium",
                "typical_length": "5-25 pages"
            }
        )
    
    @staticmethod
    def get_catalog_template() -> DocumentTemplate:
        """Product catalog template for showcasing products or services."""
        return DocumentTemplate(
            name="Product Catalog",
            description="Professional product catalog template for showcasing products or services",
            document_type=DocumentType.CATALOG,
            sections=[
                "Cover Page",
                "Table of Contents",
                "Product Categories",
                "Product Listings",
                "Specifications",
                "Pricing Information",
                "Ordering Information",
                "Contact Details"
            ],
            style=DocumentStyle(
                font_family="Arial",
                font_size=9,
                line_spacing=1.1,
                header_color="#2c3e50",
                body_color="#34495e",
                accent_color="#3498db",
                include_page_numbers=True
            ),
            metadata={
                "target_audience": "customers",
                "formality": "medium",
                "typical_length": "10-100 pages"
            }
        )
    
    @staticmethod
    def get_presentation_template() -> DocumentTemplate:
        """Presentation template for slide-based documents."""
        return DocumentTemplate(
            name="Presentation",
            description="Professional presentation template for slide-based documents",
            document_type=DocumentType.PRESENTATION,
            sections=[
                "Title Slide",
                "Agenda",
                "Introduction",
                "Main Content Slides",
                "Key Points",
                "Conclusion",
                "Q&A",
                "Contact Information"
            ],
            style=DocumentStyle(
                font_family="Arial",
                font_size=14,
                line_spacing=1.2,
                header_color="#2c3e50",
                body_color="#34495e",
                accent_color="#3498db",
                include_page_numbers=False
            ),
            metadata={
                "target_audience": "audience",
                "formality": "medium",
                "typical_length": "10-50 slides"
            }
        )


class TemplateManager:
    """Manager for document templates."""
    
    def __init__(self):
        self._templates: Dict[str, DocumentTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load all default templates."""
        templates = [
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
        
        for template in templates:
            self._templates[template.id] = template
    
    def get_template(self, template_id: str) -> DocumentTemplate:
        """Get a template by ID."""
        if template_id not in self._templates:
            raise ValueError(f"Template with ID {template_id} not found")
        return self._templates[template_id]
    
    def get_templates_by_type(self, document_type: DocumentType) -> List[DocumentTemplate]:
        """Get all templates for a specific document type."""
        return [
            template for template in self._templates.values()
            if template.document_type == document_type
        ]
    
    def get_all_templates(self) -> List[DocumentTemplate]:
        """Get all available templates."""
        return list(self._templates.values())
    
    def get_default_template(self, document_type: DocumentType) -> DocumentTemplate:
        """Get the default template for a document type."""
        templates = self.get_templates_by_type(document_type)
        if not templates:
            raise ValueError(f"No templates found for document type {document_type}")
        return templates[0]  # Return the first template as default
    
    def add_custom_template(self, template: DocumentTemplate):
        """Add a custom template."""
        self._templates[template.id] = template
    
    def remove_template(self, template_id: str):
        """Remove a template."""
        if template_id in self._templates:
            del self._templates[template_id]


# Global template manager instance
template_manager = TemplateManager()




























