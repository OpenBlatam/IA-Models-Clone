"""
Export IA Example Usage
=======================

Demonstrates how to use the Export IA model to create professional documents
in various formats with high-quality styling and validation.
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

# Import the Export IA components
from ..export_ia_engine import (
    ExportIAEngine, ExportConfig, ExportFormat, DocumentType, QualityLevel
)
from ..config.export_config import (
    ExportConfigManager, ExportTemplate, TemplateType, BrandingConfig, BrandingStyle
)
from ..quality.quality_validator import (
    QualityValidator, ValidationLevel
)
from ..styling.professional_styler import (
    ProfessionalStyler, ProfessionalLevel
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExportIAExample:
    """Example usage of the Export IA system."""
    
    def __init__(self):
        self.export_engine = ExportIAEngine()
        self.config_manager = ExportConfigManager()
        self.quality_validator = QualityValidator()
        self.professional_styler = ProfessionalStyler()
    
    async def create_business_plan_example(self):
        """Example: Create a professional business plan."""
        logger.info("Creating business plan example...")
        
        # Sample business plan content
        business_plan_content = {
            "title": "TechStart Solutions - Business Plan",
            "metadata": {
                "author": "John Smith",
                "created_date": datetime.now().isoformat(),
                "version": "1.0",
                "company": "TechStart Solutions"
            },
            "sections": [
                {
                    "heading": "Executive Summary",
                    "content": """
                    TechStart Solutions is a technology consulting company focused on helping 
                    small and medium businesses implement digital transformation solutions. 
                    Our mission is to bridge the gap between traditional business practices 
                    and modern technology, enabling our clients to compete effectively in 
                    the digital marketplace.
                    
                    We offer comprehensive services including cloud migration, process automation, 
                    data analytics, and cybersecurity solutions. Our team of certified professionals 
                    brings over 50 years of combined experience in enterprise technology implementation.
                    
                    Financial projections show strong growth potential with revenue reaching 
                    $2.5M by year three, driven by increasing demand for digital transformation 
                    services in the SME market.
                    """
                },
                {
                    "heading": "Company Description",
                    "content": """
                    TechStart Solutions was founded in 2024 with the vision of making advanced 
                    technology accessible to businesses of all sizes. We believe that every 
                    company, regardless of size, should have access to enterprise-grade 
                    technology solutions.
                    
                    Our core values include:
                    • Innovation: We stay at the forefront of technology trends
                    • Integrity: We maintain the highest ethical standards
                    • Excellence: We deliver superior results for our clients
                    • Partnership: We build long-term relationships with our clients
                    
                    The company is headquartered in San Francisco, California, with plans to 
                    expand to major metropolitan areas across the United States.
                    """
                },
                {
                    "heading": "Market Analysis",
                    "content": """
                    The digital transformation market is experiencing rapid growth, with the 
                    global market size expected to reach $1.8 trillion by 2030. Small and 
                    medium businesses represent a significant opportunity, as they often lack 
                    the internal resources to implement complex technology solutions.
                    
                    Key market trends include:
                    • Increased adoption of cloud computing
                    • Growing importance of data analytics
                    • Rising cybersecurity concerns
                    • Demand for remote work solutions
                    
                    Our target market consists of businesses with 10-500 employees that are 
                    looking to modernize their operations and improve efficiency through 
                    technology implementation.
                    """
                },
                {
                    "heading": "Financial Projections",
                    "content": """
                    Year 1: $500K revenue, $50K profit
                    Year 2: $1.2M revenue, $200K profit  
                    Year 3: $2.5M revenue, $500K profit
                    
                    Revenue streams include:
                    • Consulting services (60%)
                    • Implementation services (25%)
                    • Ongoing support and maintenance (15%)
                    
                    Key financial assumptions:
                    • Average project value: $25K
                    • Client retention rate: 85%
                    • Growth rate: 140% year-over-year
                    """
                }
            ]
        }
        
        # Create export configuration
        export_config = ExportConfig(
            format=ExportFormat.PDF,
            document_type=DocumentType.BUSINESS_PLAN,
            quality_level=QualityLevel.PREMIUM,
            template="business_plan",
            branding={
                "company_name": "TechStart Solutions",
                "logo_path": "assets/logo.png",
                "website": "www.techstartsolutions.com",
                "email": "info@techstartsolutions.com"
            },
            output_options={
                "include_page_numbers": True,
                "include_headers_footers": True,
                "table_of_contents": True
            }
        )
        
        # Export the document
        task_id = await self.export_engine.export_document(
            content=business_plan_content,
            config=export_config,
            output_path="exports/business_plan_example.pdf"
        )
        
        # Wait for completion and get results
        await asyncio.sleep(2)  # Simulate processing time
        result = self.export_engine.get_task_status(task_id)
        
        logger.info(f"Business plan export completed: {result}")
        return result
    
    async def create_professional_report_example(self):
        """Example: Create a professional report with quality validation."""
        logger.info("Creating professional report example...")
        
        # Sample report content
        report_content = {
            "title": "Q4 2024 Market Analysis Report",
            "metadata": {
                "author": "Market Research Team",
                "created_date": datetime.now().isoformat(),
                "department": "Research & Analytics",
                "confidentiality": "Internal Use Only"
            },
            "sections": [
                {
                    "heading": "Introduction",
                    "content": """
                    This report provides a comprehensive analysis of market trends and 
                    opportunities for Q4 2024. The analysis is based on data collected 
                    from multiple sources including customer surveys, industry reports, 
                    and competitive intelligence.
                    
                    The report covers key market segments, competitive landscape, 
                    customer behavior patterns, and strategic recommendations for 
                    the upcoming quarter.
                    """
                },
                {
                    "heading": "Key Findings",
                    "content": """
                    • Market growth rate increased by 15% compared to Q3 2024
                    • Customer satisfaction scores improved across all product lines
                    • New competitor entry in the premium segment
                    • Emerging opportunities in the mobile-first market
                    • Supply chain challenges affecting 20% of product categories
                    """
                },
                {
                    "heading": "Recommendations",
                    "content": """
                    1. Accelerate mobile platform development to capture emerging opportunities
                    2. Implement premium product line to compete with new market entrants
                    3. Strengthen supply chain partnerships to address current challenges
                    4. Increase marketing investment in high-growth segments
                    5. Develop customer retention programs to maintain satisfaction levels
                    """
                }
            ]
        }
        
        # Create export configuration for DOCX
        export_config = ExportConfig(
            format=ExportFormat.DOCX,
            document_type=DocumentType.REPORT,
            quality_level=QualityLevel.PROFESSIONAL,
            template="professional_report",
            output_options={
                "include_page_numbers": True,
                "include_headers_footers": True,
                "watermark": False
            }
        )
        
        # Export the document
        task_id = await self.export_engine.export_document(
            content=report_content,
            config=export_config,
            output_path="exports/professional_report_example.docx"
        )
        
        # Validate quality
        quality_report = await self.quality_validator.validate_document(
            content=report_content,
            validation_level=ValidationLevel.STANDARD
        )
        
        logger.info(f"Report export completed with quality score: {quality_report.overall_score:.2f}")
        logger.info(f"Quality validation passed: {quality_report.passed_validation}")
        
        return {
            "task_id": task_id,
            "quality_report": quality_report
        }
    
    async def create_html_newsletter_example(self):
        """Example: Create an HTML newsletter with custom styling."""
        logger.info("Creating HTML newsletter example...")
        
        # Sample newsletter content
        newsletter_content = {
            "title": "Monthly Technology Newsletter",
            "metadata": {
                "author": "Tech Team",
                "created_date": datetime.now().isoformat(),
                "issue": "December 2024",
                "audience": "Technology Professionals"
            },
            "sections": [
                {
                    "heading": "Industry News",
                    "content": """
                    <p>This month's technology landscape continues to evolve rapidly with 
                    several significant developments:</p>
                    <ul>
                        <li>AI integration reaches new milestones in enterprise applications</li>
                        <li>Cloud computing adoption accelerates among SMEs</li>
                        <li>Cybersecurity becomes a top priority for all organizations</li>
                    </ul>
                    """
                },
                {
                    "heading": "Product Updates",
                    "content": """
                    <p>We're excited to announce several new features and improvements:</p>
                    <ul>
                        <li>Enhanced security protocols for all cloud services</li>
                        <li>New mobile app with improved user experience</li>
                        <li>Advanced analytics dashboard for better insights</li>
                    </ul>
                    """
                },
                {
                    "heading": "Upcoming Events",
                    "content": """
                    <p>Don't miss these upcoming technology events:</p>
                    <ul>
                        <li>Tech Conference 2025 - January 15-17, San Francisco</li>
                        <li>Cloud Security Summit - February 8, New York</li>
                        <li>AI Innovation Workshop - March 12, Online</li>
                    </ul>
                    """
                }
            ]
        }
        
        # Get professional style
        professional_style = self.professional_styler.get_style("premium")
        
        # Apply styling to content
        styled_content = self.professional_styler.apply_style_to_content(
            content=newsletter_content,
            style=professional_style,
            format_type="html"
        )
        
        # Create export configuration
        export_config = ExportConfig(
            format=ExportFormat.HTML,
            document_type=DocumentType.NEWSLETTER,
            quality_level=QualityLevel.PREMIUM,
            custom_styles=styled_content.get("css", ""),
            output_options={
                "responsive": True,
                "include_css": True,
                "optimize_for_web": True
            }
        )
        
        # Export the document
        task_id = await self.export_engine.export_document(
            content=styled_content,
            config=export_config,
            output_path="exports/newsletter_example.html"
        )
        
        logger.info(f"Newsletter export completed: {task_id}")
        return task_id
    
    async def demonstrate_quality_validation(self):
        """Example: Demonstrate quality validation features."""
        logger.info("Demonstrating quality validation...")
        
        # Sample content with various quality issues
        test_content = {
            "title": "Test Document for Quality Validation",
            "sections": [
                {
                    "heading": "Section 1",
                    "content": "This is a short section with minimal content that might not meet quality standards."
                },
                {
                    "heading": "Section 2", 
                    "content": "This section has more substantial content that should meet quality requirements. It includes multiple sentences and provides detailed information about the topic being discussed."
                }
            ]
        }
        
        # Run validation at different levels
        validation_levels = [
            ValidationLevel.BASIC,
            ValidationLevel.STANDARD,
            ValidationLevel.STRICT,
            ValidationLevel.ENTERPRISE
        ]
        
        results = {}
        for level in validation_levels:
            quality_report = await self.quality_validator.validate_document(
                content=test_content,
                validation_level=level
            )
            
            results[level.value] = {
                "overall_score": quality_report.overall_score,
                "passed": quality_report.passed_validation,
                "rules_applied": len(quality_report.results),
                "recommendations": quality_report.recommendations[:3]  # Top 3 recommendations
            }
            
            logger.info(f"Validation Level {level.value}: Score {quality_report.overall_score:.2f}, Passed: {quality_report.passed_validation}")
        
        return results
    
    async def demonstrate_style_customization(self):
        """Example: Demonstrate style customization features."""
        logger.info("Demonstrating style customization...")
        
        # Create a custom professional style
        custom_style = self.professional_styler.create_custom_style(
            name="Corporate Blue",
            description="Custom corporate style with blue accent colors",
            level=ProfessionalLevel.PROFESSIONAL,
            base_style="standard",
            custom_colors={
                "primary": "#1E3A8A",
                "secondary": "#3B82F6", 
                "accent": "#1D4ED8",
                "background": "#FFFFFF",
                "surface": "#F8FAFC"
            }
        )
        
        # Get style recommendations
        recommendations = self.professional_styler.get_style_recommendations(
            document_type="business_plan",
            content_length=5000,
            has_images=True,
            has_tables=True,
            target_audience="enterprise"
        )
        
        logger.info(f"Custom style created: {custom_style.name}")
        logger.info(f"Style recommendations: {[s.name for s in recommendations]}")
        
        return {
            "custom_style": custom_style,
            "recommendations": recommendations
        }
    
    async def run_all_examples(self):
        """Run all examples to demonstrate Export IA capabilities."""
        logger.info("Running all Export IA examples...")
        
        try:
            # Create business plan
            business_plan_result = await self.create_business_plan_example()
            
            # Create professional report
            report_result = await self.create_professional_report_example()
            
            # Create HTML newsletter
            newsletter_result = await self.create_html_newsletter_example()
            
            # Demonstrate quality validation
            validation_results = await self.demonstrate_quality_validation()
            
            # Demonstrate style customization
            style_results = await self.demonstrate_style_customization()
            
            # Get export statistics
            export_stats = self.export_engine.get_export_statistics()
            
            logger.info("All examples completed successfully!")
            logger.info(f"Export statistics: {export_stats}")
            
            return {
                "business_plan": business_plan_result,
                "report": report_result,
                "newsletter": newsletter_result,
                "validation": validation_results,
                "styling": style_results,
                "statistics": export_stats
            }
            
        except Exception as e:
            logger.error(f"Error running examples: {e}")
            raise

async def main():
    """Main function to run the examples."""
    example = ExportIAExample()
    results = await example.run_all_examples()
    
    print("\n" + "="*50)
    print("EXPORT IA EXAMPLE RESULTS")
    print("="*50)
    
    for key, value in results.items():
        print(f"\n{key.upper()}:")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"  {value}")

if __name__ == "__main__":
    asyncio.run(main())



























