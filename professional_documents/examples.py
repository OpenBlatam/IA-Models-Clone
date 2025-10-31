"""
Professional Documents Examples
==============================

Comprehensive examples showing how to use the Professional Documents feature
in various scenarios.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any

from .models import (
    DocumentGenerationRequest,
    DocumentExportRequest,
    DocumentType,
    ExportFormat,
    DocumentStyle
)
from .services import DocumentGenerationService, DocumentExportService
from .templates import template_manager


class ProfessionalDocumentsExamples:
    """Examples for using the Professional Documents feature."""
    
    def __init__(self):
        self.doc_service = DocumentGenerationService()
        self.export_service = DocumentExportService()
    
    async def example_business_proposal(self):
        """Example: Generate a business proposal."""
        print("📋 Example: Business Proposal Generation")
        print("=" * 50)
        
        request = DocumentGenerationRequest(
            query="Create a comprehensive business proposal for implementing a cloud-based customer relationship management system for a mid-size manufacturing company. The proposal should include cost analysis, implementation timeline, and expected ROI.",
            document_type=DocumentType.PROPOSAL,
            title="Cloud CRM Implementation Proposal",
            subtitle="Enhancing Customer Relationships Through Technology",
            author="Sarah Johnson",
            company="Tech Solutions Inc.",
            tone="professional",
            length="comprehensive",
            additional_requirements="Include competitive analysis and risk assessment"
        )
        
        response = await self.doc_service.generate_document(request)
        
        if response.success:
            print(f"✅ Proposal generated successfully!")
            print(f"   Title: {response.document.title}")
            print(f"   Word Count: {response.word_count}")
            print(f"   Sections: {len(response.document.sections)}")
            print(f"   Generation Time: {response.generation_time:.2f}s")
            
            # Export to PDF
            export_request = DocumentExportRequest(
                document_id=response.document.id,
                format=ExportFormat.PDF,
                custom_filename="crm_proposal.pdf"
            )
            
            export_response = await self.export_service.export_document(response.document, export_request)
            
            if export_response.success:
                print(f"✅ Exported to PDF: {export_response.file_path}")
            
            return response.document
        else:
            print(f"❌ Proposal generation failed: {response.message}")
            return None
    
    async def example_technical_documentation(self):
        """Example: Generate technical documentation."""
        print("\n🔧 Example: Technical Documentation Generation")
        print("=" * 50)
        
        request = DocumentGenerationRequest(
            query="Create comprehensive technical documentation for a REST API that manages user authentication and authorization. Include API endpoints, authentication methods, error codes, and code examples in Python and JavaScript.",
            document_type=DocumentType.TECHNICAL_DOCUMENT,
            title="User Authentication API Documentation",
            subtitle="Complete Guide to Authentication and Authorization",
            author="Development Team",
            company="API Solutions Ltd.",
            tone="technical",
            length="long",
            additional_requirements="Include security best practices and rate limiting information"
        )
        
        response = await self.doc_service.generate_document(request)
        
        if response.success:
            print(f"✅ Technical documentation generated successfully!")
            print(f"   Title: {response.document.title}")
            print(f"   Word Count: {response.word_count}")
            print(f"   Sections: {len(response.document.sections)}")
            
            # Export to multiple formats
            formats = [ExportFormat.PDF, ExportFormat.MARKDOWN, ExportFormat.WORD]
            
            for format_type in formats:
                export_request = DocumentExportRequest(
                    document_id=response.document.id,
                    format=format_type,
                    custom_filename=f"api_docs.{format_type.value}"
                )
                
                export_response = await self.export_service.export_document(response.document, export_request)
                
                if export_response.success:
                    print(f"✅ Exported to {format_type.value.upper()}: {export_response.file_path}")
            
            return response.document
        else:
            print(f"❌ Technical documentation generation failed: {response.message}")
            return None
    
    async def example_business_report(self):
        """Example: Generate a business report."""
        print("\n📊 Example: Business Report Generation")
        print("=" * 50)
        
        request = DocumentGenerationRequest(
            query="Create a comprehensive quarterly business report analyzing market trends, sales performance, customer satisfaction metrics, and strategic recommendations for the next quarter. Focus on the e-commerce sector.",
            document_type=DocumentType.REPORT,
            title="Q4 2024 Business Performance Report",
            subtitle="Market Analysis and Strategic Recommendations",
            author="Analytics Team",
            company="E-Commerce Solutions",
            tone="formal",
            length="comprehensive",
            additional_requirements="Include charts and data visualizations descriptions"
        )
        
        response = await self.doc_service.generate_document(request)
        
        if response.success:
            print(f"✅ Business report generated successfully!")
            print(f"   Title: {response.document.title}")
            print(f"   Word Count: {response.word_count}")
            print(f"   Sections: {len(response.document.sections)}")
            
            # Export to PDF with custom styling
            export_request = DocumentExportRequest(
                document_id=response.document.id,
                format=ExportFormat.PDF,
                custom_filename="q4_business_report.pdf"
            )
            
            export_response = await self.export_service.export_document(response.document, export_request)
            
            if export_response.success:
                print(f"✅ Exported to PDF: {export_response.file_path}")
            
            return response.document
        else:
            print(f"❌ Business report generation failed: {response.message}")
            return None
    
    async def example_user_manual(self):
        """Example: Generate a user manual."""
        print("\n📖 Example: User Manual Generation")
        print("=" * 50)
        
        request = DocumentGenerationRequest(
            query="Create a comprehensive user manual for a mobile banking application. Include setup instructions, feature explanations, security guidelines, troubleshooting tips, and FAQ section.",
            document_type=DocumentType.MANUAL,
            title="Mobile Banking App User Manual",
            subtitle="Complete Guide to Mobile Banking Features",
            author="Product Team",
            company="SecureBank Mobile",
            tone="casual",
            length="long",
            additional_requirements="Include screenshots descriptions and step-by-step instructions"
        )
        
        response = await self.doc_service.generate_document(request)
        
        if response.success:
            print(f"✅ User manual generated successfully!")
            print(f"   Title: {response.document.title}")
            print(f"   Word Count: {response.word_count}")
            print(f"   Sections: {len(response.document.sections)}")
            
            # Export to multiple formats for different use cases
            export_formats = [
                (ExportFormat.PDF, "mobile_banking_manual.pdf"),
                (ExportFormat.HTML, "mobile_banking_manual.html"),
                (ExportFormat.MARKDOWN, "mobile_banking_manual.md")
            ]
            
            for format_type, filename in export_formats:
                export_request = DocumentExportRequest(
                    document_id=response.document.id,
                    format=format_type,
                    custom_filename=filename
                )
                
                export_response = await self.export_service.export_document(response.document, export_request)
                
                if export_response.success:
                    print(f"✅ Exported to {format_type.value.upper()}: {export_response.file_path}")
            
            return response.document
        else:
            print(f"❌ User manual generation failed: {response.message}")
            return None
    
    async def example_academic_paper(self):
        """Example: Generate an academic paper."""
        print("\n🎓 Example: Academic Paper Generation")
        print("=" * 50)
        
        request = DocumentGenerationRequest(
            query="Create an academic research paper on the impact of artificial intelligence on modern software development practices. Include literature review, methodology, findings, and conclusions with proper academic formatting.",
            document_type=DocumentType.ACADEMIC_PAPER,
            title="The Impact of Artificial Intelligence on Modern Software Development Practices",
            subtitle="A Comprehensive Analysis of AI Integration in Software Engineering",
            author="Dr. Maria Rodriguez",
            company="University of Technology",
            tone="academic",
            length="comprehensive",
            additional_requirements="Include proper citations format and research methodology"
        )
        
        response = await self.doc_service.generate_document(request)
        
        if response.success:
            print(f"✅ Academic paper generated successfully!")
            print(f"   Title: {response.document.title}")
            print(f"   Word Count: {response.word_count}")
            print(f"   Sections: {len(response.document.sections)}")
            
            # Export to PDF with academic formatting
            export_request = DocumentExportRequest(
                document_id=response.document.id,
                format=ExportFormat.PDF,
                custom_filename="ai_software_development_paper.pdf"
            )
            
            export_response = await self.export_service.export_document(response.document, export_request)
            
            if export_response.success:
                print(f"✅ Exported to PDF: {export_response.file_path}")
            
            return response.document
        else:
            print(f"❌ Academic paper generation failed: {response.message}")
            return None
    
    async def example_whitepaper(self):
        """Example: Generate a whitepaper."""
        print("\n📄 Example: Whitepaper Generation")
        print("=" * 50)
        
        request = DocumentGenerationRequest(
            query="Create a comprehensive whitepaper on the future of blockchain technology in supply chain management. Include market analysis, use cases, implementation challenges, and future outlook.",
            document_type=DocumentType.WHITEPAPER,
            title="Blockchain in Supply Chain Management",
            subtitle="Transforming Logistics Through Distributed Ledger Technology",
            author="Blockchain Research Team",
            company="Innovation Labs",
            tone="professional",
            length="comprehensive",
            additional_requirements="Include industry case studies and ROI analysis"
        )
        
        response = await self.doc_service.generate_document(request)
        
        if response.success:
            print(f"✅ Whitepaper generated successfully!")
            print(f"   Title: {response.document.title}")
            print(f"   Word Count: {response.word_count}")
            print(f"   Sections: {len(response.document.sections)}")
            
            # Export to PDF and HTML for web publishing
            export_formats = [
                (ExportFormat.PDF, "blockchain_supply_chain_whitepaper.pdf"),
                (ExportFormat.HTML, "blockchain_supply_chain_whitepaper.html")
            ]
            
            for format_type, filename in export_formats:
                export_request = DocumentExportRequest(
                    document_id=response.document.id,
                    format=format_type,
                    custom_filename=filename
                )
                
                export_response = await self.export_service.export_document(response.document, export_request)
                
                if export_response.success:
                    print(f"✅ Exported to {format_type.value.upper()}: {export_response.file_path}")
            
            return response.document
        else:
            print(f"❌ Whitepaper generation failed: {response.message}")
            return None
    
    def show_available_templates(self):
        """Show all available templates."""
        print("\n🎨 Available Professional Templates")
        print("=" * 50)
        
        templates = template_manager.get_all_templates()
        
        # Group by document type
        by_type = {}
        for template in templates:
            doc_type = template.document_type.value
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append(template.name)
        
        for doc_type, template_names in by_type.items():
            print(f"\n📋 {doc_type.replace('_', ' ').title()}:")
            for name in template_names:
                print(f"   • {name}")
        
        print(f"\nTotal templates available: {len(templates)}")
    
    def show_export_formats(self):
        """Show available export formats."""
        print("\n📤 Available Export Formats")
        print("=" * 50)
        
        formats = [
            ("PDF", "Professional PDF documents with proper formatting"),
            ("Word (.docx)", "Full-featured Word documents"),
            ("Markdown", "Clean markdown for easy editing"),
            ("HTML", "Web-ready HTML documents")
        ]
        
        for format_name, description in formats:
            print(f"📄 {format_name}: {description}")
    
    async def run_all_examples(self):
        """Run all examples."""
        print("🚀 Professional Documents Feature - Complete Examples")
        print("=" * 60)
        
        # Show available templates and formats
        self.show_available_templates()
        self.show_export_formats()
        
        # Run examples
        examples = [
            self.example_business_proposal,
            self.example_technical_documentation,
            self.example_business_report,
            self.example_user_manual,
            self.example_academic_paper,
            self.example_whitepaper
        ]
        
        generated_documents = []
        
        for example in examples:
            try:
                document = await example()
                if document:
                    generated_documents.append(document)
            except Exception as e:
                print(f"❌ Example failed: {str(e)}")
        
        print(f"\n🎉 Examples completed!")
        print(f"   Generated documents: {len(generated_documents)}")
        print(f"   Check the 'exports' directory for exported files.")
        
        return generated_documents


# API Usage Examples
class APIUsageExamples:
    """Examples of using the Professional Documents API."""
    
    @staticmethod
    def example_api_requests():
        """Show example API requests."""
        print("\n🌐 API Usage Examples")
        print("=" * 50)
        
        # Example 1: Generate a document
        print("1. Generate a Business Proposal:")
        print("""
POST /api/v1/professional-documents/generate
Content-Type: application/json

{
    "query": "Create a business proposal for implementing a CRM system",
    "document_type": "proposal",
    "title": "CRM Implementation Proposal",
    "author": "John Smith",
    "company": "Tech Solutions Inc.",
    "tone": "professional",
    "length": "medium"
}
        """)
        
        # Example 2: Export a document
        print("2. Export Document to PDF:")
        print("""
POST /api/v1/professional-documents/export
Content-Type: application/json

{
    "document_id": "doc_123456",
    "format": "pdf",
    "custom_filename": "my_proposal.pdf"
}
        """)
        
        # Example 3: List documents
        print("3. List User Documents:")
        print("""
GET /api/v1/professional-documents/documents?page=1&page_size=20
        """)
        
        # Example 4: Get templates
        print("4. Get Available Templates:")
        print("""
GET /api/v1/professional-documents/templates?document_type=proposal
        """)
        
        # Example 5: Get document statistics
        print("5. Get Document Statistics:")
        print("""
GET /api/v1/professional-documents/stats
        """)


async def main():
    """Main function to run examples."""
    examples = ProfessionalDocumentsExamples()
    api_examples = APIUsageExamples()
    
    # Show API examples
    api_examples.example_api_requests()
    
    # Run document generation examples
    await examples.run_all_examples()


if __name__ == "__main__":
    asyncio.run(main())




























