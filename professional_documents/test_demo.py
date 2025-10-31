"""
Demo script to test the Professional Documents feature
=====================================================

This script demonstrates the core functionality of the professional documents system.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from .models import (
    DocumentGenerationRequest,
    DocumentType,
    DocumentStyle,
    ExportFormat,
    DocumentExportRequest
)
from .services import DocumentGenerationService, DocumentExportService
from .templates import template_manager


async def test_document_generation():
    """Test document generation functionality."""
    print("🚀 Testing Document Generation...")
    
    # Initialize services
    doc_service = DocumentGenerationService()
    export_service = DocumentExportService()
    
    # Create a test request
    request = DocumentGenerationRequest(
        query="Create a comprehensive business proposal for implementing a new customer relationship management system for a mid-size company",
        document_type=DocumentType.PROPOSAL,
        title="CRM Implementation Proposal",
        subtitle="Enhancing Customer Relationships Through Technology",
        author="Jane Smith",
        company="Tech Solutions Inc.",
        tone="professional",
        length="medium",
        additional_requirements="Include cost analysis and implementation timeline"
    )
    
    print(f"📝 Generating document: {request.title}")
    
    # Generate document
    response = await doc_service.generate_document(request)
    
    if response.success:
        print(f"✅ Document generated successfully!")
        print(f"   - ID: {response.document.id}")
        print(f"   - Word Count: {response.word_count}")
        print(f"   - Estimated Pages: {response.estimated_pages}")
        print(f"   - Generation Time: {response.generation_time:.2f}s")
        print(f"   - Sections: {len(response.document.sections)}")
        
        # Display sections
        print("\n📋 Document Sections:")
        for i, section in enumerate(response.document.sections, 1):
            print(f"   {i}. {section.title} ({len(section.content.split())} words)")
        
        return response.document
    else:
        print(f"❌ Document generation failed: {response.message}")
        return None


async def test_document_export(document):
    """Test document export functionality."""
    if not document:
        print("❌ No document to export")
        return
    
    print(f"\n📤 Testing Document Export...")
    
    export_service = DocumentExportService()
    
    # Test PDF export
    print("📄 Exporting to PDF...")
    pdf_request = DocumentExportRequest(
        document_id=document.id,
        format=ExportFormat.PDF,
        custom_filename="test_proposal.pdf"
    )
    
    pdf_response = await export_service.export_document(document, pdf_request)
    
    if pdf_response.success:
        print(f"✅ PDF exported successfully!")
        print(f"   - File: {pdf_response.file_path}")
        print(f"   - Size: {pdf_response.file_size} bytes")
        print(f"   - Export Time: {pdf_response.export_time:.2f}s")
    else:
        print(f"❌ PDF export failed: {pdf_response.message}")
    
    # Test Markdown export
    print("\n📝 Exporting to Markdown...")
    md_request = DocumentExportRequest(
        document_id=document.id,
        format=ExportFormat.MARKDOWN,
        custom_filename="test_proposal.md"
    )
    
    md_response = await export_service.export_document(document, md_request)
    
    if md_response.success:
        print(f"✅ Markdown exported successfully!")
        print(f"   - File: {md_response.file_path}")
        print(f"   - Size: {md_response.file_size} bytes")
        print(f"   - Export Time: {md_response.export_time:.2f}s")
    else:
        print(f"❌ Markdown export failed: {md_response.message}")
    
    # Test Word export
    print("\n📊 Exporting to Word...")
    word_request = DocumentExportRequest(
        document_id=document.id,
        format=ExportFormat.WORD,
        custom_filename="test_proposal.docx"
    )
    
    word_response = await export_service.export_document(document, word_request)
    
    if word_response.success:
        print(f"✅ Word document exported successfully!")
        print(f"   - File: {word_response.file_path}")
        print(f"   - Size: {word_response.file_size} bytes")
        print(f"   - Export Time: {word_response.export_time:.2f}s")
    else:
        print(f"❌ Word export failed: {word_response.message}")


def test_templates():
    """Test template functionality."""
    print("\n🎨 Testing Templates...")
    
    # Get all templates
    templates = template_manager.get_all_templates()
    print(f"📚 Total templates available: {len(templates)}")
    
    # Group by document type
    by_type = {}
    for template in templates:
        doc_type = template.document_type.value
        if doc_type not in by_type:
            by_type[doc_type] = []
        by_type[doc_type].append(template.name)
    
    print("\n📋 Templates by Document Type:")
    for doc_type, template_names in by_type.items():
        print(f"   {doc_type}: {', '.join(template_names)}")
    
    # Test getting specific template
    proposal_templates = template_manager.get_templates_by_type(DocumentType.PROPOSAL)
    if proposal_templates:
        template = proposal_templates[0]
        print(f"\n🔍 Sample Proposal Template:")
        print(f"   - Name: {template.name}")
        print(f"   - Description: {template.description}")
        print(f"   - Sections: {len(template.sections)}")
        print(f"   - Font: {template.style.font_family} {template.style.font_size}pt")


def test_document_styles():
    """Test document styling functionality."""
    print("\n🎨 Testing Document Styles...")
    
    # Create custom style
    custom_style = DocumentStyle(
        font_family="Georgia",
        font_size=12,
        line_spacing=1.4,
        header_color="#1a365d",
        body_color="#2d3748",
        accent_color="#3182ce",
        margin_top=1.2,
        margin_bottom=1.2,
        margin_left=1.0,
        margin_right=1.0,
        include_page_numbers=True
    )
    
    print("🎨 Custom Style Configuration:")
    print(f"   - Font: {custom_style.font_family} {custom_style.font_size}pt")
    print(f"   - Line Spacing: {custom_style.line_spacing}")
    print(f"   - Header Color: {custom_style.header_color}")
    print(f"   - Body Color: {custom_style.body_color}")
    print(f"   - Margins: {custom_style.margin_top}\" x {custom_style.margin_bottom}\" x {custom_style.margin_left}\" x {custom_style.margin_right}\"")
    print(f"   - Page Numbers: {'Yes' if custom_style.include_page_numbers else 'No'}")


async def main():
    """Main test function."""
    print("🧪 Professional Documents Feature Test Suite")
    print("=" * 50)
    
    try:
        # Test templates
        test_templates()
        
        # Test document styles
        test_document_styles()
        
        # Test document generation
        document = await test_document_generation()
        
        # Test document export
        await test_document_export(document)
        
        print("\n🎉 All tests completed successfully!")
        print("\n📁 Check the 'exports' directory for generated files.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())




























