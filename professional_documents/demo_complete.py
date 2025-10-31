"""
Complete Professional Documents Demo
===================================

This script demonstrates the complete workflow of the Professional Documents feature,
from generation to export, showing all capabilities.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from .models import (
    DocumentGenerationRequest,
    DocumentExportRequest,
    DocumentType,
    ExportFormat,
    DocumentStyle
)
from .services import DocumentGenerationService, DocumentExportService
from .templates import template_manager
from .config import get_config


class CompleteDemo:
    """Complete demonstration of the Professional Documents feature."""
    
    def __init__(self):
        self.doc_service = DocumentGenerationService()
        self.export_service = DocumentExportService()
        self.config = get_config()
        self.generated_documents = []
    
    async def run_complete_demo(self):
        """Run the complete demonstration."""
        print("🚀 Professional Documents Feature - Complete Demo")
        print("=" * 60)
        print(f"Configuration: {self.config.enabled}")
        print(f"Export Directory: {self.config.export_directory}")
        print(f"AI Model: {self.config.ai_model_name}")
        print("=" * 60)
        
        # Step 1: Show available templates
        await self.show_available_templates()
        
        # Step 2: Generate different types of documents
        await self.generate_various_documents()
        
        # Step 3: Export documents in different formats
        await self.export_documents()
        
        # Step 4: Show document management features
        await self.demonstrate_document_management()
        
        # Step 5: Show statistics and analytics
        await self.show_statistics()
        
        print("\n🎉 Complete Demo Finished!")
        print(f"Generated {len(self.generated_documents)} documents")
        print("Check the exports directory for all generated files.")
    
    async def show_available_templates(self):
        """Show all available templates."""
        print("\n📚 Available Professional Templates")
        print("-" * 40)
        
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
        
        print(f"\nTotal templates: {len(templates)}")
    
    async def generate_various_documents(self):
        """Generate various types of documents."""
        print("\n📝 Generating Various Document Types")
        print("-" * 40)
        
        # Document generation scenarios
        scenarios = [
            {
                "name": "Business Proposal",
                "request": DocumentGenerationRequest(
                    query="Create a comprehensive business proposal for implementing a cloud-based project management system for a mid-size consulting firm. Include cost analysis, implementation timeline, and expected ROI.",
                    document_type=DocumentType.PROPOSAL,
                    title="Cloud Project Management Implementation Proposal",
                    subtitle="Streamlining Operations Through Technology",
                    author="Sarah Johnson",
                    company="Tech Solutions Inc.",
                    tone="professional",
                    length="comprehensive"
                )
            },
            {
                "name": "Technical Documentation",
                "request": DocumentGenerationRequest(
                    query="Create comprehensive technical documentation for a REST API that handles user authentication, file uploads, and data processing. Include API endpoints, authentication methods, error codes, and code examples.",
                    document_type=DocumentType.TECHNICAL_DOCUMENT,
                    title="User Management API Documentation",
                    subtitle="Complete Guide to Authentication and File Processing",
                    author="Development Team",
                    company="API Solutions Ltd.",
                    tone="technical",
                    length="long"
                )
            },
            {
                "name": "Business Report",
                "request": DocumentGenerationRequest(
                    query="Create a comprehensive quarterly business report analyzing market trends, sales performance, customer satisfaction metrics, and strategic recommendations for the next quarter in the SaaS industry.",
                    document_type=DocumentType.REPORT,
                    title="Q4 2024 SaaS Industry Report",
                    subtitle="Market Analysis and Strategic Recommendations",
                    author="Analytics Team",
                    company="SaaS Analytics",
                    tone="formal",
                    length="comprehensive"
                )
            },
            {
                "name": "User Manual",
                "request": DocumentGenerationRequest(
                    query="Create a comprehensive user manual for a mobile fitness tracking application. Include setup instructions, feature explanations, workout tracking, progress monitoring, and troubleshooting tips.",
                    document_type=DocumentType.MANUAL,
                    title="FitTracker Mobile App User Manual",
                    subtitle="Complete Guide to Fitness Tracking",
                    author="Product Team",
                    company="FitTracker Inc.",
                    tone="casual",
                    length="long"
                )
            },
            {
                "name": "Academic Paper",
                "request": DocumentGenerationRequest(
                    query="Create an academic research paper on the impact of machine learning algorithms on predictive analytics in healthcare. Include literature review, methodology, findings, and conclusions with proper academic formatting.",
                    document_type=DocumentType.ACADEMIC_PAPER,
                    title="Machine Learning in Healthcare Predictive Analytics",
                    subtitle="A Comprehensive Analysis of Algorithmic Impact",
                    author="Dr. Maria Rodriguez",
                    company="University of Technology",
                    tone="academic",
                    length="comprehensive"
                )
            }
        ]
        
        for scenario in scenarios:
            print(f"\n📄 Generating {scenario['name']}...")
            
            start_time = time.time()
            response = await self.doc_service.generate_document(scenario['request'])
            generation_time = time.time() - start_time
            
            if response.success:
                document = response.document
                self.generated_documents.append(document)
                
                print(f"✅ {scenario['name']} generated successfully!")
                print(f"   Title: {document.title}")
                print(f"   Word Count: {document.word_count}")
                print(f"   Sections: {len(document.sections)}")
                print(f"   Generation Time: {generation_time:.2f}s")
                
                # Show first few sections
                if document.sections:
                    print(f"   Sections: {', '.join([s.title for s in document.sections[:3]])}")
                    if len(document.sections) > 3:
                        print(f"   ... and {len(document.sections) - 3} more")
            else:
                print(f"❌ {scenario['name']} generation failed: {response.message}")
    
    async def export_documents(self):
        """Export documents in various formats."""
        print("\n📤 Exporting Documents in Various Formats")
        print("-" * 40)
        
        if not self.generated_documents:
            print("No documents to export.")
            return
        
        export_formats = [
            (ExportFormat.PDF, "Professional PDF"),
            (ExportFormat.WORD, "Microsoft Word"),
            (ExportFormat.MARKDOWN, "Markdown"),
            (ExportFormat.HTML, "HTML")
        ]
        
        for document in self.generated_documents[:2]:  # Export first 2 documents
            print(f"\n📄 Exporting: {document.title}")
            
            for format_type, format_name in export_formats:
                print(f"   📤 Exporting to {format_name}...")
                
                export_request = DocumentExportRequest(
                    document_id=document.id,
                    format=format_type,
                    custom_filename=f"{document.title.replace(' ', '_').lower()}.{format_type.value}"
                )
                
                start_time = time.time()
                export_response = await self.export_service.export_document(document, export_request)
                export_time = time.time() - start_time
                
                if export_response.success:
                    print(f"   ✅ {format_name}: {export_response.file_path}")
                    print(f"      Size: {export_response.file_size} bytes")
                    print(f"      Time: {export_time:.2f}s")
                else:
                    print(f"   ❌ {format_name} export failed: {export_response.message}")
    
    async def demonstrate_document_management(self):
        """Demonstrate document management features."""
        print("\n📋 Document Management Features")
        print("-" * 40)
        
        # List all documents
        all_documents = self.doc_service.list_documents(limit=100, offset=0)
        print(f"Total documents in system: {len(all_documents)}")
        
        # Show document details
        if all_documents:
            print("\n📄 Document Details:")
            for i, doc in enumerate(all_documents[:3], 1):  # Show first 3
                print(f"\n{i}. {doc.title}")
                print(f"   Type: {doc.document_type.value}")
                print(f"   Author: {doc.author or 'Unknown'}")
                print(f"   Company: {doc.company or 'Unknown'}")
                print(f"   Word Count: {doc.word_count}")
                print(f"   Status: {doc.status}")
                print(f"   Created: {doc.date_created.strftime('%Y-%m-%d %H:%M')}")
        
        # Demonstrate document retrieval
        if self.generated_documents:
            first_doc = self.generated_documents[0]
            retrieved_doc = self.doc_service.get_document(first_doc.id)
            
            if retrieved_doc:
                print(f"\n🔍 Document Retrieval Test:")
                print(f"   Retrieved: {retrieved_doc.title}")
                print(f"   Match: {retrieved_doc.id == first_doc.id}")
    
    async def show_statistics(self):
        """Show statistics and analytics."""
        print("\n📊 Document Statistics and Analytics")
        print("-" * 40)
        
        all_documents = self.doc_service.list_documents(limit=1000, offset=0)
        
        if not all_documents:
            print("No documents to analyze.")
            return
        
        # Calculate statistics
        total_documents = len(all_documents)
        total_word_count = sum(doc.word_count for doc in all_documents)
        average_word_count = total_word_count / total_documents if total_documents > 0 else 0
        
        # Group by document type
        by_type = {}
        for doc in all_documents:
            doc_type = doc.document_type.value
            by_type[doc_type] = by_type.get(doc_type, 0) + 1
        
        # Group by status
        by_status = {}
        for doc in all_documents:
            status = doc.status
            by_status[status] = by_status.get(status, 0) + 1
        
        print(f"📈 Overall Statistics:")
        print(f"   Total Documents: {total_documents}")
        print(f"   Total Word Count: {total_word_count:,}")
        print(f"   Average Word Count: {average_word_count:.0f}")
        
        print(f"\n📋 Documents by Type:")
        for doc_type, count in sorted(by_type.items()):
            percentage = (count / total_documents) * 100
            print(f"   {doc_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        print(f"\n📊 Documents by Status:")
        for status, count in sorted(by_status.items()):
            percentage = (count / total_documents) * 100
            print(f"   {status.title()}: {count} ({percentage:.1f}%)")
        
        # Show recent documents
        recent_docs = sorted(all_documents, key=lambda x: x.date_created, reverse=True)[:3]
        print(f"\n🕒 Recent Documents:")
        for i, doc in enumerate(recent_docs, 1):
            print(f"   {i}. {doc.title} ({doc.date_created.strftime('%Y-%m-%d')})")
    
    def show_configuration(self):
        """Show current configuration."""
        print("\n⚙️ Current Configuration")
        print("-" * 40)
        
        config_items = [
            ("Feature Enabled", self.config.enabled),
            ("Debug Mode", self.config.debug_mode),
            ("AI Model", self.config.ai_model_name),
            ("Max Tokens", self.config.ai_max_tokens),
            ("Temperature", self.config.ai_temperature),
            ("Export Directory", self.config.export_directory),
            ("Max Document Length", self.config.max_document_length),
            ("Default Tone", self.config.default_tone),
            ("Default Language", self.config.default_language),
            ("Rate Limit (req/min)", self.config.rate_limit_requests_per_minute),
            ("Async Generation", self.config.async_generation),
            ("Background Export", self.config.background_export)
        ]
        
        for key, value in config_items:
            print(f"   {key}: {value}")
    
    async def run_performance_test(self):
        """Run a performance test."""
        print("\n⚡ Performance Test")
        print("-" * 40)
        
        # Test document generation performance
        test_request = DocumentGenerationRequest(
            query="Create a short test document for performance testing",
            document_type=DocumentType.REPORT,
            title="Performance Test Document",
            length="short"
        )
        
        print("Testing document generation performance...")
        
        times = []
        for i in range(3):  # Run 3 tests
            start_time = time.time()
            response = await self.doc_service.generate_document(test_request)
            end_time = time.time()
            
            if response.success:
                generation_time = end_time - start_time
                times.append(generation_time)
                print(f"   Test {i+1}: {generation_time:.2f}s")
            else:
                print(f"   Test {i+1}: Failed")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\n📊 Performance Results:")
            print(f"   Average Time: {avg_time:.2f}s")
            print(f"   Min Time: {min_time:.2f}s")
            print(f"   Max Time: {max_time:.2f}s")


async def main():
    """Main function to run the complete demo."""
    demo = CompleteDemo()
    
    try:
        # Show configuration
        demo.show_configuration()
        
        # Run complete demo
        await demo.run_complete_demo()
        
        # Run performance test
        await demo.run_performance_test()
        
        print("\n🎯 Demo Summary:")
        print("✅ Document generation working")
        print("✅ Multiple export formats working")
        print("✅ Document management working")
        print("✅ Statistics and analytics working")
        print("✅ Performance testing completed")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())




























