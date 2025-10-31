"""
Basic usage examples for the refactored Export IA system.
"""

import asyncio
from src.core.engine import ExportIAEngine
from src.core.models import ExportConfig, ExportFormat, DocumentType, QualityLevel


async def basic_export_example():
    """Basic export example."""
    # Sample content
    content = {
        "title": "Sample Business Report",
        "sections": [
            {
                "heading": "Executive Summary",
                "content": "This is a comprehensive business report covering key metrics and insights."
            },
            {
                "heading": "Key Findings",
                "content": "Our analysis reveals significant growth opportunities in the market."
            },
            {
                "heading": "Recommendations",
                "content": "We recommend implementing the proposed strategies to achieve our goals."
            }
        ]
    }
    
    # Create export configuration
    config = ExportConfig(
        format=ExportFormat.PDF,
        document_type=DocumentType.REPORT,
        quality_level=QualityLevel.PROFESSIONAL
    )
    
    # Initialize engine
    async with ExportIAEngine() as engine:
        # Export document
        task_id = await engine.export_document(content, config)
        print(f"Export task created: {task_id}")
        
        # Wait for completion and check status
        while True:
            status = await engine.get_task_status(task_id)
            if status["status"] == "completed":
                print(f"Export completed! File: {status['file_path']}")
                print(f"Quality score: {status['quality_score']:.2f}")
                break
            elif status["status"] == "failed":
                print(f"Export failed: {status['error']}")
                break
            else:
                print(f"Status: {status['status']}, Progress: {status.get('progress', 0):.1%}")
                await asyncio.sleep(1)


async def multiple_format_export():
    """Export the same content to multiple formats."""
    content = {
        "title": "Multi-Format Document",
        "sections": [
            {
                "heading": "Introduction",
                "content": "This document demonstrates multi-format export capabilities."
            },
            {
                "heading": "Features",
                "content": "The system supports PDF, DOCX, HTML, Markdown, and more."
            }
        ]
    }
    
    formats = [
        ExportFormat.PDF,
        ExportFormat.DOCX,
        ExportFormat.HTML,
        ExportFormat.MARKDOWN
    ]
    
    async with ExportIAEngine() as engine:
        tasks = []
        
        # Create export tasks for each format
        for fmt in formats:
            config = ExportConfig(
                format=fmt,
                document_type=DocumentType.REPORT,
                quality_level=QualityLevel.PROFESSIONAL
            )
            task_id = await engine.export_document(content, config)
            tasks.append((task_id, fmt))
            print(f"Created {fmt.value} export task: {task_id}")
        
        # Wait for all tasks to complete
        for task_id, fmt in tasks:
            while True:
                status = await engine.get_task_status(task_id)
                if status["status"] == "completed":
                    print(f"{fmt.value} export completed: {status['file_path']}")
                    break
                elif status["status"] == "failed":
                    print(f"{fmt.value} export failed: {status['error']}")
                    break
                await asyncio.sleep(0.5)


async def quality_levels_demo():
    """Demonstrate different quality levels."""
    content = {
        "title": "Quality Level Comparison",
        "sections": [
            {
                "heading": "Overview",
                "content": "This document shows the difference between quality levels."
            }
        ]
    }
    
    quality_levels = [
        QualityLevel.BASIC,
        QualityLevel.STANDARD,
        QualityLevel.PROFESSIONAL,
        QualityLevel.PREMIUM,
        QualityLevel.ENTERPRISE
    ]
    
    async with ExportIAEngine() as engine:
        for quality in quality_levels:
            config = ExportConfig(
                format=ExportFormat.PDF,
                document_type=DocumentType.REPORT,
                quality_level=quality
            )
            
            task_id = await engine.export_document(content, config)
            print(f"Created {quality.value} quality export: {task_id}")
            
            # Wait for completion
            while True:
                status = await engine.get_task_status(task_id)
                if status["status"] == "completed":
                    print(f"{quality.value} quality completed - Score: {status['quality_score']:.2f}")
                    break
                elif status["status"] == "failed":
                    print(f"{quality.value} quality failed: {status['error']}")
                    break
                await asyncio.sleep(0.5)


async def document_types_demo():
    """Demonstrate different document types."""
    content = {
        "title": "Document Type Template",
        "sections": [
            {
                "heading": "Main Content",
                "content": "This demonstrates how different document types use different templates."
            }
        ]
    }
    
    document_types = [
        DocumentType.BUSINESS_PLAN,
        DocumentType.REPORT,
        DocumentType.PROPOSAL,
        DocumentType.MANUAL
    ]
    
    async with ExportIAEngine() as engine:
        for doc_type in document_types:
            config = ExportConfig(
                format=ExportFormat.DOCX,
                document_type=doc_type,
                quality_level=QualityLevel.PROFESSIONAL
            )
            
            task_id = await engine.export_document(content, config)
            print(f"Created {doc_type.value} document: {task_id}")
            
            # Wait for completion
            while True:
                status = await engine.get_task_status(task_id)
                if status["status"] == "completed":
                    print(f"{doc_type.value} document completed")
                    break
                elif status["status"] == "failed":
                    print(f"{doc_type.value} document failed: {status['error']}")
                    break
                await asyncio.sleep(0.5)


async def statistics_demo():
    """Demonstrate system statistics."""
    async with ExportIAEngine() as engine:
        # Get initial statistics
        stats = engine.get_export_statistics()
        print("Initial Statistics:")
        print(f"  Total tasks: {stats.total_tasks}")
        print(f"  Active tasks: {stats.active_tasks}")
        print(f"  Completed tasks: {stats.completed_tasks}")
        print(f"  Average quality score: {stats.average_quality_score:.2f}")
        
        # Get supported formats
        formats = engine.list_supported_formats()
        print("\nSupported Formats:")
        for fmt in formats:
            print(f"  {fmt['name']}: {fmt['description']}")
            print(f"    Features: {', '.join(fmt['professional_features'])}")


if __name__ == "__main__":
    print("Export IA - Basic Usage Examples")
    print("=" * 40)
    
    # Run examples
    asyncio.run(basic_export_example())
    print("\n" + "=" * 40)
    
    asyncio.run(multiple_format_export())
    print("\n" + "=" * 40)
    
    asyncio.run(quality_levels_demo())
    print("\n" + "=" * 40)
    
    asyncio.run(document_types_demo())
    print("\n" + "=" * 40)
    
    asyncio.run(statistics_demo())




