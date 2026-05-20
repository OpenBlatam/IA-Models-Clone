"""
API usage examples for the Export IA system.
"""

import asyncio
import httpx
import json
from typing import Dict, Any


class ExportIAClient:
    """Client for the Export IA API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def export_document(self, content: Dict[str, Any], format_type: str, 
                            document_type: str = "report", quality_level: str = "professional") -> str:
        """Export a document via API."""
        request_data = {
            "content": content,
            "format": format_type,
            "document_type": document_type,
            "quality_level": quality_level
        }
        
        response = await self.client.post(f"{self.base_url}/export", json=request_data)
        response.raise_for_status()
        
        result = response.json()
        return result["task_id"]
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status via API."""
        response = await self.client.get(f"{self.base_url}/export/{task_id}/status")
        response.raise_for_status()
        return response.json()
    
    async def download_file(self, task_id: str, output_path: str):
        """Download exported file via API."""
        response = await self.client.get(f"{self.base_url}/export/{task_id}/download")
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics via API."""
        response = await self.client.get(f"{self.base_url}/statistics")
        response.raise_for_status()
        return response.json()
    
    async def get_supported_formats(self) -> list:
        """Get supported formats via API."""
        response = await self.client.get(f"{self.base_url}/formats")
        response.raise_for_status()
        return response.json()
    
    async def validate_content(self, content: Dict[str, Any], format_type: str) -> Dict[str, Any]:
        """Validate content via API."""
        request_data = {
            "content": content,
            "format": format_type
        }
        
        response = await self.client.post(f"{self.base_url}/validate", json=request_data)
        response.raise_for_status()
        return response.json()


async def api_export_example():
    """Example of using the API for document export."""
    client = ExportIAClient()
    
    try:
        # Sample content
        content = {
            "title": "API Export Example",
            "sections": [
                {
                    "heading": "Introduction",
                    "content": "This document was exported using the Export IA API."
                },
                {
                    "heading": "Features",
                    "content": "The API provides a clean interface for document export operations."
                }
            ]
        }
        
        # Export to PDF
        task_id = await client.export_document(content, "pdf", "report", "professional")
        print(f"Export task created: {task_id}")
        
        # Wait for completion
        while True:
            status = await client.get_task_status(task_id)
            if status["status"] == "completed":
                print(f"Export completed! Quality score: {status['quality_score']:.2f}")
                
                # Download the file
                output_path = f"api_export_{task_id}.pdf"
                await client.download_file(task_id, output_path)
                print(f"File downloaded to: {output_path}")
                break
            elif status["status"] == "failed":
                print(f"Export failed: {status['error']}")
                break
            else:
                print(f"Status: {status['status']}, Progress: {status.get('progress', 0):.1%}")
                await asyncio.sleep(1)
    
    finally:
        await client.close()


async def api_statistics_example():
    """Example of getting API statistics."""
    client = ExportIAClient()
    
    try:
        # Get statistics
        stats = await client.get_statistics()
        print("System Statistics:")
        print(f"  Total tasks: {stats['total_tasks']}")
        print(f"  Active tasks: {stats['active_tasks']}")
        print(f"  Completed tasks: {stats['completed_tasks']}")
        print(f"  Average quality score: {stats['average_quality_score']:.2f}")
        
        # Get supported formats
        formats = await client.get_supported_formats()
        print("\nSupported Formats:")
        for fmt in formats:
            print(f"  {fmt['name']}: {fmt['description']}")
    
    finally:
        await client.close()


async def api_validation_example():
    """Example of content validation via API."""
    client = ExportIAClient()
    
    try:
        # Sample content
        content = {
            "title": "Validation Test",
            "sections": [
                {
                    "heading": "Test Section",
                    "content": "This content will be validated for quality."
                }
            ]
        }
        
        # Validate content
        validation_result = await client.validate_content(content, "pdf")
        print("Content Validation Results:")
        print(f"  Overall score: {validation_result['overall_score']:.2f}")
        print(f"  Formatting score: {validation_result['formatting_score']:.2f}")
        print(f"  Content score: {validation_result['content_score']:.2f}")
        print(f"  Accessibility score: {validation_result['accessibility_score']:.2f}")
        print(f"  Professional score: {validation_result['professional_score']:.2f}")
        
        if validation_result['issues']:
            print("\nIssues found:")
            for issue in validation_result['issues']:
                print(f"  - {issue}")
        
        if validation_result['suggestions']:
            print("\nSuggestions:")
            for suggestion in validation_result['suggestions']:
                print(f"  - {suggestion}")
    
    finally:
        await client.close()


async def batch_export_example():
    """Example of batch export via API."""
    client = ExportIAClient()
    
    try:
        # Multiple documents to export
        documents = [
            {
                "title": "Document 1",
                "sections": [{"heading": "Section 1", "content": "Content 1"}]
            },
            {
                "title": "Document 2", 
                "sections": [{"heading": "Section 2", "content": "Content 2"}]
            },
            {
                "title": "Document 3",
                "sections": [{"heading": "Section 3", "content": "Content 3"}]
            }
        ]
        
        formats = ["pdf", "docx", "html"]
        
        # Create export tasks
        tasks = []
        for i, doc in enumerate(documents):
            for fmt in formats:
                task_id = await client.export_document(doc, fmt)
                tasks.append((task_id, f"Document {i+1}", fmt))
                print(f"Created task for Document {i+1} in {fmt}: {task_id}")
        
        # Wait for all tasks to complete
        completed_tasks = []
        while len(completed_tasks) < len(tasks):
            for task_id, doc_name, fmt in tasks:
                if (task_id, doc_name, fmt) in completed_tasks:
                    continue
                
                status = await client.get_task_status(task_id)
                if status["status"] == "completed":
                    print(f"Completed: {doc_name} in {fmt}")
                    completed_tasks.append((task_id, doc_name, fmt))
                elif status["status"] == "failed":
                    print(f"Failed: {doc_name} in {fmt} - {status['error']}")
                    completed_tasks.append((task_id, doc_name, fmt))
            
            if len(completed_tasks) < len(tasks):
                await asyncio.sleep(1)
        
        print(f"\nBatch export completed: {len(completed_tasks)} tasks processed")
    
    finally:
        await client.close()


if __name__ == "__main__":
    print("Export IA - API Usage Examples")
    print("=" * 40)
    
    # Run examples
    asyncio.run(api_export_example())
    print("\n" + "=" * 40)
    
    asyncio.run(api_statistics_example())
    print("\n" + "=" * 40)
    
    asyncio.run(api_validation_example())
    print("\n" + "=" * 40)
    
    asyncio.run(batch_export_example())




