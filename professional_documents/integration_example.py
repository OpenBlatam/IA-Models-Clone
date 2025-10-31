"""
Integration Example for Professional Documents Feature
======================================================

This example shows how to integrate the Professional Documents feature
into your FastAPI application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the professional documents router
from .api import router as professional_documents_router

# Import the app factory from the core module
from ..core.app_factory import create_development_app


def create_app_with_documents() -> FastAPI:
    """Create a FastAPI app with the Professional Documents feature integrated."""
    
    # Create the base app using the existing app factory
    app = create_development_app(
        title="Blatam Academy API with Professional Documents",
        version="1.0.0",
        routers=[
            {
                "router": professional_documents_router,
                "prefix": "/api/v1",
                "tags": ["Professional Documents"]
            }
        ]
    )
    
    return app


# Example usage in your main application
def setup_professional_documents_feature(app: FastAPI):
    """Setup the professional documents feature in an existing FastAPI app."""
    
    # Include the router
    app.include_router(
        professional_documents_router,
        prefix="/api/v1/professional-documents",
        tags=["Professional Documents"]
    )
    
    # Add any additional middleware or configuration specific to this feature
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


# Example of how to use the feature programmatically
async def example_usage():
    """Example of how to use the professional documents feature programmatically."""
    
    from .models import DocumentGenerationRequest, DocumentType
    from .services import DocumentGenerationService, DocumentExportService
    from .models import ExportFormat, DocumentExportRequest
    
    # Initialize services
    doc_service = DocumentGenerationService()
    export_service = DocumentExportService()
    
    # Create a document generation request
    request = DocumentGenerationRequest(
        query="Create a comprehensive business plan for a new tech startup focused on AI-powered customer service solutions",
        document_type=DocumentType.BUSINESS_PLAN,
        title="AI Customer Service Startup Business Plan",
        subtitle="Revolutionizing Customer Support with Artificial Intelligence",
        author="Alex Johnson",
        company="AI Solutions Inc.",
        tone="professional",
        length="comprehensive"
    )
    
    # Generate the document
    response = await doc_service.generate_document(request)
    
    if response.success:
        document = response.document
        print(f"✅ Generated document: {document.title}")
        print(f"   Word count: {document.word_count}")
        print(f"   Sections: {len(document.sections)}")
        
        # Export to PDF
        export_request = DocumentExportRequest(
            document_id=document.id,
            format=ExportFormat.PDF,
            custom_filename="business_plan.pdf"
        )
        
        export_response = await export_service.export_document(document, export_request)
        
        if export_response.success:
            print(f"✅ Exported to: {export_response.file_path}")
        else:
            print(f"❌ Export failed: {export_response.message}")
    else:
        print(f"❌ Generation failed: {response.message}")


# Example API client usage
def example_api_client():
    """Example of how to use the API endpoints from a client."""
    
    import requests
    import json
    
    base_url = "http://localhost:8000/api/v1/professional-documents"
    
    # Example: Generate a document
    generation_request = {
        "query": "Create a technical documentation for a REST API that manages user authentication and authorization",
        "document_type": "technical_document",
        "title": "User Authentication API Documentation",
        "author": "Tech Team",
        "company": "Your Company",
        "tone": "technical",
        "length": "medium"
    }
    
    # Make the request (you would need proper authentication in real usage)
    response = requests.post(
        f"{base_url}/generate",
        json=generation_request,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        document = result["document"]
        print(f"✅ Document generated: {document['title']}")
        
        # Export to different formats
        export_formats = ["pdf", "docx", "md", "html"]
        
        for format_type in export_formats:
            export_request = {
                "document_id": document["id"],
                "format": format_type,
                "custom_filename": f"api_docs.{format_type}"
            }
            
            export_response = requests.post(
                f"{base_url}/export",
                json=export_request
            )
            
            if export_response.status_code == 200:
                export_result = export_response.json()
                print(f"✅ Exported to {format_type}: {export_result['download_url']}")
            else:
                print(f"❌ Export to {format_type} failed")
    else:
        print(f"❌ Generation failed: {response.status_code} - {response.text}")


if __name__ == "__main__":
    # Create and run the app
    app = create_app_with_documents()
    
    print("🚀 Professional Documents Feature Integration Example")
    print("=" * 60)
    print("The app is now configured with the Professional Documents feature.")
    print("Available endpoints:")
    print("- POST /api/v1/professional-documents/generate")
    print("- GET  /api/v1/professional-documents/documents")
    print("- POST /api/v1/professional-documents/export")
    print("- GET  /api/v1/professional-documents/templates")
    print("- GET  /api/v1/professional-documents/health")
    print("\nTo run the app:")
    print("uvicorn integration_example:app --reload")




























