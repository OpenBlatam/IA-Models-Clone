"""
Professional Documents Setup Script
==================================

Setup script for the Professional Documents feature.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing Professional Documents Dependencies...")
    
    dependencies = [
        "reportlab>=4.0.0",
        "python-docx>=0.8.11",
        "jinja2>=3.1.0",
        "aiofiles>=23.0.0",
        "markdown>=3.5.0",
        "python-multipart>=0.0.6"
    ]
    
    for dep in dependencies:
        try:
            print(f"   Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"   ✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {dep}: {e}")
            return False
    
    return True


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating Directories...")
    
    directories = [
        "exports",
        "temp",
        "templates/custom",
        "logs"
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created directory: {directory}")
        except Exception as e:
            print(f"   ❌ Failed to create directory {directory}: {e}")
            return False
    
    return True


def setup_environment():
    """Setup environment variables."""
    print("\n🔧 Setting up Environment...")
    
    env_vars = {
        "PROFESSIONAL_DOCUMENTS_ENABLED": "true",
        "PROFESSIONAL_DOCUMENTS_EXPORT_DIR": "exports",
        "PROFESSIONAL_DOCUMENTS_DEBUG": "false",
        "PROFESSIONAL_DOCUMENTS_AI_MODEL": "gpt-4"
    }
    
    env_file = Path(".env")
    env_content = []
    
    # Read existing .env file if it exists
    if env_file.exists():
        with open(env_file, "r") as f:
            env_content = f.readlines()
    
    # Add new environment variables
    for key, value in env_vars.items():
        # Check if variable already exists
        exists = any(line.startswith(f"{key}=") for line in env_content)
        
        if not exists:
            env_content.append(f"{key}={value}\n")
            print(f"   ✅ Added environment variable: {key}={value}")
        else:
            print(f"   ⚠️  Environment variable already exists: {key}")
    
    # Write updated .env file
    try:
        with open(env_file, "w") as f:
            f.writelines(env_content)
        print(f"   ✅ Updated .env file")
    except Exception as e:
        print(f"   ❌ Failed to update .env file: {e}")
        return False
    
    return True


def verify_installation():
    """Verify the installation."""
    print("\n🔍 Verifying Installation...")
    
    try:
        # Test imports
        from .models import DocumentGenerationRequest, DocumentType
        from .services import DocumentGenerationService
        from .templates import template_manager
        print("   ✅ All modules imported successfully")
        
        # Test template manager
        templates = template_manager.get_all_templates()
        print(f"   ✅ Template manager working: {len(templates)} templates loaded")
        
        # Test document service
        doc_service = DocumentGenerationService()
        print("   ✅ Document generation service initialized")
        
        # Test export service
        export_service = DocumentExportService()
        print("   ✅ Document export service initialized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return False


def run_tests():
    """Run basic tests."""
    print("\n🧪 Running Basic Tests...")
    
    try:
        # Test document generation request
        from .models import DocumentGenerationRequest, DocumentType
        
        request = DocumentGenerationRequest(
            query="Create a test document for verification",
            document_type=DocumentType.REPORT,
            title="Test Document"
        )
        
        print("   ✅ Document generation request created")
        
        # Test template retrieval
        from .templates import template_manager
        
        templates = template_manager.get_templates_by_type(DocumentType.REPORT)
        if templates:
            print(f"   ✅ Report templates found: {len(templates)}")
        else:
            print("   ⚠️  No report templates found")
        
        # Test export formats
        from .models import ExportFormat
        
        formats = [ExportFormat.PDF, ExportFormat.WORD, ExportFormat.MARKDOWN, ExportFormat.HTML]
        print(f"   ✅ Export formats available: {len(formats)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Tests failed: {e}")
        return False


def show_usage_instructions():
    """Show usage instructions."""
    print("\n📖 Usage Instructions")
    print("=" * 50)
    
    print("""
🚀 Professional Documents Feature Setup Complete!

📋 Quick Start:
1. Import the feature in your FastAPI app:
   from features.professional_documents.api import router as professional_documents_router
   app.include_router(professional_documents_router, prefix="/api/v1")

2. Generate a document:
   POST /api/v1/professional-documents/generate
   {
       "query": "Create a business proposal for CRM implementation",
       "document_type": "proposal",
       "title": "CRM Proposal",
       "author": "John Smith"
   }

3. Export a document:
   POST /api/v1/professional-documents/export
   {
       "document_id": "doc_123",
       "format": "pdf",
       "custom_filename": "my_proposal.pdf"
   }

📚 Available Document Types:
   • report - Business reports and analysis
   • proposal - Business proposals
   • manual - User manuals and guides
   • technical_document - Technical documentation
   • academic_paper - Academic research papers
   • whitepaper - Industry whitepapers
   • business_plan - Business plans
   • newsletter - Newsletters
   • brochure - Marketing brochures
   • guide - How-to guides
   • catalog - Product catalogs
   • presentation - Presentations

📤 Export Formats:
   • pdf - Professional PDF documents
   • docx - Microsoft Word documents
   • md - Markdown documents
   • html - HTML documents

🔧 Configuration:
   • Set PROFESSIONAL_DOCUMENTS_ENABLED=true to enable
   • Set PROFESSIONAL_DOCUMENTS_AI_MODEL to configure AI model
   • Set PROFESSIONAL_DOCUMENTS_EXPORT_DIR for export directory

📁 File Structure:
   • exports/ - Generated document files
   • temp/ - Temporary processing files
   • templates/custom/ - Custom templates
   • logs/ - Application logs

🧪 Testing:
   Run the demo script to test functionality:
   python -m features.professional_documents.demo_complete

📖 Documentation:
   See README.md for detailed documentation and examples.
    """)


def main():
    """Main setup function."""
    print("🚀 Professional Documents Feature Setup")
    print("=" * 50)
    
    success = True
    
    # Install dependencies
    if not install_dependencies():
        success = False
    
    # Create directories
    if not create_directories():
        success = False
    
    # Setup environment
    if not setup_environment():
        success = False
    
    # Verify installation
    if not verify_installation():
        success = False
    
    # Run tests
    if not run_tests():
        success = False
    
    if success:
        print("\n🎉 Setup completed successfully!")
        show_usage_instructions()
    else:
        print("\n❌ Setup completed with errors. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()




























