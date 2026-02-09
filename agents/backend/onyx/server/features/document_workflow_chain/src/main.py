"""
Main Application - Simple and Clear
===================================

Simple and clear main application for the Document Workflow Chain system.
"""

from fastapi import FastAPI

from .core import create_app

# Create application
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )