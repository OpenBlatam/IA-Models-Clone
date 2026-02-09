"""Format and language endpoints"""
from fastapi import APIRouter
from utils.i18n import Language

router = APIRouter(prefix="/formats", tags=["Formats"])


@router.get("")
async def get_supported_formats():
    """Get list of supported output formats"""
    return {
        "formats": {
            "excel": {
                "description": "Microsoft Excel (.xlsx)",
                "features": ["tables", "charts", "formulas", "formatting"],
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            },
            "pdf": {
                "description": "Portable Document Format (.pdf)",
                "features": ["text", "tables", "charts", "diagrams", "images"],
                "mime_type": "application/pdf"
            },
            "word": {
                "description": "Microsoft Word (.docx)",
                "features": ["text", "tables", "images", "formatting", "styles"],
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            },
            "tableau": {
                "description": "Tableau Workbook (.twb)",
                "features": ["data_connections", "visualizations", "dashboards"],
                "mime_type": "application/xml"
            },
            "powerbi": {
                "description": "Power BI Report (.pbix)",
                "features": ["data_models", "visualizations", "reports"],
                "mime_type": "application/x-msdownload"
            },
            "html": {
                "description": "HyperText Markup Language (.html)",
                "features": ["interactive", "charts", "responsive", "css"],
                "mime_type": "text/html"
            },
            "ppt": {
                "description": "Microsoft PowerPoint (.pptx)",
                "features": ["slides", "charts", "images", "animations"],
                "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            },
            "latex": {
                "description": "LaTeX Document (.tex)",
                "features": ["formulas", "equations", "academic_formatting"],
                "mime_type": "application/x-latex"
            },
            "rtf": {
                "description": "Rich Text Format (.rtf)",
                "features": ["text", "tables", "colors"],
                "mime_type": "application/rtf"
            },
            "epub": {
                "description": "Electronic Publication (.epub)",
                "features": ["ebook", "reflowable", "metadata"],
                "mime_type": "application/epub+zip"
            },
            "odt": {
                "description": "OpenDocument Text (.odt)",
                "features": ["text", "tables", "images"],
                "mime_type": "application/vnd.oasis.opendocument.text"
            }
        }
    }


@router.get("/languages")
async def get_languages():
    """Get list of supported languages"""
    return {
        "languages": [lang.value for lang in Language],
        "default": "en",
        "auto_detect": True
    }

