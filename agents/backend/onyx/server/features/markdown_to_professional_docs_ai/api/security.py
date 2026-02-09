"""Security endpoints"""
from fastapi import APIRouter, Form
from utils.security import get_security_sanitizer

router = APIRouter(prefix="/security", tags=["Security"])


@router.post("/sanitize")
async def sanitize_content(
    content: str = Form(...),
    content_type: str = Form("markdown")  # markdown, html, filename, url
):
    """Sanitize content"""
    sanitizer = get_security_sanitizer()
    
    if content_type == "markdown":
        sanitized = sanitizer.sanitize_markdown(content)
    elif content_type == "html":
        sanitized = sanitizer.sanitize_html(content)
    elif content_type == "filename":
        sanitized = sanitizer.sanitize_filename(content)
    elif content_type == "url":
        sanitized = sanitizer.sanitize_url(content)
    else:
        sanitized = sanitizer.sanitize_markdown(content)
    
    return {
        "original": content,
        "sanitized": sanitized,
        "content_type": content_type
    }

