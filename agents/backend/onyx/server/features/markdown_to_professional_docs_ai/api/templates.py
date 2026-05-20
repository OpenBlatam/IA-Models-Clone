"""Template management endpoints"""
from fastapi import APIRouter, HTTPException, Form
from utils.templates import get_template_manager
from utils.advanced_templates import get_advanced_template_manager
import json

router = APIRouter(prefix="/templates", tags=["Templates"])


@router.get("")
async def get_templates():
    """Get list of available templates"""
    template_manager = get_template_manager()
    advanced_manager = get_advanced_template_manager()
    
    return {
        "basic_templates": template_manager.list_templates(),
        "advanced_templates": advanced_manager.list_templates(),
        "default": "professional"
    }


@router.post("/create")
async def create_template(
    template_name: str = Form(...),
    template_config: str = Form(...)  # JSON string
):
    """Create a new template"""
    advanced_manager = get_advanced_template_manager()
    try:
        config = json.loads(template_config)
        success = advanced_manager.create_template(template_name, config)
        
        if success:
            return {
                "status": "success",
                "template_name": template_name
            }
        else:
            raise HTTPException(status_code=400, detail="Template creation failed")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in template_config")


@router.delete("/{template_name}")
async def delete_template(template_name: str):
    """Delete a template"""
    advanced_manager = get_advanced_template_manager()
    success = advanced_manager.delete_template(template_name)
    
    if success:
        return {
            "status": "success",
            "template_name": template_name
        }
    else:
        raise HTTPException(status_code=400, detail="Template deletion failed or template is built-in")

