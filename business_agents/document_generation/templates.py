"""
Document Templates
==================

Template management system for document generation.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from .types import DocumentTemplate, DocumentType, DocumentFormat, TemplateType

logger = logging.getLogger(__name__)

class TemplateManager:
    """Manages document templates."""
    
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, DocumentTemplate] = {}
        self.templates_dir.mkdir(exist_ok=True)
    
    async def load_templates(self):
        """Load all templates from the templates directory."""
        try:
            template_files = list(self.templates_dir.glob("*.json"))
            
            for template_file in template_files:
                try:
                    with open(template_file, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)
                    
                    template = self._create_template_from_data(template_data)
                    self.templates[template.template_id] = template
                    
                except Exception as e:
                    logger.error(f"Failed to load template {template_file}: {str(e)}")
            
            logger.info(f"Loaded {len(self.templates)} templates")
            
        except Exception as e:
            logger.error(f"Failed to load templates: {str(e)}")
    
    def _create_template_from_data(self, data: Dict[str, Any]) -> DocumentTemplate:
        """Create a DocumentTemplate from dictionary data."""
        return DocumentTemplate(
            template_id=data["template_id"],
            name=data["name"],
            description=data["description"],
            document_type=DocumentType(data["document_type"]),
            template_type=TemplateType(data["template_type"]),
            format=DocumentFormat(data["format"]),
            content=data["content"],
            variables=data.get("variables", {}),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
            is_active=data.get("is_active", True)
        )
    
    def get_template(self, template_id: str) -> Optional[DocumentTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def get_templates_by_type(self, document_type: DocumentType) -> List[DocumentTemplate]:
        """Get all templates for a specific document type."""
        return [
            template for template in self.templates.values()
            if template.document_type == document_type and template.is_active
        ]
    
    def get_templates_by_format(self, format: DocumentFormat) -> List[DocumentTemplate]:
        """Get all templates for a specific format."""
        return [
            template for template in self.templates.values()
            if template.format == format and template.is_active
        ]
    
    def list_templates(self) -> List[DocumentTemplate]:
        """List all active templates."""
        return [template for template in self.templates.values() if template.is_active]
    
    async def create_template(self, template: DocumentTemplate) -> bool:
        """Create a new template."""
        try:
            self.templates[template.template_id] = template
            await self._save_template(template)
            logger.info(f"Created template: {template.template_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create template: {str(e)}")
            return False
    
    async def update_template(self, template_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing template."""
        try:
            if template_id not in self.templates:
                return False
            
            template = self.templates[template_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(template, key):
                    setattr(template, key, value)
            
            template.updated_at = datetime.now()
            await self._save_template(template)
            logger.info(f"Updated template: {template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update template {template_id}: {str(e)}")
            return False
    
    async def delete_template(self, template_id: str) -> bool:
        """Delete a template."""
        try:
            if template_id not in self.templates:
                return False
            
            del self.templates[template_id]
            
            # Remove template file
            template_file = self.templates_dir / f"{template_id}.json"
            if template_file.exists():
                template_file.unlink()
            
            logger.info(f"Deleted template: {template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete template {template_id}: {str(e)}")
            return False
    
    async def _save_template(self, template: DocumentTemplate):
        """Save a template to disk."""
        template_data = {
            "template_id": template.template_id,
            "name": template.name,
            "description": template.description,
            "document_type": template.document_type.value,
            "template_type": template.template_type.value,
            "format": template.format.value,
            "content": template.content,
            "variables": template.variables,
            "metadata": template.metadata,
            "created_at": template.created_at.isoformat() if template.created_at else None,
            "updated_at": template.updated_at.isoformat() if template.updated_at else None,
            "is_active": template.is_active
        }
        
        template_file = self.templates_dir / f"{template.template_id}.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2, ensure_ascii=False)
    
    def get_default_template(self, document_type: DocumentType, format: DocumentFormat) -> Optional[DocumentTemplate]:
        """Get the default template for a document type and format."""
        templates = self.get_templates_by_type(document_type)
        format_templates = [t for t in templates if t.format == format]
        
        if format_templates:
            # Return the first template (could be enhanced with priority logic)
            return format_templates[0]
        
        return None

class BuiltinTemplates:
    """Built-in template definitions."""
    
    @staticmethod
    def get_business_plan_template() -> DocumentTemplate:
        """Get business plan template."""
        return DocumentTemplate(
            template_id="business_plan_default",
            name="Business Plan Template",
            description="Standard business plan template",
            document_type=DocumentType.BUSINESS_PLAN,
            template_type=TemplateType.STATIC,
            format=DocumentFormat.MARKDOWN,
            content="""# {{ title }}

## Executive Summary
{{ executive_summary }}

## Company Description
{{ company_description }}

## Market Analysis
{{ market_analysis }}

## Organization & Management
{{ organization_management }}

## Service or Product Line
{{ service_product_line }}

## Marketing & Sales
{{ marketing_sales }}

## Financial Projections
{{ financial_projections }}

## Funding Request
{{ funding_request }}

## Appendix
{{ appendix }}
""",
            variables={
                "title": "Business Plan",
                "executive_summary": "",
                "company_description": "",
                "market_analysis": "",
                "organization_management": "",
                "service_product_line": "",
                "marketing_sales": "",
                "financial_projections": "",
                "funding_request": "",
                "appendix": ""
            }
        )
    
    @staticmethod
    def get_marketing_report_template() -> DocumentTemplate:
        """Get marketing report template."""
        return DocumentTemplate(
            template_id="marketing_report_default",
            name="Marketing Report Template",
            description="Standard marketing report template",
            document_type=DocumentType.REPORT,
            template_type=TemplateType.STATIC,
            format=DocumentFormat.MARKDOWN,
            content="""# {{ title }}

## Report Overview
{{ report_overview }}

## Campaign Performance
{{ campaign_performance }}

## Key Metrics
{{ key_metrics }}

## Analysis
{{ analysis }}

## Recommendations
{{ recommendations }}

## Next Steps
{{ next_steps }}
""",
            variables={
                "title": "Marketing Report",
                "report_overview": "",
                "campaign_performance": "",
                "key_metrics": "",
                "analysis": "",
                "recommendations": "",
                "next_steps": ""
            }
        )
    
    @staticmethod
    def get_technical_spec_template() -> DocumentTemplate:
        """Get technical specification template."""
        return DocumentTemplate(
            template_id="technical_spec_default",
            name="Technical Specification Template",
            description="Standard technical specification template",
            document_type=DocumentType.TECHNICAL_SPEC,
            template_type=TemplateType.STATIC,
            format=DocumentFormat.MARKDOWN,
            content="""# {{ title }}

## Overview
{{ overview }}

## Requirements
{{ requirements }}

## Architecture
{{ architecture }}

## Implementation
{{ implementation }}

## Testing
{{ testing }}

## Deployment
{{ deployment }}

## Maintenance
{{ maintenance }}
""",
            variables={
                "title": "Technical Specification",
                "overview": "",
                "requirements": "",
                "architecture": "",
                "implementation": "",
                "testing": "",
                "deployment": "",
                "maintenance": ""
            }
        )
