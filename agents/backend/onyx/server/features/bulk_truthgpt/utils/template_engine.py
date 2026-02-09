"""
Template Engine
==============

Advanced template engine for document generation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import re
from pathlib import Path
import aiofiles
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

logger = logging.getLogger(__name__)

class TemplateEngine:
    """
    Advanced template engine with Jinja2 backend.
    
    Features:
    - Jinja2 templates
    - Custom filters
    - Template inheritance
    - Caching
    - Hot reloading
    """
    
    def __init__(self, template_dir: str = "./templates"):
        self.template_dir = Path(template_dir)
        self.jinja_env = None
        self.template_cache = {}
        
    async def initialize(self):
        """Initialize template engine."""
        logger.info("Initializing Template Engine...")
        
        try:
            # Create template directory
            self.template_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize Jinja2 environment
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Add custom filters
            self._add_custom_filters()
            
            # Load default templates
            await self._load_default_templates()
            
            logger.info("Template Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Template Engine: {str(e)}")
            raise
    
    def _add_custom_filters(self):
        """Add custom Jinja2 filters."""
        def format_date(value, format='%Y-%m-%d'):
            if isinstance(value, str):
                try:
                    dt = datetime.fromisoformat(value)
                    return dt.strftime(format)
                except:
                    return value
            return value
        
        def truncate_text(value, length=100, suffix='...'):
            if len(str(value)) <= length:
                return value
            return str(value)[:length] + suffix
        
        def word_count(value):
            return len(str(value).split())
        
        def sentence_count(value):
            return len(str(value).split('.'))
        
        def paragraph_count(value):
            return len(str(value).split('\n\n'))
        
        def reading_time(value, words_per_minute=200):
            words = word_count(value)
            minutes = words / words_per_minute
            return f"{int(minutes)} min"
        
        # Register filters
        self.jinja_env.filters['format_date'] = format_date
        self.jinja_env.filters['truncate'] = truncate_text
        self.jinja_env.filters['word_count'] = word_count
        self.jinja_env.filters['sentence_count'] = sentence_count
        self.jinja_env.filters['paragraph_count'] = paragraph_count
        self.jinja_env.filters['reading_time'] = reading_time
    
    async def _load_default_templates(self):
        """Load default templates."""
        try:
            # Create default templates
            default_templates = {
                'markdown': '''# {{ title }}

{{ content }}

---
*Generated on {{ created_at|format_date }} | Reading time: {{ content|reading_time }}*''',
                
                'html': '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1, h2, h3 { color: #333; }
        .metadata { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <div class="content">{{ content|safe }}</div>
    <div class="metadata">
        <p>Generated on {{ created_at|format_date }} | Reading time: {{ content|reading_time }}</p>
    </div>
</body>
</html>''',
                
                'json': '''{
    "title": "{{ title }}",
    "content": "{{ content|replace('"', '\\"') }}",
    "metadata": {
        "created_at": "{{ created_at }}",
        "word_count": {{ content|word_count }},
        "sentence_count": {{ content|sentence_count }},
        "paragraph_count": {{ content|paragraph_count }},
        "reading_time": "{{ content|reading_time }}"
    }
}''',
                
                'pdf': '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <style>
        @page { margin: 2cm; }
        body { font-family: 'Times New Roman', serif; font-size: 12pt; line-height: 1.6; }
        h1 { color: #000; font-size: 18pt; margin-bottom: 20pt; }
        h2 { color: #333; font-size: 14pt; margin-top: 20pt; margin-bottom: 10pt; }
        h3 { color: #666; font-size: 12pt; margin-top: 15pt; margin-bottom: 8pt; }
        p { margin-bottom: 10pt; text-align: justify; }
        .metadata { font-size: 10pt; color: #666; margin-top: 30pt; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <div>{{ content|safe }}</div>
    <div class="metadata">
        <p>Generated on {{ created_at|format_date }} | Reading time: {{ content|reading_time }}</p>
    </div>
</body>
</html>'''
            }
            
            # Save default templates
            for name, content in default_templates.items():
                template_file = self.template_dir / f"{name}.jinja2"
                async with aiofiles.open(template_file, 'w', encoding='utf-8') as f:
                    await f.write(content)
            
            logger.info("Default templates loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load default templates: {str(e)}")
    
    async def get_template(self, template_name: str) -> Optional[Template]:
        """Get template by name."""
        try:
            # Check cache first
            if template_name in self.template_cache:
                return self.template_cache[template_name]
            
            # Load template
            template = self.jinja_env.get_template(f"{template_name}.jinja2")
            
            # Cache template
            self.template_cache[template_name] = template
            
            return template
            
        except Exception as e:
            logger.error(f"Failed to get template {template_name}: {str(e)}")
            return None
    
    async def render_template(
        self, 
        template: Template, 
        content: str, 
        metadata: Dict[str, Any],
        **kwargs
    ) -> str:
        """Render template with data."""
        try:
            # Prepare template data
            template_data = {
                'content': content,
                'title': metadata.get('title', 'Generated Document'),
                'created_at': metadata.get('created_at', datetime.utcnow().isoformat()),
                'author': metadata.get('author', 'Bulk TruthGPT'),
                'tags': metadata.get('tags', []),
                'quality_score': metadata.get('quality_score', 0.0),
                'optimization_level': metadata.get('optimization_level', 'basic'),
                **kwargs
            }
            
            # Render template
            rendered = template.render(**template_data)
            
            return rendered
            
        except Exception as e:
            logger.error(f"Failed to render template: {str(e)}")
            return content  # Return original content if rendering fails
    
    async def create_template(
        self, 
        template_name: str, 
        template_content: str
    ) -> bool:
        """Create new template."""
        try:
            template_file = self.template_dir / f"{template_name}.jinja2"
            
            async with aiofiles.open(template_file, 'w', encoding='utf-8') as f:
                await f.write(template_content)
            
            # Clear cache for this template
            if template_name in self.template_cache:
                del self.template_cache[template_name]
            
            logger.info(f"Created template {template_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create template {template_name}: {str(e)}")
            return False
    
    async def update_template(
        self, 
        template_name: str, 
        template_content: str
    ) -> bool:
        """Update existing template."""
        try:
            template_file = self.template_dir / f"{template_name}.jinja2"
            
            if not template_file.exists():
                logger.warning(f"Template {template_name} does not exist")
                return False
            
            async with aiofiles.open(template_file, 'w', encoding='utf-8') as f:
                await f.write(template_content)
            
            # Clear cache for this template
            if template_name in self.template_cache:
                del self.template_cache[template_name]
            
            logger.info(f"Updated template {template_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update template {template_name}: {str(e)}")
            return False
    
    async def delete_template(self, template_name: str) -> bool:
        """Delete template."""
        try:
            template_file = self.template_dir / f"{template_name}.jinja2"
            
            if not template_file.exists():
                logger.warning(f"Template {template_name} does not exist")
                return False
            
            template_file.unlink()
            
            # Clear cache for this template
            if template_name in self.template_cache:
                del self.template_cache[template_name]
            
            logger.info(f"Deleted template {template_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete template {template_name}: {str(e)}")
            return False
    
    async def list_templates(self) -> List[str]:
        """List available templates."""
        try:
            templates = []
            for template_file in self.template_dir.glob("*.jinja2"):
                template_name = template_file.stem
                templates.append(template_name)
            
            return templates
            
        except Exception as e:
            logger.error(f"Failed to list templates: {str(e)}")
            return []
    
    async def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get template information."""
        try:
            template_file = self.template_dir / f"{template_name}.jinja2"
            
            if not template_file.exists():
                return None
            
            # Get file stats
            stat = template_file.stat()
            
            # Read template content
            async with aiofiles.open(template_file, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            return {
                'name': template_name,
                'size': stat.st_size,
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'content_preview': content[:200] + '...' if len(content) > 200 else content,
                'variables': self._extract_template_variables(content)
            }
            
        except Exception as e:
            logger.error(f"Failed to get template info {template_name}: {str(e)}")
            return None
    
    def _extract_template_variables(self, template_content: str) -> List[str]:
        """Extract variables from template content."""
        try:
            # Find all {{ variable }} patterns
            variables = re.findall(r'\{\{\s*([^}]+)\s*\}\}', template_content)
            
            # Clean up variables
            cleaned_variables = []
            for var in variables:
                # Remove filters and functions
                var = var.split('|')[0].split('.')[0].strip()
                if var and var not in cleaned_variables:
                    cleaned_variables.append(var)
            
            return cleaned_variables
            
        except Exception as e:
            logger.error(f"Failed to extract template variables: {str(e)}")
            return []
    
    async def validate_template(self, template_content: str) -> Dict[str, Any]:
        """Validate template syntax."""
        try:
            # Try to compile template
            template = self.jinja_env.from_string(template_content)
            
            # Test with sample data
            test_data = {
                'content': 'Sample content',
                'title': 'Sample Title',
                'created_at': datetime.utcnow().isoformat(),
                'author': 'Test Author',
                'tags': ['test'],
                'quality_score': 0.8,
                'optimization_level': 'basic'
            }
            
            rendered = template.render(**test_data)
            
            return {
                'valid': True,
                'error': None,
                'rendered_preview': rendered[:200] + '...' if len(rendered) > 200 else rendered
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'rendered_preview': None
            }
    
    async def cleanup(self):
        """Cleanup template engine."""
        try:
            # Clear template cache
            self.template_cache.clear()
            
            logger.info("Template Engine cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Template Engine: {str(e)}")











