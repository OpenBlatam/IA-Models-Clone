"""
Template Engine Module
Template rendering using Jinja2 for document generation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional import
try:
    from jinja2 import Environment, FileSystemLoader, BaseLoader, DictLoader, select_autoescape
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logger.warning("jinja2 not available. Template engine disabled.")


class TemplateEngine:
    """
    Template engine for generating documents from templates.
    Uses Jinja2 for powerful template rendering.
    """
    
    def __init__(self, templates_dir: str = None):
        self.templates_dir = Path(templates_dir) if templates_dir else None
        self._env: Optional["Environment"] = None
        self._string_templates: Dict[str, str] = {}
        self._init_builtin_templates()
        
        if self.templates_dir and self.templates_dir.exists():
            self._init_file_loader()
        
        logger.info(f"TemplateEngine initialized. Templates dir: {templates_dir or 'none'}")
    
    def _init_builtin_templates(self):
        """Initialize built-in templates."""
        self._string_templates = {
            "report": """# {{ title }}
*Generated: {{ timestamp }}*

## Summary
{{ summary }}

## Details
{% for item in items %}
- **{{ item.name }}**: {{ item.value }}
{% endfor %}

## Conclusion
{{ conclusion }}
""",
            "email": """Subject: {{ subject }}

Dear {{ recipient }},

{{ body }}

Best regards,
{{ sender }}
""",
            "api_response": """{
    "status": "{{ status }}",
    "message": "{{ message }}",
    "data": {{ data | tojson }},
    "timestamp": "{{ timestamp }}"
}
""",
            "html_page": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 2rem; }
        h1 { color: #333; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    {{ content }}
</body>
</html>
""",
            "task_summary": """# Task Summary: {{ task_name }}

**Status**: {{ status }}
**Created**: {{ created_at }}
**Completed**: {{ completed_at | default('In Progress') }}

## Description
{{ description }}

## Results
{% if results %}
{% for key, value in results.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% else %}
No results yet.
{% endif %}
"""
        }
    
    def _init_file_loader(self):
        """Initialize Jinja2 environment with file loader."""
        if not JINJA2_AVAILABLE:
            return
        
        self._env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(['html', 'xml'])
        )
        self._add_custom_filters()
    
    def _get_string_env(self) -> "Environment":
        """Get Jinja2 environment for string templates."""
        if not JINJA2_AVAILABLE:
            raise ImportError("jinja2 required for template rendering")
        
        env = Environment(
            loader=DictLoader(self._string_templates),
            autoescape=select_autoescape(['html', 'xml'])
        )
        self._add_custom_filters(env)
        return env
    
    def _add_custom_filters(self, env: "Environment" = None):
        """Add custom Jinja2 filters."""
        target_env = env or self._env
        if not target_env:
            return
        
        target_env.filters['datetime'] = lambda d, fmt='%Y-%m-%d %H:%M': d.strftime(fmt) if d else ''
        target_env.filters['truncate_words'] = lambda s, n: ' '.join(str(s).split()[:n]) + '...' if len(str(s).split()) > n else s
    
    def add_template(self, name: str, template_str: str) -> None:
        """Add a string template."""
        self._string_templates[name] = template_str
        logger.debug(f"Added template: {name}")
    
    def render(
        self,
        template_name: str,
        context: Dict[str, Any] = None,
        from_string: bool = False
    ) -> str:
        """Render a template with context."""
        if not JINJA2_AVAILABLE:
            # Simple fallback
            result = self._string_templates.get(template_name, template_name)
            for key, value in (context or {}).items():
                result = result.replace(f"{{{{ {key} }}}}", str(value))
            return result
        
        context = context or {}
        context.setdefault('timestamp', datetime.now().isoformat())
        
        try:
            if from_string or template_name in self._string_templates:
                env = self._get_string_env()
                template = env.get_template(template_name)
            elif self._env:
                template = self._env.get_template(template_name)
            else:
                raise ValueError(f"Template not found: {template_name}")
            
            return template.render(**context)
        
        except Exception as e:
            logger.error(f"Template rendering failed: {e}")
            raise
    
    def render_string(self, template_str: str, context: Dict[str, Any] = None) -> str:
        """Render a template string directly."""
        if not JINJA2_AVAILABLE:
            result = template_str
            for key, value in (context or {}).items():
                result = result.replace(f"{{{{ {key} }}}}", str(value))
            return result
        
        from jinja2 import Template
        context = context or {}
        context.setdefault('timestamp', datetime.now().isoformat())
        
        template = Template(template_str)
        return template.render(**context)
    
    def list_templates(self) -> List[str]:
        """List available templates."""
        templates = list(self._string_templates.keys())
        if self._env and self.templates_dir:
            templates.extend(self._env.list_templates())
        return templates
