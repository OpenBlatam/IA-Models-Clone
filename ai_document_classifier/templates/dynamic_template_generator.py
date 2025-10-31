"""
Dynamic Template Generator
==========================

Advanced template generation system that creates customized document templates
based on user requirements, document type, and context.
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import re
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class TemplateComplexity(Enum):
    """Template complexity levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PROFESSIONAL = "professional"

class DocumentFormat(Enum):
    """Document output formats"""
    WORD = "docx"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    LATEX = "latex"
    PLAIN_TEXT = "txt"

@dataclass
class TemplateSection:
    """Individual template section"""
    name: str
    description: str
    required: bool = True
    repeatable: bool = False
    order: int = 0
    content_type: str = "text"  # text, list, table, image, code
    placeholder: str = ""
    validation_rules: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    conditional: Optional[str] = None  # Condition for including this section

@dataclass
class TemplateStyle:
    """Template styling and formatting"""
    font_family: str = "Arial"
    font_size: int = 11
    line_spacing: float = 1.2
    margins: Dict[str, float] = field(default_factory=lambda: {
        "top": 1.0, "bottom": 1.0, "left": 1.0, "right": 1.0
    })
    colors: Dict[str, str] = field(default_factory=lambda: {
        "primary": "#000000",
        "secondary": "#666666",
        "accent": "#0066CC"
    })
    headers: bool = True
    page_numbers: bool = True
    table_of_contents: bool = False

@dataclass
class DynamicTemplate:
    """Dynamic template with customizable sections and styling"""
    name: str
    document_type: str
    complexity: TemplateComplexity
    sections: List[TemplateSection]
    style: TemplateStyle
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

class DynamicTemplateGenerator:
    """
    Advanced template generator that creates customized document templates
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the dynamic template generator
        
        Args:
            templates_dir: Directory containing template definitions
        """
        self.templates_dir = Path(templates_dir) if templates_dir else Path(__file__).parent
        self.templates_dir.mkdir(exist_ok=True)
        
        # Template patterns and rules
        self.template_patterns = self._load_template_patterns()
        self.style_presets = self._load_style_presets()
        self.section_templates = self._load_section_templates()
        
        # Customization options
        self.customization_options = self._load_customization_options()
    
    def _load_template_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load template patterns for different document types"""
        return {
            "novel": {
                "basic_sections": [
                    "Title Page", "Copyright", "Table of Contents", "Chapters", "About Author"
                ],
                "advanced_sections": [
                    "Title Page", "Copyright", "Dedication", "Acknowledgments", 
                    "Table of Contents", "Prologue", "Chapters", "Epilogue", 
                    "Glossary", "About Author", "Other Books"
                ],
                "professional_sections": [
                    "Title Page", "Copyright", "Dedication", "Acknowledgments",
                    "Table of Contents", "Prologue", "Chapters", "Epilogue",
                    "Glossary", "Character List", "Timeline", "About Author",
                    "Other Books", "Contact Information"
                ],
                "genre_specific": {
                    "science_fiction": ["Glossary", "Timeline", "World Building"],
                    "romance": ["Character Profiles", "Relationship Timeline"],
                    "mystery": ["Suspect List", "Clue Tracker", "Timeline"],
                    "fantasy": ["Glossary", "Map", "Character List", "Magic System"]
                }
            },
            "contract": {
                "basic_sections": [
                    "Header", "Parties", "Terms", "Signatures"
                ],
                "advanced_sections": [
                    "Header", "Parties", "Recitals", "Terms and Conditions",
                    "Payment Terms", "Termination", "Dispute Resolution", "Signatures"
                ],
                "professional_sections": [
                    "Header", "Parties", "Recitals", "Terms and Conditions",
                    "Payment Terms", "Intellectual Property", "Confidentiality",
                    "Liability", "Insurance", "Termination", "Dispute Resolution",
                    "Governing Law", "Signatures", "Exhibits"
                ],
                "type_specific": {
                    "employment": ["Position Description", "Compensation", "Benefits", "Non-Compete"],
                    "service": ["Service Description", "Scope of Work", "Timeline", "Deliverables"],
                    "partnership": ["Partnership Terms", "Profit Sharing", "Decision Making", "Exit Strategy"]
                }
            },
            "design": {
                "basic_sections": [
                    "Title Page", "Overview", "Specifications", "Implementation"
                ],
                "advanced_sections": [
                    "Title Page", "Executive Summary", "Design Overview", "Technical Specifications",
                    "System Architecture", "Drawings", "Materials", "Implementation Plan"
                ],
                "professional_sections": [
                    "Title Page", "Executive Summary", "Design Overview", "Technical Specifications",
                    "System Architecture", "Drawings", "Materials", "Implementation Plan",
                    "Testing", "Quality Assurance", "Risk Assessment", "Timeline", "Budget"
                ],
                "type_specific": {
                    "architectural": ["Site Analysis", "Floor Plans", "Elevations", "Sections", "Specifications"],
                    "product": ["User Requirements", "Concept Sketches", "3D Models", "Prototype Plans"],
                    "software": ["Requirements", "Architecture", "UI/UX", "Database Design", "API Design"]
                }
            }
        }
    
    def _load_style_presets(self) -> Dict[str, TemplateStyle]:
        """Load predefined style presets"""
        return {
            "academic": TemplateStyle(
                font_family="Times New Roman",
                font_size=12,
                line_spacing=1.5,
                margins={"top": 1.0, "bottom": 1.0, "left": 1.25, "right": 1.0},
                colors={"primary": "#000000", "secondary": "#333333", "accent": "#0000FF"},
                headers=True,
                page_numbers=True,
                table_of_contents=True
            ),
            "business": TemplateStyle(
                font_family="Arial",
                font_size=11,
                line_spacing=1.2,
                margins={"top": 1.0, "bottom": 1.0, "left": 1.0, "right": 1.0},
                colors={"primary": "#000000", "secondary": "#666666", "accent": "#0066CC"},
                headers=True,
                page_numbers=True,
                table_of_contents=True
            ),
            "creative": TemplateStyle(
                font_family="Georgia",
                font_size=12,
                line_spacing=1.4,
                margins={"top": 1.0, "bottom": 1.0, "left": 1.0, "right": 1.0},
                colors={"primary": "#000000", "secondary": "#444444", "accent": "#8B4513"},
                headers=False,
                page_numbers=True,
                table_of_contents=False
            ),
            "technical": TemplateStyle(
                font_family="Courier New",
                font_size=10,
                line_spacing=1.15,
                margins={"top": 0.75, "bottom": 0.75, "left": 0.75, "right": 0.75},
                colors={"primary": "#000000", "secondary": "#333333", "accent": "#006600"},
                headers=True,
                page_numbers=True,
                table_of_contents=True
            )
        }
    
    def _load_section_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load reusable section templates"""
        return {
            "title_page": {
                "description": "Document title page with title, author, and publication info",
                "content_type": "structured",
                "fields": ["title", "subtitle", "author", "date", "publisher"],
                "validation": ["title_required", "author_required"]
            },
            "table_of_contents": {
                "description": "Table of contents with page numbers",
                "content_type": "list",
                "auto_generate": True,
                "format": "numbered"
            },
            "executive_summary": {
                "description": "High-level overview of the document",
                "content_type": "text",
                "max_length": 500,
                "placeholder": "Provide a concise summary of the main points..."
            },
            "introduction": {
                "description": "Introduction section with background and objectives",
                "content_type": "text",
                "placeholder": "Introduce the topic and state the objectives..."
            },
            "conclusion": {
                "description": "Conclusion section with summary and recommendations",
                "content_type": "text",
                "placeholder": "Summarize key points and provide recommendations..."
            },
            "references": {
                "description": "Reference list or bibliography",
                "content_type": "list",
                "format": "academic",
                "validation": ["citation_format"]
            }
        }
    
    def _load_customization_options(self) -> Dict[str, List[str]]:
        """Load customization options for templates"""
        return {
            "languages": ["English", "Spanish", "French", "German", "Italian", "Portuguese"],
            "industries": ["Technology", "Healthcare", "Finance", "Education", "Legal", "Creative"],
            "audiences": ["General", "Technical", "Academic", "Business", "Consumer"],
            "formats": ["Formal", "Informal", "Creative", "Technical", "Academic"],
            "lengths": ["Short", "Medium", "Long", "Comprehensive"]
        }
    
    def generate_template(
        self,
        document_type: str,
        complexity: TemplateComplexity = TemplateComplexity.INTERMEDIATE,
        style_preset: str = "business",
        custom_requirements: Optional[Dict[str, Any]] = None,
        genre: Optional[str] = None,
        industry: Optional[str] = None
    ) -> DynamicTemplate:
        """
        Generate a dynamic template based on requirements
        
        Args:
            document_type: Type of document (novel, contract, design, etc.)
            complexity: Template complexity level
            style_preset: Style preset to use
            custom_requirements: Custom requirements and preferences
            genre: Document genre (for novels, contracts, etc.)
            industry: Target industry
            
        Returns:
            DynamicTemplate object
        """
        if custom_requirements is None:
            custom_requirements = {}
        
        # Get base sections for document type and complexity
        sections = self._get_sections_for_complexity(document_type, complexity)
        
        # Add genre-specific sections
        if genre and document_type in self.template_patterns:
            genre_sections = self.template_patterns[document_type].get("genre_specific", {}).get(genre, [])
            sections.extend(genre_sections)
        
        # Add industry-specific sections
        if industry:
            sections.extend(self._get_industry_sections(industry))
        
        # Apply custom requirements
        sections = self._apply_custom_requirements(sections, custom_requirements)
        
        # Create template sections
        template_sections = []
        for i, section_name in enumerate(sections):
            section = self._create_template_section(section_name, i, custom_requirements)
            template_sections.append(section)
        
        # Get style
        style = self.style_presets.get(style_preset, self.style_presets["business"])
        if custom_requirements.get("custom_style"):
            style = self._customize_style(style, custom_requirements["custom_style"])
        
        # Create template name
        template_name = self._generate_template_name(document_type, complexity, genre, industry)
        
        # Create metadata
        metadata = {
            "document_type": document_type,
            "complexity": complexity.value,
            "style_preset": style_preset,
            "genre": genre,
            "industry": industry,
            "custom_requirements": custom_requirements,
            "generated_at": datetime.now().isoformat(),
            "section_count": len(template_sections)
        }
        
        # Create tags
        tags = [document_type, complexity.value, style_preset]
        if genre:
            tags.append(genre)
        if industry:
            tags.append(industry)
        
        return DynamicTemplate(
            name=template_name,
            document_type=document_type,
            complexity=complexity,
            sections=template_sections,
            style=style,
            metadata=metadata,
            tags=tags
        )
    
    def _get_sections_for_complexity(self, document_type: str, complexity: TemplateComplexity) -> List[str]:
        """Get sections based on document type and complexity"""
        if document_type not in self.template_patterns:
            return ["Introduction", "Main Content", "Conclusion"]
        
        patterns = self.template_patterns[document_type]
        
        if complexity == TemplateComplexity.BASIC:
            return patterns.get("basic_sections", ["Introduction", "Main Content", "Conclusion"])
        elif complexity == TemplateComplexity.INTERMEDIATE:
            return patterns.get("advanced_sections", patterns.get("basic_sections", []))
        else:  # ADVANCED or PROFESSIONAL
            return patterns.get("professional_sections", patterns.get("advanced_sections", []))
    
    def _get_industry_sections(self, industry: str) -> List[str]:
        """Get industry-specific sections"""
        industry_sections = {
            "Technology": ["Technical Specifications", "System Requirements", "API Documentation"],
            "Healthcare": ["Medical Disclaimer", "Patient Information", "Compliance"],
            "Finance": ["Financial Disclaimers", "Risk Assessment", "Regulatory Compliance"],
            "Education": ["Learning Objectives", "Assessment Criteria", "References"],
            "Legal": ["Legal Disclaimers", "Jurisdiction", "Governing Law"],
            "Creative": ["Creative Brief", "Inspiration", "Style Guide"]
        }
        return industry_sections.get(industry, [])
    
    def _apply_custom_requirements(self, sections: List[str], requirements: Dict[str, Any]) -> List[str]:
        """Apply custom requirements to sections"""
        # Add required sections
        if "required_sections" in requirements:
            sections.extend(requirements["required_sections"])
        
        # Remove excluded sections
        if "excluded_sections" in requirements:
            sections = [s for s in sections if s not in requirements["excluded_sections"]]
        
        # Reorder sections
        if "section_order" in requirements:
            custom_order = requirements["section_order"]
            ordered_sections = []
            for section in custom_order:
                if section in sections:
                    ordered_sections.append(section)
            # Add remaining sections
            for section in sections:
                if section not in ordered_sections:
                    ordered_sections.append(section)
            sections = ordered_sections
        
        return list(dict.fromkeys(sections))  # Remove duplicates while preserving order
    
    def _create_template_section(
        self, 
        section_name: str, 
        order: int, 
        requirements: Dict[str, Any]
    ) -> TemplateSection:
        """Create a template section with appropriate properties"""
        # Get section template if available
        section_template = self.section_templates.get(section_name.lower().replace(" ", "_"), {})
        
        # Determine if section is required
        required = section_name.lower() in [
            "title page", "introduction", "main content", "conclusion", "signatures"
        ]
        
        # Determine if section is repeatable
        repeatable = section_name.lower() in [
            "chapter", "section", "appendix", "exhibit", "attachment"
        ]
        
        # Get placeholder text
        placeholder = section_template.get("placeholder", f"Enter content for {section_name}...")
        
        # Get validation rules
        validation_rules = section_template.get("validation", [])
        
        # Get examples
        examples = section_template.get("examples", [])
        
        return TemplateSection(
            name=section_name,
            description=section_template.get("description", f"Section for {section_name}"),
            required=required,
            repeatable=repeatable,
            order=order,
            content_type=section_template.get("content_type", "text"),
            placeholder=placeholder,
            validation_rules=validation_rules,
            examples=examples
        )
    
    def _customize_style(self, base_style: TemplateStyle, custom_style: Dict[str, Any]) -> TemplateStyle:
        """Customize style based on user preferences"""
        style_dict = {
            "font_family": base_style.font_family,
            "font_size": base_style.font_size,
            "line_spacing": base_style.line_spacing,
            "margins": base_style.margins.copy(),
            "colors": base_style.colors.copy(),
            "headers": base_style.headers,
            "page_numbers": base_style.page_numbers,
            "table_of_contents": base_style.table_of_contents
        }
        
        # Apply customizations
        for key, value in custom_style.items():
            if key in style_dict:
                style_dict[key] = value
        
        return TemplateStyle(**style_dict)
    
    def _generate_template_name(
        self, 
        document_type: str, 
        complexity: TemplateComplexity, 
        genre: Optional[str], 
        industry: Optional[str]
    ) -> str:
        """Generate a descriptive template name"""
        name_parts = [document_type.replace("_", " ").title()]
        
        if genre:
            name_parts.append(genre.replace("_", " ").title())
        
        if industry:
            name_parts.append(f"({industry})")
        
        name_parts.append(complexity.value.title())
        
        return " ".join(name_parts)
    
    def export_template(
        self, 
        template: DynamicTemplate, 
        format: DocumentFormat = DocumentFormat.MARKDOWN,
        include_metadata: bool = True
    ) -> str:
        """
        Export template in specified format
        
        Args:
            template: DynamicTemplate to export
            format: Output format
            include_metadata: Whether to include metadata
            
        Returns:
            Exported template as string
        """
        if format == DocumentFormat.MARKDOWN:
            return self._export_markdown(template, include_metadata)
        elif format == DocumentFormat.JSON:
            return self._export_json(template, include_metadata)
        elif format == DocumentFormat.YAML:
            return self._export_yaml(template, include_metadata)
        elif format == DocumentFormat.HTML:
            return self._export_html(template, include_metadata)
        else:
            return self._export_plain_text(template, include_metadata)
    
    def _export_markdown(self, template: DynamicTemplate, include_metadata: bool) -> str:
        """Export template as Markdown"""
        md = f"# {template.name}\n\n"
        
        if include_metadata:
            md += f"**Document Type:** {template.document_type}\n"
            md += f"**Complexity:** {template.complexity.value}\n"
            md += f"**Version:** {template.version}\n"
            md += f"**Created:** {template.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        md += "## Template Sections\n\n"
        
        for section in template.sections:
            md += f"### {section.name}\n"
            md += f"{section.description}\n\n"
            
            if section.placeholder:
                md += f"*Placeholder:* {section.placeholder}\n\n"
            
            if section.validation_rules:
                md += f"*Validation Rules:* {', '.join(section.validation_rules)}\n\n"
            
            if section.examples:
                md += f"*Examples:*\n"
                for example in section.examples:
                    md += f"- {example}\n"
                md += "\n"
        
        md += "## Style Configuration\n\n"
        md += f"- **Font:** {template.style.font_family}, {template.style.font_size}pt\n"
        md += f"- **Line Spacing:** {template.style.line_spacing}\n"
        md += f"- **Margins:** {template.style.margins}\n"
        md += f"- **Headers:** {'Yes' if template.style.headers else 'No'}\n"
        md += f"- **Page Numbers:** {'Yes' if template.style.page_numbers else 'No'}\n"
        
        return md
    
    def _export_json(self, template: DynamicTemplate, include_metadata: bool) -> str:
        """Export template as JSON"""
        data = {
            "name": template.name,
            "document_type": template.document_type,
            "complexity": template.complexity.value,
            "sections": [
                {
                    "name": section.name,
                    "description": section.description,
                    "required": section.required,
                    "repeatable": section.repeatable,
                    "order": section.order,
                    "content_type": section.content_type,
                    "placeholder": section.placeholder,
                    "validation_rules": section.validation_rules,
                    "examples": section.examples
                }
                for section in template.sections
            ],
            "style": {
                "font_family": template.style.font_family,
                "font_size": template.style.font_size,
                "line_spacing": template.style.line_spacing,
                "margins": template.style.margins,
                "colors": template.style.colors,
                "headers": template.style.headers,
                "page_numbers": template.style.page_numbers,
                "table_of_contents": template.style.table_of_contents
            }
        }
        
        if include_metadata:
            data["metadata"] = template.metadata
            data["version"] = template.version
            data["created_at"] = template.created_at.isoformat()
            data["tags"] = template.tags
        
        return json.dumps(data, indent=2)
    
    def _export_yaml(self, template: DynamicTemplate, include_metadata: bool) -> str:
        """Export template as YAML"""
        data = {
            "name": template.name,
            "document_type": template.document_type,
            "complexity": template.complexity.value,
            "sections": [
                {
                    "name": section.name,
                    "description": section.description,
                    "required": section.required,
                    "repeatable": section.repeatable,
                    "order": section.order,
                    "content_type": section.content_type,
                    "placeholder": section.placeholder,
                    "validation_rules": section.validation_rules,
                    "examples": section.examples
                }
                for section in template.sections
            ],
            "style": {
                "font_family": template.style.font_family,
                "font_size": template.style.font_size,
                "line_spacing": template.style.line_spacing,
                "margins": template.style.margins,
                "colors": template.style.colors,
                "headers": template.style.headers,
                "page_numbers": template.style.page_numbers,
                "table_of_contents": template.style.table_of_contents
            }
        }
        
        if include_metadata:
            data["metadata"] = template.metadata
            data["version"] = template.version
            data["created_at"] = template.created_at.isoformat()
            data["tags"] = template.tags
        
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    
    def _export_html(self, template: DynamicTemplate, include_metadata: bool) -> str:
        """Export template as HTML"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{template.name}</title>
    <style>
        body {{ font-family: {template.style.font_family}; font-size: {template.style.font_size}px; line-height: {template.style.line_spacing}; }}
        .header {{ color: {template.style.colors['primary']}; }}
        .section {{ margin: 20px 0; }}
        .metadata {{ background: #f5f5f5; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1 class="header">{template.name}</h1>
"""
        
        if include_metadata:
            html += f"""
    <div class="metadata">
        <p><strong>Document Type:</strong> {template.document_type}</p>
        <p><strong>Complexity:</strong> {template.complexity.value}</p>
        <p><strong>Version:</strong> {template.version}</p>
        <p><strong>Created:</strong> {template.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
        
        html += "    <h2>Template Sections</h2>\n"
        
        for section in template.sections:
            html += f"""
    <div class="section">
        <h3>{section.name}</h3>
        <p>{section.description}</p>
"""
            if section.placeholder:
                html += f"        <p><em>Placeholder:</em> {section.placeholder}</p>\n"
            
            if section.validation_rules:
                html += f"        <p><em>Validation Rules:</em> {', '.join(section.validation_rules)}</p>\n"
            
            html += "    </div>\n"
        
        html += """
</body>
</html>
"""
        return html
    
    def _export_plain_text(self, template: DynamicTemplate, include_metadata: bool) -> str:
        """Export template as plain text"""
        text = f"{template.name}\n"
        text += "=" * len(template.name) + "\n\n"
        
        if include_metadata:
            text += f"Document Type: {template.document_type}\n"
            text += f"Complexity: {template.complexity.value}\n"
            text += f"Version: {template.version}\n"
            text += f"Created: {template.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        text += "Template Sections:\n"
        text += "-" * 20 + "\n\n"
        
        for section in template.sections:
            text += f"{section.name}\n"
            text += f"  {section.description}\n"
            if section.placeholder:
                text += f"  Placeholder: {section.placeholder}\n"
            text += "\n"
        
        return text
    
    def save_template(self, template: DynamicTemplate, filename: Optional[str] = None) -> str:
        """Save template to file"""
        if not filename:
            safe_name = re.sub(r'[^\w\-_\.]', '_', template.name.lower())
            filename = f"{safe_name}_{template.document_type}.yaml"
        
        filepath = self.templates_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.export_template(template, DocumentFormat.YAML))
        
        return str(filepath)
    
    def load_template(self, filepath: str) -> DynamicTemplate:
        """Load template from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Reconstruct template object
        sections = [
            TemplateSection(**section_data) 
            for section_data in data['sections']
        ]
        
        style = TemplateStyle(**data['style'])
        
        return DynamicTemplate(
            name=data['name'],
            document_type=data['document_type'],
            complexity=TemplateComplexity(data['complexity']),
            sections=sections,
            style=style,
            metadata=data.get('metadata', {}),
            version=data.get('version', '1.0'),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            tags=data.get('tags', [])
        )

# Example usage
if __name__ == "__main__":
    generator = DynamicTemplateGenerator()
    
    # Generate a novel template
    novel_template = generator.generate_template(
        document_type="novel",
        complexity=TemplateComplexity.ADVANCED,
        style_preset="creative",
        genre="science_fiction",
        custom_requirements={
            "required_sections": ["Character Profiles", "World Building"],
            "excluded_sections": ["Glossary"]
        }
    )
    
    print("Generated Novel Template:")
    print(generator.export_template(novel_template, DocumentFormat.MARKDOWN))
    
    # Generate a contract template
    contract_template = generator.generate_template(
        document_type="contract",
        complexity=TemplateComplexity.PROFESSIONAL,
        style_preset="business",
        industry="Technology",
        custom_requirements={
            "section_order": ["Header", "Parties", "Service Description", "Terms", "Signatures"]
        }
    )
    
    print("\nGenerated Contract Template:")
    print(generator.export_template(contract_template, DocumentFormat.MARKDOWN))



























