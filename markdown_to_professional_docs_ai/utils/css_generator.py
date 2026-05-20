"""CSS generator for HTML documents"""
from typing import Dict, Any, Optional
from utils.templates import get_template_manager


class CSSGenerator:
    """Generate CSS styles for HTML documents"""
    
    def generate_css(
        self,
        template_name: str = "professional",
        custom_styles: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate CSS from template
        
        Args:
            template_name: Template name
            custom_styles: Custom style overrides
            
        Returns:
            CSS string
        """
        template_manager = get_template_manager()
        template = template_manager.get_template(template_name)
        
        if custom_styles:
            template = template_manager.merge_template(template_name, custom_styles)
        
        colors = template.get("colors", {})
        fonts = template.get("fonts", {})
        spacing = template.get("spacing", {})
        table_styles = template.get("table", {})
        
        css = f"""
/* Generated CSS for {template_name} template */

:root {{
    --primary-color: {colors.get('primary', '#366092')};
    --secondary-color: {colors.get('secondary', '#2c4a6b')};
    --accent-color: {colors.get('accent', '#4a90e2')};
    --text-color: {colors.get('text', '#1a1a1a')};
    --background-color: {colors.get('background', '#ffffff')};
    --header-bg: {colors.get('header_bg', colors.get('primary', '#366092'))};
    --header-text: {colors.get('header_text', '#ffffff')};
}}

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: {fonts.get('body', 'Calibri, sans-serif')};
    color: var(--text-color);
    background-color: var(--background-color);
    line-height: 1.6;
    padding: 20px;
    max-width: 1200px;
    margin: 0 auto;
}}

h1 {{
    font-family: {fonts.get('heading', 'Arial, sans-serif')};
    color: var(--primary-color);
    font-size: 2.5em;
    margin-bottom: {spacing.get('heading', 20)}px;
    border-bottom: 3px solid var(--primary-color);
    padding-bottom: 10px;
}}

h2 {{
    font-family: {fonts.get('heading', 'Arial, sans-serif')};
    color: var(--secondary-color);
    font-size: 2em;
    margin-top: {spacing.get('section', 30)}px;
    margin-bottom: {spacing.get('heading', 20)}px;
}}

h3 {{
    font-family: {fonts.get('heading', 'Arial, sans-serif')};
    color: var(--secondary-color);
    font-size: 1.5em;
    margin-top: {spacing.get('section', 30)}px;
    margin-bottom: {spacing.get('heading', 20)}px;
}}

p {{
    margin-bottom: {spacing.get('paragraph', 12)}px;
    text-align: justify;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}

th {{
    background-color: {table_styles.get('header_bg', colors.get('primary', '#366092'))};
    color: {table_styles.get('header_text', '#ffffff')};
    padding: 12px;
    text-align: left;
    font-weight: bold;
    border-bottom: 2px solid {table_styles.get('border', '#dddddd')};
}}

td {{
    padding: 10px;
    border-bottom: 1px solid {table_styles.get('border', '#dddddd')};
}}

tr:nth-child(even) {{
    background-color: {table_styles.get('row_alt', '#f2f2f2')};
}}

tr:hover {{
    background-color: rgba(54, 96, 146, 0.1);
}}

code {{
    font-family: {fonts.get('monospace', 'Courier New, monospace')};
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.9em;
}}

pre {{
    background-color: #f4f4f4;
    padding: 15px;
    border-radius: 5px;
    overflow-x: auto;
    border-left: 4px solid var(--primary-color);
}}

blockquote {{
    border-left: 4px solid var(--accent-color);
    padding-left: 20px;
    margin: 20px 0;
    font-style: italic;
    color: #666;
}}

a {{
    color: var(--accent-color);
    text-decoration: none;
}}

a:hover {{
    text-decoration: underline;
}}

ul, ol {{
    margin-left: 20px;
    margin-bottom: {spacing.get('paragraph', 12)}px;
}}

li {{
    margin-bottom: 5px;
}}

.chart-container {{
    margin: 30px 0;
    padding: 20px;
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}}

@media print {{
    body {{
        padding: 0;
    }}
    
    table {{
        page-break-inside: avoid;
    }}
    
    h1, h2, h3 {{
        page-break-after: avoid;
    }}
}}

@media (max-width: 768px) {{
    body {{
        padding: 10px;
    }}
    
    h1 {{
        font-size: 2em;
    }}
    
    h2 {{
        font-size: 1.5em;
    }}
    
    table {{
        font-size: 0.9em;
    }}
}}
"""
        return css.strip()


# Global CSS generator
_css_generator: Optional[CSSGenerator] = None


def get_css_generator() -> CSSGenerator:
    """Get global CSS generator"""
    global _css_generator
    if _css_generator is None:
        _css_generator = CSSGenerator()
    return _css_generator

