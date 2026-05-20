"""HTML Converter - Convert Markdown to HTML format"""
from typing import Dict, Any, Optional
import json

from .base_converter import BaseConverter
from ...utils.chart_generator import ChartGenerator
from ...utils.css_generator import get_css_generator


class HTMLConverter(BaseConverter):
    """Convert Markdown to HTML format with interactive charts"""
    
    def __init__(self):
        self.chart_generator = ChartGenerator()
    
    async def convert(
        self,
        parsed_content: Dict[str, Any],
        output_path: str,
        include_charts: bool = True,
        include_tables: bool = True,
        custom_styling: Optional[Dict[str, Any]] = None
    ) -> None:
        """Convert to HTML format"""
        html_parts = []
        
        # HTML header with styles
        # Get CSS from template
        css_generator = get_css_generator()
        template_name = custom_styling.get("template", "professional") if custom_styling else "professional"
        css = css_generator.generate_css(template_name, custom_styling)
        
        html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>""")
        html_parts.append(parsed_content.get("title", "Document"))
        html_parts.append("""</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
""")
        html_parts.append(css)
        html_parts.append("""
    </style>
</head>
<body>
""")
        
        # Add title
        if parsed_content.get("title"):
            html_parts.append(f'<h1>{parsed_content["title"]}</h1>')
        
        # Add headings
        for heading in parsed_content.get("headings", []):
            level = heading["level"]
            html_parts.append(f'<h{level}>{heading["text"]}</h{level}>')
        
        # Add tables
        if include_tables:
            for table in parsed_content.get("tables", []):
                html_parts.append('<table>')
                # Headers
                html_parts.append('<thead><tr>')
                for header in table["headers"]:
                    html_parts.append(f'<th>{header}</th>')
                html_parts.append('</tr></thead>')
                # Rows
                html_parts.append('<tbody>')
                for row in table["rows"]:
                    html_parts.append('<tr>')
                    for cell in row:
                        html_parts.append(f'<td>{cell}</td>')
                    html_parts.append('</tr>')
                html_parts.append('</tbody></table>')
        
        # Add charts
        if include_charts and parsed_content.get("tables"):
            for idx, table in enumerate(parsed_content.get("tables", [])):
                try:
                    chart_data = self.chart_generator.create_plotly_chart_data(table)
                    if chart_data:
                        chart_id = f"chart_{idx}"
                        html_parts.append(f'<div class="chart-container" id="{chart_id}"></div>')
                        html_parts.append(f"""
<script>
    var data = {json.dumps(chart_data['data'])};
    var layout = {json.dumps(chart_data['layout'])};
    Plotly.newPlot('{chart_id}', data, layout);
</script>
""")
                except:
                    pass
        
        # Add paragraphs
        for para in parsed_content.get("paragraphs", []):
            html_parts.append(f'<p>{para}</p>')
        
        # Add lists
        for list_data in parsed_content.get("lists", []):
            tag = "ul" if list_data["type"] == "unordered" else "ol"
            html_parts.append(f'<{tag}>')
            for item in list_data["items"]:
                html_parts.append(f'<li>{item}</li>')
            html_parts.append(f'</{tag}>')
        
        # Close HTML
        html_parts.append("""
</body>
</html>
""")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(html_parts))

