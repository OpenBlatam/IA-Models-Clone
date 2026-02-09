from typing import Dict, Optional
from jinja2 import Template
import structlog

logger = structlog.get_logger()

class DynamicUIGenerator:
    """
    Generates dynamic UI layouts and content.
    Reference: "Generative AI Tools in Web Design"
    """
    
    def generate_page(self, prompt: str, style: str = "modern") -> str:
        """
        Generates a basic HTML page based on the prompt and style.
        """
        logger.info("Generating page", prompt=prompt, style=style)
        
        # In a real system, this would call an LLM.
        # Here we use a simple template system for demonstration.
        
        title = prompt.capitalize()
        css_styles = self._get_style(style)
        
        template_str = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        {{ css_styles }}
    </style>
</head>
<body>
    <header>
        <nav>
            <div class="logo">{{ title }}</div>
            <ul>
                <li><a href="#home">Home</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
    </header>
    
    <main>
        <section id="hero">
            <h1>Welcome to {{ title }}</h1>
            <p>Generated based on: {{ prompt }}</p>
            <button class="cta-button">Get Started</button>
        </section>
        
        <section id="content">
            <h2>About Us</h2>
            <p>This is a dynamically generated section.</p>
            <img src="placeholder.jpg" alt="Placeholder image">
        </section>
    </main>
    
    <footer>
        <p>&copy; 2024 {{ title }}. All rights reserved.</p>
    </footer>
</body>
</html>"""

        template = Template(template_str)
        return template.render(title=title, css_styles=css_styles, prompt=prompt)

    def _get_style(self, style: str) -> str:
        if style == "modern":
            return """
                body { font-family: 'Inter', sans-serif; margin: 0; padding: 0; color: #333; }
                header { background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 1rem 2rem; }
                nav { display: flex; justify-content: space-between; align-items: center; }
                .logo { font-weight: bold; font-size: 1.5rem; }
                nav ul { display: flex; list-style: none; gap: 1rem; }
                nav a { text-decoration: none; color: #333; }
                #hero { background: #f0f4f8; padding: 4rem 2rem; text-align: center; }
                .cta-button { background: #007bff; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 4px; cursor: pointer; }
                #content { padding: 2rem; max-width: 800px; margin: 0 auto; }
                footer { background: #333; color: white; text-align: center; padding: 1rem; margin-top: 2rem; }
            """
        elif style == "minimal":
             return """
                body { font-family: 'Helvetica', sans-serif; margin: 0; padding: 0; color: #111; }
                header { padding: 2rem; }
                nav { display: flex; flex-direction: column; align-items: center; }
                .logo { font-size: 2rem; margin-bottom: 1rem; }
                nav ul { list-style: none; padding: 0; display: flex; gap: 2rem; }
                nav a { text-decoration: none; color: #111; text-transform: uppercase; letter-spacing: 1px; }
                #hero { padding: 6rem 2rem; text-align: center; }
                .cta-button { border: 1px solid #111; background: transparent; padding: 1rem 2rem; cursor: pointer; }
                #content { padding: 2rem; max-width: 600px; margin: 0 auto; }
                footer { border-top: 1px solid #eee; padding: 2rem; text-align: center; }
            """
        else:
            return ""

    def apply_design_system(self, html_content: str, design_system: Dict) -> str:
        """
        Applies a specific design system (colors, fonts) to existing HTML.
        """
        if not html_content:
            return ""
            
        # Basic implementation: Inject CSS variables based on design system
        # In a real system, this would parse CSS or use a CSS-in-JS generator
        
        primary_color = design_system.get("primary_color", "#007bff")
        font_family = design_system.get("font_family", "Inter, sans-serif")
        
        style_injection = f"""
        <style>
            :root {{
                --primary-color: {primary_color};
                --font-family: {font_family};
            }}
            body {{
                font-family: var(--font-family);
            }}
            .cta-button {{
                background-color: var(--primary-color) !important;
            }}
        </style>
        """
        
        # Inject before </head>
        if "</head>" in html_content:
            return html_content.replace("</head>", f"{style_injection}</head>")
        else:
            return html_content + style_injection
