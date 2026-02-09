import logging
from typing import Dict, Any
from .ai_base_service import AIBaseService

logger = logging.getLogger(__name__)

class UIGeneratorService(AIBaseService):
    """
    Service for generating UI components using Generative AI.
    Based on papers:
    - "Generative UI: LLMs are Effective UI Generators"
    - "Web Sculptor-Generative AI Based Comprehensive Web Development Framework"
    """

    def generate_component(self, description: str, style_guide: str = "Tailwind CSS") -> Dict[str, str]:
        """
        Generates HTML/CSS for a UI component based on a description.
        """
        prompt = (
            f"Generate HTML code for a UI component based on the following description.\n"
            f"Use {style_guide} for styling.\n"
            f"Description: {description}\n\n"
            f"HTML Code:"
        )

        try:
            result = self.generate_text(prompt, max_new_tokens=500)
            return {
                "description": description,
                "code": result
            }
        except Exception as e:
            logger.error(f"Error generating UI component: {e}")
            return {"error": str(e)}

    def generate_page_layout(self, page_description: str) -> Dict[str, str]:
        """
        Generates a basic HTML structure for a page.
        """
        prompt = (
            f"Generate a semantic HTML5 page layout based on the following description.\n"
            f"Include header, main, and footer sections.\n"
            f"Description: {page_description}\n\n"
            f"HTML Layout:"
        )

        try:
            result = self.generate_text(prompt, max_new_tokens=600)
            return {
                "description": page_description,
                "layout": result
            }
        except Exception as e:
            logger.error(f"Error generating page layout: {e}")
            return {"error": str(e)}
