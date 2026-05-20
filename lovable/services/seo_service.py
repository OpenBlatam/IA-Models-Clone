import logging
from typing import Dict, Any, List
from .ai_base_service import AIBaseService

logger = logging.getLogger(__name__)

class SEOService(AIBaseService):
    """
    Service for optimizing SEO using Generative AI.
    Based on papers:
    - "Empowering Web Editors with Generative AI: Creating a Tool for Efficient Search Engine Optimization"
    - "Generative AI in content SEO processes"
    """

    def generate_meta_tags(self, content: str) -> Dict[str, str]:
        """
        Generates meta title and description for the given content.
        """
        prompt = (
            f"Generate an SEO-optimized meta title (max 60 chars) and meta description (max 160 chars) for the following content:\n\n"
            f"Content:\n{content[:1000]}...\n\n"
            f"Output Format:\nTitle: ...\nDescription: ..."
        )

        try:
            result = self.generate_text(prompt, max_new_tokens=100)
            
            # Simple parsing
            lines = result.split('\n')
            title = ""
            description = ""
            
            for line in lines:
                if line.startswith("Title:"):
                    title = line.replace("Title:", "").strip()
                elif line.startswith("Description:"):
                    description = line.replace("Description:", "").strip()
            
            return {
                "title": title,
                "description": description
            }
        except Exception as e:
            logger.error(f"Error generating meta tags: {e}")
            return {"error": str(e)}

    def suggest_keywords(self, content: str) -> List[str]:
        """
        Suggests relevant keywords for the content.
        """
        prompt = (
            f"Suggest 5 relevant SEO keywords for the following content, separated by commas:\n\n"
            f"Content:\n{content[:1000]}...\n\n"
            f"Keywords:"
        )

        try:
            result = self.generate_text(prompt, max_new_tokens=50)
            keywords = [k.strip() for k in result.split(',')]
            return keywords
        except Exception as e:
            logger.error(f"Error suggesting keywords: {e}")
            return []
