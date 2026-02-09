"""
Prompt building for code explanations
"""

from typing import Optional, Dict


class PromptBuilder:
    """Builds prompts for code explanation"""
    
    DEFAULT_TEMPLATE = "Explain this code: {code}"
    
    TEMPLATES = {
        "brief": "Briefly explain this {language} code: {code}",
        "medium": "Explain this {language} code: {code}",
        "detailed": "Provide a detailed explanation of this {language} code, "
                   "including what each part does: {code}"
    }
    
    def __init__(self, default_template: str = DEFAULT_TEMPLATE):
        """Initialize prompt builder
        
        Args:
            default_template: Default prompt template
        """
        self.default_template = default_template
    
    def build(
        self,
        code: str,
        language: Optional[str] = None,
        detail_level: str = "medium"
    ) -> str:
        """
        Build prompt for code explanation.
        
        Args:
            code: Code to explain
            language: Programming language (optional)
            detail_level: Detail level ("brief", "medium", "detailed")
            
        Returns:
            Formatted prompt
            
        Raises:
            ValueError: If detail_level is invalid
        """
        if detail_level not in self.TEMPLATES:
            if detail_level != "medium":  # Allow default for medium
                raise ValueError(
                    f"detail_level must be one of {list(self.TEMPLATES.keys())}, "
                    f"got {detail_level}"
                )
            template = self.default_template
        else:
            language_str = language or "programming"
            template = self.TEMPLATES[detail_level].format(
                language=language_str,
                code="{code}"
            )
        
        return template.format(code=code)
    
    def build_batch(
        self,
        codes: list,
        language: Optional[str] = None,
        detail_level: str = "medium"
    ) -> list:
        """Build prompts for batch processing
        
        Args:
            codes: List of codes to explain
            language: Programming language (optional)
            detail_level: Detail level
            
        Returns:
            List of formatted prompts
        """
        return [self.build(code, language, detail_level) for code in codes]
    
    def get_max_length_for_detail(self, base_length: int, detail_level: str) -> int:
        """Get max length based on detail level
        
        Args:
            base_length: Base maximum length
            detail_level: Detail level
            
        Returns:
            Adjusted maximum length
        """
        max_length_map = {
            "brief": base_length // 2,
            "medium": base_length,
            "detailed": base_length * 2
        }
        return max_length_map.get(detail_level, base_length)

