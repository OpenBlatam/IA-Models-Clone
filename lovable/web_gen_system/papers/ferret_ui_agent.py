from typing import Dict, Any, List
from ..agents.base import BaseAgent

class FerretUIAgent(BaseAgent):
    """
    Agent for grounded mobile UI understanding.
    Inspired by 'Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs' (Apple, 2024).
    """

    def __init__(self, name: str = "FerretBot"):
        super().__init__(name, "Mobile UI Analyst")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Describes a UI element or screen region with high precision.
        """
        try:
            element_region = context.shared_memory.get("region")
            if not element_region:
                return {"status": "skipped", "reason": "No region provided"}

            self.log(f"Analyzing UI region: {element_region}")
            
            description = self.describe_ui_element(element_region)
            
            return {"status": "success", "description": description}
            
        except Exception as e:
            self.log(f"Error in FerretUIAgent: {str(e)}", level="error")
            return {"status": "error", "error": str(e)}

    def describe_ui_element(self, region: List[int]) -> str:
        """
        Simulates generating a detailed description for a UI region.
        """
        return "A red 'Submit' button with rounded corners, located at the bottom right."
