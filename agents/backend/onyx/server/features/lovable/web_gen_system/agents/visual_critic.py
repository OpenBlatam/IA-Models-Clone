from typing import Dict, Any, List
from .base import BaseAgent

class VisualCriticAgent(BaseAgent):
    """
    Simulates a Visual Language Model (VLM) to provide feedback on the
    visual appearance and layout of the generated code (WebGen-Agent concept).
    """
    def __init__(self, name: str = "VisualCritic", role: str = "Visual Critic"):
        super().__init__(name, role)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the visual layout and aesthetics of the generated code.
        """
        try:
            code_structure = context.shared_memory.get("code_structure", {})
            self.log("Analyzing visual layout and aesthetics...")
            
            feedback = self.analyze_layout(code_structure)
            
            if feedback:
                self.log(f"Visual feedback generated: {len(feedback)} items.", level="warning")
                return {"status": "critique", "visual_feedback": feedback}
            else:
                self.log("Visual check passed. Layout looks good.")
                return {"status": "passed", "visual_feedback": []}
                
        except Exception as e:
            self.log(f"Error in VisualCriticAgent: {str(e)}", level="error")
            return {"status": "error", "error": str(e)}

    def analyze_layout(self, code_structure: Dict[str, str]) -> List[str]:
        """
        Heuristic-based visual analysis. In a real system, this would use a VLM
        to analyze screenshots.
        """
        feedback = []
        
        for path, content in code_structure.items():
            if not (path.endswith(".tsx") or path.endswith(".jsx") or path.endswith(".html")):
                continue
                
            # Heuristic 1: Check for responsive design patterns (Tailwind)
            if "className" in content and "md:" not in content and "lg:" not in content:
                feedback.append(f"File {path} might lack responsive design (missing md:/lg: breakpoints).")
                
            # Heuristic 2: Check for image accessibility/styling
            if "<img" in content:
                if "className" not in content.split("<img")[1].split(">")[0]:
                    feedback.append(f"Image in {path} appears unstyled (missing className).")
                    
            # Heuristic 3: Check for empty containers which might look like visual bugs
            if "<div className=" in content and "></div>" in content:
                # Very naive check for empty divs
                feedback.append(f"Potential empty container found in {path}, might cause layout shifts.")
                
            # Heuristic 4: Check for hardcoded colors instead of theme tokens
            if "bg-[#]" in content or "text-[#]" in content:
                 feedback.append(f"Hardcoded hex colors found in {path}. Use theme tokens for visual consistency.")

        return feedback
