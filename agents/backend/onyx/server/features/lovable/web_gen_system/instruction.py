from typing import Dict, Any, List

class InstructionParser:
    """
    Parses complex instructions into structured sub-tasks (WaveCoder concept).
    """
    def parse(self, prompt: str) -> Dict[str, Any]:
        """
        Decomposes a complex prompt into actionable requirements.
        """
        parsed = {
            "original_prompt": prompt,
            "tasks": [],
            "constraints": []
        }
        
        # Heuristic parsing for demo purposes
        # Real implementation would use an LLM for this
        
        lower_prompt = prompt.lower()
        
        # Detect main tasks
        if "login" in lower_prompt:
            parsed["tasks"].append("Implement Login Page")
            parsed["tasks"].append("Implement Authentication Logic")
            
        if "blog" in lower_prompt:
            parsed["tasks"].append("Implement Post List")
            parsed["tasks"].append("Implement Post Detail View")
            parsed["tasks"].append("Implement Markdown Rendering")
            
        if "dashboard" in lower_prompt:
            parsed["tasks"].append("Implement Sidebar Navigation")
            parsed["tasks"].append("Implement Analytics Widgets")
            
        # Detect constraints
        if "mobile" in lower_prompt:
            parsed["constraints"].append("Mobile-First Design")
        if "dark mode" in lower_prompt:
            parsed["constraints"].append("Support Dark Mode")
        if "typescript" in lower_prompt:
            parsed["constraints"].append("Use TypeScript Strict Mode")
            
        # Default task if nothing specific found
        if not parsed["tasks"]:
            parsed["tasks"].append("Implement Main Landing Page")
            
        return parsed
