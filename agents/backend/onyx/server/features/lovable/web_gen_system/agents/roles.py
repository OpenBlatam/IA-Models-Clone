from typing import Dict, Any, List, Optional
from .base import BaseAgent
from ..frameworks.nextjs import NextJSGenerator
from ..frameworks.expo import ExpoGenerator
from ..accessibility import WebAccessibilityEnhancer
from ..instruction import InstructionParser
from ..context import RepositoryContext

class ProductManagerAgent(BaseAgent):
    """
    Refines the user prompt into clear requirements using InstructionParser.
    """
    def __init__(self, name: str, role: str):
        super().__init__(name, role)
        self.parser = InstructionParser()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the user prompt and extracts requirements.
        """
        try:
            prompt = context.get("prompt", "")
            if not prompt:
                self.log("No prompt provided in context.", level="warning")
                return {"requirements": {}, "status": "failed", "error": "No prompt provided"}

            self.log(f"Analyzing request: '{prompt}'")
            
            # Use WaveCoder-style parsing
            parsed_instruction = self.parser.parse(prompt)
            
            requirements = {
                "original_prompt": prompt,
                "core_features": parsed_instruction.get("tasks", []),
                "constraints": parsed_instruction.get("constraints", []),
                "target_audience": "General",
                "tone": "Professional"
            }
            
            if "mobile" in prompt.lower() or "app" in prompt.lower():
                requirements["platform"] = "mobile"
            else:
                requirements["platform"] = "web"
                
            self.log(f"Requirements defined: {len(requirements['core_features'])} tasks identified.")
            return {"requirements": requirements, "status": "success"}
            
        except Exception as e:
            self.log(f"Error in ProductManagerAgent: {str(e)}", level="error")
            return {"status": "error", "error": str(e)}

class ArchitectAgent(BaseAgent):
    """
    Decides on the technical stack and file structure.
    """
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Designs the architecture based on requirements.
        """
        try:
            requirements = context.get("requirements", {})
            platform = requirements.get("platform", "web")
            
            self.log(f"Designing architecture for {platform}...")
            
            if platform == "mobile":
                stack = "Expo React Native"
                structure_type = "expo"
            else:
                stack = "Next.js"
                structure_type = "nextjs"
                
            architecture = {
                "stack": stack,
                "structure_type": structure_type,
                "components": ["Header", "Hero", "Footer"]
            }
            
            self.log(f"Architecture decided: {stack}")
            return {"architecture": architecture, "status": "success"}
            
        except Exception as e:
            self.log(f"Error in ArchitectAgent: {str(e)}", level="error")
            return {"status": "error", "error": str(e)}

class EngineerAgent(BaseAgent):
    """
    Generates the actual code based on architecture, using RepositoryContext.
    """
    def __init__(self, name: str):
        super().__init__(name, "Engineer")
        self.next_gen = NextJSGenerator()
        self.expo_gen = ExpoGenerator()
        self.repo_context = RepositoryContext()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates code structure and populates the repository context.
        """
        try:
            architecture = context.get("architecture", {})
            structure_type = architecture.get("structure_type", "nextjs")
            
            self.log(f"Generating code for {structure_type}...")
            
            code_structure: Dict[str, str] = {}
            if structure_type == "expo":
                code_structure = self.expo_gen.generate_project_structure("generated-app")
            else:
                code_structure = self.next_gen.generate_project_structure("generated-app")
                
            # Populate RepositoryContext (CodePlan)
            for path, content in code_structure.items():
                self.repo_context.add_file(path, content)
                
            # Check for consistency
            issues = self.repo_context.check_consistency()
            if issues:
                self.log(f"Warning: Found consistency issues: {issues}", level="warning")
            else:
                self.log("Repository consistency check passed.")
                
            self.log("Code generation complete.")
            return {"code_structure": code_structure, "status": "success"}
            
        except Exception as e:
            self.log(f"Error in EngineerAgent: {str(e)}", level="error")
            return {"status": "error", "error": str(e)}

class QAAgent(BaseAgent):
    """
    Reviews code and provides feedback (Critic Loop).
    """
    def __init__(self, name: str):
        super().__init__(name, "QA")
        self.accessibility = WebAccessibilityEnhancer()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reviews the generated code for issues.
        """
        try:
            code_structure = context.get("code_structure", {})
            self.log("Reviewing code...")
            
            issues: List[str] = []
            
            # Simple heuristic check for demo purposes
            # In a real system, this would parse AST or run linters
            for file_path, content in code_structure.items():
                if "page.tsx" in file_path or "App.tsx" in file_path:
                    # Check for empty text blocks
                    if "<Text></Text>" in content:
                        issues.append(f"Empty Text component in {file_path}")
                    
                    # Check for accessibility (using our enhancer logic on string content)
                    # This is a bit hacky for TSX but works for basic checks
                    if "img" in content and "alt=" not in content:
                         issues.append(f"Missing alt text in {file_path}")

            if issues:
                self.log(f"Found {len(issues)} issues: {', '.join(issues)}", level="warning")
                return {"status": "failed", "issues": issues}
            else:
                self.log("QA Passed. No critical issues found.")
                return {"status": "passed", "issues": []}
                
        except Exception as e:
            self.log(f"Error in QAAgent: {str(e)}", level="error")
            return {"status": "error", "error": str(e)}
