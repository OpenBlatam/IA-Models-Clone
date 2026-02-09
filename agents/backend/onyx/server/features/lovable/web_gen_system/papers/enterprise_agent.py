from typing import List, Dict, Any
from ..agents.base import BaseAgent

class EnterpriseTaskPlanner(BaseAgent):
    """
    Agent responsible for long-horizon task planning and complex workflow management.
    Inspired by 'TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks'.
    """

    def __init__(self, name: str = "EnterprisePlanner"):
        super().__init__(name, "Strategic Planner")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a long-term plan based on the high-level goal.
        """
        try:
            goal = context.get("goal") or context.get("prompt")
            if not goal:
                 return {"status": "skipped", "plan": []}

            self.log(f"Creating enterprise plan for: '{goal}'")
            
            plan = self.create_long_term_plan(goal)
            return {"status": "success", "plan": plan}
            
        except Exception as e:
            self.log(f"Error in EnterpriseTaskPlanner: {str(e)}", level="error")
            return {"status": "error", "error": str(e)}

    def create_long_term_plan(self, goal: str) -> List[Dict[str, Any]]:
        """
        Decomposes a goal into a multi-phase enterprise plan.
        """
        # Heuristic planning logic
        plan = [
            {"phase": 1, "name": "Research & Requirements", "agent": "ProductManager", "status": "pending"},
            {"phase": 2, "name": "System Architecture", "agent": "Architect", "status": "pending"},
            {"phase": 3, "name": "Implementation", "agent": "Engineer", "status": "pending"},
            {"phase": 4, "name": "Quality Assurance", "agent": "QA", "status": "pending"},
            {"phase": 5, "name": "Security Audit", "agent": "SecurityScanner", "status": "pending"},
            {"phase": 6, "name": "Deployment", "agent": "DevOps", "status": "pending"}
        ]
        
        self.log(f"Generated {len(plan)}-phase plan.")
        return plan
