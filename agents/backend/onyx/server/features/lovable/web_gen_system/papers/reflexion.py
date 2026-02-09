from typing import Dict, Any, Callable
from ..agents.base import BaseAgent

class ReflexionAgent(BaseAgent):
    """
    A wrapper agent that implements the 'Reflexion' pattern (NeurIPS 2023).
    It wraps another agent's execution, evaluates the result, and triggers retries
    with verbal reinforcement if the result is unsatisfactory.
    """

    def __init__(self, name: str, wrapped_agent: BaseAgent):
        super().__init__(name, "Reflexion Wrapper")
        self.wrapped_agent = wrapped_agent

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the wrapped agent with self-correction logic.
        """
        max_retries = 3
        history = []
        
        for attempt in range(max_retries):
            self.log(f"Attempt {attempt + 1}/{max_retries} for {self.wrapped_agent.name}")
            
            # Add reflection history to context
            if history:
                context["reflection_history"] = history
                self.log(f"Providing reflection history: {history}")

            # Run wrapped agent
            result = self.wrapped_agent.run(context)
            
            # Simple self-evaluation (Simulated)
            # In a real system, this would use an Evaluator or LLM check
            if result.get("status") == "success":
                self.log("Result successful. No reflection needed.")
                return result
            
            # If failed, reflect
            error = result.get("error", "Unknown failure")
            reflection = f"Attempt {attempt+1} failed due to: {error}. I need to fix this."
            history.append(reflection)
            self.log(f"Reflecting: {reflection}")
            
        self.log("Max retries reached. Reflexion failed.", level="error")
        return {"status": "failed", "error": "Max retries exceeded", "history": history}
