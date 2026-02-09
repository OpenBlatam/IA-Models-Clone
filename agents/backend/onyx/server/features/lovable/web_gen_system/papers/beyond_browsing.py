from typing import List, Dict, Any
from ..agents.base import BaseAgent

class HybridResearchAgent(BaseAgent):
    """
    Agent responsible for researching topics using a hybrid approach (API vs Browsing).
    Inspired by 'Beyond Browsing: API-Based Web Agents' paper.
    """

    def __init__(self, name: str = "HybridResearcher"):
        super().__init__(name, "Research Scientist")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conducts research on a given topic, deciding between API and Browsing.
        """
        try:
            topic = context.get("topic")
            if not topic:
                return {"status": "skipped", "reason": "No topic provided"}

            self.log(f"Researching topic: '{topic}'")
            
            # Decision logic (Simulated)
            method = self._decide_method(topic)
            self.log(f"Selected method: {method}")

            results = {}
            if method == "API":
                results = self._research_via_api(topic)
            else:
                results = self._research_via_browsing(topic)

            return {"status": "success", "method": method, "results": results}
            
        except Exception as e:
            self.log(f"Error in HybridResearchAgent: {str(e)}", level="error")
            return {"status": "error", "error": str(e)}

    def _decide_method(self, topic: str) -> str:
        """
        Decides whether to use API or Browsing based on the topic.
        Heuristic: Technical/Fact-based -> API, News/Trends -> Browsing.
        """
        api_keywords = ["docs", "api", "reference", "syntax", "library"]
        if any(k in topic.lower() for k in api_keywords):
            return "API"
        return "BROWSING"

    def _research_via_api(self, topic: str) -> Dict[str, str]:
        """
        Simulates research using an API (e.g., StackOverflow, GitHub).
        """
        self.log(f"Querying APIs for {topic}...")
        return {
            "source": "Simulated API",
            "summary": f"API results for {topic}: Found 5 relevant endpoints."
        }

    def _research_via_browsing(self, topic: str) -> Dict[str, str]:
        """
        Simulates research using a browser.
        """
        self.log(f"Browsing web for {topic}...")
        return {
            "source": "Simulated Browser",
            "summary": f"Browser results for {topic}: Found 3 relevant articles."
        }
