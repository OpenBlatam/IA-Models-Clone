# truthgpt_enhanced.py - Improved TruthGPT with memory, logging, and self-optimization
import json
import time
import logging
from collections import namedtuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TruthGPT")

class Memory:
    def __init__(self):
        self.short_term = []
        self.long_term = {}  # key: topic, value: list of entries
    
    def add(self, content: str, topic: str = "general"):
        entry = {"content": content, "timestamp": time.time(), "utility": 1.0}
        self.short_term.append(entry)
        if topic not in self.long_term:
            self.long_term[topic] = []
        self.long_term[topic].append(entry)
        logger.info(f"Stored memory: {content[:50]}...")
    
    def recall(self, query: str, top_k: int = 3):
        # Simple keyword-based retrieval
        results = []
        for entry in self.short_term:
            if query.lower() in entry["content"].lower():
                results.append(entry)
        return results[:top_k]

class CodeInjector:
    def improve(self, original_code: str):
        # Adds error handling and logging
        improved = f"""import logging
logger = logging.getLogger(__name__)
def execute():
    try:
        {original_code}
        logger.info("Code executed successfully")
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise
"""
        return improved

class TruthGPT:
    def __init__(self):
        self.memory = Memory()
        self.injector = CodeInjector()
        self.actions_log = []
    
    def think(self, input_text: str):
        logger.info(f"Processing: {input_text[:50]}...")
        # Simulate reasoning
        self.memory.add(input_text, topic="user_input")
        self.actions_log.append({"action": "think", "input": input_text, "time": time.time()})
        return "I have processed your request."
    
    def improve_code(self, code: str):
        better = self.injector.improve(code)
        self.memory.add("Improved code injection pattern", topic="code")
        return better
    
    def get_stats(self):
        return {
            "memory_entries": len(self.memory.short_term),
            "actions": len(self.actions_log),
            "topics": list(self.memory.long_term.keys())
        }

# Example usage
if __name__ == "__main__":
    agent = TruthGPT()
    response = agent.think("Make TruthGPT not a simple script")
    print(response)
    improved = agent.improve_code("print('Hello World')")
    print("Improved code:\n", improved)
    print("Stats:", agent.get_stats())