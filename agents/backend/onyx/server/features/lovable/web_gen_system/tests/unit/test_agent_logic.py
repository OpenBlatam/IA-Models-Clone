import unittest
from unittest.mock import MagicMock
from ...agents.base import BaseAgent
from ...papers.reflexion import ReflexionAgent

class TestAgentLogic(unittest.TestCase):
    def setUp(self):
        self.mock_bus = MagicMock()
        self.mock_memory = MagicMock()
        self.agent = BaseAgent("TestAgent", "Tester", self.mock_bus, self.mock_memory)

    def test_agent_publish_event(self):
        self.agent.publish_event("test_topic", {"status": "ok"})
        self.mock_bus.publish.assert_called_once()
        args = self.mock_bus.publish.call_args[0]
        self.assertEqual(args[0], "test_topic")
        self.assertEqual(args[1]["source"], "TestAgent")

    def test_agent_memory(self):
        self.agent.remember("Something important")
        self.mock_memory.add_memory.assert_called_once()
        
        self.agent.recall("query")
        self.mock_memory.retrieve_relevant.assert_called_once_with("query")

    def test_reflexion_agent(self):
        mock_inner = MagicMock()
        mock_inner.run.side_effect = [{"status": "failed", "error": "bug"}, {"status": "success"}]
        
        reflexion = ReflexionAgent("ReflexionWrapper", mock_inner)
        result = reflexion.run({"task": "do something"})
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(mock_inner.run.call_count, 2)

if __name__ == "__main__":
    unittest.main()
