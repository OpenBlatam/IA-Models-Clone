import unittest
from unittest.mock import MagicMock
from ..papers.safe_arena import SecurityScannerAgent
from ..papers.browser_agent import PlaywrightTestGenerator
from ..papers.beyond_browsing import HybridResearchAgent
from ..papers.agentic_web import AgenticInterconnector
from ..papers.enterprise_agent import EnterpriseTaskPlanner
from ..papers.visual_web_arena import MultimodalWebAgent
from ..papers.web_voyager import WebVoyagerAgent
from ..papers.mobile_agent import MobileDeviceAgent
from ..papers.os_world import OSWorldAgent
from ..papers.web_arena import WebArenaEvaluator
from ..context import RepositoryContext

class TestTop10Agents(unittest.TestCase):
    
    def setUp(self):
        self.repo_context = RepositoryContext()

    def test_security_scanner_pass(self):
        agent = SecurityScannerAgent()
        self.repo_context.add_file("safe.js", "console.log('Hello');")
        result = agent.run({"repository_context": self.repo_context})
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["issues"]), 0)

    def test_security_scanner_fail(self):
        agent = SecurityScannerAgent()
        self.repo_context.add_file("unsafe.js", "eval('2 + 2');")
        result = agent.run({"repository_context": self.repo_context})
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("eval" in issue for issue in result["issues"]))

    def test_playwright_generator(self):
        agent = PlaywrightTestGenerator()
        result = agent.run({"repository_context": self.repo_context})
        self.assertEqual(result["status"], "success")
        self.assertIn("tests/e2e.spec.ts", self.repo_context.get_all_files())
        content = self.repo_context.get_file("tests/e2e.spec.ts")
        self.assertIn("import { test, expect }", content)

    def test_hybrid_researcher_api(self):
        agent = HybridResearchAgent()
        result = agent.run({"topic": "React API docs"})
        self.assertEqual(result["method"], "API")
        self.assertEqual(result["status"], "success")

    def test_hybrid_researcher_browsing(self):
        agent = HybridResearchAgent()
        result = agent.run({"topic": "Latest web design trends"})
        self.assertEqual(result["method"], "BROWSING")
        self.assertEqual(result["status"], "success")

    def test_agentic_interconnector(self):
        agent = AgenticInterconnector()
        msg = agent.format_message("Sender", "Receiver", "Content")
        self.assertEqual(msg["header"]["sender"], "Sender")
        self.assertEqual(msg["header"]["protocol"], "A2A/1.0")
        self.assertEqual(agent.parse_message(msg), "Content")

    def test_enterprise_planner(self):
        agent = EnterpriseTaskPlanner()
        result = agent.run({"goal": "Build a complex enterprise CRM"})
        self.assertEqual(result["status"], "success")
        self.assertTrue(len(result["plan"]) > 0)
        self.assertEqual(result["plan"][0]["phase"], 1)

    def test_multimodal_agent(self):
        agent = MultimodalWebAgent()
        result = agent.run({"screenshot_path": "test.png"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["analysis"]["visual_hierarchy"], "good")

    def test_web_voyager(self):
        agent = WebVoyagerAgent()
        result = agent.run({"goal": "Buy a coffee"})
        self.assertEqual(result["status"], "success")
        self.assertIn("Go to Homepage", result["navigation_plan"])

    def test_mobile_agent(self):
        agent = MobileDeviceAgent()
        result = agent.run({"action": "click button"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["gesture"], "tap")

    def test_os_world_agent(self):
        agent = OSWorldAgent()
        result = agent.run({"command": "ls -la"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["exit_code"], 0)

    def test_web_arena_evaluator(self):
        agent = WebArenaEvaluator()
        result = agent.run({"task_id": "task1", "trajectory": ["step1", "step2"]})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["score"], 1.0)

if __name__ == "__main__":
    unittest.main()
