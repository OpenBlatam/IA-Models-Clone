import unittest
from ..agents import ProductManagerAgent, ArchitectAgent, EngineerAgent, QAAgent

class TestAgents(unittest.TestCase):
    def test_pm_agent(self):
        agent = ProductManagerAgent("TestPM", "PM")
        output = agent.run({"prompt": "Create a mobile app"})
        self.assertEqual(output["requirements"]["platform"], "mobile")

    def test_architect_agent(self):
        agent = ArchitectAgent("TestArch", "Architect")
        output = agent.run({"requirements": {"platform": "web"}})
        self.assertEqual(output["architecture"]["stack"], "Next.js")

    def test_engineer_agent(self):
        agent = EngineerAgent("TestEng")
        context = {
            "architecture": {"structure_type": "nextjs"},
            "prompt": "Test App"
        }
        output = agent.run(context)
        self.assertIn("generated-app/package.json", output["code_structure"])

    def test_qa_agent_pass(self):
        agent = QAAgent("TestQA")
        # Mock valid code
        context = {"code_structure": {"page.tsx": "<div><img alt='test' /></div>"}}
        output = agent.run(context)
        self.assertEqual(output["status"], "passed")

    def test_qa_agent_fail(self):
        agent = QAAgent("TestQA")
        # Mock invalid code (missing alt)
        context = {"code_structure": {"page.tsx": "<div><img src='test.jpg' /></div>"}}
        output = agent.run(context)
        self.assertEqual(output["status"], "failed")
        self.assertIn("Missing alt text", output["issues"][0])

if __name__ == '__main__':
    unittest.main()
