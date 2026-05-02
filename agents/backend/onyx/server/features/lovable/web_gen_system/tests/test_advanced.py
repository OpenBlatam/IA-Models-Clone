import unittest
from ..context import RepositoryContext
from ..instruction import InstructionParser
from ..agents import ProductManagerAgent, EngineerAgent

class TestAdvancedConcepts(unittest.TestCase):
    
    def test_repository_context(self):
        repo = RepositoryContext()
        repo.add_file("utils.ts", "export const add = (a, b) => a + b;")
        repo.add_file("main.ts", "import { add } from './utils';")
        
        self.assertTrue(repo.file_exists("utils.ts"))
        self.assertEqual(len(repo.check_consistency()), 0)
        
    def test_repository_context_missing_dependency(self):
        repo = RepositoryContext()
        repo.add_file("main.ts", "import { add } from './missing';")
        
        issues = repo.check_consistency()
        self.assertEqual(len(issues), 1)
        self.assertIn("missing", issues[0])

    def test_instruction_parser(self):
        parser = InstructionParser()
        parsed = parser.parse("Create a login page with dark mode")
        
        self.assertIn("Implement Login Page", parsed["tasks"])
        self.assertIn("Support Dark Mode", parsed["constraints"])

    def test_pm_agent_with_parser(self):
        agent = ProductManagerAgent("TestPM", "PM")
        output = agent.run({"prompt": "Create a blog"})
        
        self.assertIn("Implement Post List", output["requirements"]["core_features"])

    def test_engineer_agent_with_context(self):
        agent = EngineerAgent("TestEng")
        context = {
            "architecture": {"structure_type": "nextjs"},
            "prompt": "Test App"
        }
        output = agent.run(context)
        # Engineer agent logs consistency checks, we just verify it runs without error
        self.assertIsNotNone(output["code_structure"])

if __name__ == '__main__':
    unittest.main()
