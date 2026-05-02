import unittest
from unittest.mock import MagicMock, patch
from ..pipeline import WebGenPipeline
from ..agents.roles import (
    ProductManagerAgent, ArchitectAgent, EngineerAgent, QAAgent
)
from ..agents.visual_critic import VisualCriticAgent

class TestMultiAgentSOP(unittest.TestCase):
    def setUp(self):
        # Mock all agents
        self.mock_pm = MagicMock(spec=ProductManagerAgent)
        self.mock_arch = MagicMock(spec=ArchitectAgent)
        self.mock_eng = MagicMock(spec=EngineerAgent)
        self.mock_qa = MagicMock(spec=QAAgent)
        self.mock_visual = MagicMock(spec=VisualCriticAgent)
        
        # Setup default return values
        self.mock_pm.run.return_value = {"requirements": {"platform": "web"}}
        self.mock_arch.run.return_value = {"architecture": {"structure_type": "nextjs"}}
        self.mock_eng.run.return_value = {"code_structure": {"page.tsx": "<div>Hello</div>"}}
        self.mock_qa.run.return_value = {"status": "passed", "issues": []}
        self.mock_visual.run.return_value = {"status": "passed", "visual_feedback": []}
        
        # Mock the repo context on the engineer agent
        self.mock_eng.repo_context = MagicMock()
        
        self.pipeline = WebGenPipeline(
            pm_agent=self.mock_pm,
            architect_agent=self.mock_arch,
            engineer_agent=self.mock_eng,
            qa_agent=self.mock_qa,
            visual_critic=self.mock_visual
        )

    @patch('..pipeline.logger')
    def test_run_multi_agent_sop_success(self, mock_logger):
        """Test the happy path of the multi-agent SOP."""
        result = self.pipeline.run("Build a landing page", use_agents=True)
        
        # Verify agent calls
        self.mock_pm.run.assert_called_once()
        self.mock_arch.run.assert_called_once()
        self.mock_eng.run.assert_called_once()
        self.mock_qa.run.assert_called_once()
        self.mock_visual.run.assert_called_once()
        
        self.assertEqual(result, {"page.tsx": "<div>Hello</div>"})

    def test_run_multi_agent_sop_retry_loop(self):
        """Test that the loop retries on failure."""
        # First call fails, second succeeds
        self.mock_qa.run.side_effect = [
            {"status": "failed", "issues": ["Bug"]},
            {"status": "passed", "issues": []}
        ]
        
        self.pipeline.run("Build a landing page", use_agents=True)
        
        self.assertEqual(self.mock_qa.run.call_count, 2)

if __name__ == '__main__':
    unittest.main()
