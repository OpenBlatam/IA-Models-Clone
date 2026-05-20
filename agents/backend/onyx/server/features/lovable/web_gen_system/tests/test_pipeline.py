import unittest
from unittest.mock import MagicMock
from ..pipeline import WebGenPipeline
from ..schemas import WebGenRequest

class TestWebGenPipeline(unittest.TestCase):
    def setUp(self):
        # Mock agents to avoid heavy initialization
        self.pipeline = WebGenPipeline(
            pm_agent=MagicMock(),
            architect_agent=MagicMock(),
            engineer_agent=MagicMock(),
            qa_agent=MagicMock(),
            visual_critic=MagicMock(),
            security_agent=MagicMock(),
            test_generator=MagicMock(),
            researcher=MagicMock(),
            interconnector=MagicMock(),
            planner=MagicMock(),
            multimodal_agent=MagicMock(),
            voyager_agent=MagicMock(),
            mobile_agent=MagicMock(),
            os_agent=MagicMock(),
            arena_evaluator=MagicMock(),
            benchmarker=MagicMock()
        )

    def test_run_pipeline_integration(self):
        # Test with Pydantic model
        req = WebGenRequest(prompt="coffee shop", style="modern")
        result = self.pipeline.run(req)
        self.assertIn("Coffee shop", result.content)
        
        # Test with dict (legacy support)
        result_dict = self.pipeline.run({"prompt": "coffee shop", "style": "modern"})
        self.assertIn("Coffee shop", result_dict.content)

if __name__ == '__main__':
    unittest.main()
