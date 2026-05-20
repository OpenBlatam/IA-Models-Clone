import unittest
from unittest.mock import patch
from ..fixtures import create_mock_pipeline

class TestSystemFlow(unittest.TestCase):
    def setUp(self):
        self.pipeline = create_mock_pipeline()

    def test_web_track_routing(self):
        # Use patch to mock the internal methods properly
        with patch.object(self.pipeline, '_run_web_track', return_value={"type": "web_project"}) as mock_web, \
             patch.object(self.pipeline, '_run_mobile_track', return_value={"type": "mobile_project"}) as mock_mobile:
            
            result = self.pipeline.run("Create a website for a bakery", use_agents=True)
            
            self.assertEqual(result.get("type"), "web_project")
            mock_web.assert_called_once()
            mock_mobile.assert_not_called()

    def test_mobile_track_routing(self):
        with patch.object(self.pipeline, '_run_web_track', return_value={"type": "web_project"}) as mock_web, \
             patch.object(self.pipeline, '_run_mobile_track', return_value={"type": "mobile_project"}) as mock_mobile:
            
            result = self.pipeline.run("Create a mobile app for fitness", use_agents=True)
            
            self.assertEqual(result.get("type"), "mobile_project")
            mock_mobile.assert_called_once()
            mock_web.assert_not_called()

if __name__ == "__main__":
    unittest.main()
