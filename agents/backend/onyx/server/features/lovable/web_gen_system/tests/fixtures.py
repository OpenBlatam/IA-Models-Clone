from unittest.mock import MagicMock
from ..pipeline import WebGenPipeline

def create_mock_pipeline():
    """
    Creates a WebGenPipeline with all agents mocked.
    """
    return WebGenPipeline(
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
        benchmarker=MagicMock(),
        digirl_agent=MagicMock(),
        cog_agent=MagicMock(),
        seeclick_agent=MagicMock(),
        mind2web_agent=MagicMock(),
        ferret_agent=MagicMock(),
        app_agent=MagicMock(),
        omni_parser=MagicMock()
    )
