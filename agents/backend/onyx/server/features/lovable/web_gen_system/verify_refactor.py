import logging
import sys
import os

# Add the project root to the python path
# Add the parent of 'lovable' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lovable.web_gen_system.agents.base import BaseAgent
from lovable.web_gen_system.agents.roles import (
    ProductManagerAgent, ArchitectAgent, EngineerAgent, QAAgent
)
from lovable.web_gen_system.agents.visual_critic import VisualCriticAgent
from lovable.web_gen_system.schemas import AgentContext
from lovable.web_gen_system.papers.web_arena import WebArenaEvaluator
from lovable.web_gen_system.papers.safe_arena import SecurityScannerAgent
from lovable.web_gen_system.papers.beyond_browsing import HybridResearchAgent
from lovable.web_gen_system.papers.agentic_web import AgenticInterconnector
from lovable.web_gen_system.papers.enterprise_agent import EnterpriseTaskPlanner
from lovable.web_gen_system.papers.mobile_agent import MobileDeviceAgent
from lovable.web_gen_system.papers.os_world import OSWorldAgent
from lovable.web_gen_system.papers.visual_web_arena import MultimodalWebAgent
from lovable.web_gen_system.papers.web_voyager import WebVoyagerAgent
from lovable.web_gen_system.papers.browser_agent import PlaywrightTestGenerator
from lovable.web_gen_system.papers.digirl_agent import DigiRLAgent
from lovable.web_gen_system.papers.cog_agent import CogAgent
from lovable.web_gen_system.papers.seeclick_agent import SeeClickAgent
from lovable.web_gen_system.papers.mind2web_agent import Mind2WebAgent
from lovable.web_gen_system.papers.ferret_ui_agent import FerretUIAgent
from lovable.web_gen_system.papers.app_agent import AppAgent
from lovable.web_gen_system.papers.omni_parser_agent import OmniParserAgent

# Configure logging to show info level
logging.basicConfig(level=logging.INFO)

def verify_agents():
    print("--- Verifying Agents ---")
    
    agents = [
        ProductManagerAgent("Alice", "PM"),
        ArchitectAgent("Bob", "Architect"),
        EngineerAgent("Charlie"),
        QAAgent("Dave"),
        VisualCriticAgent("Eve"),
        WebArenaEvaluator("Olivia"),
        SecurityScannerAgent("Frank"),
        HybridResearchAgent("Heidi"),
        AgenticInterconnector("Ivan"),
        EnterpriseTaskPlanner("Judy"),
        MobileDeviceAgent("Mia"),
        OSWorldAgent("Noah"),
        MultimodalWebAgent("Kevin"),
        WebVoyagerAgent("Liam"),
        PlaywrightTestGenerator("Grace"),
        DigiRLAgent("Paul"),
        CogAgent("Quinn"),
        SeeClickAgent("Rachel"),
        Mind2WebAgent("Sam"),
        FerretUIAgent("Tina"),
        AppAgent("Uma"),
        OmniParserAgent("Victor")
    ]

    for agent in agents:
        print(f"Testing {agent.name} ({agent.__class__.__name__})...")
        try:
            # Run with empty or minimal context to check for crashes
            # Most agents handle missing keys gracefully now due to refactoring
            # Run with AgentContext
            context = AgentContext(
                task_id="test_task",
                shared_memory={
                    "prompt": "test", "goal": "test", "topic": "test", 
                    "action": "click", "command": "ls", "task": "test",
                    "screenshot_path": "test.png", "element_description": "button",
                    "region": [0,0,10,10], "app_name": "TestApp",
                    # For roles
                    "requirements": {"platform": "web"},
                    "architecture": {"structure_type": "nextjs"},
                    "code_structure": {"page.tsx": "<div></div>"} 
                }
            )
            result = agent.run(context)
            print(f"  Status: {result.get('status', 'unknown')}")
        except Exception as e:
            print(f"  FAILED: {e}")
            # raise e # Uncomment to stop on first error

    print("--- Verification Complete ---")

    print("\n--- Testing System Orchestration ---")
    from lovable.web_gen_system.pipeline import WebGenPipeline
    
    pipeline = WebGenPipeline()
    
    print("1. Testing Web Track...")
    try:
        pipeline.run({"prompt": "Create a modern landing page for a coffee shop", "use_agents": True})
        print("  Web Track: SUCCESS")
    except Exception as e:
        print(f"  Web Track FAILED: {e}")

    print("2. Testing Mobile Track...")
    try:
        pipeline.run({"prompt": "Automate a mobile app to buy coffee", "use_agents": True})
        print("  Mobile Track: SUCCESS")
    except Exception as e:
        print(f"  Mobile Track FAILED: {e}")

    print("\n--- Testing Advanced Architecture ---")
    print("1. Testing Event Bus...")
    try:
        pipeline.event_bus.publish("TEST_TOPIC", {"message": "Hello Event Bus"})
        print("  Event Bus: SUCCESS")
    except Exception as e:
        print(f"  Event Bus FAILED: {e}")

    print("2. Testing Shared Memory...")
    try:
        pipeline.memory.add_memory("User prefers dark mode", "System", ["preference"])
        memories = pipeline.memory.retrieve_relevant("dark mode")
        if memories and "dark mode" in memories[0]["content"]:
            print("  Shared Memory: SUCCESS")
        else:
            print("  Shared Memory: FAILED (Retrieval mismatch)")
    except Exception as e:
        print(f"  Shared Memory FAILED: {e}")

if __name__ == "__main__":
    verify_agents()
