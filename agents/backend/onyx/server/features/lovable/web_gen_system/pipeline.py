import structlog
from typing import Dict, Union, Any, Optional

from .generator import DynamicUIGenerator
from .accessibility import WebAccessibilityEnhancer
from .seo import SEOOptimizer
from .frameworks.nextjs import NextJSGenerator
from .frameworks.expo import ExpoGenerator
from .schemas import WebGenRequest, WebGenResponse, AgentContext
from .agents import (
    ProductManagerAgent, ArchitectAgent, EngineerAgent, QAAgent, VisualCriticAgent
)
from .papers import (
    SecurityScannerAgent, PlaywrightTestGenerator, HybridResearchAgent,
    AgenticInterconnector, EnterpriseTaskPlanner,
    MultimodalWebAgent, WebVoyagerAgent, MobileDeviceAgent,
    OSWorldAgent, WebArenaEvaluator,
    DigiRLAgent, CogAgent, SeeClickAgent, Mind2WebAgent,
    FerretUIAgent, AppAgent, OmniParserAgent
)
from .evaluation.benchmarker import WebGenBench
from .context import RepositoryContext
from ..constants import (
    SIMULATED_SCREENSHOT_PATH,
    TASK_TYPE_WEB,
    TASK_TYPE_MOBILE
)

logger = structlog.get_logger()

from .core.event_bus import EventBus
from .core.memory import SharedMemory
from .papers.reflexion import ReflexionAgent

class WebGenPipeline:
    """
    Orchestrates the web generation process using a Multi-Agent System.
    """
    
    def __init__(
        self,
        pm_agent: Optional[ProductManagerAgent] = None,
        architect_agent: Optional[ArchitectAgent] = None,
        engineer_agent: Optional[EngineerAgent] = None,
        qa_agent: Optional[QAAgent] = None,
        visual_critic: Optional[VisualCriticAgent] = None,
        security_agent: Optional[SecurityScannerAgent] = None,
        test_generator: Optional[PlaywrightTestGenerator] = None,
        researcher: Optional[HybridResearchAgent] = None,
        interconnector: Optional[AgenticInterconnector] = None,
        planner: Optional[EnterpriseTaskPlanner] = None,
        multimodal_agent: Optional[MultimodalWebAgent] = None,
        voyager_agent: Optional[WebVoyagerAgent] = None,
        mobile_agent: Optional[MobileDeviceAgent] = None,
        os_agent: Optional[OSWorldAgent] = None,
        arena_evaluator: Optional[WebArenaEvaluator] = None,
        benchmarker: Optional[WebGenBench] = None,
        # New Agents
        digirl_agent: Optional[DigiRLAgent] = None,
        cog_agent: Optional[CogAgent] = None,
        seeclick_agent: Optional[SeeClickAgent] = None,
        mind2web_agent: Optional[Mind2WebAgent] = None,
        ferret_agent: Optional[FerretUIAgent] = None,
        app_agent: Optional[AppAgent] = None,
        omni_parser: Optional[OmniParserAgent] = None
    ):
        # Core Architecture Components
        self.event_bus = EventBus()
        self.memory = SharedMemory()

        # Legacy components
        self.generator = DynamicUIGenerator()
        self.accessibility = WebAccessibilityEnhancer()
        self.seo = SEOOptimizer()
        self.nextjs_generator = NextJSGenerator()
        self.expo_generator = ExpoGenerator()
        
        # Helper to inject core components
        def inject_core(agent):
            if agent:
                agent.event_bus = self.event_bus
                agent.memory = self.memory
            return agent

        # Agents (Dependency Injection with defaults)
        self.pm_agent = inject_core(pm_agent or ProductManagerAgent("Alice", "PM"))
        self.architect_agent = inject_core(architect_agent or ArchitectAgent("Bob", "Architect"))
        self.engineer_agent = inject_core(engineer_agent or EngineerAgent("Charlie"))
        self.qa_agent = inject_core(qa_agent or QAAgent("Dave"))
        self.visual_critic = inject_core(visual_critic or VisualCriticAgent("Eve"))
        self.security_agent = inject_core(security_agent or SecurityScannerAgent("Frank"))
        self.test_generator = inject_core(test_generator or PlaywrightTestGenerator("Grace"))
        self.researcher = inject_core(researcher or HybridResearchAgent("Heidi"))
        self.interconnector = inject_core(interconnector or AgenticInterconnector("Ivan"))
        self.planner = inject_core(planner or EnterpriseTaskPlanner("Judy"))
        self.multimodal_agent = inject_core(multimodal_agent or MultimodalWebAgent("Kevin"))
        self.voyager_agent = inject_core(voyager_agent or WebVoyagerAgent("Liam"))
        self.mobile_agent = inject_core(mobile_agent or MobileDeviceAgent("Mia"))
        self.os_agent = inject_core(os_agent or OSWorldAgent("Noah"))
        self.arena_evaluator = inject_core(arena_evaluator or WebArenaEvaluator("Olivia"))
        
        # New Top 10 Agents
        self.digirl_agent = inject_core(digirl_agent or DigiRLAgent("Paul"))
        self.cog_agent = inject_core(cog_agent or CogAgent("Quinn"))
        self.seeclick_agent = inject_core(seeclick_agent or SeeClickAgent("Rachel"))
        self.mind2web_agent = inject_core(mind2web_agent or Mind2WebAgent("Sam"))
        self.ferret_agent = inject_core(ferret_agent or FerretUIAgent("Tina"))
        self.app_agent = inject_core(app_agent or AppAgent("Uma"))
        self.omni_parser = inject_core(omni_parser or OmniParserAgent("Victor"))
        
        # Evaluation
        self.benchmarker = benchmarker or WebGenBench()

    def run(self, request: Union[WebGenRequest, Dict[str, Any]]) -> Union[str, Dict[str, str], WebGenResponse]:
        """
        Runs the generation pipeline.
        Accepts either a WebGenRequest object or a dictionary (for backward compatibility).
        """
        # Validate input
        if isinstance(request, dict):
            # Handle legacy signature: prompt, style, etc.
            # If 'prompt' is not in dict, assume it's kwargs from legacy call
            if 'prompt' not in request:
                 # This handles the case where run is called with positional args mapped to kwargs
                 # But since we changed signature, we need to be careful.
                 # For now, let's assume the caller adapts or we support legacy kwargs if needed.
                 pass
            req = WebGenRequest(**request)
        else:
            req = request

        logger.info("Running pipeline", target=req.target, use_agents=req.use_agents)
        
        if req.use_agents:
            return self._run_system_orchestration(req)
            
        # Legacy/Direct Mode
        if req.target == "nextjs":
            if "project" in req.prompt.lower() or "app" in req.prompt.lower():
                return self.nextjs_generator.generate_project_structure("my-next-app")
            else:
                return self.nextjs_generator.generate_page(req.prompt, "/")
                
        elif req.target == "expo":
            if "project" in req.prompt.lower() or "app" in req.prompt.lower():
                return self.expo_generator.generate_project_structure("my-expo-app")
            else:
                return self.expo_generator.generate_screen(req.prompt, "GeneratedScreen")
        
        else: # Default to HTML
            # 1. Generate
            html = self.generator.generate_page(req.prompt, req.style)
            
            # 2. SEO Optimization
            if req.optimize_seo:
                title = req.prompt.capitalize()
                description = f"A generated page about {req.prompt}"
                keywords = req.prompt.split()
                html = self.seo.optimize_meta_tags(html, title, description, keywords)
                
            # 3. Accessibility Enhancement
            issues = []
            if req.fix_accessibility:
                html = self.accessibility.fix_accessibility(html)
                issues = self.accessibility.analyze_html(html)
                
            return WebGenResponse(content=html, issues=issues)

    def _run_system_orchestration(self, req: WebGenRequest) -> Dict[str, Any]:
        """
        Executes the Dynamic Agent System Orchestration.
        Routes tasks to specialized teams (Web vs Mobile) and uses shared visual services.
        """
        logger.info("Starting Dynamic System Orchestration")
        
        # 1. Intent Analysis & Routing
        intent = self._classify_intent(req.prompt)
        logger.info("Detected Intent", intent=intent)
        
        # Context for agents
        context = AgentContext(task_id="task_001", shared_memory={"topic": req.prompt})
        
        # 2. Shared Research Phase
        # Note: Agents need to be updated to accept AgentContext
        # For now, we adapt manually or update agents
        research_output = self.researcher.run(context) 
        
        # 3. Execution Track
        if intent == TASK_TYPE_MOBILE:
            return self._run_mobile_track(req.prompt, research_output)
        else:
            return self._run_web_track(req.prompt, research_output)

    def _classify_intent(self, prompt: str) -> str:
        """
        Uses ProductManager to classify the user intent.
        """
        # Simple heuristic for now, could be an LLM call
        if "mobile" in prompt.lower() or "app" in prompt.lower() or "android" in prompt.lower() or "ios" in prompt.lower():
            return TASK_TYPE_MOBILE
        return TASK_TYPE_WEB

    def _run_web_track(self, prompt: str, research: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates the Web Team: Voyager -> Mind2Web -> Engineer -> Evaluator.
        """
        logger.info("Activating Web Team")
        
        context = AgentContext(task_id="web_track", shared_memory={"goal": prompt, "research": research})
        
        # Planning (WebVoyager)
        plan = self.voyager_agent.run(context)
        logger.info("Web Plan Generated", steps=len(plan.get('navigation_plan', [])))
        
        # Execution (Mind2Web + Engineer)
        m2w_output = self.mind2web_agent.run(context)
        
        # Engineer builds the actual artifact
        # Update context
        context.shared_memory.update({"plan": plan, "interaction_logic": m2w_output})
        eng_output = self.engineer_agent.run(context)
        
        # Visual Analysis (Shared Service: OmniParser + CogAgent)
        screenshot_path = SIMULATED_SCREENSHOT_PATH
        context.shared_memory["screenshot_path"] = screenshot_path
        
        omni_output = self.omni_parser.run(context)
        cog_output = self.cog_agent.run(context)
        
        logger.info("Visual Analysis", description=cog_output.get('analysis', {}).get('layout_description', 'N/A'))
        
        # Evaluation (WebArena)
        context.shared_memory["trajectory"] = plan.get("navigation_plan", [])
        arena_score = self.arena_evaluator.run(context)
        logger.info("WebArena Score", score=arena_score.get('score'))
        
        return eng_output.get("code_structure", {})

    def _run_mobile_track(self, prompt: str, research: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates the Mobile Team: AppAgent -> FerretUI -> MobileDevice -> DigiRL.
        """
        logger.info("Activating Mobile Team")
        
        context = AgentContext(task_id="mobile_track", shared_memory={"app_name": "TargetApp", "goal": prompt})
        
        # Planning (AppAgent)
        app_plan = self.app_agent.run(context)
        
        # Perception (FerretUI)
        context.shared_memory["region"] = [0, 0, 100, 100]
        ferret_output = self.ferret_agent.run(context)
        
        # Action (MobileDevice + DigiRL)
        digirl_output = self.digirl_agent.run(context)
        
        context.shared_memory["action"] = digirl_output.get("action", {}).get("type", "tap")
        mobile_output = self.mobile_agent.run(context)
        
        logger.info("Mobile Action Result", status=mobile_output.get('status'))
        
        return {"mobile_app": "generated_structure"}
