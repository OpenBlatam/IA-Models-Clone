"""
Autonomous Agents Implementations
==================================

Implementations of autonomous agent frameworks based on research papers.

Available frameworks:
- ReAct: Reasoning + Acting
- Tree of Thoughts: Deliberate problem solving
- Generative Agents: Human-like behavior simulation
- LATS: Language Agent Tree Search
"""

from .react import ReActAgent
from .tree_of_thoughts import TreeOfThoughts, ToTNode, ToTSearchStrategy
from .generative_agents import GenerativeAgent, AgentProfile
from .lats import LATSAgent, LATSTree
from .toolformer import Toolformer, APICall, ToolformerTrainer
from .multi_agent_rl import MultiAgentRL, Agent, MARLEnvironment, MARLTrainer
from .theory_of_mind import TheoryOfMindAgent, AgentModel, BeliefState, IntentionTracker
from .llm_to_autonomous import LLMToAutonomousAgent, AutonomyLevel, ReasoningStep, ActionPlan, Goal
from .personality_driven import PersonalityDrivenAgent, PersonalityProfile, PersonalityTrait, EmotionalState, DecisionOption, Decision
from .mobile_agent import MobileAgent, ModalityType, ActionType, ScreenElement, ScreenState, MobileAction
from .autonomous_driving import AutonomousVehicleAgent, VehicleStatus, Position, VehicleState, TrafficMessage, TrafficEvent
from .multimodal_interactive import MultimodalInteractiveAgent, InteractionType, MultimodalInput, MultimodalOutput, Interaction
from .coordinated_paths import CoordinatedPathPlanner, Point, PathSegment, AgentPath, PathStatus, Conflict
from .ai_population_dynamics import PopulationDynamicsSystem, AgentRole, PopulationMetrics, AgentInteraction
from .self_initiated_learning import SelfInitiatedLearningAgent, LearningTrigger, LearningTask, LearningExperience
from .altruistic_planning import AltruisticPlanner, AltruismLevel, Maneuver, CooperativePlan
from .autonomous_driving_rl import AutonomousDrivingRL, DrivingAction, DrivingState, DrivingReward
from .model_free_motion import ModelFreeMotionPlanner, MotionType, MotionPlan, Obstacle
from .distributed_intersection import DistributedIntersectionManager, IntersectionState, ReservationStatus, IntersectionZone, ReservationRequest
from .ai_agents_evolution import AIAgentsEvolution, ArchitecturePattern, EvolutionOperator, AgentArchitecture, EvolutionGeneration
from .driving_olympics import DrivingOlympicsBenchmark, TrackType, ChallengeType, TrackMetrics, ChallengeResult
from .comfybench import ComfyBench, TaskCategory, EvaluationMetric, TaskResult, BenchmarkResult
from .generative_ai_machines import GenerativeAIMachineAgent, MachineType, OperationMode, MachineState, ActionPlan
from .sparks_agi import SparksAGIAgent, AGICapability, EmergenceLevel, AGITask, AGIPerformance
from .emergent_abilities import EmergentAbilitiesAnalyzer, EmergenceType, EvaluationMetric as EmergentEvaluationMetric, EmergenceObservation, ScalingCurve
from .resource_abstraction import ResourceAbstractionManager, ResourceType, AbstractionLevel, Resource, ResourceAllocation
from .language_perception import LanguagePerceptionAgent, ModalityType, AlignmentLevel, PerceptualInput, AlignedRepresentation
from .causal_explanations import CausalExplanationsAgent, CausalRelation, ExplanationType, CausalLink, CausalExplanation
from .emergent_cognitive_synergy import EmergentCognitiveSynergySystem, SynergyType, CognitiveCapability, CognitiveState, SynergisticInteraction
from .ultimate_brain import UltimateBrainAgent, BrainModule, CognitiveLevel, BrainState, CognitiveTask
from .llm_potential import LLMPotentialAnalyzer, CapabilityDomain, PotentialLevel, CapabilityAssessment, LLMPotential
from .agents_framework import AgentsFramework, FrameworkComponent, AgentType, FrameworkConfig, AgentInstance
from .legalbench import LegalBench, LegalTaskCategory, LegalDifficulty, LegalTask, LegalTaskResult
from .fully_autonomous_limitations import FullyAutonomousLimitationsAgent, RiskLevel, AutonomyConstraint, SafetyCheck, ActionLimitation
from .safe_honest_agents import SafeHonestAgent, HonestyLevel, SafetyStatus, TruthfulnessCheck, SafetyMeasure
from .web_agent_security import WebAgentSecurityAnalyzer, VulnerabilityType, ThreatLevel, Vulnerability, SecurityEvent
from .ethics_framework import EthicsFrameworkAgent, EthicalPrinciple, EthicalConcern, EthicalStatus, EthicalAssessment, BiasDetection
from .debias_me import DeBiasMeAgent, BiasType, DebiasingStrategy, BiasDetection as DeBiasDetection, DebiasingAction
from .morpheus_agent import MorpheusAgent, RoleType, DialogueContext, DialogueTurn, RoleProfile
from .research_education_agent import ResearchEducationAgent, ResearchTaskType, EducationTaskType, ResearchTask, EducationTask, KnowledgeSynthesis
from .seamless_multimodal import SeamlessMultimodalAgent, Modality, FusionStrategy, MultimodalInput, MultimodalOutput

__all__ = [
    # ReAct
    "ReActAgent",
    # Tree of Thoughts
    "TreeOfThoughts",
    "ToTNode",
    "ToTSearchStrategy",
    # Generative Agents
    "GenerativeAgent",
    "AgentProfile",
    # LATS
    "LATSAgent",
    "LATSTree",
    # Toolformer
    "Toolformer",
    "APICall",
    "ToolformerTrainer",
    # Multi-Agent RL
    "MultiAgentRL",
    "Agent",
    "MARLEnvironment",
    "MARLTrainer",
    # Theory of Mind
    "TheoryOfMindAgent",
    "AgentModel",
    "BeliefState",
    "IntentionTracker",
    # LLM to Autonomous
    "LLMToAutonomousAgent",
    "AutonomyLevel",
    "ReasoningStep",
    "ActionPlan",
    "Goal",
    # Personality-Driven
    "PersonalityDrivenAgent",
    "PersonalityProfile",
    "PersonalityTrait",
    "EmotionalState",
    "DecisionOption",
    "Decision",
    # Mobile Agent
    "MobileAgent",
    "ModalityType",
    "ActionType",
    "ScreenElement",
    "ScreenState",
    "MobileAction",
    # Autonomous Driving
    "AutonomousVehicleAgent",
    "VehicleStatus",
    "Position",
    "VehicleState",
    "TrafficMessage",
    "TrafficEvent",
    # Multimodal Interactive
    "MultimodalInteractiveAgent",
    "InteractionType",
    "MultimodalInput",
    "MultimodalOutput",
    "Interaction",
    # Coordinated Paths
    "CoordinatedPathPlanner",
    "Point",
    "PathSegment",
    "AgentPath",
    "PathStatus",
    "Conflict",
    # AI Population Dynamics
    "PopulationDynamicsSystem",
    "AgentRole",
    "PopulationMetrics",
    "AgentInteraction",
    # Self-Initiated Learning
    "SelfInitiatedLearningAgent",
    "LearningTrigger",
    "LearningTask",
    "LearningExperience",
    # Altruistic Planning
    "AltruisticPlanner",
    "AltruismLevel",
    "Maneuver",
    "CooperativePlan",
    # Autonomous Driving RL
    "AutonomousDrivingRL",
    "DrivingAction",
    "DrivingState",
    "DrivingReward",
    # Model-free Motion Planning
    "ModelFreeMotionPlanner",
    "MotionType",
    "MotionPlan",
    "Obstacle",
    # Distributed Intersection
    "DistributedIntersectionManager",
    "IntersectionState",
    "ReservationStatus",
    "IntersectionZone",
    "ReservationRequest",
    # AI Agents Evolution
    "AIAgentsEvolution",
    "ArchitecturePattern",
    "EvolutionOperator",
    "AgentArchitecture",
    "EvolutionGeneration",
    # Driving Olympics
    "DrivingOlympicsBenchmark",
    "TrackType",
    "ChallengeType",
    "TrackMetrics",
    "ChallengeResult",
    # ComfyBench
    "ComfyBench",
    "TaskCategory",
    "EvaluationMetric",
    "TaskResult",
    "BenchmarkResult",
    # Generative AI Machines
    "GenerativeAIMachineAgent",
    "MachineType",
    "OperationMode",
    "MachineState",
    "ActionPlan",
    # Sparks of AGI
    "SparksAGIAgent",
    "AGICapability",
    "EmergenceLevel",
    "AGITask",
    "AGIPerformance",
    # Emergent Abilities
    "EmergentAbilitiesAnalyzer",
    "EmergenceType",
    "EmergentEvaluationMetric",
    "EmergenceObservation",
    "ScalingCurve",
    # Resource Abstraction
    "ResourceAbstractionManager",
    "ResourceType",
    "AbstractionLevel",
    "Resource",
    "ResourceAllocation",
    # Language Perception
    "LanguagePerceptionAgent",
    "ModalityType",
    "AlignmentLevel",
    "PerceptualInput",
    "AlignedRepresentation",
    # Causal Explanations
    "CausalExplanationsAgent",
    "CausalRelation",
    "ExplanationType",
    "CausalLink",
    "CausalExplanation",
    # Emergent Cognitive Synergy
    "EmergentCognitiveSynergySystem",
    "SynergyType",
    "CognitiveCapability",
    "CognitiveState",
    "SynergisticInteraction",
    # Ultimate Brain
    "UltimateBrainAgent",
    "BrainModule",
    "CognitiveLevel",
    "BrainState",
    "CognitiveTask",
    # LLM Potential
    "LLMPotentialAnalyzer",
    "CapabilityDomain",
    "PotentialLevel",
    "CapabilityAssessment",
    "LLMPotential",
    # Agents Framework
    "AgentsFramework",
    "FrameworkComponent",
    "AgentType",
    "FrameworkConfig",
    "AgentInstance",
    # LegalBench
    "LegalBench",
    "LegalTaskCategory",
    "LegalDifficulty",
    "LegalTask",
    "LegalTaskResult",
    # Fully Autonomous Limitations
    "FullyAutonomousLimitationsAgent",
    "RiskLevel",
    "AutonomyConstraint",
    "SafetyCheck",
    "ActionLimitation",
    # Safe and Honest Agents
    "SafeHonestAgent",
    "HonestyLevel",
    "SafetyStatus",
    "TruthfulnessCheck",
    "SafetyMeasure",
    # Web Agent Security
    "WebAgentSecurityAnalyzer",
    "VulnerabilityType",
    "ThreatLevel",
    "Vulnerability",
    "SecurityEvent",
    # Ethics Framework
    "EthicsFrameworkAgent",
    "EthicalPrinciple",
    "EthicalConcern",
    "EthicalStatus",
    "EthicalAssessment",
    "BiasDetection",
    # DeBiasMe
    "DeBiasMeAgent",
    "BiasType",
    "DebiasingStrategy",
    "DeBiasDetection",
    "DebiasingAction",
    # MORPHEUS Agent
    "MorpheusAgent",
    "RoleType",
    "DialogueContext",
    "DialogueTurn",
    "RoleProfile",
    # Research and Education Agent
    "ResearchEducationAgent",
    "ResearchTaskType",
    "EducationTaskType",
    "ResearchTask",
    "EducationTask",
    "KnowledgeSynthesis",
    # Seamless Multimodal
    "SeamlessMultimodalAgent",
    "Modality",
    "FusionStrategy",
    "MultimodalInput",
    "MultimodalOutput",
    # GenIR
    "GenIRAgent",
    "GenIRTaskType",
    "RetrievalStrategy",
    "Query",
    "GeneratedDocument",
    "RetrievalResult",
    # Action Conventions
    "ActionConventionsAgent",
    "ConventionType",
    "ActionProtocol",
    "Convention",
    "ActionConvention",
    # Situation Coverage
    "SituationCoverageAgent",
    "SituationType",
    "CoverageMetric",
    "Situation",
    "CoverageResult",
    # Human Control
    "HumanControlAgent",
    "ControlLevel",
    "InterventionType",
    "ControlRequest",
    "HumanIntervention",
    # COLM 2025
    "COLM2025Agent",
    "CapabilityType",
    "SafetyLevel",
    "CapabilityAssessment",
    "SafetyEvaluation",
    # Human-Computer Interaction
    "HCIAgent",
    "InteractionModality",
    "FeedbackType",
    "Interaction",
    "UsabilityMetric",
    # Lewis Hammond
    "LewisHammondAgent",
    "AnalysisType",
    "MetricCategory",
    "AnalysisResult",
    "PerformanceMetric",
]

__version__ = "12.0.0"



