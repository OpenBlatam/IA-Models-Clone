"""
Exploring Utilization of Generative AI for Research and Education
==================================================================

Paper: "Exploring utilization of generative AI for research and education in..."

Key concepts:
- AI agents for research assistance
- Educational AI applications
- Research workflow automation
- Knowledge synthesis
- Learning support
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory


class ResearchTaskType(Enum):
    """Types of research tasks."""
    LITERATURE_REVIEW = "literature_review"
    DATA_ANALYSIS = "data_analysis"
    HYPOTHESIS_FORMULATION = "hypothesis_formulation"
    EXPERIMENT_DESIGN = "experiment_design"
    PAPER_WRITING = "paper_writing"
    CITATION_MANAGEMENT = "citation_management"


class EducationTaskType(Enum):
    """Types of education tasks."""
    EXPLANATION = "explanation"
    QUIZ_GENERATION = "quiz_generation"
    CONCEPT_CLARIFICATION = "concept_clarification"
    LEARNING_PATH = "learning_path"
    ASSESSMENT = "assessment"
    TUTORING = "tutoring"


@dataclass
class ResearchTask:
    """Research task representation."""
    task_id: str
    task_type: ResearchTaskType
    description: str
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class EducationTask:
    """Education task representation."""
    task_id: str
    task_type: EducationTaskType
    description: str
    difficulty_level: str = "medium"
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeSynthesis:
    """Knowledge synthesis result."""
    synthesis_id: str
    topic: str
    sources: List[str] = field(default_factory=list)
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class ResearchEducationAgent(BaseAgent):
    """
    Agent for research and education assistance.
    
    Supports research workflows, knowledge synthesis,
    and educational content generation.
    """
    
    def __init__(
        self,
        name: str,
        mode: str = "research",  # "research" or "education"
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize research/education agent.
        
        Args:
            name: Agent name
            mode: Operating mode ("research" or "education")
            config: Additional configuration
        """
        super().__init__(name, config)
        self.mode = mode
        
        # Task management
        self.research_tasks: List[ResearchTask] = []
        self.education_tasks: List[EducationTask] = []
        
        # Knowledge management
        self.knowledge_syntheses: List[KnowledgeSynthesis] = []
        self.citations: List[Dict[str, Any]] = []
        
        # Metrics
        self.research_tasks_completed = 0
        self.education_tasks_completed = 0
        self.knowledge_syntheses_created = 0
        
        # Memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Learning support
        self.learning_paths: List[Dict[str, Any]] = []
        self.explanations_generated = 0
        self.assessments_created = 0
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about task in research/education context.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        if self.mode == "research":
            analysis = self._analyze_research_task(task, context)
        else:
            analysis = self._analyze_education_task(task, context)
        
        result = {
            "task": task,
            "mode": self.mode,
            "analysis": analysis,
            "recommendations": self._generate_recommendations(task, analysis)
        }
        
        self.state.add_step("think", result)
        return result
    
    def _analyze_research_task(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze research task."""
        # Determine task type
        task_type = ResearchTaskType.LITERATURE_REVIEW
        
        if "analyze" in task.lower() or "data" in task.lower():
            task_type = ResearchTaskType.DATA_ANALYSIS
        elif "hypothesis" in task.lower() or "hypothesize" in task.lower():
            task_type = ResearchTaskType.HYPOTHESIS_FORMULATION
        elif "experiment" in task.lower() or "design" in task.lower():
            task_type = ResearchTaskType.EXPERIMENT_DESIGN
        elif "write" in task.lower() or "paper" in task.lower():
            task_type = ResearchTaskType.PAPER_WRITING
        
        return {
            "task_type": task_type.value,
            "complexity": "medium",
            "estimated_time": "2-4 hours",
            "requires_sources": True
        }
    
    def _analyze_education_task(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze education task."""
        # Determine task type
        task_type = EducationTaskType.EXPLANATION
        
        if "quiz" in task.lower() or "test" in task.lower():
            task_type = EducationTaskType.QUIZ_GENERATION
        elif "explain" in task.lower() or "clarify" in task.lower():
            task_type = EducationTaskType.CONCEPT_CLARIFICATION
        elif "learn" in task.lower() or "path" in task.lower():
            task_type = EducationTaskType.LEARNING_PATH
        elif "assess" in task.lower() or "evaluate" in task.lower():
            task_type = EducationTaskType.ASSESSMENT
        
        return {
            "task_type": task_type.value,
            "difficulty_level": context.get("difficulty", "medium") if context else "medium",
            "target_audience": context.get("audience", "general") if context else "general"
        }
    
    def _generate_recommendations(self, task: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for task."""
        recommendations = []
        
        if self.mode == "research":
            if analysis["task_type"] == ResearchTaskType.LITERATURE_REVIEW.value:
                recommendations.append("Search relevant academic databases")
                recommendations.append("Organize findings by theme")
                recommendations.append("Identify gaps in existing research")
            elif analysis["task_type"] == ResearchTaskType.DATA_ANALYSIS.value:
                recommendations.append("Prepare data cleaning steps")
                recommendations.append("Select appropriate analysis methods")
                recommendations.append("Plan visualization approach")
        else:
            if analysis["task_type"] == EducationTaskType.EXPLANATION.value:
                recommendations.append("Break down complex concepts")
                recommendations.append("Use examples and analogies")
                recommendations.append("Adapt to learner's level")
        
        return recommendations
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research/education action.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        action_type = action.get("type", "execute")
        
        if action_type == "synthesize_knowledge":
            result = self._synthesize_knowledge(action)
        elif action_type == "generate_explanation":
            result = self._generate_explanation(action)
        elif action_type == "create_learning_path":
            result = self._create_learning_path(action)
        else:
            result = self._execute_generic_action(action)
        
        self.state.add_step("act", {
            "action": action,
            "result": result
        })
        
        return result
    
    def _synthesize_knowledge(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize knowledge from multiple sources."""
        topic = action.get("topic", "unknown")
        sources = action.get("sources", [])
        
        synthesis = KnowledgeSynthesis(
            synthesis_id=f"synthesis_{datetime.now().timestamp()}",
            topic=topic,
            sources=sources,
            summary=f"Knowledge synthesis for {topic}",
            key_points=[
                f"Key point 1 about {topic}",
                f"Key point 2 about {topic}",
                f"Key point 3 about {topic}"
            ]
        )
        
        self.knowledge_syntheses.append(synthesis)
        self.knowledge_syntheses_created += 1
        
        return {
            "status": "completed",
            "synthesis_id": synthesis.synthesis_id,
            "topic": topic,
            "key_points": synthesis.key_points
        }
    
    def _generate_explanation(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Generate educational explanation."""
        concept = action.get("concept", "unknown")
        level = action.get("level", "intermediate")
        
        explanation = {
            "concept": concept,
            "level": level,
            "explanation": f"Explanation of {concept} at {level} level",
            "examples": [f"Example 1 for {concept}", f"Example 2 for {concept}"],
            "analogies": [f"Analogy for {concept}"]
        }
        
        self.explanations_generated += 1
        
        return {
            "status": "completed",
            "explanation": explanation
        }
    
    def _create_learning_path(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Create learning path."""
        topic = action.get("topic", "unknown")
        duration = action.get("duration", "4 weeks")
        
        learning_path = {
            "topic": topic,
            "duration": duration,
            "steps": [
                {"step": 1, "title": f"Introduction to {topic}", "duration": "1 week"},
                {"step": 2, "title": f"Core concepts of {topic}", "duration": "1 week"},
                {"step": 3, "title": f"Advanced {topic}", "duration": "1 week"},
                {"step": 4, "title": f"Practice and assessment", "duration": "1 week"}
            ]
        }
        
        self.learning_paths.append(learning_path)
        
        return {
            "status": "completed",
            "learning_path": learning_path
        }
    
    def _execute_generic_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic action."""
        return {
            "status": "executed",
            "action": action,
            "timestamp": datetime.now().isoformat()
        }
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update research/education state.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Update task status if applicable
        if isinstance(observation, dict):
            if observation.get("task_completed"):
                if self.mode == "research":
                    self.research_tasks_completed += 1
                else:
                    self.education_tasks_completed += 1
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.7,
            additional_data={
                "mode": self.mode,
                "research_tasks_completed": self.research_tasks_completed,
                "education_tasks_completed": self.education_tasks_completed
            }
        )
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run research/education task.
        
        Args:
            task: Task description
            context: Optional context
            
        Returns:
            Final result
        """
        from ..common.agent_utils import standard_run_pattern
        
        # Prepare context
        if context is None:
            context = {}
        
        context["mode"] = self.mode
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add mode-specific information
        result["mode_summary"] = {
            "mode": self.mode,
            "research_tasks_completed": self.research_tasks_completed,
            "education_tasks_completed": self.education_tasks_completed,
            "knowledge_syntheses": self.knowledge_syntheses_created,
            "explanations_generated": self.explanations_generated,
            "learning_paths_created": len(self.learning_paths)
        }
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "mode": self.mode,
            "research_tasks_completed": self.research_tasks_completed,
            "education_tasks_completed": self.education_tasks_completed,
            "knowledge_syntheses": self.knowledge_syntheses_created,
            "explanations_generated": self.explanations_generated,
            "learning_paths": len(self.learning_paths)
        })
    
    def add_research_task(self, task_type: ResearchTaskType, description: str) -> ResearchTask:
        """Add a new research task."""
        task = ResearchTask(
            task_id=f"task_{datetime.now().timestamp()}",
            task_type=task_type,
            description=description
        )
        self.research_tasks.append(task)
        return task
    
    def add_education_task(self, task_type: EducationTaskType, description: str) -> EducationTask:
        """Add a new education task."""
        task = EducationTask(
            task_id=f"task_{datetime.now().timestamp()}",
            task_type=task_type,
            description=description
        )
        self.education_tasks.append(task)
        return task


