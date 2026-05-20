"""
LEGALBENCH: A COLLABORATIVELY BUILT BENCHMARK FOR
=================================================

Paper: "LEGALBENCH: A COLLABORATIVELY BUILT BENCHMARK FOR"

Key concepts:
- Legal domain benchmarking
- Legal reasoning tasks
- Collaborative benchmark construction
- Legal AI evaluation
- Task categories
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus


class LegalTaskCategory(Enum):
    """Legal task categories."""
    CONTRACT_ANALYSIS = "contract_analysis"
    CASE_LAW = "case_law"
    STATUTORY_INTERPRETATION = "statutory_interpretation"
    LEGAL_RESEARCH = "legal_research"
    DOCUMENT_REVIEW = "document_review"
    COMPLIANCE = "compliance"
    LITIGATION = "litigation"


class LegalDifficulty(Enum):
    """Legal task difficulty."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class LegalTask:
    """A legal task."""
    task_id: str
    category: LegalTaskCategory
    difficulty: LegalDifficulty
    description: str
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LegalTaskResult:
    """Result of a legal task."""
    task_id: str
    agent_name: str
    output: str
    accuracy: float
    completeness: float
    legal_correctness: float
    overall_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate overall score."""
        self.overall_score = (
            self.accuracy * 0.4 +
            self.completeness * 0.3 +
            self.legal_correctness * 0.3
        )


class LegalBench:
    """
    LEGALBENCH benchmark system for legal AI.
    
    Evaluates agents on legal reasoning tasks.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LegalBench.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.tasks: Dict[str, LegalTask] = {}
        self.results: List[LegalTaskResult] = []
        
        # Initialize tasks
        self._initialize_tasks()
    
    def _initialize_tasks(self):
        """Initialize legal benchmark tasks."""
        task_templates = [
            {
                "category": LegalTaskCategory.CONTRACT_ANALYSIS,
                "difficulty": LegalDifficulty.MEDIUM,
                "description": "Analyze contract terms and identify key obligations"
            },
            {
                "category": LegalTaskCategory.CASE_LAW,
                "difficulty": LegalDifficulty.HARD,
                "description": "Find relevant case law for a given legal issue"
            },
            {
                "category": LegalTaskCategory.STATUTORY_INTERPRETATION,
                "difficulty": LegalDifficulty.HARD,
                "description": "Interpret statute and apply to facts"
            },
            {
                "category": LegalTaskCategory.LEGAL_RESEARCH,
                "difficulty": LegalDifficulty.MEDIUM,
                "description": "Research legal precedents and authorities"
            },
            {
                "category": LegalTaskCategory.DOCUMENT_REVIEW,
                "difficulty": LegalDifficulty.EASY,
                "description": "Review legal document for key information"
            },
            {
                "category": LegalTaskCategory.COMPLIANCE,
                "difficulty": LegalDifficulty.MEDIUM,
                "description": "Assess compliance with regulations"
            },
            {
                "category": LegalTaskCategory.LITIGATION,
                "difficulty": LegalDifficulty.EXPERT,
                "description": "Analyze litigation strategy and risks"
            }
        ]
        
        for i, template in enumerate(task_templates):
            task = LegalTask(
                task_id=f"legal_task_{i+1}",
                category=template["category"],
                difficulty=template["difficulty"],
                description=template["description"]
            )
            self.tasks[task.task_id] = task
    
    def evaluate_agent(
        self,
        agent: BaseAgent,
        task_id: str,
        evaluation_data: Optional[Dict[str, Any]] = None
    ) -> LegalTaskResult:
        """
        Evaluate an agent on a legal task.
        
        Args:
            agent: Agent to evaluate
            task_id: Task identifier
            evaluation_data: Optional evaluation data
            
        Returns:
            Task result
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        # Execute task
        result = agent.run(task.description)
        
        # Evaluate output
        output = str(result.get("action", {}).get("status", ""))
        
        # Calculate scores (simplified)
        accuracy = evaluation_data.get("accuracy", 0.7) if evaluation_data else 0.7
        completeness = evaluation_data.get("completeness", 0.6) if evaluation_data else 0.6
        legal_correctness = evaluation_data.get("legal_correctness", 0.65) if evaluation_data else 0.65
        
        task_result = LegalTaskResult(
            task_id=task_id,
            agent_name=agent.name,
            output=output,
            accuracy=accuracy,
            completeness=completeness,
            legal_correctness=legal_correctness
        )
        
        self.results.append(task_result)
        return task_result
    
    def run_benchmark(
        self,
        agent: BaseAgent,
        task_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run full benchmark on an agent.
        
        Args:
            agent: Agent to benchmark
            task_ids: Optional list of task IDs (default: all)
            
        Returns:
            Benchmark results
        """
        if task_ids is None:
            task_ids = list(self.tasks.keys())
        
        task_results = []
        
        for task_id in task_ids:
            result = self.evaluate_agent(agent, task_id)
            task_results.append(result)
        
        # Calculate aggregate scores
        if task_results:
            avg_accuracy = sum(r.accuracy for r in task_results) / len(task_results)
            avg_completeness = sum(r.completeness for r in task_results) / len(task_results)
            avg_legal_correctness = sum(r.legal_correctness for r in task_results) / len(task_results)
            avg_overall = sum(r.overall_score for r in task_results) / len(task_results)
        else:
            avg_accuracy = avg_completeness = avg_legal_correctness = avg_overall = 0.0
        
        return {
            "agent_name": agent.name,
            "tasks_completed": len(task_results),
            "average_accuracy": avg_accuracy,
            "average_completeness": avg_completeness,
            "average_legal_correctness": avg_legal_correctness,
            "average_overall_score": avg_overall,
            "task_results": [r.__dict__ for r in task_results]
        }
    
    def get_leaderboard(self, category: Optional[LegalTaskCategory] = None) -> List[Dict[str, Any]]:
        """
        Get leaderboard for legal tasks.
        
        Args:
            category: Optional category filter
            
        Returns:
            Leaderboard entries
        """
        results = self.results
        if category:
            # Filter by category
            filtered_results = []
            for result in results:
                task = self.tasks.get(result.task_id)
                if task and task.category == category:
                    filtered_results.append(result)
            results = filtered_results
        
        # Group by agent
        agent_scores = {}
        for result in results:
            if result.agent_name not in agent_scores:
                agent_scores[result.agent_name] = []
            agent_scores[result.agent_name].append(result.overall_score)
        
        # Calculate averages and sort
        leaderboard = []
        for agent_name, scores in agent_scores.items():
            avg_score = sum(scores) / len(scores)
            leaderboard.append({
                "agent_name": agent_name,
                "average_score": avg_score,
                "tasks_completed": len(scores)
            })
        
        leaderboard.sort(key=lambda x: x["average_score"], reverse=True)
        return leaderboard
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get benchmark statistics."""
        if not self.results:
            return {}
        
        all_scores = [r.overall_score for r in self.results]
        
        category_counts = {}
        for result in self.results:
            task = self.tasks.get(result.task_id)
            if task:
                category = task.category.value
                category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "total_evaluations": len(self.results),
            "average_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "best_score": max(all_scores) if all_scores else 0.0,
            "worst_score": min(all_scores) if all_scores else 0.0,
            "category_distribution": category_counts
        }



