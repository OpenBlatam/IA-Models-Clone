"""
ComfyBench: Benchmarking LLM-based Agents in ComfyUI
=====================================================

Paper: "ComfyBench: Benchmarking LLM-based Agents in ComfyUI"

Key concepts:
- Benchmarking framework for LLM-based agents
- ComfyUI integration
- Task evaluation
- Performance metrics
- Agent comparison
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus


class TaskCategory(Enum):
    """Task categories."""
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDITING = "image_editing"
    WORKFLOW_CREATION = "workflow_creation"
    COMPLEX_REASONING = "complex_reasoning"
    MULTI_STEP = "multi_step"


class EvaluationMetric(Enum):
    """Evaluation metrics."""
    ACCURACY = "accuracy"
    COMPLETION_RATE = "completion_rate"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    USER_SATISFACTION = "user_satisfaction"


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    task_category: TaskCategory
    success: bool
    execution_time: float
    quality_score: float
    metrics: Dict[str, float]
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    agent_name: str
    task_results: List[TaskResult]
    overall_score: float = 0.0
    category_scores: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Calculate scores."""
        if not self.task_results:
            return
        
        # Overall score
        self.overall_score = sum(
            task.quality_score for task in self.task_results
        ) / len(self.task_results)
        
        # Category scores
        category_results = {}
        for task in self.task_results:
            category = task.task_category.value
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(task.quality_score)
        
        self.category_scores = {
            category: sum(scores) / len(scores)
            for category, scores in category_results.items()
        }


class ComfyBench:
    """
    ComfyBench benchmarking system for LLM-based agents.
    
    Evaluates agents on various ComfyUI-related tasks.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ComfyBench.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.results: List[BenchmarkResult] = []
        
        # Initialize tasks
        self._initialize_tasks()
    
    def _initialize_tasks(self):
        """Initialize benchmark tasks."""
        self.tasks = {
            "task_1": {
                "task_id": "task_1",
                "category": TaskCategory.IMAGE_GENERATION,
                "description": "Generate image from text prompt",
                "difficulty": "easy"
            },
            "task_2": {
                "task_id": "task_2",
                "category": TaskCategory.IMAGE_EDITING,
                "description": "Edit image based on instructions",
                "difficulty": "medium"
            },
            "task_3": {
                "task_id": "task_3",
                "category": TaskCategory.WORKFLOW_CREATION,
                "description": "Create ComfyUI workflow from description",
                "difficulty": "hard"
            },
            "task_4": {
                "task_id": "task_4",
                "category": TaskCategory.COMPLEX_REASONING,
                "description": "Solve complex multi-step reasoning task",
                "difficulty": "hard"
            },
            "task_5": {
                "task_id": "task_5",
                "category": TaskCategory.MULTI_STEP,
                "description": "Execute multi-step workflow",
                "difficulty": "expert"
            }
        }
    
    def evaluate_task(
        self,
        agent: BaseAgent,
        task_id: str,
        task_input: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """
        Evaluate agent on a specific task.
        
        Args:
            agent: Agent to evaluate
            task_id: Task identifier
            task_input: Task input data
            
        Returns:
            Task result
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task_info = self.tasks[task_id]
        start_time = datetime.now()
        
        try:
            # Execute task
            result = agent.run(task_info["description"])
            
            # Calculate metrics
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Evaluate quality (simplified)
            quality_score = self._evaluate_quality(result, task_info)
            
            metrics = {
                "execution_time": execution_time,
                "quality_score": quality_score
            }
            
            return TaskResult(
                task_id=task_id,
                task_category=task_info["category"],
                success=True,
                execution_time=execution_time,
                quality_score=quality_score,
                metrics=metrics
            )
        
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return TaskResult(
                task_id=task_id,
                task_category=task_info["category"],
                success=False,
                execution_time=execution_time,
                quality_score=0.0,
                metrics={"execution_time": execution_time},
                error_message=str(e)
            )
    
    def _evaluate_quality(
        self,
        result: Dict[str, Any],
        task_info: Dict[str, Any]
    ) -> float:
        """Evaluate quality of task result."""
        # Simplified quality evaluation
        # In production, this would use specific metrics per task type
        
        if not result:
            return 0.0
        
        # Base score
        score = 0.5
        
        # Check if result has expected structure
        if "task" in result:
            score += 0.2
        
        if "action" in result or "observation" in result:
            score += 0.2
        
        # Task-specific evaluation
        if task_info["category"] == TaskCategory.IMAGE_GENERATION:
            # Check for image-related output
            if "output" in result or "image" in result:
                score += 0.1
        
        return min(1.0, score)
    
    def run_benchmark(
        self,
        agent: BaseAgent,
        task_ids: Optional[List[str]] = None
    ) -> BenchmarkResult:
        """
        Run full benchmark on an agent.
        
        Args:
            agent: Agent to benchmark
            task_ids: Optional list of task IDs to run (default: all)
            
        Returns:
            Benchmark result
        """
        if task_ids is None:
            task_ids = list(self.tasks.keys())
        
        task_results = []
        
        for task_id in task_ids:
            result = self.evaluate_task(agent, task_id)
            task_results.append(result)
        
        benchmark_result = BenchmarkResult(
            agent_name=agent.name,
            task_results=task_results
        )
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def compare_agents(
        self,
        agent_results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """
        Compare multiple agents.
        
        Args:
            agent_results: List of benchmark results
            
        Returns:
            Comparison data
        """
        comparison = {
            "agents": [],
            "overall_ranking": [],
            "category_rankings": {}
        }
        
        for result in agent_results:
            comparison["agents"].append({
                "name": result.agent_name,
                "overall_score": result.overall_score,
                "category_scores": result.category_scores
            })
        
        # Overall ranking
        comparison["overall_ranking"] = sorted(
            comparison["agents"],
            key=lambda x: x["overall_score"],
            reverse=True
        )
        
        # Category rankings
        categories = set()
        for result in agent_results:
            categories.update(result.category_scores.keys())
        
        for category in categories:
            category_agents = [
                {
                    "name": result.agent_name,
                    "score": result.category_scores.get(category, 0.0)
                }
                for result in agent_results
                if category in result.category_scores
            ]
            comparison["category_rankings"][category] = sorted(
                category_agents,
                key=lambda x: x["score"],
                reverse=True
            )
        
        return comparison
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get benchmark statistics."""
        if not self.results:
            return {}
        
        all_scores = [r.overall_score for r in self.results]
        
        return {
            "total_benchmarks": len(self.results),
            "average_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "best_score": max(all_scores) if all_scores else 0.0,
            "worst_score": min(all_scores) if all_scores else 0.0,
            "tasks_available": len(self.tasks)
        }



