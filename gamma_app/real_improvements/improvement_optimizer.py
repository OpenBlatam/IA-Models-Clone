"""
Gamma App - Real Improvement Optimizer
Advanced optimization system for real improvements that actually work
"""

import asyncio
import logging
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import optuna

logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    """Optimization types"""
    PERFORMANCE = "performance"
    COST = "cost"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    RESOURCE = "resource"
    SCHEDULE = "schedule"

class OptimizationAlgorithm(Enum):
    """Optimization algorithms"""
    GENETIC = "genetic"
    GRADIENT = "gradient"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    PARTICLE_SWARM = "particle_swarm"

@dataclass
class OptimizationGoal:
    """Optimization goal"""
    goal_id: str
    name: str
    type: OptimizationType
    target_value: float
    weight: float = 1.0
    constraints: List[Dict[str, Any]] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.constraints is None:
            self.constraints = []

@dataclass
class OptimizationResult:
    """Optimization result"""
    result_id: str
    goal_id: str
    algorithm: OptimizationAlgorithm
    best_solution: Dict[str, Any]
    best_value: float
    iterations: int
    execution_time: float
    convergence: bool
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class RealImprovementOptimizer:
    """
    Advanced optimization system for real improvements
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize improvement optimizer"""
        self.project_root = Path(project_root)
        self.goals: Dict[str, OptimizationGoal] = {}
        self.results: Dict[str, OptimizationResult] = {}
        self.optimization_logs: Dict[str, List[Dict[str, Any]]] = {}
        self.optimization_models: Dict[str, Any] = {}
        
        # Initialize with default goals
        self._initialize_default_goals()
        
        logger.info(f"Real Improvement Optimizer initialized for {self.project_root}")
    
    def _initialize_default_goals(self):
        """Initialize default optimization goals"""
        # Performance optimization goal
        performance_goal = OptimizationGoal(
            goal_id="performance_optimization",
            name="Performance Optimization",
            type=OptimizationType.PERFORMANCE,
            target_value=0.95,  # 95% performance improvement
            weight=0.3,
            constraints=[
                {"type": "max_cost", "value": 10000},
                {"type": "max_time", "value": 168}  # 1 week
            ]
        )
        self.goals[performance_goal.goal_id] = performance_goal
        
        # Cost optimization goal
        cost_goal = OptimizationGoal(
            goal_id="cost_optimization",
            name="Cost Optimization",
            type=OptimizationType.COST,
            target_value=0.5,  # 50% cost reduction
            weight=0.2,
            constraints=[
                {"type": "min_quality", "value": 0.8},
                {"type": "max_time", "value": 240}  # 10 days
            ]
        )
        self.goals[cost_goal.goal_id] = cost_goal
        
        # Quality optimization goal
        quality_goal = OptimizationGoal(
            goal_id="quality_optimization",
            name="Quality Optimization",
            type=OptimizationType.QUALITY,
            target_value=0.9,  # 90% quality score
            weight=0.25,
            constraints=[
                {"type": "max_cost", "value": 15000},
                {"type": "max_time", "value": 336}  # 2 weeks
            ]
        )
        self.goals[quality_goal.goal_id] = quality_goal
        
        # Efficiency optimization goal
        efficiency_goal = OptimizationGoal(
            goal_id="efficiency_optimization",
            name="Efficiency Optimization",
            type=OptimizationType.EFFICIENCY,
            target_value=0.85,  # 85% efficiency
            weight=0.15,
            constraints=[
                {"type": "min_quality", "value": 0.7},
                {"type": "max_cost", "value": 8000}
            ]
        )
        self.goals[efficiency_goal.goal_id] = efficiency_goal
        
        # Resource optimization goal
        resource_goal = OptimizationGoal(
            goal_id="resource_optimization",
            name="Resource Optimization",
            type=OptimizationType.RESOURCE,
            target_value=0.8,  # 80% resource utilization
            weight=0.1,
            constraints=[
                {"type": "min_quality", "value": 0.75},
                {"type": "max_time", "value": 120}  # 5 days
            ]
        )
        self.goals[resource_goal.goal_id] = resource_goal
    
    def create_optimization_goal(self, name: str, type: OptimizationType, 
                               target_value: float, weight: float = 1.0,
                               constraints: List[Dict[str, Any]] = None) -> str:
        """Create optimization goal"""
        try:
            goal_id = f"goal_{int(time.time() * 1000)}"
            
            goal = OptimizationGoal(
                goal_id=goal_id,
                name=name,
                type=type,
                target_value=target_value,
                weight=weight,
                constraints=constraints or []
            )
            
            self.goals[goal_id] = goal
            self.optimization_logs[goal_id] = []
            
            logger.info(f"Optimization goal created: {name}")
            return goal_id
            
        except Exception as e:
            logger.error(f"Failed to create optimization goal: {e}")
            raise
    
    async def optimize_improvements(self, goal_id: str, 
                                  improvement_data: pd.DataFrame,
                                  algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GENETIC) -> Dict[str, Any]:
        """Optimize improvements"""
        try:
            if goal_id not in self.goals:
                return {"success": False, "error": "Goal not found"}
            
            goal = self.goals[goal_id]
            start_time = time.time()
            
            self._log_optimization(goal_id, "optimization_started", f"Optimization started for goal {goal.name}")
            
            # Prepare optimization problem
            problem = self._prepare_optimization_problem(goal, improvement_data)
            
            # Run optimization based on algorithm
            if algorithm == OptimizationAlgorithm.GENETIC:
                result = await self._genetic_optimization(problem, goal)
            elif algorithm == OptimizationAlgorithm.GRADIENT:
                result = await self._gradient_optimization(problem, goal)
            elif algorithm == OptimizationAlgorithm.RANDOM_SEARCH:
                result = await self._random_search_optimization(problem, goal)
            elif algorithm == OptimizationAlgorithm.BAYESIAN:
                result = await self._bayesian_optimization(problem, goal)
            elif algorithm == OptimizationAlgorithm.PARTICLE_SWARM:
                result = await self._particle_swarm_optimization(problem, goal)
            else:
                return {"success": False, "error": f"Unknown algorithm: {algorithm}"}
            
            # Create optimization result
            result_id = f"result_{int(time.time() * 1000)}"
            execution_time = time.time() - start_time
            
            optimization_result = OptimizationResult(
                result_id=result_id,
                goal_id=goal_id,
                algorithm=algorithm,
                best_solution=result["solution"],
                best_value=result["value"],
                iterations=result["iterations"],
                execution_time=execution_time,
                convergence=result["convergence"]
            )
            
            self.results[result_id] = optimization_result
            
            self._log_optimization(goal_id, "optimization_completed", 
                                f"Optimization completed: {result['value']:.4f}")
            
            return {
                "success": True,
                "result_id": result_id,
                "best_solution": result["solution"],
                "best_value": result["value"],
                "iterations": result["iterations"],
                "execution_time": execution_time,
                "convergence": result["convergence"]
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize improvements: {e}")
            return {"success": False, "error": str(e)}
    
    def _prepare_optimization_problem(self, goal: OptimizationGoal, 
                                    improvement_data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare optimization problem"""
        try:
            # Define objective function
            def objective_function(x):
                # x is a vector of improvement selections (0 or 1)
                selected_improvements = improvement_data.iloc[np.where(x == 1)[0]]
                
                if len(selected_improvements) == 0:
                    return float('inf')
                
                # Calculate objective based on goal type
                if goal.type == OptimizationType.PERFORMANCE:
                    return -selected_improvements['impact_score'].sum()
                elif goal.type == OptimizationType.COST:
                    return selected_improvements['effort_hours'].sum() * 100  # Cost per hour
                elif goal.type == OptimizationType.QUALITY:
                    return -selected_improvements['quality_score'].mean() if 'quality_score' in selected_improvements.columns else -selected_improvements['impact_score'].mean()
                elif goal.type == OptimizationType.EFFICIENCY:
                    efficiency = selected_improvements['impact_score'].sum() / selected_improvements['effort_hours'].sum()
                    return -efficiency
                elif goal.type == OptimizationType.RESOURCE:
                    return -selected_improvements['impact_score'].sum() / selected_improvements['effort_hours'].sum()
                else:
                    return -selected_improvements['impact_score'].sum()
            
            # Define constraints
            constraints = []
            for constraint in goal.constraints:
                if constraint["type"] == "max_cost":
                    def cost_constraint(x):
                        selected = improvement_data.iloc[np.where(x == 1)[0]]
                        return constraint["value"] - (selected['effort_hours'].sum() * 100)
                    constraints.append(cost_constraint)
                elif constraint["type"] == "max_time":
                    def time_constraint(x):
                        selected = improvement_data.iloc[np.where(x == 1)[0]]
                        return constraint["value"] - selected['effort_hours'].sum()
                    constraints.append(time_constraint)
                elif constraint["type"] == "min_quality":
                    def quality_constraint(x):
                        selected = improvement_data.iloc[np.where(x == 1)[0]]
                        if len(selected) == 0:
                            return -1
                        quality = selected['impact_score'].mean() / 10  # Normalize to 0-1
                        return quality - constraint["value"]
                    constraints.append(quality_constraint)
            
            return {
                "objective": objective_function,
                "constraints": constraints,
                "bounds": [(0, 1) for _ in range(len(improvement_data))],
                "data": improvement_data
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare optimization problem: {e}")
            raise
    
    async def _genetic_optimization(self, problem: Dict[str, Any], goal: OptimizationGoal) -> Dict[str, Any]:
        """Genetic algorithm optimization"""
        try:
            from scipy.optimize import differential_evolution
            
            def objective_wrapper(x):
                return problem["objective"](x)
            
            # Run genetic algorithm
            result = differential_evolution(
                objective_wrapper,
                bounds=problem["bounds"],
                maxiter=100,
                popsize=15,
                seed=42
            )
            
            return {
                "solution": {"improvements": np.where(result.x > 0.5)[0].tolist()},
                "value": result.fun,
                "iterations": result.nit,
                "convergence": result.success
            }
            
        except Exception as e:
            logger.error(f"Genetic optimization failed: {e}")
            return {"solution": {}, "value": float('inf'), "iterations": 0, "convergence": False}
    
    async def _gradient_optimization(self, problem: Dict[str, Any], goal: OptimizationGoal) -> Dict[str, Any]:
        """Gradient-based optimization"""
        try:
            from scipy.optimize import minimize
            
            def objective_wrapper(x):
                return problem["objective"](x)
            
            # Initial guess
            x0 = np.random.random(len(problem["bounds"]))
            
            # Run gradient optimization
            result = minimize(
                objective_wrapper,
                x0,
                method='L-BFGS-B',
                bounds=problem["bounds"],
                options={'maxiter': 100}
            )
            
            return {
                "solution": {"improvements": np.where(result.x > 0.5)[0].tolist()},
                "value": result.fun,
                "iterations": result.nit,
                "convergence": result.success
            }
            
        except Exception as e:
            logger.error(f"Gradient optimization failed: {e}")
            return {"solution": {}, "value": float('inf'), "iterations": 0, "convergence": False}
    
    async def _random_search_optimization(self, problem: Dict[str, Any], goal: OptimizationGoal) -> Dict[str, Any]:
        """Random search optimization"""
        try:
            best_solution = None
            best_value = float('inf')
            iterations = 1000
            
            for i in range(iterations):
                # Generate random solution
                x = np.random.random(len(problem["bounds"]))
                x = (x > 0.5).astype(int)
                
                # Check constraints
                valid = True
                for constraint in problem["constraints"]:
                    if constraint(x) < 0:
                        valid = False
                        break
                
                if valid:
                    value = problem["objective"](x)
                    if value < best_value:
                        best_value = value
                        best_solution = x
            
            return {
                "solution": {"improvements": np.where(best_solution > 0.5)[0].tolist() if best_solution is not None else []},
                "value": best_value,
                "iterations": iterations,
                "convergence": best_solution is not None
            }
            
        except Exception as e:
            logger.error(f"Random search optimization failed: {e}")
            return {"solution": {}, "value": float('inf'), "iterations": 0, "convergence": False}
    
    async def _bayesian_optimization(self, problem: Dict[str, Any], goal: OptimizationGoal) -> Dict[str, Any]:
        """Bayesian optimization"""
        try:
            import optuna
            
            def objective(trial):
                # Sample improvement selections
                x = []
                for i in range(len(problem["bounds"])):
                    x.append(trial.suggest_int(f'improvement_{i}', 0, 1))
                
                x = np.array(x)
                
                # Check constraints
                for constraint in problem["constraints"]:
                    if constraint(x) < 0:
                        return float('inf')
                
                return problem["objective"](x)
            
            # Create study
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=100)
            
            best_params = study.best_params
            best_solution = np.array([best_params[f'improvement_{i}'] for i in range(len(problem["bounds"]))])
            
            return {
                "solution": {"improvements": np.where(best_solution > 0.5)[0].tolist()},
                "value": study.best_value,
                "iterations": len(study.trials),
                "convergence": True
            }
            
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            return {"solution": {}, "value": float('inf'), "iterations": 0, "convergence": False}
    
    async def _particle_swarm_optimization(self, problem: Dict[str, Any], goal: OptimizationGoal) -> Dict[str, Any]:
        """Particle swarm optimization"""
        try:
            from pyswarm import pso
            
            def objective_wrapper(x):
                return problem["objective"](x)
            
            # Define constraint function
            def constraint_wrapper(x):
                constraints = []
                for constraint in problem["constraints"]:
                    constraints.append(constraint(x))
                return constraints
            
            # Run PSO
            xopt, fopt = pso(
                objective_wrapper,
                [0] * len(problem["bounds"]),
                [1] * len(problem["bounds"]),
                ieqcons=constraint_wrapper,
                swarmsize=20,
                maxiter=100
            )
            
            return {
                "solution": {"improvements": np.where(xopt > 0.5)[0].tolist()},
                "value": fopt,
                "iterations": 100,
                "convergence": True
            }
            
        except Exception as e:
            logger.error(f"Particle swarm optimization failed: {e}")
            return {"solution": {}, "value": float('inf'), "iterations": 0, "convergence": False}
    
    async def multi_objective_optimization(self, goal_ids: List[str], 
                                         improvement_data: pd.DataFrame) -> Dict[str, Any]:
        """Multi-objective optimization"""
        try:
            if not all(goal_id in self.goals for goal_id in goal_ids):
                return {"success": False, "error": "One or more goals not found"}
            
            goals = [self.goals[goal_id] for goal_id in goal_ids]
            
            # Prepare multi-objective problem
            def multi_objective_function(x):
                objectives = []
                for goal in goals:
                    problem = self._prepare_optimization_problem(goal, improvement_data)
                    objectives.append(problem["objective"](x))
                return objectives
            
            # Use NSGA-II or similar multi-objective algorithm
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.core.problem import Problem
            from pymoo.optimize import minimize
            
            class MultiObjectiveProblem(Problem):
                def __init__(self, n_vars, n_obj):
                    super().__init__(n_var=n_vars, n_obj=n_obj, n_constr=0, xl=0, xu=1)
                    self.goals = goals
                    self.improvement_data = improvement_data
                
                def _evaluate(self, x, out, *args, **kwargs):
                    objectives = []
                    for goal in self.goals:
                        problem = self._prepare_optimization_problem(goal, self.improvement_data)
                        objectives.append([problem["objective"](x[i]) for i in range(len(x))])
                    
                    out["F"] = np.column_stack(objectives)
            
            problem = MultiObjectiveProblem(len(improvement_data), len(goals))
            algorithm = NSGA2(pop_size=100)
            
            res = minimize(problem, algorithm, ('n_gen', 100), verbose=False)
            
            return {
                "success": True,
                "pareto_front": res.F.tolist(),
                "solutions": res.X.tolist(),
                "objectives": [goal.name for goal in goals]
            }
            
        except Exception as e:
            logger.error(f"Multi-objective optimization failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _log_optimization(self, goal_id: str, event: str, message: str):
        """Log optimization event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "message": message
        }
        
        if goal_id not in self.optimization_logs:
            self.optimization_logs[goal_id] = []
        
        self.optimization_logs[goal_id].append(log_entry)
        
        logger.info(f"Optimization {goal_id}: {event} - {message}")
    
    def get_optimization_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get optimization result"""
        if result_id not in self.results:
            return None
        
        result = self.results[result_id]
        
        return {
            "result_id": result_id,
            "goal_id": result.goal_id,
            "algorithm": result.algorithm.value,
            "best_solution": result.best_solution,
            "best_value": result.best_value,
            "iterations": result.iterations,
            "execution_time": result.execution_time,
            "convergence": result.convergence,
            "created_at": result.created_at.isoformat()
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary"""
        total_goals = len(self.goals)
        total_results = len(self.results)
        successful_results = len([r for r in self.results.values() if r.convergence])
        
        return {
            "total_goals": total_goals,
            "total_results": total_results,
            "successful_results": successful_results,
            "success_rate": (successful_results / total_results * 100) if total_results > 0 else 0,
            "optimization_types": list(set(g.type.value for g in self.goals.values())),
            "algorithms_used": list(set(r.algorithm.value for r in self.results.values()))
        }
    
    def get_optimization_logs(self, goal_id: str) -> List[Dict[str, Any]]:
        """Get optimization logs"""
        return self.optimization_logs.get(goal_id, [])
    
    def create_optimization_model(self, goal_id: str, training_data: pd.DataFrame) -> str:
        """Create optimization model"""
        try:
            if goal_id not in self.goals:
                raise ValueError(f"Goal {goal_id} not found")
            
            goal = self.goals[goal_id]
            
            # Train model based on goal type
            if goal.type == OptimizationType.PERFORMANCE:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                X = training_data[['effort_hours', 'complexity', 'team_size']]
                y = training_data['impact_score']
            elif goal.type == OptimizationType.COST:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                X = training_data[['impact_score', 'complexity', 'team_size']]
                y = training_data['effort_hours']
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                X = training_data[['effort_hours', 'impact_score', 'complexity']]
                y = training_data['quality_score'] if 'quality_score' in training_data.columns else training_data['impact_score']
            
            model.fit(X, y)
            
            # Store model
            model_id = f"model_{int(time.time() * 1000)}"
            self.optimization_models[model_id] = {
                "goal_id": goal_id,
                "model": model,
                "features": X.columns.tolist(),
                "target": y.name if hasattr(y, 'name') else 'target'
            }
            
            logger.info(f"Optimization model created: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to create optimization model: {e}")
            raise

# Global optimizer instance
improvement_optimizer = None

def get_improvement_optimizer() -> RealImprovementOptimizer:
    """Get improvement optimizer instance"""
    global improvement_optimizer
    if not improvement_optimizer:
        improvement_optimizer = RealImprovementOptimizer()
    return improvement_optimizer













