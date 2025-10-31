"""
Ultra-Advanced Swarm Intelligence System
========================================

Ultra-advanced swarm intelligence system with cutting-edge features.
"""

import time
import functools
import logging
import asyncio
import threading
from typing import Dict, Any, Optional, List, Callable, Union, TypeVar, Generic
from flask import request, jsonify, g, current_app
import concurrent.futures
from threading import Lock, RLock
import queue
import heapq
import json
import hashlib
import uuid
from datetime import datetime, timedelta
import psutil
import os
import gc
import weakref
from collections import defaultdict, deque
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)
T = TypeVar('T')

class UltraSwarm:
    """
    Ultra-advanced swarm intelligence system.
    """
    
    def __init__(self):
        # Swarm algorithms
        self.swarm_algorithms = {}
        self.algorithm_lock = RLock()
        
        # Swarm behaviors
        self.swarm_behaviors = {}
        self.behavior_lock = RLock()
        
        # Swarm coordination
        self.swarm_coordination = {}
        self.coordination_lock = RLock()
        
        # Swarm optimization
        self.swarm_optimization = {}
        self.optimization_lock = RLock()
        
        # Swarm communication
        self.swarm_communication = {}
        self.communication_lock = RLock()
        
        # Swarm learning
        self.swarm_learning = {}
        self.learning_lock = RLock()
        
        # Initialize swarm system
        self._initialize_swarm_system()
    
    def _initialize_swarm_system(self):
        """Initialize swarm system."""
        try:
            # Initialize swarm algorithms
            self._initialize_swarm_algorithms()
            
            # Initialize swarm behaviors
            self._initialize_swarm_behaviors()
            
            # Initialize swarm coordination
            self._initialize_swarm_coordination()
            
            # Initialize swarm optimization
            self._initialize_swarm_optimization()
            
            # Initialize swarm communication
            self._initialize_swarm_communication()
            
            # Initialize swarm learning
            self._initialize_swarm_learning()
            
            logger.info("Ultra swarm system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize swarm system: {str(e)}")
    
    def _initialize_swarm_algorithms(self):
        """Initialize swarm algorithms."""
        try:
            # Initialize swarm algorithms
            self.swarm_algorithms['pso'] = self._create_pso_algorithm()
            self.swarm_algorithms['aco'] = self._create_aco_algorithm()
            self.swarm_algorithms['abc'] = self._create_abc_algorithm()
            self.swarm_algorithms['firefly'] = self._create_firefly_algorithm()
            self.swarm_algorithms['bat'] = self._create_bat_algorithm()
            self.swarm_algorithms['cuckoo'] = self._create_cuckoo_algorithm()
            
            logger.info("Swarm algorithms initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize swarm algorithms: {str(e)}")
    
    def _initialize_swarm_behaviors(self):
        """Initialize swarm behaviors."""
        try:
            # Initialize swarm behaviors
            self.swarm_behaviors['flocking'] = self._create_flocking_behavior()
            self.swarm_behaviors['foraging'] = self._create_foraging_behavior()
            self.swarm_behaviors['nesting'] = self._create_nesting_behavior()
            self.swarm_behaviors['migration'] = self._create_migration_behavior()
            self.swarm_behaviors['hunting'] = self._create_hunting_behavior()
            self.swarm_behaviors['defense'] = self._create_defense_behavior()
            
            logger.info("Swarm behaviors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize swarm behaviors: {str(e)}")
    
    def _initialize_swarm_coordination(self):
        """Initialize swarm coordination."""
        try:
            # Initialize swarm coordination
            self.swarm_coordination['consensus'] = self._create_consensus_coordination()
            self.swarm_coordination['formation'] = self._create_formation_coordination()
            self.swarm_coordination['task_allocation'] = self._create_task_allocation_coordination()
            self.swarm_coordination['resource_sharing'] = self._create_resource_sharing_coordination()
            self.swarm_coordination['collision_avoidance'] = self._create_collision_avoidance_coordination()
            self.swarm_coordination['path_planning'] = self._create_path_planning_coordination()
            
            logger.info("Swarm coordination initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize swarm coordination: {str(e)}")
    
    def _initialize_swarm_optimization(self):
        """Initialize swarm optimization."""
        try:
            # Initialize swarm optimization
            self.swarm_optimization['global_optimization'] = self._create_global_optimization()
            self.swarm_optimization['local_optimization'] = self._create_local_optimization()
            self.swarm_optimization['multi_objective'] = self._create_multi_objective_optimization()
            self.swarm_optimization['dynamic_optimization'] = self._create_dynamic_optimization()
            self.swarm_optimization['constrained_optimization'] = self._create_constrained_optimization()
            self.swarm_optimization['robust_optimization'] = self._create_robust_optimization()
            
            logger.info("Swarm optimization initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize swarm optimization: {str(e)}")
    
    def _initialize_swarm_communication(self):
        """Initialize swarm communication."""
        try:
            # Initialize swarm communication
            self.swarm_communication['pheromone'] = self._create_pheromone_communication()
            self.swarm_communication['visual'] = self._create_visual_communication()
            self.swarm_communication['acoustic'] = self._create_acoustic_communication()
            self.swarm_communication['tactile'] = self._create_tactile_communication()
            self.swarm_communication['chemical'] = self._create_chemical_communication()
            self.swarm_communication['electromagnetic'] = self._create_electromagnetic_communication()
            
            logger.info("Swarm communication initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize swarm communication: {str(e)}")
    
    def _initialize_swarm_learning(self):
        """Initialize swarm learning."""
        try:
            # Initialize swarm learning
            self.swarm_learning['collective_learning'] = self._create_collective_learning()
            self.swarm_learning['distributed_learning'] = self._create_distributed_learning()
            self.swarm_learning['reinforcement_learning'] = self._create_reinforcement_learning()
            self.swarm_learning['evolutionary_learning'] = self._create_evolutionary_learning()
            self.swarm_learning['adaptive_learning'] = self._create_adaptive_learning()
            self.swarm_learning['emergent_learning'] = self._create_emergent_learning()
            
            logger.info("Swarm learning initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize swarm learning: {str(e)}")
    
    # Swarm algorithm creation methods
    def _create_pso_algorithm(self):
        """Create PSO algorithm."""
        return {'name': 'PSO', 'type': 'algorithm', 'features': ['particle_swarm', 'optimization', 'global_search']}
    
    def _create_aco_algorithm(self):
        """Create ACO algorithm."""
        return {'name': 'ACO', 'type': 'algorithm', 'features': ['ant_colony', 'optimization', 'path_finding']}
    
    def _create_abc_algorithm(self):
        """Create ABC algorithm."""
        return {'name': 'ABC', 'type': 'algorithm', 'features': ['artificial_bee', 'optimization', 'foraging']}
    
    def _create_firefly_algorithm(self):
        """Create Firefly algorithm."""
        return {'name': 'Firefly', 'type': 'algorithm', 'features': ['firefly', 'optimization', 'attraction']}
    
    def _create_bat_algorithm(self):
        """Create Bat algorithm."""
        return {'name': 'Bat', 'type': 'algorithm', 'features': ['bat', 'optimization', 'echolocation']}
    
    def _create_cuckoo_algorithm(self):
        """Create Cuckoo algorithm."""
        return {'name': 'Cuckoo', 'type': 'algorithm', 'features': ['cuckoo', 'optimization', 'nest_parasitism']}
    
    # Swarm behavior creation methods
    def _create_flocking_behavior(self):
        """Create flocking behavior."""
        return {'name': 'Flocking', 'type': 'behavior', 'features': ['alignment', 'cohesion', 'separation']}
    
    def _create_foraging_behavior(self):
        """Create foraging behavior."""
        return {'name': 'Foraging', 'type': 'behavior', 'features': ['food_search', 'resource_exploitation', 'efficiency']}
    
    def _create_nesting_behavior(self):
        """Create nesting behavior."""
        return {'name': 'Nesting', 'type': 'behavior', 'features': ['nest_building', 'territory', 'protection']}
    
    def _create_migration_behavior(self):
        """Create migration behavior."""
        return {'name': 'Migration', 'type': 'behavior', 'features': ['seasonal_movement', 'navigation', 'endurance']}
    
    def _create_hunting_behavior(self):
        """Create hunting behavior."""
        return {'name': 'Hunting', 'type': 'behavior', 'features': ['prey_tracking', 'coordination', 'strategy']}
    
    def _create_defense_behavior(self):
        """Create defense behavior."""
        return {'name': 'Defense', 'type': 'behavior', 'features': ['threat_response', 'protection', 'coordination']}
    
    # Swarm coordination creation methods
    def _create_consensus_coordination(self):
        """Create consensus coordination."""
        return {'name': 'Consensus', 'type': 'coordination', 'features': ['agreement', 'decision_making', 'democracy']}
    
    def _create_formation_coordination(self):
        """Create formation coordination."""
        return {'name': 'Formation', 'type': 'coordination', 'features': ['spatial_arrangement', 'pattern', 'structure']}
    
    def _create_task_allocation_coordination(self):
        """Create task allocation coordination."""
        return {'name': 'Task Allocation', 'type': 'coordination', 'features': ['work_distribution', 'efficiency', 'specialization']}
    
    def _create_resource_sharing_coordination(self):
        """Create resource sharing coordination."""
        return {'name': 'Resource Sharing', 'type': 'coordination', 'features': ['resource_distribution', 'cooperation', 'fairness']}
    
    def _create_collision_avoidance_coordination(self):
        """Create collision avoidance coordination."""
        return {'name': 'Collision Avoidance', 'type': 'coordination', 'features': ['safety', 'navigation', 'prevention']}
    
    def _create_path_planning_coordination(self):
        """Create path planning coordination."""
        return {'name': 'Path Planning', 'type': 'coordination', 'features': ['route_optimization', 'navigation', 'efficiency']}
    
    # Swarm optimization creation methods
    def _create_global_optimization(self):
        """Create global optimization."""
        return {'name': 'Global Optimization', 'type': 'optimization', 'features': ['global_search', 'exploration', 'convergence']}
    
    def _create_local_optimization(self):
        """Create local optimization."""
        return {'name': 'Local Optimization', 'type': 'optimization', 'features': ['local_search', 'exploitation', 'refinement']}
    
    def _create_multi_objective_optimization(self):
        """Create multi-objective optimization."""
        return {'name': 'Multi-objective', 'type': 'optimization', 'features': ['pareto_front', 'trade_offs', 'diversity']}
    
    def _create_dynamic_optimization(self):
        """Create dynamic optimization."""
        return {'name': 'Dynamic', 'type': 'optimization', 'features': ['adaptation', 'change_response', 'flexibility']}
    
    def _create_constrained_optimization(self):
        """Create constrained optimization."""
        return {'name': 'Constrained', 'type': 'optimization', 'features': ['constraints', 'feasibility', 'penalty']}
    
    def _create_robust_optimization(self):
        """Create robust optimization."""
        return {'name': 'Robust', 'type': 'optimization', 'features': ['uncertainty', 'robustness', 'reliability']}
    
    # Swarm communication creation methods
    def _create_pheromone_communication(self):
        """Create pheromone communication."""
        return {'name': 'Pheromone', 'type': 'communication', 'features': ['chemical_signals', 'trail_marking', 'stigmergy']}
    
    def _create_visual_communication(self):
        """Create visual communication."""
        return {'name': 'Visual', 'type': 'communication', 'features': ['visual_signals', 'gestures', 'appearance']}
    
    def _create_acoustic_communication(self):
        """Create acoustic communication."""
        return {'name': 'Acoustic', 'type': 'communication', 'features': ['sound_signals', 'vibrations', 'frequency']}
    
    def _create_tactile_communication(self):
        """Create tactile communication."""
        return {'name': 'Tactile', 'type': 'communication', 'features': ['touch_signals', 'contact', 'pressure']}
    
    def _create_chemical_communication(self):
        """Create chemical communication."""
        return {'name': 'Chemical', 'type': 'communication', 'features': ['chemical_signals', 'molecules', 'receptors']}
    
    def _create_electromagnetic_communication(self):
        """Create electromagnetic communication."""
        return {'name': 'Electromagnetic', 'type': 'communication', 'features': ['electromagnetic_signals', 'radio', 'frequency']}
    
    # Swarm learning creation methods
    def _create_collective_learning(self):
        """Create collective learning."""
        return {'name': 'Collective Learning', 'type': 'learning', 'features': ['group_learning', 'knowledge_sharing', 'wisdom']}
    
    def _create_distributed_learning(self):
        """Create distributed learning."""
        return {'name': 'Distributed Learning', 'type': 'learning', 'features': ['decentralized', 'parallel', 'scalable']}
    
    def _create_reinforcement_learning(self):
        """Create reinforcement learning."""
        return {'name': 'Reinforcement Learning', 'type': 'learning', 'features': ['reward_based', 'trial_error', 'adaptation']}
    
    def _create_evolutionary_learning(self):
        """Create evolutionary learning."""
        return {'name': 'Evolutionary Learning', 'type': 'learning', 'features': ['genetic_algorithm', 'selection', 'mutation']}
    
    def _create_adaptive_learning(self):
        """Create adaptive learning."""
        return {'name': 'Adaptive Learning', 'type': 'learning', 'features': ['adaptation', 'flexibility', 'change_response']}
    
    def _create_emergent_learning(self):
        """Create emergent learning."""
        return {'name': 'Emergent Learning', 'type': 'learning', 'features': ['emergence', 'complexity', 'self_organization']}
    
    # Swarm operations
    def execute_swarm_algorithm(self, algorithm_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute swarm algorithm."""
        try:
            with self.algorithm_lock:
                if algorithm_type in self.swarm_algorithms:
                    # Execute swarm algorithm
                    result = {
                        'algorithm_type': algorithm_type,
                        'parameters': parameters,
                        'result': self._simulate_swarm_algorithm(parameters, algorithm_type),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    return result
                else:
                    return {'error': f'Swarm algorithm type {algorithm_type} not supported'}
        except Exception as e:
            logger.error(f"Swarm algorithm execution error: {str(e)}")
            return {'error': str(e)}
    
    def apply_swarm_behavior(self, behavior_type: str, swarm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply swarm behavior."""
        try:
            with self.behavior_lock:
                if behavior_type in self.swarm_behaviors:
                    # Apply swarm behavior
                    result = {
                        'behavior_type': behavior_type,
                        'swarm_data': swarm_data,
                        'result': self._simulate_swarm_behavior(swarm_data, behavior_type),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    return result
                else:
                    return {'error': f'Swarm behavior type {behavior_type} not supported'}
        except Exception as e:
            logger.error(f"Swarm behavior application error: {str(e)}")
            return {'error': str(e)}
    
    def coordinate_swarm(self, coordination_type: str, swarm_config: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate swarm."""
        try:
            with self.coordination_lock:
                if coordination_type in self.swarm_coordination:
                    # Coordinate swarm
                    result = {
                        'coordination_type': coordination_type,
                        'swarm_config': swarm_config,
                        'result': self._simulate_swarm_coordination(swarm_config, coordination_type),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    return result
                else:
                    return {'error': f'Swarm coordination type {coordination_type} not supported'}
        except Exception as e:
            logger.error(f"Swarm coordination error: {str(e)}")
            return {'error': str(e)}
    
    def optimize_swarm(self, optimization_type: str, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize swarm."""
        try:
            with self.optimization_lock:
                if optimization_type in self.swarm_optimization:
                    # Optimize swarm
                    result = {
                        'optimization_type': optimization_type,
                        'problem': problem,
                        'result': self._simulate_swarm_optimization(problem, optimization_type),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    return result
                else:
                    return {'error': f'Swarm optimization type {optimization_type} not supported'}
        except Exception as e:
            logger.error(f"Swarm optimization error: {str(e)}")
            return {'error': str(e)}
    
    def communicate_swarm(self, communication_type: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Communicate with swarm."""
        try:
            with self.communication_lock:
                if communication_type in self.swarm_communication:
                    # Communicate with swarm
                    result = {
                        'communication_type': communication_type,
                        'message': message,
                        'result': self._simulate_swarm_communication(message, communication_type),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    return result
                else:
                    return {'error': f'Swarm communication type {communication_type} not supported'}
        except Exception as e:
            logger.error(f"Swarm communication error: {str(e)}")
            return {'error': str(e)}
    
    def learn_swarm(self, learning_type: str, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn with swarm."""
        try:
            with self.learning_lock:
                if learning_type in self.swarm_learning:
                    # Learn with swarm
                    result = {
                        'learning_type': learning_type,
                        'learning_data': learning_data,
                        'result': self._simulate_swarm_learning(learning_data, learning_type),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    return result
                else:
                    return {'error': f'Swarm learning type {learning_type} not supported'}
        except Exception as e:
            logger.error(f"Swarm learning error: {str(e)}")
            return {'error': str(e)}
    
    def get_swarm_analytics(self, time_range: str = '24h') -> Dict[str, Any]:
        """Get swarm analytics."""
        try:
            # Get analytics
            analytics = {
                'time_range': time_range,
                'total_algorithms': len(self.swarm_algorithms),
                'total_behaviors': len(self.swarm_behaviors),
                'total_coordination_types': len(self.swarm_coordination),
                'total_optimization_types': len(self.swarm_optimization),
                'total_communication_types': len(self.swarm_communication),
                'total_learning_types': len(self.swarm_learning),
                'timestamp': datetime.utcnow().isoformat()
            }
            return analytics
        except Exception as e:
            logger.error(f"Swarm analytics error: {str(e)}")
            return {'error': str(e)}
    
    def _simulate_swarm_algorithm(self, parameters: Dict[str, Any], algorithm_type: str) -> Dict[str, Any]:
        """Simulate swarm algorithm."""
        # Implementation would perform actual swarm algorithm
        return {'executed': True, 'algorithm_type': algorithm_type, 'fitness': 0.95}
    
    def _simulate_swarm_behavior(self, swarm_data: Dict[str, Any], behavior_type: str) -> Dict[str, Any]:
        """Simulate swarm behavior."""
        # Implementation would perform actual swarm behavior
        return {'applied': True, 'behavior_type': behavior_type, 'efficiency': 0.90}
    
    def _simulate_swarm_coordination(self, swarm_config: Dict[str, Any], coordination_type: str) -> Dict[str, Any]:
        """Simulate swarm coordination."""
        # Implementation would perform actual swarm coordination
        return {'coordinated': True, 'coordination_type': coordination_type, 'synergy': 0.88}
    
    def _simulate_swarm_optimization(self, problem: Dict[str, Any], optimization_type: str) -> Dict[str, Any]:
        """Simulate swarm optimization."""
        # Implementation would perform actual swarm optimization
        return {'optimized': True, 'optimization_type': optimization_type, 'improvement': 0.25}
    
    def _simulate_swarm_communication(self, message: Dict[str, Any], communication_type: str) -> Dict[str, Any]:
        """Simulate swarm communication."""
        # Implementation would perform actual swarm communication
        return {'communicated': True, 'communication_type': communication_type, 'clarity': 0.92}
    
    def _simulate_swarm_learning(self, learning_data: Dict[str, Any], learning_type: str) -> Dict[str, Any]:
        """Simulate swarm learning."""
        # Implementation would perform actual swarm learning
        return {'learned': True, 'learning_type': learning_type, 'knowledge': 0.87}
    
    def cleanup(self):
        """Cleanup swarm system."""
        try:
            # Clear swarm algorithms
            with self.algorithm_lock:
                self.swarm_algorithms.clear()
            
            # Clear swarm behaviors
            with self.behavior_lock:
                self.swarm_behaviors.clear()
            
            # Clear swarm coordination
            with self.coordination_lock:
                self.swarm_coordination.clear()
            
            # Clear swarm optimization
            with self.optimization_lock:
                self.swarm_optimization.clear()
            
            # Clear swarm communication
            with self.communication_lock:
                self.swarm_communication.clear()
            
            # Clear swarm learning
            with self.learning_lock:
                self.swarm_learning.clear()
            
            logger.info("Swarm system cleaned up successfully")
        except Exception as e:
            logger.error(f"Swarm system cleanup error: {str(e)}")

# Global swarm instance
ultra_swarm = UltraSwarm()

# Decorators for swarm
def swarm_algorithm_execution(algorithm_type: str = 'pso'):
    """Swarm algorithm execution decorator."""
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Execute swarm algorithm if parameters are present
                if hasattr(request, 'json') and request.json:
                    parameters = request.json.get('algorithm_parameters', {})
                    if parameters:
                        result = ultra_swarm.execute_swarm_algorithm(algorithm_type, parameters)
                        kwargs['swarm_algorithm_execution'] = result
                
                return f(*args, **kwargs)
            except Exception as e:
                logger.error(f"Swarm algorithm execution error: {str(e)}")
                return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def swarm_behavior_application(behavior_type: str = 'flocking'):
    """Swarm behavior application decorator."""
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Apply swarm behavior if swarm data is present
                if hasattr(request, 'json') and request.json:
                    swarm_data = request.json.get('swarm_data', {})
                    if swarm_data:
                        result = ultra_swarm.apply_swarm_behavior(behavior_type, swarm_data)
                        kwargs['swarm_behavior_application'] = result
                
                return f(*args, **kwargs)
            except Exception as e:
                logger.error(f"Swarm behavior application error: {str(e)}")
                return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def swarm_coordination(coordination_type: str = 'consensus'):
    """Swarm coordination decorator."""
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Coordinate swarm if swarm config is present
                if hasattr(request, 'json') and request.json:
                    swarm_config = request.json.get('swarm_config', {})
                    if swarm_config:
                        result = ultra_swarm.coordinate_swarm(coordination_type, swarm_config)
                        kwargs['swarm_coordination'] = result
                
                return f(*args, **kwargs)
            except Exception as e:
                logger.error(f"Swarm coordination error: {str(e)}")
                return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def swarm_optimization(optimization_type: str = 'global_optimization'):
    """Swarm optimization decorator."""
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Optimize swarm if problem is present
                if hasattr(request, 'json') and request.json:
                    problem = request.json.get('optimization_problem', {})
                    if problem:
                        result = ultra_swarm.optimize_swarm(optimization_type, problem)
                        kwargs['swarm_optimization'] = result
                
                return f(*args, **kwargs)
            except Exception as e:
                logger.error(f"Swarm optimization error: {str(e)}")
                return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def swarm_communication(communication_type: str = 'pheromone'):
    """Swarm communication decorator."""
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Communicate with swarm if message is present
                if hasattr(request, 'json') and request.json:
                    message = request.json.get('swarm_message', {})
                    if message:
                        result = ultra_swarm.communicate_swarm(communication_type, message)
                        kwargs['swarm_communication'] = result
                
                return f(*args, **kwargs)
            except Exception as e:
                logger.error(f"Swarm communication error: {str(e)}")
                return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def swarm_learning(learning_type: str = 'collective_learning'):
    """Swarm learning decorator."""
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # Learn with swarm if learning data is present
                if hasattr(request, 'json') and request.json:
                    learning_data = request.json.get('learning_data', {})
                    if learning_data:
                        result = ultra_swarm.learn_swarm(learning_type, learning_data)
                        kwargs['swarm_learning'] = result
                
                return f(*args, **kwargs)
            except Exception as e:
                logger.error(f"Swarm learning error: {str(e)}")
                return f(*args, **kwargs)
        
        return decorated_function
    return decorator









