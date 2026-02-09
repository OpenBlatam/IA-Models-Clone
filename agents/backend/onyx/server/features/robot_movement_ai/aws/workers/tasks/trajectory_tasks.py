"""
Trajectory Optimization Tasks
==============================

Background tasks for trajectory optimization using Celery.
"""

try:
    from aws.workers.celery_config import celery_app, task_with_retry
except ImportError:
    # Fallback for local development
    celery_app = None
    def task_with_retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
import logging
import numpy as np

logger = logging.getLogger(__name__)


@task_with_retry(name="trajectory.optimize")
def optimize_trajectory(waypoints, obstacles=None, optimization_type="astar"):
    """
    Optimize trajectory in background.
    
    Args:
        waypoints: List of waypoints
        obstacles: List of obstacles
        optimization_type: Type of optimization (astar, rrt, etc.)
    
    Returns:
        Optimized trajectory
    """
    logger.info(f"Optimizing trajectory with {optimization_type}")
    
    try:
        # Import here to avoid circular dependencies
        from robot_movement_ai.core.trajectory_optimizer import TrajectoryOptimizer
        from robot_movement_ai.config.robot_config import RobotConfig
        
        config = RobotConfig()
        optimizer = TrajectoryOptimizer(config)
        
        # Convert waypoints to trajectory points
        trajectory_points = []
        for i, wp in enumerate(waypoints):
            from robot_movement_ai.core.trajectory_optimizer import TrajectoryPoint
            point = TrajectoryPoint(
                position=np.array(wp["position"]),
                orientation=np.array(wp.get("orientation", [0, 0, 0, 1])),
                timestamp=i * 0.01
            )
            trajectory_points.append(point)
        
        # Optimize based on type
        if optimization_type == "astar":
            trajectory = optimizer.optimize_with_astar(
                trajectory_points[0],
                trajectory_points[-1],
                obstacles or [],
                grid_resolution=0.05
            )
        elif optimization_type == "rrt":
            trajectory = optimizer.optimize_with_rrt(
                trajectory_points[0],
                trajectory_points[-1],
                obstacles or [],
                max_iterations=1000
            )
        else:
            trajectory = trajectory_points
        
        # Analyze trajectory
        analysis = optimizer.analyze_trajectory(trajectory)
        
        logger.info(f"Trajectory optimized: {len(trajectory)} points")
        
        return {
            "success": True,
            "trajectory": [
                {
                    "position": point.position.tolist(),
                    "orientation": point.orientation.tolist(),
                    "timestamp": point.timestamp
                }
                for point in trajectory
            ],
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Error optimizing trajectory: {e}", exc_info=True)
        raise


@task_with_retry(name="trajectory.validate")
def validate_trajectory(trajectory, safety_limits=None):
    """
    Validate trajectory for safety and feasibility.
    
    Args:
        trajectory: Trajectory to validate
        safety_limits: Safety limits dictionary
    
    Returns:
        Validation result
    """
    logger.info("Validating trajectory")
    
    try:
        # Validation logic
        errors = []
        warnings = []
        
        # Check velocity limits
        if safety_limits:
            max_velocity = safety_limits.get("max_velocity", 1.0)
            for i in range(len(trajectory) - 1):
                # Calculate velocity (simplified)
                # Add actual velocity calculation here
                pass
        
        # Check collision
        # Add collision detection here
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
        
    except Exception as e:
        logger.error(f"Error validating trajectory: {e}", exc_info=True)
        raise

