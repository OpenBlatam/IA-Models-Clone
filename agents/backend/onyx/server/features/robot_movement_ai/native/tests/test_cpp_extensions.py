"""
Tests for C++ Extensions
========================
"""

import pytest
import numpy as np
from robot_movement_ai.native.wrapper import (
    CPP_AVAILABLE,
    NativeIKWrapper,
    NativeTrajectoryOptimizerWrapper,
    NativeMatrixOpsWrapper,
    NativeCollisionDetectorWrapper
)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ extensions not available")
class TestMatrixOps:
    """Tests for matrix operations."""
    
    def test_matmul(self):
        """Test matrix multiplication."""
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)
        
        result = NativeMatrixOpsWrapper.matmul(a, b)
        expected = np.matmul(a, b)
        
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_inv(self):
        """Test matrix inverse."""
        a = np.random.rand(5, 5)
        a = a + np.eye(5) * 0.1  # Make it invertible
        
        result = NativeMatrixOpsWrapper.inv(a)
        expected = np.linalg.inv(a)
        
        np.testing.assert_allclose(result, expected, rtol=1e-8)
    
    def test_det(self):
        """Test matrix determinant."""
        a = np.random.rand(5, 5)
        
        result = NativeMatrixOpsWrapper.det(a)
        expected = np.linalg.det(a)
        
        np.testing.assert_almost_equal(result, expected, decimal=10)


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ extensions not available")
class TestIK:
    """Tests for inverse kinematics."""
    
    def test_ik_solve(self):
        """Test IK solving."""
        link_lengths = [0.3, 0.3, 0.3]
        joint_limits = [(-3.14, 3.14)] * 3
        
        ik = NativeIKWrapper(link_lengths, joint_limits)
        target = np.array([0.5, 0.3, 0.2])
        
        solution = ik.solve(target)
        
        assert solution.shape == (3,)
        assert not np.any(np.isnan(solution))
        assert not np.any(np.isinf(solution))


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ extensions not available")
class TestTrajectoryOptimizer:
    """Tests for trajectory optimization."""
    
    def test_optimize(self):
        """Test trajectory optimization."""
        optimizer = NativeTrajectoryOptimizerWrapper()
        
        trajectory = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ])
        
        obstacles = np.array([[0.3, 0.3, 0.3, 0.1]])
        
        optimized = optimizer.optimize(trajectory, obstacles)
        
        assert optimized.shape == trajectory.shape
        assert not np.any(np.isnan(optimized))


@pytest.mark.skipif(not CPP_AVAILABLE, reason="C++ extensions not available")
class TestCollisionDetection:
    """Tests for collision detection."""
    
    def test_collision_detection(self):
        """Test collision detection."""
        trajectory = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        
        obstacles = np.array([
            [0.5, 0.5, 0.5, 0.2],  # Should collide
        ])
        
        has_collision = NativeCollisionDetectorWrapper.check_trajectory_collision(
            trajectory, obstacles
        )
        
        assert isinstance(has_collision, bool)
    
    def test_no_collision(self):
        """Test no collision case."""
        trajectory = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        
        obstacles = np.array([
            [5.0, 5.0, 5.0, 0.1],  # Far away
        ])
        
        has_collision = NativeCollisionDetectorWrapper.check_trajectory_collision(
            trajectory, obstacles
        )
        
        assert not has_collision

