
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath("c:/blatam-academy"))

from agents.backend.onyx.server.features.robot_movement_ai.core.algorithms.sac_algorithm import SACAlgorithm
from agents.backend.onyx.server.features.robot_movement_ai.core.algorithms.td3_algorithm import TD3Algorithm
from agents.backend.onyx.server.features.robot_movement_ai.core.types.types import TrajectoryPoint

def test_sac_instantiation():
    print("Testing SAC Instantiation...")
    sac = SACAlgorithm()
    assert sac.name == "SAC"
    print("SAC Instantiation: PASSED")

def test_td3_instantiation():
    print("Testing TD3 Instantiation...")
    td3 = TD3Algorithm()
    assert td3.name == "TD3"
    print("TD3 Instantiation: PASSED")

def test_sac_optimization():
    print("Testing SAC Optimization...")
    sac = SACAlgorithm()
    start = TrajectoryPoint(position=np.array([0.0, 0.0, 0.0]), orientation=np.array([0.0, 0.0, 0.0, 1.0]))
    goal = TrajectoryPoint(position=np.array([1.0, 1.0, 1.0]), orientation=np.array([0.0, 0.0, 0.0, 1.0]))
    
    trajectory = sac.optimize(start, goal)
    assert len(trajectory) > 0
    assert np.allclose(trajectory[0].position, start.position)
    assert np.allclose(trajectory[-1].position, goal.position)
    print("SAC Optimization: PASSED")

def test_td3_optimization():
    print("Testing TD3 Optimization...")
    td3 = TD3Algorithm()
    start = TrajectoryPoint(position=np.array([0.0, 0.0, 0.0]), orientation=np.array([0.0, 0.0, 0.0, 1.0]))
    goal = TrajectoryPoint(position=np.array([1.0, 1.0, 1.0]), orientation=np.array([0.0, 0.0, 0.0, 1.0]))
    
    trajectory = td3.optimize(start, goal)
    assert len(trajectory) > 0
    assert np.allclose(trajectory[0].position, start.position)
    assert np.allclose(trajectory[-1].position, goal.position)
    print("TD3 Optimization: PASSED")

if __name__ == "__main__":
    try:
        test_sac_instantiation()
        test_td3_instantiation()
        test_sac_optimization()
        test_td3_optimization()
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
