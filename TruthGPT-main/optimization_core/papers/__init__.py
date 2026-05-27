"""
TruthGPT SOTA Paper Implementations — 6 Foundational Papers.

Each module implements the exact algorithms from the referenced paper:

1. chain_of_draft      — Paper 2506.10987v1: Concise reasoning steps (≤5 words)
2. elastic_reasoning    — Paper 2505.05315v2: Dynamic think/solution budget allocation
3. fp16_stability       — Paper 2510.26788v1: FP16 numerical stability monitoring
4. self_consistency     — Paper 2203.11171:  Majority voting over sampled reasoning paths
5. speculative_decoding — Paper 2302.01318:  API-level draft/target cost optimization
6. mcts_reasoning       — Paper 2305.20050:  MCTS tree search for complex reasoning
"""

from .chain_of_draft import ChainOfDraft
from .elastic_reasoning import ElasticReasoning
from .fp16_stability import FP16Stability
from .self_consistency import SelfConsistency
from .speculative_decoding import SpeculativeDecoder
from .mcts_reasoning import MCTSReasoner, RewardEstimator, ThoughtNode

__all__ = [
    "ChainOfDraft",
    "ElasticReasoning",
    "FP16Stability",
    "SelfConsistency",
    "SpeculativeDecoder",
    "MCTSReasoner",
    "RewardEstimator",
    "ThoughtNode",
]
