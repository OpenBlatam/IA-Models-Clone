"""
Paper 2305.20050 — MCTS-based Reward-Guided Tree Search (Yao et al., 2023 + Feng et al., 2024)
'Tree of Thoughts' + 'AlphaCode-style MCTS for LLM Reasoning'

Combines:
  - Tree of Thoughts (Yao et al., 2023): Deliberate search over reasoning paths
  - MCTS for LLMs (Feng et al., 2024): Monte-Carlo Tree Search with value estimation
  - Process Reward Models (Lightman et al., 2023): Step-level verification

Core Algorithm:
  1. Generate multiple reasoning "branches" from the current state
  2. Evaluate each branch with a reward/value heuristic
  3. Select the best branch (UCB1 exploration-exploitation)
  4. Expand and continue until terminal state
  5. Backpropagate rewards

This is the foundational technique behind DeepSeek-R1, OpenAI o1/o3, 
and all frontier reasoning models.
"""

from __future__ import annotations
import math
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ThoughtNode:
    """A node in the reasoning tree."""
    content: str
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    depth: int = 0
    is_terminal: bool = False

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(self.visits, 1)

    @property
    def ucb1(self) -> float:
        """Upper Confidence Bound for exploration-exploitation balance."""
        if self.visits == 0:
            return float('inf')
        if self.parent is None or self.parent.visits == 0:
            return self.avg_reward
        exploration = math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return self.avg_reward + 1.414 * exploration


class RewardEstimator:
    """
    Process Reward Model (PRM) approximation.

    Uses heuristic signals to estimate the quality of intermediate
    reasoning steps without requiring a trained reward model.

    Based on Lightman et al., 2023: "Let's Verify Step by Step"
    """

    # Quality signal weights
    WEIGHTS = {
        "coherence": 0.25,
        "specificity": 0.20,
        "progress": 0.25,
        "correctness_signals": 0.15,
        "conciseness": 0.15,
    }

    @staticmethod
    def estimate(thought: str, prompt: str, depth: int = 0) -> float:
        """
        Estimate the reward for a reasoning step.
        Returns a score in [0, 1].
        """
        if not thought or len(thought.strip()) < 3:
            return 0.0

        score = 0.0
        import re

        # 1. Coherence: Does the thought flow logically?
        coherence_markers = [
            "therefore", "because", "since", "given that",
            "first", "next", "then", "finally",
            "we can", "this means", "so",
        ]
        coherence_hits = sum(1 for m in coherence_markers if m in thought.lower())
        coherence = min(1.0, coherence_hits / 3.0)
        score += coherence * RewardEstimator.WEIGHTS["coherence"]

        # 2. Specificity: Contains concrete details, numbers, code?
        has_numbers = bool(re.search(r'\d+\.?\d*', thought))
        has_code = "```" in thought or "def " in thought or "import " in thought
        has_quotes = '"' in thought or "'" in thought
        specificity = (0.4 * has_numbers + 0.4 * has_code + 0.2 * has_quotes)
        score += specificity * RewardEstimator.WEIGHTS["specificity"]

        # 3. Progress: Is this step advancing toward a solution?
        progress_markers = [
            "answer", "solution", "result", "conclusion",
            "found", "determined", "calculated", "verified",
        ]
        progress_hits = sum(1 for m in progress_markers if m in thought.lower())
        progress = min(1.0, progress_hits / 2.0)
        # Deeper nodes with progress signals are more valuable
        depth_bonus = min(0.3, depth * 0.05)
        score += (progress + depth_bonus) * RewardEstimator.WEIGHTS["progress"]

        # 4. Correctness signals: No self-contradiction or uncertainty
        uncertainty_markers = [
            "i'm not sure", "maybe", "possibly", "i think",
            "uncertain", "don't know", "cannot determine",
        ]
        uncertainty_hits = sum(1 for m in uncertainty_markers if m in thought.lower())
        correctness = max(0.0, 1.0 - uncertainty_hits * 0.3)
        score += correctness * RewardEstimator.WEIGHTS["correctness_signals"]

        # 5. Conciseness: Reward focused reasoning, penalize rambling
        word_count = len(thought.split())
        if 20 <= word_count <= 200:
            conciseness = 1.0
        elif word_count < 20:
            conciseness = word_count / 20.0
        else:
            conciseness = max(0.3, 1.0 - (word_count - 200) / 500.0)
        score += conciseness * RewardEstimator.WEIGHTS["conciseness"]

        return max(0.0, min(1.0, score))


class MCTSReasoner:
    """
    Monte-Carlo Tree Search for LLM Reasoning.

    Implements a simplified MCTS loop adapted for API-based LLM inference:
    1. SELECT: Choose the most promising node (UCB1)
    2. EXPAND: Generate new reasoning branches via LLM
    3. EVALUATE: Score each branch with the reward estimator
    4. BACKPROPAGATE: Update ancestor rewards

    This is computationally expensive (multiple LLM calls) but produces
    significantly higher quality reasoning for complex problems.
    """

    def __init__(
        self,
        max_iterations: int = 8,
        max_depth: int = 5,
        branching_factor: int = 3,
        min_reward_threshold: float = 0.4,
    ):
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.min_reward_threshold = min_reward_threshold
        self.reward_estimator = RewardEstimator()

    # ── MCTS Phases ────────────────────────────────────────────

    def select(self, node: ThoughtNode) -> ThoughtNode:
        """Select the most promising leaf node using UCB1."""
        current = node
        while current.children and not current.is_terminal:
            current = max(current.children, key=lambda n: n.ucb1)
        return current

    async def expand(
        self,
        node: ThoughtNode,
        prompt: str,
        llm_func: Callable,
        **kwargs,
    ) -> List[ThoughtNode]:
        """Generate new reasoning branches from a node."""
        if node.depth >= self.max_depth:
            node.is_terminal = True
            return []

        # Build expansion prompt
        path = self._get_path_text(node)
        expansion_prompt = (
            f"{prompt}\n\n"
            f"Current reasoning so far:\n{path}\n\n"
            f"Generate the next step of reasoning. Be specific and concise. "
            f"If you have enough information, provide the final answer."
        )

        children = []
        for i in range(self.branching_factor):
            try:
                response = await llm_func(
                    expansion_prompt,
                    temperature=0.7 + (i * 0.1),  # Vary temperature for diversity
                    **kwargs,
                )
                child = ThoughtNode(
                    content=str(response),
                    parent=node,
                    depth=node.depth + 1,
                )
                # Check if this is a terminal (final answer) node
                response_lower = str(response).lower()
                if any(marker in response_lower for marker in [
                    "final answer", "therefore the answer", "in conclusion",
                    "the result is", "solution:", "answer:"
                ]):
                    child.is_terminal = True
                children.append(child)
            except Exception as e:
                logger.warning("MCTS expansion branch %d failed: %s", i, e)
                continue

        node.children = children
        return children

    def evaluate(self, node: ThoughtNode, prompt: str) -> float:
        """Evaluate a node using the reward estimator."""
        return self.reward_estimator.estimate(node.content, prompt, node.depth)

    def backpropagate(self, node: ThoughtNode, reward: float) -> None:
        """Update rewards up the tree."""
        current: Optional[ThoughtNode] = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    # ── Main Search Loop ───────────────────────────────────────

    async def search(
        self,
        prompt: str,
        llm_func: Callable,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute the full MCTS reasoning search.

        Returns the best reasoning path found.
        """
        t0 = time.time()
        root = ThoughtNode(content=prompt, depth=0)

        best_path: Optional[ThoughtNode] = None
        best_reward = 0.0

        for iteration in range(self.max_iterations):
            # SELECT
            leaf = self.select(root)

            # EXPAND
            if not leaf.is_terminal and leaf.depth < self.max_depth:
                children = await self.expand(leaf, prompt, llm_func, **kwargs)
                if not children:
                    continue
                leaf = children[0]  # Evaluate first child

            # EVALUATE
            reward = self.evaluate(leaf, prompt)

            # BACKPROPAGATE
            self.backpropagate(leaf, reward)

            # Track best
            if reward > best_reward:
                best_reward = reward
                best_path = leaf

            logger.info(
                "MCTS iter %d/%d: depth=%d, reward=%.3f, best=%.3f",
                iteration + 1, self.max_iterations, leaf.depth, reward, best_reward,
            )

            # Early exit if we found a high-quality terminal node
            if leaf.is_terminal and reward > 0.8:
                break

        elapsed = time.time() - t0

        if best_path is None:
            return {
                "best_path": prompt,
                "best_reward": 0.0,
                "iterations": 0,
                "elapsed": elapsed,
            }

        return {
            "best_path": self._get_path_text(best_path),
            "best_answer": best_path.content,
            "best_reward": best_reward,
            "iterations": root.visits,
            "tree_depth": best_path.depth,
            "elapsed": elapsed,
        }

    # ── Utilities ──────────────────────────────────────────────

    def _get_path_text(self, node: ThoughtNode) -> str:
        """Reconstruct the full reasoning path from root to this node."""
        path = []
        current: Optional[ThoughtNode] = node
        while current is not None:
            if current.depth > 0:  # Skip root (it's just the prompt)
                path.append(f"Step {current.depth}: {current.content}")
            current = current.parent
        path.reverse()
        return "\n".join(path) if path else ""
