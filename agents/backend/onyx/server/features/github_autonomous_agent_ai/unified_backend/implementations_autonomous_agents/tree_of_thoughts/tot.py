"""
Tree of Thoughts (ToT) Implementation
======================================

Paper: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
arXiv: 2305.10601

ToT maintains a tree of thoughts where:
- Each node is a state representing a partial solution
- Each edge represents a thought (intermediate step)
- The agent explores multiple paths and evaluates states
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import heapq


class ToTSearchStrategy(Enum):
    """Search strategies for ToT."""
    BFS = "breadth_first"  # Breadth-first search
    DFS = "depth_first"    # Depth-first search


@dataclass
class ToTNode:
    """
    Represents a node in the Tree of Thoughts.
    
    Each node contains:
    - state: Current partial solution state
    - thought: The thought that led to this state
    - value: Heuristic value (higher = more promising)
    - parent: Parent node
    - children: Child nodes
    """
    state: str
    thought: str
    value: float = 0.0
    parent: Optional['ToTNode'] = None
    children: List['ToTNode'] = field(default_factory=list)
    depth: int = 0
    
    def __lt__(self, other):
        """For priority queue (higher value = higher priority)."""
        return self.value > other.value


class TreeOfThoughts:
    """
    Tree of Thoughts framework for deliberate problem solving.
    
    The framework involves:
    1. Thought decomposition: Break problem into intermediate steps
    2. Thought generation: Generate k candidate thoughts from current state
    3. State evaluation: Evaluate states using heuristics
    4. Search algorithm: Explore tree using BFS or DFS
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        thought_generator: Optional[Callable] = None,
        state_evaluator: Optional[Callable] = None,
        search_strategy: ToTSearchStrategy = ToTSearchStrategy.BFS,
        max_depth: int = 5,
        beam_width: int = 5,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Tree of Thoughts.
        
        Args:
            llm: Language model for thought generation/evaluation
            thought_generator: Function to generate candidate thoughts
            state_evaluator: Function to evaluate state values
            search_strategy: BFS or DFS
            max_depth: Maximum tree depth
            beam_width: Number of states to keep per level (BFS)
            config: Additional configuration
        """
        self.llm = llm
        self.thought_generator = thought_generator or self._default_thought_generator
        self.state_evaluator = state_evaluator or self._default_state_evaluator
        self.search_strategy = search_strategy
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.config = config or {}
        
        self.root: Optional[ToTNode] = None
        self.all_nodes: List[ToTNode] = []
    
    def solve(
        self,
        problem: str,
        initial_state: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solve a problem using Tree of Thoughts.
        
        Args:
            problem: Problem description
            initial_state: Initial state (defaults to problem)
            
        Returns:
            Solution dictionary with best path and final state
        """
        initial_state = initial_state or problem
        self.root = ToTNode(state=initial_state, thought="Initial state", depth=0)
        self.all_nodes = [self.root]
        
        if self.search_strategy == ToTSearchStrategy.BFS:
            return self._bfs_search(problem)
        else:
            return self._dfs_search(problem)
    
    def _bfs_search(self, problem: str) -> Dict[str, Any]:
        """
        Breadth-first search through tree of thoughts.
        
        Maintains beam_width most promising states at each level.
        """
        current_level = [self.root]
        
        for depth in range(1, self.max_depth + 1):
            next_level = []
            
            # Generate thoughts for all nodes in current level
            for node in current_level:
                # Generate candidate thoughts
                thoughts = self.thought_generator(node.state, problem, k=self.beam_width)
                
                # Create child nodes
                for thought in thoughts:
                    new_state = self._apply_thought(node.state, thought)
                    child = ToTNode(
                        state=new_state,
                        thought=thought,
                        parent=node,
                        depth=depth
                    )
                    node.children.append(child)
                    next_level.append(child)
                    self.all_nodes.append(child)
                
                # Evaluate children
                if node.children:
                    values = self.state_evaluator([c.state for c in node.children], problem)
                    for child, value in zip(node.children, values):
                        child.value = value
            
            # Keep only beam_width best nodes
            next_level.sort(key=lambda n: n.value, reverse=True)
            current_level = next_level[:self.beam_width]
            
            # Check if we found a solution
            best_node = max(current_level, key=lambda n: n.value) if current_level else None
            if best_node and self._is_solution(best_node.state, problem):
                return self._extract_solution(best_node)
        
        # Return best node found
        best_node = max(self.all_nodes, key=lambda n: n.value)
        return self._extract_solution(best_node)
    
    def _dfs_search(self, problem: str) -> Dict[str, Any]:
        """
        Depth-first search through tree of thoughts.
        
        Explores most promising path first, backtracks when needed.
        """
        best_solution = None
        best_value = float('-inf')
        
        def dfs(node: ToTNode, depth: int):
            nonlocal best_solution, best_value
            
            if depth >= self.max_depth:
                return
            
            # Generate candidate thoughts
            thoughts = self.thought_generator(node.state, problem, k=self.beam_width)
            
            # Create and evaluate children
            children = []
            for thought in thoughts:
                new_state = self._apply_thought(node.state, thought)
                child = ToTNode(
                    state=new_state,
                    thought=thought,
                    parent=node,
                    depth=depth + 1
                )
                child.value = self.state_evaluator([child.state], problem)[0]
                node.children.append(child)
                children.append(child)
                self.all_nodes.append(child)
            
            # Sort by value (most promising first)
            children.sort(key=lambda n: n.value, reverse=True)
            
            # Explore children
            for child in children:
                if child.value > best_value:
                    best_value = child.value
                    best_solution = child
                
                if self._is_solution(child.state, problem):
                    return child
                
                # Prune if value too low
                if child.value < -10:  # Threshold
                    continue
                
                result = dfs(child, depth + 1)
                if result:
                    return result
            
            return None
        
        solution_node = dfs(self.root, 0)
        if solution_node:
            return self._extract_solution(solution_node)
        elif best_solution:
            return self._extract_solution(best_solution)
        else:
            return self._extract_solution(self.root)
    
    def _default_thought_generator(self, state: str, problem: str, k: int = 5) -> List[str]:
        """
        Default thought generator.
        
        Generates k candidate thoughts from current state.
        """
        if self.llm:
            # Use LLM to generate thoughts
            prompt = f"""Problem: {problem}
Current state: {state}

Generate {k} different thoughts (intermediate steps) to progress toward solving the problem.
Each thought should be a coherent step that builds on the current state.

Thoughts:"""
            # In production, call LLM here
            thoughts = [f"Thought {i+1}: Consider {state} from perspective {i+1}" for i in range(k)]
        else:
            # Simple template-based generation
            thoughts = [
                f"Analyze {state}",
                f"Break down {state} into components",
                f"Consider alternative approach to {state}",
                f"Refine {state}",
                f"Evaluate {state}"
            ][:k]
        
        return thoughts
    
    def _default_state_evaluator(self, states: List[str], problem: str) -> List[float]:
        """
        Default state evaluator.
        
        Evaluates states and returns heuristic values (higher = more promising).
        """
        if self.llm:
            # Use LLM to evaluate states
            prompt = f"""Problem: {problem}

Evaluate these states on a scale of 1-10 based on how promising they are for solving the problem.
States:
{chr(10).join(f"{i+1}. {state}" for i, state in enumerate(states))}

Provide scores:"""
            # In production, call LLM here
            values = [5.0 + (i % 3) for i in range(len(states))]  # Placeholder
        else:
            # Simple heuristic: longer states might be more complete
            values = [len(state) / 100.0 for state in states]
        
        return values
    
    def _apply_thought(self, state: str, thought: str) -> str:
        """Apply a thought to a state, producing a new state."""
        # Simple concatenation (can be customized)
        return f"{state}\n{thought}"
    
    def _is_solution(self, state: str, problem: str) -> bool:
        """Check if state represents a solution."""
        # Simple heuristic: check for completion keywords
        completion_keywords = ["solution", "answer", "complete", "solved", "result"]
        return any(keyword in state.lower() for keyword in completion_keywords)
    
    def _extract_solution(self, node: ToTNode) -> Dict[str, Any]:
        """Extract solution path from root to node."""
        path = []
        current = node
        
        while current:
            path.insert(0, {
                "thought": current.thought,
                "state": current.state,
                "value": current.value,
                "depth": current.depth
            })
            current = current.parent
        
        return {
            "solution_state": node.state,
            "final_value": node.value,
            "path": path,
            "path_length": len(path),
            "total_nodes_explored": len(self.all_nodes)
        }
    
    def get_tree_structure(self) -> Dict[str, Any]:
        """Get tree structure for visualization."""
        def node_to_dict(node: ToTNode) -> Dict[str, Any]:
            return {
                "state": node.state[:50] + "..." if len(node.state) > 50 else node.state,
                "thought": node.thought,
                "value": node.value,
                "depth": node.depth,
                "children": [node_to_dict(child) for child in node.children]
            }
        
        if self.root:
            return node_to_dict(self.root)
        return {}



