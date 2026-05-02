"""
Language Agent Tree Search (LATS) Implementation
=================================================

Paper: "Language Agent Tree Search Unifies Reasoning, Acting, and Planning"

LATS uses tree search with LLM-based evaluation to:
1. Generate candidate actions
2. Evaluate states using LLM
3. Search tree for optimal solution path
4. Execute actions and observe results
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from ..common.agent_base import BaseAgent, AgentStatus
from ..common.tools import ToolRegistry
from ..tree_of_thoughts.tot import ToTNode, ToTSearchStrategy
import heapq


@dataclass
class LATSTreeNode:
    """Node in LATS tree."""
    state: str
    action: Optional[str] = None
    observation: Optional[str] = None
    value: float = 0.0
    parent: Optional['LATSTreeNode'] = None
    children: List['LATSTreeNode'] = field(default_factory=list)
    depth: int = 0
    visited: bool = False
    
    def __lt__(self, other):
        return self.value > other.value


class LATSTree:
    """Tree structure for LATS search."""
    
    def __init__(self, root_state: str):
        """Initialize tree with root state."""
        self.root = LATSTreeNode(state=root_state, depth=0)
        self.all_nodes: List[LATSTreeNode] = [self.root]
        self.leaf_nodes: List[LATSTreeNode] = [self.root]
    
    def add_node(self, parent: LATSTreeNode, state: str, action: str, observation: str) -> LATSTreeNode:
        """Add a new node to the tree."""
        node = LATSTreeNode(
            state=state,
            action=action,
            observation=observation,
            parent=parent,
            depth=parent.depth + 1
        )
        parent.children.append(node)
        self.all_nodes.append(node)
        
        # Update leaf nodes
        if parent in self.leaf_nodes:
            self.leaf_nodes.remove(parent)
        self.leaf_nodes.append(node)
        
        return node
    
    def get_best_path(self) -> List[LATSTreeNode]:
        """Get best path from root to best leaf."""
        best_leaf = max(self.leaf_nodes, key=lambda n: n.value)
        path = []
        current = best_leaf
        
        while current:
            path.insert(0, current)
            current = current.parent
        
        return path


class LATSAgent(BaseAgent):
    """
    Language Agent Tree Search agent.
    
    Combines:
    - Tree search for exploration
    - LLM for state evaluation
    - Tool execution for actions
    - Observation processing
    """
    
    def __init__(
        self,
        name: str = "LATSAgent",
        llm: Optional[Any] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_depth: int = 5,
        beam_width: int = 3,
        max_iterations: int = 20,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LATS agent.
        
        Args:
            name: Agent name
            llm: Language model
            tool_registry: Available tools
            max_depth: Maximum search depth
            beam_width: Number of paths to explore
            max_iterations: Maximum search iterations
            config: Additional configuration
        """
        super().__init__(name=name, llm=llm, config=config)
        self.tool_registry = tool_registry
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.max_iterations = max_iterations
        self.tree: Optional[LATSTree] = None
    
    def think(self, observation: str, context: Optional[Dict] = None) -> str:
        """
        Generate reasoning about current state.
        
        In LATS, thinking involves evaluating the state
        and determining promising actions.
        """
        if self.llm:
            prompt = f"""Current state: {observation}

Evaluate this state and suggest the most promising next actions.
Consider:
1. How close is this to solving the problem?
2. What actions would be most helpful?
3. What are the risks/benefits of different actions?

Reasoning:"""
            reasoning = self._call_llm(prompt)
        else:
            reasoning = f"Evaluating state: {observation}. Considering next actions."
        
        return reasoning
    
    def act(self, thought: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate and execute action.
        
        In LATS, actions are generated based on tree search
        and executed using tools.
        """
        # Extract action from thought
        action_info = self._extract_action(thought, context)
        
        if not action_info:
            return {
                "action": "finish",
                "tool": None,
                "parameters": {},
                "result": "No action needed",
                "complete": True
            }
        
        tool_name = action_info.get("tool")
        parameters = action_info.get("parameters", {})
        
        # Execute tool
        if self.tool_registry and tool_name:
            result = self.tool_registry.execute(tool_name, **parameters)
        else:
            result = {
                "success": False,
                "error": f"Tool '{tool_name}' not available",
                "result": None
            }
        
        return {
            "action": "tool_call",
            "tool": tool_name,
            "parameters": parameters,
            "result": result,
            "complete": False
        }
    
    def observe(self, action_result: Dict[str, Any]) -> str:
        """Generate observation from action result."""
        if action_result.get("complete"):
            return "Task completed."
        
        tool = action_result.get("tool")
        result = action_result.get("result", {})
        
        if result.get("success"):
            return f"Action '{tool}' succeeded: {result.get('result')}"
        else:
            return f"Action '{tool}' failed: {result.get('error')}"
    
    def solve(self, task: str) -> Dict[str, Any]:
        """
        Solve a task using LATS tree search.
        
        Args:
            task: Task description
            
        Returns:
            Solution with best path
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        self.state.add_step("task_start", {"task": task})
        
        # Initialize tree
        self.tree = LATSTree(root_state=task)
        
        # Tree search
        for iteration in range(self.max_iterations):
            # Select promising leaf nodes to expand
            candidates = self._select_candidates()
            
            if not candidates:
                break
            
            # Expand candidates
            for node in candidates:
                if node.depth >= self.max_depth:
                    continue
                
                # Generate actions from state
                actions = self._generate_actions(node.state, task)
                
                # Execute actions and create children
                for action_info in actions[:self.beam_width]:
                    action_result = self._execute_action(action_info)
                    observation = self.observe(action_result)
                    
                    # Create new state
                    new_state = self._update_state(node.state, action_info, observation)
                    
                    # Create child node
                    child = self.tree.add_node(
                        node,
                        new_state,
                        action_info.get("tool", "unknown"),
                        observation
                    )
                    
                    # Evaluate child state
                    child.value = self._evaluate_state(child.state, task)
            
            # Check for solution
            best_path = self.tree.get_best_path()
            best_leaf = best_path[-1] if best_path else None
            
            if best_leaf and self._is_solution(best_leaf.state, task):
                self.state.status = AgentStatus.COMPLETED
                return {
                    "task": task,
                    "solution": best_leaf.state,
                    "path": [n.state for n in best_path],
                    "completed": True,
                    "iterations": iteration + 1
                }
        
        # Return best found solution
        best_path = self.tree.get_best_path()
        best_leaf = best_path[-1] if best_path else self.tree.root
        
        self.state.status = AgentStatus.COMPLETED
        return {
            "task": task,
            "solution": best_leaf.state,
            "path": [n.state for n in best_path] if best_path else [task],
            "completed": self._is_solution(best_leaf.state, task),
            "iterations": self.max_iterations
        }
    
    def _select_candidates(self) -> List[LATSTreeNode]:
        """Select most promising leaf nodes to expand."""
        # Select top beam_width unvisited leaf nodes
        unvisited = [n for n in self.tree.leaf_nodes if not n.visited]
        unvisited.sort(key=lambda n: n.value, reverse=True)
        
        candidates = unvisited[:self.beam_width]
        for candidate in candidates:
            candidate.visited = True
        
        return candidates
    
    def _generate_actions(self, state: str, task: str) -> List[Dict[str, Any]]:
        """Generate candidate actions from state."""
        if self.llm:
            prompt = f"""Task: {task}
Current state: {state}

Generate {self.beam_width} different actions that could help solve the task.
Format: tool_name(parameter1=value1, parameter2=value2)

Actions:"""
            # In production, parse LLM response
            actions = [
                {"tool": "search", "parameters": {"query": task}},
                {"tool": "calculator", "parameters": {"expression": "2+2"}},
            ]
        else:
            # Simple action generation
            actions = [
                {"tool": "search", "parameters": {"query": state}},
            ]
        
        return actions
    
    def _execute_action(self, action_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action."""
        tool_name = action_info.get("tool")
        parameters = action_info.get("parameters", {})
        
        if self.tool_registry:
            return self.tool_registry.execute(tool_name, **parameters)
        else:
            return {"success": False, "error": "No tool registry", "result": None}
    
    def _update_state(self, current_state: str, action: Dict, observation: str) -> str:
        """Update state based on action and observation."""
        return f"{current_state}\nAction: {action.get('tool')}\nObservation: {observation}"
    
    def _evaluate_state(self, state: str, task: str) -> float:
        """Evaluate state value using LLM."""
        if self.llm:
            prompt = f"""Task: {task}
Current state: {state}

Rate how promising this state is for solving the task (0-10):"""
            # In production, parse LLM response
            value = 5.0
        else:
            # Simple heuristic
            value = len(state) / 100.0
        
        return value
    
    def _is_solution(self, state: str, task: str) -> bool:
        """Check if state is a solution."""
        completion_keywords = ["solution", "answer", "complete", "solved"]
        return any(keyword in state.lower() for keyword in completion_keywords)
    
    def _extract_action(self, thought: str, context: Optional[Dict]) -> Optional[Dict[str, Any]]:
        """Extract action information from thought."""
        # Simple parsing (in production, use more sophisticated parsing)
        if "search" in thought.lower():
            return {"tool": "search", "parameters": {"query": thought}}
        elif "calculate" in thought.lower():
            return {"tool": "calculator", "parameters": {"expression": "2+2"}}
        return None
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM (placeholder)."""
        return f"LLM response to: {prompt[:50]}"



