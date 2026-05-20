"""
Toolformer Implementation
==========================

Paper: "Toolformer: Language Models Can Teach Themselves to Use Tools"
arXiv: 2302.04761

Key components:
1. Self-supervised API call sampling
2. API call execution
3. Loss-based filtering of useful API calls
4. Model finetuning on filtered API calls
"""

from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from ..common.agent_base import BaseAgent, AgentStatus


@dataclass
class APICall:
    """
    Represents an API call in Toolformer format.
    
    Format: <API> api_name(input) → result </API>
    """
    api_name: str
    input_text: str
    result: Optional[str] = None
    position: int = 0
    loss_reduction: float = 0.0
    
    def to_string(self, include_result: bool = True) -> str:
        """Convert API call to string format."""
        if include_result and self.result:
            return f"<API> {self.api_name}({self.input_text}) → {self.result} </API>"
        else:
            return f"<API> {self.api_name}({self.input_text}) </API>"
    
    @classmethod
    def from_string(cls, text: str) -> Optional['APICall']:
        """Parse API call from string."""
        # Pattern: <API> api_name(input) → result </API>
        pattern = r'<API>\s*(\w+)\(([^)]+)\)\s*(?:→\s*([^<]+))?\s*</API>'
        match = re.search(pattern, text)
        if match:
            return cls(
                api_name=match.group(1),
                input_text=match.group(2).strip(),
                result=match.group(3).strip() if match.group(3) else None
            )
        return None


class Toolformer:
    """
    Toolformer: Language model that learns to use tools.
    
    The model learns to:
    - Decide which APIs to call
    - When to call them
    - What arguments to pass
    - How to incorporate results
    """
    
    def __init__(
        self,
        llm: Optional[Any] = None,
        tools: Optional[Dict[str, Callable]] = None,
        sampling_threshold: float = 0.05,
        filtering_threshold: float = 1.0,
        max_api_calls_per_text: int = 5,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Toolformer.
        
        Args:
            llm: Language model
            tools: Dictionary of available tools (name -> function)
            sampling_threshold: Threshold for sampling API calls
            filtering_threshold: Threshold for filtering useful API calls
            max_api_calls_per_text: Maximum API calls per text
            config: Additional configuration
        """
        self.llm = llm
        self.tools = tools or {}
        self.sampling_threshold = sampling_threshold
        self.filtering_threshold = filtering_threshold
        self.max_api_calls_per_text = max_api_calls_per_text
        self.config = config or {}
        
        # Special tokens
        self.API_START = "<API>"
        self.API_END = "</API>"
        self.API_RESULT = "→"
    
    def sample_api_calls(self, text: str, tool_name: str) -> List[APICall]:
        """
        Sample potential API calls for a given text and tool.
        
        Uses in-context learning to generate API call candidates.
        """
        if not self.llm:
            return []
        
        # Build prompt for API call generation
        prompt = self._build_sampling_prompt(text, tool_name)
        
        # Generate API calls using LLM
        # In production, call LLM here
        api_calls = self._generate_api_calls_from_prompt(prompt, tool_name)
        
        return api_calls[:self.max_api_calls_per_text]
    
    def execute_api_call(self, api_call: APICall) -> str:
        """
        Execute an API call and return result.
        
        Args:
            api_call: API call to execute
            
        Returns:
            Result as string
        """
        tool_func = self.tools.get(api_call.api_name)
        if not tool_func:
            return f"Error: Tool '{api_call.api_name}' not found"
        
        try:
            # Execute tool
            result = tool_func(api_call.input_text)
            api_call.result = str(result)
            return str(result)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            api_call.result = error_msg
            return error_msg
    
    def filter_api_calls(
        self,
        text: str,
        api_calls: List[APICall],
        llm: Optional[Any] = None
    ) -> List[APICall]:
        """
        Filter API calls based on loss reduction.
        
        Keeps only API calls that reduce the loss by at least filtering_threshold.
        """
        filtered = []
        
        for api_call in api_calls:
            # Execute API call if not already executed
            if not api_call.result:
                self.execute_api_call(api_call)
            
            # Calculate loss reduction
            loss_reduction = self._calculate_loss_reduction(text, api_call, llm)
            api_call.loss_reduction = loss_reduction
            
            # Keep if loss reduction is significant
            if loss_reduction >= self.filtering_threshold:
                filtered.append(api_call)
        
        return filtered
    
    def augment_text(self, text: str, api_calls: List[APICall]) -> str:
        """
        Augment text with API calls.
        
        Inserts API calls at appropriate positions in the text.
        """
        if not api_calls:
            return text
        
        # Sort API calls by position
        sorted_calls = sorted(api_calls, key=lambda x: x.position)
        
        # Build augmented text
        result = []
        last_pos = 0
        
        for api_call in sorted_calls:
            # Add text before API call
            if api_call.position > last_pos:
                result.append(text[last_pos:api_call.position])
            
            # Add API call
            result.append(api_call.to_string(include_result=True))
            last_pos = api_call.position
        
        # Add remaining text
        if last_pos < len(text):
            result.append(text[last_pos:])
        
        return "".join(result)
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """
        Generate text with tool use.
        
        During generation, the model can decide to call APIs.
        When it generates the API_RESULT token, we interrupt,
        execute the API call, and continue generation.
        """
        if not self.llm:
            return prompt
        
        generated = prompt
        api_call_buffer = ""
        in_api_call = False
        
        for _ in range(max_tokens):
            # Generate next token
            next_token = self._generate_next_token(generated)
            
            if next_token == self.API_START:
                in_api_call = True
                api_call_buffer = self.API_START
            elif in_api_call:
                api_call_buffer += next_token
                
                if next_token == self.API_END:
                    # Parse and execute API call
                    api_call = APICall.from_string(api_call_buffer)
                    if api_call:
                        result = self.execute_api_call(api_call)
                        generated += api_call_buffer + " " + result + " "
                    else:
                        generated += api_call_buffer
                    
                    api_call_buffer = ""
                    in_api_call = False
                elif next_token == self.API_RESULT:
                    # API call complete, need to execute
                    # Extract API call without result
                    api_call = APICall.from_string(api_call_buffer.replace(self.API_RESULT, ""))
                    if api_call:
                        result = self.execute_api_call(api_call)
                        generated += api_call_buffer + " " + result + " " + self.API_END
                        api_call_buffer = ""
                        in_api_call = False
            else:
                generated += next_token
                
                # Check for completion
                if self._is_complete(generated):
                    break
        
        return generated
    
    def _build_sampling_prompt(self, text: str, tool_name: str) -> str:
        """Build prompt for sampling API calls."""
        tool_examples = self._get_tool_examples(tool_name)
        
        prompt = f"""Your task is to add calls to a {tool_name} API to a piece of text.
The calls should help you get information required to complete the text.
You can call the API by writing "[{tool_name}(input)]" where "input" is the input.

Here are some examples:
{tool_examples}

Input: {text}
Output:"""
        
        return prompt
    
    def _get_tool_examples(self, tool_name: str) -> str:
        """Get example API calls for a tool."""
        examples = {
            "QA": """Input: Joe Biden was born in Scranton, Pennsylvania.
Output: Joe Biden was born in [QA("Where was Joe Biden born?")] Scranton, [QA("In which state is Scranton?")] Pennsylvania.""",
            
            "Calculator": """Input: The number in the next term is 18 + 12 x 3 = 54.
Output: The number in the next term is 18 + 12 x 3 = [Calculator(18 + 12 * 3)] 54.""",
            
            "WikiSearch": """Input: The colors on the flag of Ghana have the following meanings.
Output: The colors on the flag of Ghana [WikiSearch("Ghana flag colors")] have the following meanings.""",
            
            "MT": """Input: He has published one book: O homem suprimido
Output: He has published one book: O homem suprimido [MT(O homem suprimido)] ("The Suppressed Man")""",
            
            "Calendar": """Input: Today is the first Friday of the year.
Output: Today is the first [Calendar()] Friday of the year."""
        }
        
        return examples.get(tool_name, "")
    
    def _generate_api_calls_from_prompt(self, prompt: str, tool_name: str) -> List[APICall]:
        """Generate API calls from prompt (placeholder)."""
        # In production, use LLM to generate
        # For now, return empty list
        return []
    
    def _calculate_loss_reduction(
        self,
        text: str,
        api_call: APICall,
        llm: Optional[Any] = None
    ) -> float:
        """
        Calculate loss reduction from using API call.
        
        Compares loss with and without API call result.
        """
        # Simplified calculation (in production, use actual LLM loss)
        # Higher loss reduction = more useful API call
        
        # Simple heuristic: longer results might be more useful
        if api_call.result and len(api_call.result) > 10:
            return 2.0
        elif api_call.result:
            return 1.0
        else:
            return 0.0
    
    def _generate_next_token(self, text: str) -> str:
        """Generate next token (placeholder)."""
        # In production, use LLM
        return " "
    
    def _is_complete(self, text: str) -> bool:
        """Check if generation is complete."""
        return text.endswith(".") or len(text) > 1000


class ToolformerTrainer:
    """
    Trainer for Toolformer.
    
    Handles the self-supervised training process:
    1. Sample API calls from dataset
    2. Execute API calls
    3. Filter based on loss reduction
    4. Augment dataset
    5. Finetune model
    """
    
    def __init__(
        self,
        toolformer: Toolformer,
        dataset: List[str],
        tools: Dict[str, Callable]
    ):
        """
        Initialize trainer.
        
        Args:
            toolformer: Toolformer instance
            dataset: List of text examples
            tools: Available tools
        """
        self.toolformer = toolformer
        self.dataset = dataset
        self.tools = tools
        self.augmented_dataset: List[str] = []
    
    def train(
        self,
        epochs: int = 1,
        tools_to_use: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train Toolformer on dataset.
        
        Args:
            epochs: Number of training epochs
            tools_to_use: List of tool names to use (None = all)
            
        Returns:
            Training statistics
        """
        tools_to_use = tools_to_use or list(self.tools.keys())
        stats = {
            "total_texts": len(self.dataset),
            "augmented_texts": 0,
            "api_calls_sampled": 0,
            "api_calls_filtered": 0,
            "tools_used": {}
        }
        
        for text in self.dataset:
            text_api_calls = []
            
            # Sample API calls for each tool
            for tool_name in tools_to_use:
                api_calls = self.toolformer.sample_api_calls(text, tool_name)
                stats["api_calls_sampled"] += len(api_calls)
                
                # Execute API calls
                for api_call in api_calls:
                    self.toolformer.execute_api_call(api_call)
                
                # Filter API calls
                filtered = self.toolformer.filter_api_calls(text, api_calls)
                stats["api_calls_filtered"] += len(filtered)
                text_api_calls.extend(filtered)
                
                if filtered:
                    stats["tools_used"][tool_name] = stats["tools_used"].get(tool_name, 0) + len(filtered)
            
            # Augment text with filtered API calls
            if text_api_calls:
                augmented = self.toolformer.augment_text(text, text_api_calls)
                self.augmented_dataset.append(augmented)
                stats["augmented_texts"] += 1
            else:
                self.augmented_dataset.append(text)
        
        return stats
    
    def get_augmented_dataset(self) -> List[str]:
        """Get augmented dataset."""
        return self.augmented_dataset



