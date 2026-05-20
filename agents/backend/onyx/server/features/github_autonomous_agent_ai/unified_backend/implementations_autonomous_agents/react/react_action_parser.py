"""
ReAct Action Parser

Handles parsing of actions from thoughts using multiple format strategies.
Consolidates all action parsing logic into a single, testable class.
"""

import re
import json
from typing import Dict, Any, Optional, List
from enum import Enum

from .react_constants import ReActPatterns, FinishKeywords


class ActionFormat(Enum):
    """Supported action formats for parsing."""
    FUNCTION_CALL = "function_call"  # tool_name(param1=value1, param2=value2)
    JSON = "json"  # {"tool": "name", "parameters": {...}}
    NATURAL = "natural"  # "I will use tool_name with parameters..."


class ActionParser:
    """
    Parser for extracting actions from thoughts.
    
    Supports multiple parsing strategies and formats, with fallback
    mechanisms for robust action extraction.
    """
    
    def __init__(self, formats: Optional[List[ActionFormat]] = None):
        """
        Initialize action parser.
        
        Args:
            formats: List of formats to try (in order). Defaults to all formats.
        """
        self.formats = formats or [
            ActionFormat.FUNCTION_CALL,
            ActionFormat.JSON,
            ActionFormat.NATURAL
        ]
        self.patterns = ReActPatterns()
    
    def parse(self, thought: str) -> Optional[Dict[str, Any]]:
        """
        Parse action from thought using configured formats.
        
        Tries each format in order until one succeeds.
        
        Args:
            thought: Thought string containing action
            
        Returns:
            Parsed action dict with 'tool' and 'parameters' keys, or None if no action found
        """
        # Check for finish/completion keywords
        if self._is_finish(thought):
            return None
        
        # Try each format in order
        for format_type in self.formats:
            action = self._parse_with_format(thought, format_type)
            if action:
                return action
        
        return None
    
    def _is_finish(self, thought: str) -> bool:
        """
        Check if thought indicates task completion.
        
        Args:
            thought: Thought string to check
            
        Returns:
            True if thought indicates completion
        """
        thought_lower = thought.lower()
        return any(keyword in thought_lower for keyword in FinishKeywords.KEYWORDS)
    
    def _parse_with_format(
        self, 
        thought: str, 
        format_type: ActionFormat
    ) -> Optional[Dict[str, Any]]:
        """
        Parse action using specific format.
        
        Args:
            thought: Thought string
            format_type: Format to use for parsing
            
        Returns:
            Parsed action dict or None
        """
        if format_type == ActionFormat.FUNCTION_CALL:
            return self._parse_function_call(thought)
        elif format_type == ActionFormat.JSON:
            return self._parse_json_action(thought)
        elif format_type == ActionFormat.NATURAL:
            return self._parse_natural_action(thought)
        
        return None
    
    def _parse_function_call(self, thought: str) -> Optional[Dict[str, Any]]:
        """
        Parse function call format: tool_name(param1=value1, param2=value2).
        
        Args:
            thought: Thought string containing function call
            
        Returns:
            Parsed action dict or None
        """
        for pattern in self.patterns.FUNCTION_CALL_PATTERNS:
            match = re.search(pattern, thought, re.IGNORECASE)
            if match:
                tool_name = match.group(1)
                params_str = match.group(2)
                
                # Parse parameters
                parameters = self._parse_parameters(params_str)
                
                return {
                    "tool": tool_name,
                    "parameters": parameters
                }
        
        return None
    
    def _parse_json_action(self, thought: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON format: {"tool": "name", "parameters": {...}}.
        
        Args:
            thought: Thought string containing JSON action
            
        Returns:
            Parsed action dict or None
        """
        # Look for JSON object
        matches = re.findall(
            self.patterns.JSON_ACTION_PATTERN, 
            thought, 
            re.DOTALL
        )
        
        for match in matches:
            try:
                action_dict = json.loads(match)
                if "tool" in action_dict:
                    return {
                        "tool": action_dict["tool"],
                        "parameters": action_dict.get("parameters", {})
                    }
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _parse_natural_action(self, thought: str) -> Optional[Dict[str, Any]]:
        """
        Parse natural language action description.
        
        Looks for patterns like "use tool_name", "call tool_name", etc.
        
        Args:
            thought: Thought string containing natural language action
            
        Returns:
            Parsed action dict or None
        """
        for pattern in self.patterns.NATURAL_ACTION_PATTERNS:
            match = re.search(pattern, thought, re.IGNORECASE)
            if match:
                tool_name = match.group(1)
                
                # Try to extract parameters from context
                parameters = self._extract_parameters_from_text(thought)
                
                return {
                    "tool": tool_name,
                    "parameters": parameters
                }
        
        return None
    
    def _parse_parameters(self, params_str: str) -> Dict[str, Any]:
        """
        Parse parameters from string.
        
        Supports:
        - key="value"
        - key='value'
        - key=value
        - key=123 (numbers)
        - key=true/false (booleans)
        
        Args:
            params_str: Parameter string to parse
            
        Returns:
            Dictionary of parsed parameters
        """
        parameters = {}
        
        if not params_str or not params_str.strip():
            return parameters
        
        # Split by comma, handling nested quotes
        param_parts = self._split_parameters(params_str)
        
        # Parse each parameter
        for param in param_parts:
            param = param.strip()
            if '=' in param:
                key, value = param.split('=', 1)
                key = key.strip().strip('"\'')
                value = value.strip().strip('"\'')
                
                # Convert to appropriate type
                value = self._convert_value(value)
                parameters[key] = value
        
        return parameters
    
    def _split_parameters(self, params_str: str) -> List[str]:
        """
        Split parameter string by comma, handling quoted values.
        
        Args:
            params_str: Parameter string
            
        Returns:
            List of parameter strings
        """
        param_parts = []
        current = ""
        in_quotes = False
        quote_char = None
        
        for char in params_str:
            if char in ('"', "'") and (not in_quotes or char == quote_char):
                in_quotes = not in_quotes
                quote_char = char if in_quotes else None
                current += char
            elif char == ',' and not in_quotes:
                if current.strip():
                    param_parts.append(current.strip())
                current = ""
            else:
                current += char
        
        if current.strip():
            param_parts.append(current.strip())
        
        return param_parts
    
    def _convert_value(self, value: str) -> Any:
        """
        Convert string value to appropriate type.
        
        Args:
            value: String value to convert
            
        Returns:
            Converted value (bool, int, float, or str)
        """
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # String (default)
        return value
    
    def _extract_parameters_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract parameters from natural language text.
        
        Args:
            text: Text to extract parameters from
            
        Returns:
            Dictionary of extracted parameters
        """
        parameters = {}
        
        for pattern in self.patterns.PARAMETER_EXTRACTION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for key, value in matches:
                parameters[key] = self._convert_value(value)
        
        return parameters



