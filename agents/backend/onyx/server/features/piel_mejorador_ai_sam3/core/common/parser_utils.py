"""
Parser Utilities for Piel Mejorador AI SAM3
==========================================

Unified parser pattern utilities.
"""

import json
import logging
from typing import TypeVar, Callable, Any, Optional, Dict, List
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class Parser(ABC):
    """Base parser interface."""
    
    @abstractmethod
    def parse(self, data: str) -> Any:
        """Parse data."""
        pass


class JSONParser(Parser):
    """JSON parser."""
    
    def parse(self, data: str) -> Any:
        """
        Parse JSON data.
        
        Args:
            data: JSON string
            
        Returns:
            Parsed data
        """
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            raise ValueError(f"Invalid JSON: {e}") from e


class FunctionParser(Parser):
    """Parser using a function."""
    
    def __init__(
        self,
        parse_func: Callable[[str], R],
        name: Optional[str] = None
    ):
        """
        Initialize function parser.
        
        Args:
            parse_func: Parsing function
            name: Optional parser name
        """
        self._parse_func = parse_func
        self.name = name or parse_func.__name__
    
    def parse(self, data: str) -> R:
        """Parse data."""
        return self._parse_func(data)


class CSVParser(Parser):
    """CSV parser."""
    
    def __init__(self, delimiter: str = ",", has_header: bool = True):
        """
        Initialize CSV parser.
        
        Args:
            delimiter: CSV delimiter
            has_header: Whether CSV has header row
        """
        self.delimiter = delimiter
        self.has_header = has_header
        self._header: Optional[List[str]] = None
    
    def parse(self, data: str) -> List[Dict[str, str]]:
        """
        Parse CSV data.
        
        Args:
            data: CSV string
            
        Returns:
            List of dictionaries
        """
        lines = data.strip().split('\n')
        if not lines:
            return []
        
        if self.has_header:
            self._header = lines[0].split(self.delimiter)
            lines = lines[1:]
        else:
            # Generate header if not provided
            first_line = lines[0].split(self.delimiter)
            self._header = [f"col_{i}" for i in range(len(first_line))]
        
        result = []
        for line in lines:
            if not line.strip():
                continue
            values = line.split(self.delimiter)
            row = dict(zip(self._header, values))
            result.append(row)
        
        return result


class KeyValueParser(Parser):
    """Key-value parser."""
    
    def __init__(self, separator: str = "=", delimiter: str = "\n"):
        """
        Initialize key-value parser.
        
        Args:
            separator: Key-value separator
            delimiter: Line delimiter
        """
        self.separator = separator
        self.delimiter = delimiter
    
    def parse(self, data: str) -> Dict[str, str]:
        """
        Parse key-value data.
        
        Args:
            data: Key-value string
            
        Returns:
            Dictionary of key-value pairs
        """
        result = {}
        for line in data.split(self.delimiter):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if self.separator in line:
                key, value = line.split(self.separator, 1)
                result[key.strip()] = value.strip()
        return result


class ParserUtils:
    """Unified parser utilities."""
    
    @staticmethod
    def create_json_parser() -> JSONParser:
        """
        Create JSON parser.
        
        Returns:
            JSONParser
        """
        return JSONParser()
    
    @staticmethod
    def create_csv_parser(delimiter: str = ",", has_header: bool = True) -> CSVParser:
        """
        Create CSV parser.
        
        Args:
            delimiter: CSV delimiter
            has_header: Whether CSV has header
            
        Returns:
            CSVParser
        """
        return CSVParser(delimiter, has_header)
    
    @staticmethod
    def create_key_value_parser(separator: str = "=", delimiter: str = "\n") -> KeyValueParser:
        """
        Create key-value parser.
        
        Args:
            separator: Key-value separator
            delimiter: Line delimiter
            
        Returns:
            KeyValueParser
        """
        return KeyValueParser(separator, delimiter)
    
    @staticmethod
    def create_function_parser(
        parse_func: Callable[[str], R],
        name: Optional[str] = None
    ) -> FunctionParser:
        """
        Create function parser.
        
        Args:
            parse_func: Parsing function
            name: Optional parser name
            
        Returns:
            FunctionParser
        """
        return FunctionParser(parse_func, name)
    
    @staticmethod
    def parse_json(data: str) -> Any:
        """
        Parse JSON data.
        
        Args:
            data: JSON string
            
        Returns:
            Parsed data
        """
        return JSONParser().parse(data)
    
    @staticmethod
    def parse_csv(data: str, delimiter: str = ",", has_header: bool = True) -> List[Dict[str, str]]:
        """
        Parse CSV data.
        
        Args:
            data: CSV string
            delimiter: CSV delimiter
            has_header: Whether CSV has header
            
        Returns:
            List of dictionaries
        """
        return CSVParser(delimiter, has_header).parse(data)


# Convenience functions
def create_json_parser() -> JSONParser:
    """Create JSON parser."""
    return ParserUtils.create_json_parser()


def create_csv_parser(**kwargs) -> CSVParser:
    """Create CSV parser."""
    return ParserUtils.create_csv_parser(**kwargs)


def parse_json(data: str) -> Any:
    """Parse JSON data."""
    return ParserUtils.parse_json(data)




