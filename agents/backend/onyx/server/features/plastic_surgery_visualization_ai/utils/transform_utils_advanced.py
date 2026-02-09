"""Advanced request/response transformation utilities."""

from typing import Any, Dict, List, Optional, Callable, TypeVar
from datetime import datetime
import json
from functools import reduce
from collections import defaultdict

T = TypeVar('T')


class RequestTransformer:
    """Transform incoming requests."""
    
    @staticmethod
    def normalize_keys(data: Dict[str, Any], case: str = 'snake') -> Dict[str, Any]:
        """Normalize dictionary keys."""
        try:
            from utils.string_utils import camel_to_snake, snake_to_camel
        except ImportError:
            def camel_to_snake(s: str) -> str:
                import re
                return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()
            
            def snake_to_camel(s: str) -> str:
                components = s.split('_')
                return components[0] + ''.join(x.capitalize() for x in components[1:])
        
        if case == 'snake':
            return {camel_to_snake(k): v for k, v in data.items()}
        elif case == 'camel':
            return {snake_to_camel(k): v for k, v in data.items()}
        return data
    
    @staticmethod
    def filter_fields(data: Dict[str, Any], allowed_fields: List[str]) -> Dict[str, Any]:
        """Filter dictionary to only allowed fields."""
        return {k: v for k, v in data.items() if k in allowed_fields}
    
    @staticmethod
    def exclude_fields(data: Dict[str, Any], excluded_fields: List[str]) -> Dict[str, Any]:
        """Exclude fields from dictionary."""
        return {k: v for k, v in data.items() if k not in excluded_fields}
    
    @staticmethod
    def add_defaults(data: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Add default values for missing keys."""
        return {**defaults, **data}
    
    @staticmethod
    def transform_values(
        data: Dict[str, Any],
        transformers: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """Transform values using provided functions."""
        result = data.copy()
        for key, transformer in transformers.items():
            if key in result:
                result[key] = transformer(result[key])
        return result


class ResponseTransformer:
    """Transform outgoing responses."""
    
    @staticmethod
    def format_response(
        data: Any,
        success: bool = True,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format standard API response."""
        response = {
            'success': success,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if message:
            response['message'] = message
        
        if metadata:
            response['metadata'] = metadata
        
        return response
    
    @staticmethod
    def paginate_response(
        items: List[Any],
        page: int,
        page_size: int,
        total: Optional[int] = None
    ) -> Dict[str, Any]:
        """Format paginated response."""
        if total is None:
            total = len(items)
        
        return {
            'items': items,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total': total,
                'pages': (total + page_size - 1) // page_size
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def error_response(
        error: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format error response."""
        response = {
            'success': False,
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if code:
            response['code'] = code
        
        if details:
            response['details'] = details
        
        return response


class DataMapper:
    """Map data between different structures."""
    
    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping
        self.reverse_mapping = {v: k for k, v in mapping.items()}
    
    def map_forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Map keys forward."""
        return {self.mapping.get(k, k): v for k, v in data.items()}
    
    def map_backward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Map keys backward."""
        return {self.reverse_mapping.get(k, k): v for k, v in data.items()}
    
    def map_nested(self, data: Dict[str, Any], forward: bool = True) -> Dict[str, Any]:
        """Map nested dictionaries."""
        
        mapper = self.map_forward if forward else self.map_backward
        result = {}
        
        for key, value in data.items():
            mapped_key = mapper({key: None}).popitem()[0]
            
            if isinstance(value, dict):
                result[mapped_key] = self.map_nested(value, forward)
            elif isinstance(value, list):
                result[mapped_key] = [
                    self.map_nested(item, forward) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[mapped_key] = value
        
        return result


class FieldValidator:
    """Validate and transform fields."""
    
    def __init__(self):
        self.validators: Dict[str, Callable] = {}
        self.transformers: Dict[str, Callable] = {}
    
    def add_validator(self, field: str, validator: Callable):
        """Add field validator."""
        self.validators[field] = validator
    
    def add_transformer(self, field: str, transformer: Callable):
        """Add field transformer."""
        self.transformers[field] = transformer
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate data."""
        for field, validator in self.validators.items():
            if field in data:
                try:
                    if not validator(data[field]):
                        return False, f"Validation failed for field: {field}"
                except Exception as e:
                    return False, f"Validation error for {field}: {str(e)}"
        return True, None
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data."""
        result = data.copy()
        for field, transformer in self.transformers.items():
            if field in result:
                result[field] = transformer(result[field])
        return result


def transform_request(
    data: Dict[str, Any],
    normalize: bool = True,
    filter_fields: Optional[List[str]] = None,
    add_defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Transform request data."""
    transformer = RequestTransformer()
    
    if normalize:
        data = transformer.normalize_keys(data)
    
    if filter_fields:
        data = transformer.filter_fields(data, filter_fields)
    
    if add_defaults:
        data = transformer.add_defaults(data, add_defaults)
    
    return data


def format_success_response(
    data: Any,
    message: Optional[str] = None
) -> Dict[str, Any]:
    """Format success response."""
    return ResponseTransformer.format_response(data, success=True, message=message)


def format_error_response(
    error: str,
    code: Optional[str] = None
) -> Dict[str, Any]:
    """Format error response."""
    return ResponseTransformer.error_response(error, code=code)


# Data transformation functions from transform_utils.py
def map_list(func: Callable, items: List) -> List:
    """Map function over list."""
    return [func(item) for item in items]


def filter_list(predicate: Callable, items: List) -> List:
    """Filter list by predicate."""
    return [item for item in items if predicate(item)]


def reduce_list(func: Callable, items: List, initial: Any = None) -> Any:
    """Reduce list using function."""
    if initial is not None:
        return reduce(func, items, initial)
    return reduce(func, items)


def transform_dict(
    d: Dict,
    key_transform: Optional[Callable] = None,
    value_transform: Optional[Callable] = None
) -> Dict:
    """Transform dictionary keys and/or values."""
    result = {}
    for key, value in d.items():
        new_key = key_transform(key) if key_transform else key
        new_value = value_transform(value) if value_transform else value
        result[new_key] = new_value
    return result


def pivot_list(items: List[Dict], key_field: str, value_field: str) -> Dict:
    """Pivot list of dictionaries."""
    return {item[key_field]: item[value_field] for item in items}


def transpose_dict(d: Dict[str, List]) -> List[Dict]:
    """Transpose dictionary of lists to list of dictionaries."""
    keys = list(d.keys())
    lengths = [len(d[k]) for k in keys]
    
    if not lengths or len(set(lengths)) != 1:
        return []
    
    return [
        {keys[i]: d[keys[i]][j] for i in range(len(keys))}
        for j in range(lengths[0])
    ]


def normalize_list(items: List[Dict], default_keys: List[str]) -> List[Dict]:
    """Normalize list of dictionaries to have same keys."""
    return [
        {key: item.get(key) for key in default_keys}
        for item in items
    ]


def aggregate_list(
    items: List[Dict],
    group_by: str,
    aggregate: Dict[str, Callable]
) -> List[Dict]:
    """Aggregate list of dictionaries."""
    grouped = defaultdict(list)
    for item in items:
        grouped[item[group_by]].append(item)
    
    result = []
    for key, group_items in grouped.items():
        agg_dict = {group_by: key}
        for field, func in aggregate.items():
            values = [item.get(field) for item in group_items if field in item]
            agg_dict[field] = func(values) if values else None
        result.append(agg_dict)
    
    return result

