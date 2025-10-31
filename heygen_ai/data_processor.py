from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import json
import csv
import hashlib
import re
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from typing import Any, List, Dict, Optional
"""
Data Processor Module
====================

Data processing utilities following lowercase_underscores naming convention.
"""


class DataFormat(Enum):
    """Supported data formats."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    YAML = "yaml"
    TEXT = "text"

class ProcessingStatus(Enum):
    """Data processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class DataProcessingResult:
    """Result of data processing operation."""
    processing_id: str
    input_data: Any
    processed_data: Any
    processing_status: ProcessingStatus
    processing_timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_duration: float = 0.0
    error_message: Optional[str] = None
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataValidationRule:
    """Data validation rule definition."""
    rule_name: str
    validation_function: Callable
    rule_description: str
    is_rule_enabled: bool = True
    error_message: str = "Validation failed"

class DataProcessor:
    """Data processor with descriptive naming and comprehensive functionality."""
    
    def __init__(self, max_concurrent_processors: int = 10):
        
    """__init__ function."""
self.max_concurrent_processors = max_concurrent_processors
        self.processing_semaphore = asyncio.Semaphore(max_concurrent_processors)
        
        # Data validation rules
        self.validation_rules = {
            'required_field_check': DataValidationRule(
                rule_name='required_field_check',
                validation_function=self._validate_required_fields,
                rule_description='Check if required fields are present',
                is_rule_enabled=True,
                error_message='Required field is missing'
            ),
            'data_type_validation': DataValidationRule(
                rule_name='data_type_validation',
                validation_function=self._validate_data_types,
                rule_description='Validate data types of fields',
                is_rule_enabled=True,
                error_message='Invalid data type'
            ),
            'string_length_validation': DataValidationRule(
                rule_name='string_length_validation',
                validation_function=self._validate_string_lengths,
                rule_description='Validate string field lengths',
                is_rule_enabled=True,
                error_message='String length validation failed'
            ),
            'email_format_validation': DataValidationRule(
                rule_name='email_format_validation',
                validation_function=self._validate_email_format,
                rule_description='Validate email format',
                is_rule_enabled=True,
                error_message='Invalid email format'
            )
        }
        
        # Data transformation functions
        self.transformation_functions = {
            'sanitize_string': self._sanitize_string_data,
            'hash_sensitive_data': self._hash_sensitive_data,
            'normalize_whitespace': self._normalize_whitespace,
            'convert_to_lowercase': self._convert_to_lowercase,
            'add_timestamp': self._add_processing_timestamp,
            'generate_unique_id': self._generate_unique_identifier
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    async def process_data_record(
        self, 
        input_record: Dict[str, Any],
        transformation_rules: List[str],
        validation_rules: List[str]
    ) -> DataProcessingResult:
        """Process a single data record with transformations and validations."""
        async with self.processing_semaphore:
            processing_start_time = asyncio.get_event_loop().time()
            processing_id = self._generate_unique_identifier()
            
            processing_result = DataProcessingResult(
                processing_id=processing_id,
                input_data=input_record,
                processed_data=input_record.copy(),
                processing_status=ProcessingStatus.PROCESSING
            )
            
            try:
                # Apply transformations
                for rule_name in transformation_rules:
                    if rule_name in self.transformation_functions:
                        processing_result.processed_data = await self.transformation_functions[rule_name](
                            processing_result.processed_data
                        )
                
                # Apply validations
                validation_errors = []
                for rule_name in validation_rules:
                    if rule_name in self.validation_rules:
                        rule = self.validation_rules[rule_name]
                        if rule.is_rule_enabled:
                            is_valid = await rule.validation_function(processing_result.processed_data)
                            if not is_valid:
                                validation_errors.append(rule.error_message)
                
                if validation_errors:
                    processing_result.processing_status = ProcessingStatus.FAILED
                    processing_result.error_message = "; ".join(validation_errors)
                else:
                    processing_result.processing_status = ProcessingStatus.COMPLETED
                
                processing_result.processing_duration = asyncio.get_event_loop().time() - processing_start_time
                
            except Exception as e:
                processing_result.processing_status = ProcessingStatus.FAILED
                processing_result.error_message = str(e)
                self.logger.error(f"Data processing failed: {str(e)}")
            
            return processing_result
    
    async def process_data_batch(
        self,
        input_records: List[Dict[str, Any]],
        transformation_rules: List[str],
        validation_rules: List[str]
    ) -> List[DataProcessingResult]:
        """Process a batch of data records concurrently."""
        processing_tasks = [
            self.process_data_record(record, transformation_rules, validation_rules)
            for record in input_records
        ]
        
        processing_results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Handle exceptions
        valid_results = []
        for result in processing_results:
            if isinstance(result, Exception):
                # Create error result
                error_result = DataProcessingResult(
                    processing_id=self._generate_unique_identifier(),
                    input_data=None,
                    processed_data=None,
                    processing_status=ProcessingStatus.FAILED,
                    error_message=str(result)
                )
                valid_results.append(error_result)
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _sanitize_string_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize string data to prevent injection attacks."""
        sanitized_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # Remove potentially dangerous characters
                sanitized_value = re.sub(r'[<>"\']', '', value.strip())
                sanitized_data[key] = sanitized_value
            else:
                sanitized_data[key] = value
        
        return sanitized_data
    
    async def _hash_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Hash sensitive data fields."""
        sensitive_fields = ['password', 'ssn', 'credit_card', 'api_key']
        hashed_data = data.copy()
        
        for field in sensitive_fields:
            if field in hashed_data and hashed_data[field]:
                hashed_data[field] = hashlib.sha256(
                    str(hashed_data[field]).encode()
                ).hexdigest()
        
        return hashed_data
    
    async def _normalize_whitespace(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize whitespace in string fields."""
        normalized_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                normalized_data[key] = ' '.join(value.split())
            else:
                normalized_data[key] = value
        
        return normalized_data
    
    async def _convert_to_lowercase(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert string fields to lowercase."""
        lowercase_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                lowercase_data[key] = value.lower()
            else:
                lowercase_data[key] = value
        
        return lowercase_data
    
    async def _add_processing_timestamp(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing timestamp to data."""
        data_with_timestamp = data.copy()
        data_with_timestamp['processing_timestamp'] = datetime.utcnow().isoformat()
        return data_with_timestamp
    
    def _generate_unique_identifier(self) -> str:
        """Generate a unique identifier for processing."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        random_suffix = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"proc_{timestamp}_{random_suffix}"
    
    async def _validate_required_fields(self, data: Dict[str, Any]) -> bool:
        """Validate that required fields are present."""
        required_fields = ['id', 'name', 'email']  # Example required fields
        
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == "":
                return False
        
        return True
    
    async def _validate_data_types(self, data: Dict[str, Any]) -> bool:
        """Validate data types of fields."""
        type_validations = {
            'id': int,
            'name': str,
            'email': str,
            'age': int
        }
        
        for field, expected_type in type_validations.items():
            if field in data:
                try:
                    if expected_type == int:
                        int(data[field])
                    elif expected_type == str:
                        str(data[field])
                except (ValueError, TypeError):
                    return False
        
        return True
    
    async def _validate_string_lengths(self, data: Dict[str, Any]) -> bool:
        """Validate string field lengths."""
        length_validations = {
            'name': {'min': 1, 'max': 100},
            'email': {'min': 5, 'max': 255}
        }
        
        for field, constraints in length_validations.items():
            if field in data and isinstance(data[field], str):
                field_length = len(data[field])
                if field_length < constraints['min'] or field_length > constraints['max']:
                    return False
        
        return True
    
    async def _validate_email_format(self, data: Dict[str, Any]) -> bool:
        """Validate email format."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if 'email' in data and isinstance(data['email'], str):
            return bool(re.match(email_pattern, data['email']))
        
        return True
    
    async def export_processed_data(
        self,
        processed_results: List[DataProcessingResult],
        output_format: DataFormat,
        output_file_path: str
    ) -> Dict[str, Any]:
        """Export processed data to various formats."""
        export_result = {
            'is_export_successful': False,
            'exported_records_count': 0,
            'output_file_path': output_file_path,
            'export_format': output_format.value,
            'error_message': None
        }
        
        try:
            # Filter successful results
            successful_results = [
                result for result in processed_results
                if result.processing_status == ProcessingStatus.COMPLETED
            ]
            
            if output_format == DataFormat.JSON:
                await self._export_to_json(successful_results, output_file_path)
            elif output_format == DataFormat.CSV:
                await self._export_to_csv(successful_results, output_file_path)
            else:
                raise ValueError(f"Unsupported export format: {output_format}")
            
            export_result['is_export_successful'] = True
            export_result['exported_records_count'] = len(successful_results)
            
        except Exception as e:
            export_result['error_message'] = str(e)
            self.logger.error(f"Data export failed: {str(e)}")
        
        return export_result
    
    async def _export_to_json(self, results: List[DataProcessingResult], file_path: str):
        """Export results to JSON format."""
        export_data = []
        
        for result in results:
            export_data.append({
                'processing_id': result.processing_id,
                'processed_data': result.processed_data,
                'processing_timestamp': result.processing_timestamp.isoformat(),
                'processing_duration': result.processing_duration
            })
        
        # Use asyncio to write file
        await asyncio.get_event_loop().run_in_executor(
            None, self._write_json_file, file_path, export_data
        )
    
    async def _export_to_csv(self, results: List[DataProcessingResult], file_path: str):
        """Export results to CSV format."""
        if not results:
            return
        
        # Get all unique keys from processed data
        all_keys = set()
        for result in results:
            if result.processed_data:
                all_keys.update(result.processed_data.keys())
        
        # Prepare CSV data
        csv_data = []
        for result in results:
            if result.processed_data:
                row = result.processed_data.copy()
                row['processing_id'] = result.processing_id
                row['processing_timestamp'] = result.processing_timestamp.isoformat()
                csv_data.append(row)
        
        # Use asyncio to write file
        await asyncio.get_event_loop().run_in_executor(
            None, self._write_csv_file, file_path, csv_data, list(all_keys)
        )
    
    def _write_json_file(self, file_path: str, data: List[Dict[str, Any]]):
        """Write data to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _write_csv_file(self, file_path: str, data: List[Dict[str, Any]], fieldnames: List[str]):
        """Write data to CSV file."""
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

# Usage example
async def main():
    """Example usage of the data processor."""
    data_processor = DataProcessor(max_concurrent_processors=5)
    
    # Sample input data
    input_records = [
        {
            'id': 1,
            'name': '  John Doe  ',
            'email': 'john@example.com',
            'password': 'secret123',
            'age': 30
        },
        {
            'id': 2,
            'name': 'Jane Smith',
            'email': 'jane@example.com',
            'password': 'password456',
            'age': 25
        }
    ]
    
    # Define processing rules
    transformation_rules = ['sanitize_string', 'hash_sensitive_data', 'normalize_whitespace', 'add_timestamp']
    validation_rules = ['required_field_check', 'data_type_validation', 'email_format_validation']
    
    # Process data
    processing_results = await data_processor.process_data_batch(
        input_records, transformation_rules, validation_rules
    )
    
    # Export results
    export_result = await data_processor.export_processed_data(
        processing_results, DataFormat.JSON, 'processed_data.json'
    )
    
    print(f"Processing completed: {export_result['exported_records_count']} records exported")
    print(f"Export successful: {export_result['is_export_successful']}")

# Run: asyncio.run(main()) 