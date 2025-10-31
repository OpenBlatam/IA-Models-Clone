"""
BUL Data Processing Engine
=========================

Advanced data processing engine for the BUL system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import yaml
import re
import csv
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config_optimized import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFormat(Enum):
    """Supported data formats."""
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    EXCEL = "excel"
    PARQUET = "parquet"
    TEXT = "text"
    MARKDOWN = "markdown"

class ProcessingOperation(Enum):
    """Data processing operations."""
    CLEAN = "clean"
    TRANSFORM = "transform"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    MERGE = "merge"
    SPLIT = "split"
    VALIDATE = "validate"
    ENRICH = "enrich"
    NORMALIZE = "normalize"
    DEDUPLICATE = "deduplicate"

@dataclass
class DataProcessor:
    """Data processor definition."""
    id: str
    name: str
    operation: ProcessingOperation
    parameters: Dict[str, Any]
    input_format: DataFormat
    output_format: DataFormat
    created_at: datetime
    description: str = ""

@dataclass
class ProcessingJob:
    """Data processing job definition."""
    id: str
    name: str
    processor_id: str
    input_path: str
    output_path: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None

class DataProcessingEngine:
    """Advanced data processing engine for BUL system."""
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.processors = {}
        self.jobs = {}
        self.processing_history = []
        self.init_processing_environment()
        self.load_processors()
        self.load_jobs()
    
    def init_processing_environment(self):
        """Initialize data processing environment."""
        print("⚙️ Initializing data processing environment...")
        
        # Create processing directories
        self.processing_dir = Path("data_processing")
        self.processing_dir.mkdir(exist_ok=True)
        
        self.input_dir = Path("data_input")
        self.input_dir.mkdir(exist_ok=True)
        
        self.output_dir = Path("data_output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.temp_dir = Path("data_temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        print("✅ Data processing environment initialized")
    
    def load_processors(self):
        """Load existing data processors."""
        processors_file = self.processing_dir / "processors.json"
        if processors_file.exists():
            with open(processors_file, 'r') as f:
                processors_data = json.load(f)
            
            for processor_id, processor_data in processors_data.items():
                processor = DataProcessor(
                    id=processor_id,
                    name=processor_data['name'],
                    operation=ProcessingOperation(processor_data['operation']),
                    parameters=processor_data['parameters'],
                    input_format=DataFormat(processor_data['input_format']),
                    output_format=DataFormat(processor_data['output_format']),
                    created_at=datetime.fromisoformat(processor_data['created_at']),
                    description=processor_data.get('description', '')
                )
                self.processors[processor_id] = processor
        
        print(f"✅ Loaded {len(self.processors)} data processors")
    
    def load_jobs(self):
        """Load existing processing jobs."""
        jobs_file = self.processing_dir / "jobs.json"
        if jobs_file.exists():
            with open(jobs_file, 'r') as f:
                jobs_data = json.load(f)
            
            for job_id, job_data in jobs_data.items():
                job = ProcessingJob(
                    id=job_id,
                    name=job_data['name'],
                    processor_id=job_data['processor_id'],
                    input_path=job_data['input_path'],
                    output_path=job_data['output_path'],
                    status=job_data['status'],
                    created_at=datetime.fromisoformat(job_data['created_at']),
                    started_at=datetime.fromisoformat(job_data['started_at']) if job_data.get('started_at') else None,
                    completed_at=datetime.fromisoformat(job_data['completed_at']) if job_data.get('completed_at') else None,
                    error_message=job_data.get('error_message'),
                    result_summary=job_data.get('result_summary')
                )
                self.jobs[job_id] = job
        
        print(f"✅ Loaded {len(self.jobs)} processing jobs")
    
    def create_processor(self, processor_id: str, name: str, operation: ProcessingOperation,
                        parameters: Dict[str, Any], input_format: DataFormat,
                        output_format: DataFormat, description: str = "") -> DataProcessor:
        """Create a new data processor."""
        processor = DataProcessor(
            id=processor_id,
            name=name,
            operation=operation,
            parameters=parameters,
            input_format=input_format,
            output_format=output_format,
            created_at=datetime.now(),
            description=description
        )
        
        self.processors[processor_id] = processor
        self._save_processors()
        
        print(f"✅ Created data processor: {name}")
        return processor
    
    def create_processing_job(self, job_id: str, name: str, processor_id: str,
                             input_path: str, output_path: str) -> ProcessingJob:
        """Create a new processing job."""
        if processor_id not in self.processors:
            raise ValueError(f"Processor {processor_id} not found")
        
        job = ProcessingJob(
            id=job_id,
            name=name,
            processor_id=processor_id,
            input_path=input_path,
            output_path=output_path,
            status="pending",
            created_at=datetime.now()
        )
        
        self.jobs[job_id] = job
        self._save_jobs()
        
        print(f"✅ Created processing job: {name}")
        return job
    
    async def execute_job(self, job_id: str) -> Dict[str, Any]:
        """Execute a processing job."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        processor = self.processors[job.processor_id]
        
        print(f"⚙️ Executing job: {job.name}")
        print(f"   Processor: {processor.name}")
        print(f"   Operation: {processor.operation.value}")
        
        job.status = "running"
        job.started_at = datetime.now()
        self._save_jobs()
        
        try:
            # Load input data
            input_data = await self._load_data(job.input_path, processor.input_format)
            
            # Process data
            result = await self._process_data(input_data, processor)
            
            # Save output data
            await self._save_data(result, job.output_path, processor.output_format)
            
            # Update job status
            job.status = "completed"
            job.completed_at = datetime.now()
            job.result_summary = {
                'input_size': len(input_data) if hasattr(input_data, '__len__') else 1,
                'output_size': len(result) if hasattr(result, '__len__') else 1,
                'processing_time': (job.completed_at - job.started_at).total_seconds()
            }
            
            self._save_jobs()
            self._log_processing(job_id, processor.id, "completed", job.result_summary)
            
            print(f"✅ Job completed successfully")
            print(f"   Processing time: {job.result_summary['processing_time']:.2f}s")
            
            return {
                'job_id': job_id,
                'status': 'completed',
                'result_summary': job.result_summary
            }
            
        except Exception as e:
            job.status = "failed"
            job.completed_at = datetime.now()
            job.error_message = str(e)
            
            self._save_jobs()
            self._log_processing(job_id, processor.id, "failed", {'error': str(e)})
            
            logger.error(f"Job {job_id} failed: {e}")
            raise
    
    async def _load_data(self, file_path: str, data_format: DataFormat) -> Any:
        """Load data from file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        if data_format == DataFormat.CSV:
            return pd.read_csv(file_path)
        elif data_format == DataFormat.JSON:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif data_format == DataFormat.XML:
            tree = ET.parse(file_path)
            return tree.getroot()
        elif data_format == DataFormat.EXCEL:
            return pd.read_excel(file_path)
        elif data_format == DataFormat.PARQUET:
            return pd.read_parquet(file_path)
        elif data_format == DataFormat.TEXT:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif data_format == DataFormat.MARKDOWN:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported input format: {data_format}")
    
    async def _save_data(self, data: Any, file_path: str, data_format: DataFormat):
        """Save data to file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if data_format == DataFormat.CSV:
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
            else:
                raise ValueError("Data must be a pandas DataFrame for CSV format")
        elif data_format == DataFormat.JSON:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        elif data_format == DataFormat.XML:
            if hasattr(data, 'write'):
                data.write(file_path, encoding='utf-8', xml_declaration=True)
            else:
                raise ValueError("Data must be an XML element for XML format")
        elif data_format == DataFormat.EXCEL:
            if isinstance(data, pd.DataFrame):
                data.to_excel(file_path, index=False)
            else:
                raise ValueError("Data must be a pandas DataFrame for Excel format")
        elif data_format == DataFormat.PARQUET:
            if isinstance(data, pd.DataFrame):
                data.to_parquet(file_path, index=False)
            else:
                raise ValueError("Data must be a pandas DataFrame for Parquet format")
        elif data_format == DataFormat.TEXT:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
        elif data_format == DataFormat.MARKDOWN:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
        else:
            raise ValueError(f"Unsupported output format: {data_format}")
    
    async def _process_data(self, data: Any, processor: DataProcessor) -> Any:
        """Process data using the specified processor."""
        operation = processor.operation
        parameters = processor.parameters
        
        if operation == ProcessingOperation.CLEAN:
            return await self._clean_data(data, parameters)
        elif operation == ProcessingOperation.TRANSFORM:
            return await self._transform_data(data, parameters)
        elif operation == ProcessingOperation.FILTER:
            return await self._filter_data(data, parameters)
        elif operation == ProcessingOperation.AGGREGATE:
            return await self._aggregate_data(data, parameters)
        elif operation == ProcessingOperation.MERGE:
            return await self._merge_data(data, parameters)
        elif operation == ProcessingOperation.SPLIT:
            return await self._split_data(data, parameters)
        elif operation == ProcessingOperation.VALIDATE:
            return await self._validate_data(data, parameters)
        elif operation == ProcessingOperation.ENRICH:
            return await self._enrich_data(data, parameters)
        elif operation == ProcessingOperation.NORMALIZE:
            return await self._normalize_data(data, parameters)
        elif operation == ProcessingOperation.DEDUPLICATE:
            return await self._deduplicate_data(data, parameters)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    async def _clean_data(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Clean data."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            
            # Remove duplicates
            if parameters.get('remove_duplicates', False):
                df = df.drop_duplicates()
            
            # Handle missing values
            if parameters.get('handle_missing'):
                method = parameters['handle_missing']
                if method == 'drop':
                    df = df.dropna()
                elif method == 'fill':
                    fill_value = parameters.get('fill_value', 0)
                    df = df.fillna(fill_value)
                elif method == 'forward_fill':
                    df = df.fillna(method='ffill')
                elif method == 'backward_fill':
                    df = df.fillna(method='bfill')
            
            # Remove outliers
            if parameters.get('remove_outliers', False):
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            # Clean text columns
            if parameters.get('clean_text', False):
                text_columns = df.select_dtypes(include=['object']).columns
                for col in text_columns:
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            
            return df
        
        elif isinstance(data, str):
            # Clean text data
            text = data
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters if specified
            if parameters.get('remove_special_chars', False):
                text = re.sub(r'[^\w\s]', '', text)
            
            # Convert to lowercase if specified
            if parameters.get('lowercase', False):
                text = text.lower()
            
            return text
        
        else:
            return data
    
    async def _transform_data(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Transform data."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            
            # Apply transformations to columns
            transformations = parameters.get('transformations', {})
            for column, transformation in transformations.items():
                if column in df.columns:
                    if transformation['type'] == 'scale':
                        # Min-max scaling
                        min_val = df[column].min()
                        max_val = df[column].max()
                        df[column] = (df[column] - min_val) / (max_val - min_val)
                    
                    elif transformation['type'] == 'normalize':
                        # Z-score normalization
                        mean_val = df[column].mean()
                        std_val = df[column].std()
                        df[column] = (df[column] - mean_val) / std_val
                    
                    elif transformation['type'] == 'log':
                        # Logarithmic transformation
                        df[column] = np.log1p(df[column])
                    
                    elif transformation['type'] == 'square_root':
                        # Square root transformation
                        df[column] = np.sqrt(df[column])
            
            # Create new columns
            new_columns = parameters.get('new_columns', {})
            for new_col, expression in new_columns.items():
                try:
                    df[new_col] = df.eval(expression)
                except Exception as e:
                    logger.warning(f"Could not create column {new_col}: {e}")
            
            return df
        
        else:
            return data
    
    async def _filter_data(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Filter data."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            
            # Apply filters
            filters = parameters.get('filters', [])
            for filter_condition in filters:
                try:
                    df = df.query(filter_condition)
                except Exception as e:
                    logger.warning(f"Could not apply filter {filter_condition}: {e}")
            
            # Filter by column values
            column_filters = parameters.get('column_filters', {})
            for column, filter_value in column_filters.items():
                if column in df.columns:
                    if isinstance(filter_value, list):
                        df = df[df[column].isin(filter_value)]
                    else:
                        df = df[df[column] == filter_value]
            
            return df
        
        else:
            return data
    
    async def _aggregate_data(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Aggregate data."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            
            # Group by columns
            group_by = parameters.get('group_by', [])
            if group_by:
                grouped = df.groupby(group_by)
                
                # Apply aggregations
                aggregations = parameters.get('aggregations', {})
                if aggregations:
                    result = grouped.agg(aggregations)
                else:
                    # Default aggregations
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    result = grouped[numeric_columns].agg(['mean', 'sum', 'count'])
                
                return result.reset_index()
            
            return df
        
        else:
            return data
    
    async def _merge_data(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Merge data."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            
            # Load second dataset
            second_data_path = parameters.get('second_data_path')
            if second_data_path:
                second_df = pd.read_csv(second_data_path)
                
                # Merge datasets
                merge_type = parameters.get('merge_type', 'inner')
                merge_on = parameters.get('merge_on', [])
                
                if merge_on:
                    result = pd.merge(df, second_df, on=merge_on, how=merge_type)
                else:
                    result = pd.concat([df, second_df], ignore_index=True)
                
                return result
            
            return df
        
        else:
            return data
    
    async def _split_data(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Split data."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            
            # Split by ratio
            split_ratio = parameters.get('split_ratio', 0.8)
            split_size = int(len(df) * split_ratio)
            
            train_data = df[:split_size]
            test_data = df[split_size:]
            
            return {
                'train': train_data,
                'test': test_data
            }
        
        else:
            return data
    
    async def _validate_data(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Validate data."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            
            validation_results = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.to_dict(),
                'duplicates': df.duplicated().sum()
            }
            
            # Check data quality rules
            rules = parameters.get('rules', [])
            for rule in rules:
                rule_result = self._apply_validation_rule(df, rule)
                validation_results[f"rule_{rule['name']}"] = rule_result
            
            return validation_results
        
        else:
            return {'validation': 'completed', 'data_type': type(data).__name__}
    
    async def _enrich_data(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Enrich data with additional information."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            
            # Add timestamp
            if parameters.get('add_timestamp', False):
                df['processed_at'] = datetime.now()
            
            # Add row numbers
            if parameters.get('add_row_numbers', False):
                df['row_number'] = range(1, len(df) + 1)
            
            # Add calculated fields
            calculated_fields = parameters.get('calculated_fields', {})
            for field_name, expression in calculated_fields.items():
                try:
                    df[field_name] = df.eval(expression)
                except Exception as e:
                    logger.warning(f"Could not calculate field {field_name}: {e}")
            
            return df
        
        else:
            return data
    
    async def _normalize_data(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Normalize data."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            
            # Normalize numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            normalization_method = parameters.get('method', 'minmax')
            
            for col in numeric_columns:
                if normalization_method == 'minmax':
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val != min_val:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                
                elif normalization_method == 'zscore':
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val != 0:
                        df[col] = (df[col] - mean_val) / std_val
            
            return df
        
        else:
            return data
    
    async def _deduplicate_data(self, data: Any, parameters: Dict[str, Any]) -> Any:
        """Remove duplicate data."""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            
            # Remove duplicates based on specified columns
            subset = parameters.get('subset', None)
            keep = parameters.get('keep', 'first')
            
            if subset:
                df = df.drop_duplicates(subset=subset, keep=keep)
            else:
                df = df.drop_duplicates(keep=keep)
            
            return df
        
        else:
            return data
    
    def _apply_validation_rule(self, df: pd.DataFrame, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a validation rule to the dataframe."""
        rule_name = rule['name']
        rule_type = rule['type']
        
        try:
            if rule_type == 'not_null':
                column = rule['column']
                null_count = df[column].isnull().sum()
                return {
                    'passed': null_count == 0,
                    'null_count': null_count,
                    'message': f"Column {column} has {null_count} null values"
                }
            
            elif rule_type == 'unique':
                column = rule['column']
                unique_count = df[column].nunique()
                total_count = len(df)
                return {
                    'passed': unique_count == total_count,
                    'unique_count': unique_count,
                    'total_count': total_count,
                    'message': f"Column {column} has {unique_count} unique values out of {total_count}"
                }
            
            elif rule_type == 'range':
                column = rule['column']
                min_val = rule.get('min')
                max_val = rule.get('max')
                
                if min_val is not None:
                    below_min = (df[column] < min_val).sum()
                else:
                    below_min = 0
                
                if max_val is not None:
                    above_max = (df[column] > max_val).sum()
                else:
                    above_max = 0
                
                return {
                    'passed': below_min == 0 and above_max == 0,
                    'below_min': below_min,
                    'above_max': above_max,
                    'message': f"Column {column} has {below_min} values below min and {above_max} values above max"
                }
            
            else:
                return {
                    'passed': False,
                    'message': f"Unknown rule type: {rule_type}"
                }
        
        except Exception as e:
            return {
                'passed': False,
                'message': f"Error applying rule {rule_name}: {str(e)}"
            }
    
    def _save_processors(self):
        """Save processors to file."""
        processors_data = {}
        for processor_id, processor in self.processors.items():
            processors_data[processor_id] = {
                'id': processor.id,
                'name': processor.name,
                'operation': processor.operation.value,
                'parameters': processor.parameters,
                'input_format': processor.input_format.value,
                'output_format': processor.output_format.value,
                'created_at': processor.created_at.isoformat(),
                'description': processor.description
            }
        
        with open(self.processing_dir / "processors.json", 'w') as f:
            json.dump(processors_data, f, indent=2)
    
    def _save_jobs(self):
        """Save jobs to file."""
        jobs_data = {}
        for job_id, job in self.jobs.items():
            jobs_data[job_id] = {
                'id': job.id,
                'name': job.name,
                'processor_id': job.processor_id,
                'input_path': job.input_path,
                'output_path': job.output_path,
                'status': job.status,
                'created_at': job.created_at.isoformat(),
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                'error_message': job.error_message,
                'result_summary': job.result_summary
            }
        
        with open(self.processing_dir / "jobs.json", 'w') as f:
            json.dump(jobs_data, f, indent=2)
    
    def _log_processing(self, job_id: str, processor_id: str, status: str, result: Dict[str, Any]):
        """Log processing event."""
        processing_event = {
            'job_id': job_id,
            'processor_id': processor_id,
            'status': status,
            'result': result,
            'processed_at': datetime.now().isoformat()
        }
        
        self.processing_history.append(processing_event)
        
        # Save processing history
        with open(self.processing_dir / "processing_history.json", 'w') as f:
            json.dump(self.processing_history, f, indent=2)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_jobs = len(self.jobs)
        completed_jobs = len([j for j in self.jobs.values() if j.status == 'completed'])
        failed_jobs = len([j for j in self.jobs.values() if j.status == 'failed'])
        running_jobs = len([j for j in self.jobs.values() if j.status == 'running'])
        
        return {
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'running_jobs': running_jobs,
            'success_rate': (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0,
            'total_processors': len(self.processors),
            'total_processing_events': len(self.processing_history)
        }
    
    def generate_processing_report(self) -> str:
        """Generate data processing engine report."""
        stats = self.get_processing_stats()
        
        report = f"""
BUL Data Processing Engine Report
=================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROCESSORS
----------
Total Processors: {stats['total_processors']}
"""
        
        for processor_id, processor in self.processors.items():
            report += f"""
{processor.name} ({processor_id}):
  Operation: {processor.operation.value}
  Input Format: {processor.input_format.value}
  Output Format: {processor.output_format.value}
  Created: {processor.created_at.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        report += f"""
JOBS
----
Total Jobs: {stats['total_jobs']}
Completed: {stats['completed_jobs']}
Failed: {stats['failed_jobs']}
Running: {stats['running_jobs']}
Success Rate: {stats['success_rate']:.1f}%
"""
        
        # Show recent jobs
        recent_jobs = sorted(
            self.jobs.values(),
            key=lambda x: x.created_at,
            reverse=True
        )[:10]
        
        if recent_jobs:
            report += f"""
RECENT JOBS
-----------
"""
            for job in recent_jobs:
                report += f"""
{job.name} ({job.id}):
  Status: {job.status}
  Created: {job.created_at.strftime('%Y-%m-%d %H:%M:%S')}
  Processor: {job.processor_id}
"""
        
        return report

def main():
    """Main data processing engine function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Data Processing Engine")
    parser.add_argument("--create-processor", help="Create a new data processor")
    parser.add_argument("--create-job", help="Create a new processing job")
    parser.add_argument("--execute-job", help="Execute a processing job")
    parser.add_argument("--list-processors", action="store_true", help="List all processors")
    parser.add_argument("--list-jobs", action="store_true", help="List all jobs")
    parser.add_argument("--stats", action="store_true", help="Show processing statistics")
    parser.add_argument("--report", action="store_true", help="Generate processing report")
    parser.add_argument("--operation", choices=['clean', 'transform', 'filter', 'aggregate', 'merge', 'split', 'validate', 'enrich', 'normalize', 'deduplicate'],
                       help="Processing operation")
    parser.add_argument("--input-format", choices=['csv', 'json', 'xml', 'excel', 'parquet', 'text', 'markdown'],
                       help="Input data format")
    parser.add_argument("--output-format", choices=['csv', 'json', 'xml', 'excel', 'parquet', 'text', 'markdown'],
                       help="Output data format")
    parser.add_argument("--input-path", help="Input file path")
    parser.add_argument("--output-path", help="Output file path")
    parser.add_argument("--processor-id", help="Processor ID for job creation")
    parser.add_argument("--name", help="Name for processor/job")
    parser.add_argument("--parameters", help="JSON string of parameters")
    
    args = parser.parse_args()
    
    engine = DataProcessingEngine()
    
    print("⚙️ BUL Data Processing Engine")
    print("=" * 40)
    
    if args.create_processor:
        if not all([args.operation, args.input_format, args.output_format]):
            print("❌ Error: --operation, --input-format, and --output-format are required")
            return 1
        
        parameters = {}
        if args.parameters:
            try:
                parameters = json.loads(args.parameters)
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON in --parameters")
                return 1
        
        processor = engine.create_processor(
            processor_id=args.create_processor,
            name=args.name or f"Processor {args.create_processor}",
            operation=ProcessingOperation(args.operation),
            parameters=parameters,
            input_format=DataFormat(args.input_format),
            output_format=DataFormat(args.output_format)
        )
        print(f"✅ Created processor: {processor.name}")
    
    elif args.create_job:
        if not all([args.processor_id, args.input_path, args.output_path]):
            print("❌ Error: --processor-id, --input-path, and --output-path are required")
            return 1
        
        job = engine.create_processing_job(
            job_id=args.create_job,
            name=args.name or f"Job {args.create_job}",
            processor_id=args.processor_id,
            input_path=args.input_path,
            output_path=args.output_path
        )
        print(f"✅ Created job: {job.name}")
    
    elif args.execute_job:
        async def execute_job():
            try:
                result = await engine.execute_job(args.execute_job)
                print(f"✅ Job executed successfully")
                print(f"   Status: {result['status']}")
                if result.get('result_summary'):
                    print(f"   Processing time: {result['result_summary']['processing_time']:.2f}s")
            except Exception as e:
                print(f"❌ Job execution failed: {e}")
        
        asyncio.run(execute_job())
    
    elif args.list_processors:
        processors = engine.processors
        if processors:
            print(f"\n⚙️ Data Processors ({len(processors)}):")
            print("-" * 50)
            for processor_id, processor in processors.items():
                print(f"{processor.name} ({processor_id}):")
                print(f"  Operation: {processor.operation.value}")
                print(f"  Input: {processor.input_format.value}")
                print(f"  Output: {processor.output_format.value}")
                print()
        else:
            print("No processors found.")
    
    elif args.list_jobs:
        jobs = engine.jobs
        if jobs:
            print(f"\n📋 Processing Jobs ({len(jobs)}):")
            print("-" * 50)
            for job_id, job in jobs.items():
                print(f"{job.name} ({job_id}):")
                print(f"  Status: {job.status}")
                print(f"  Processor: {job.processor_id}")
                print(f"  Created: {job.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
        else:
            print("No jobs found.")
    
    elif args.stats:
        stats = engine.get_processing_stats()
        print(f"\n📊 Processing Statistics:")
        print(f"   Total Jobs: {stats['total_jobs']}")
        print(f"   Completed: {stats['completed_jobs']}")
        print(f"   Failed: {stats['failed_jobs']}")
        print(f"   Running: {stats['running_jobs']}")
        print(f"   Success Rate: {stats['success_rate']:.1f}%")
        print(f"   Total Processors: {stats['total_processors']}")
    
    elif args.report:
        report = engine.generate_processing_report()
        print(report)
        
        # Save report
        report_file = f"data_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")
    
    else:
        # Show quick overview
        stats = engine.get_processing_stats()
        print(f"⚙️ Processors: {stats['total_processors']}")
        print(f"📋 Jobs: {stats['total_jobs']}")
        print(f"✅ Completed: {stats['completed_jobs']}")
        print(f"❌ Failed: {stats['failed_jobs']}")
        print(f"📊 Success Rate: {stats['success_rate']:.1f}%")
        print(f"\n💡 Use --create-processor to create a new processor")
        print(f"💡 Use --create-job to create a new job")
        print(f"💡 Use --execute-job to run a job")
        print(f"💡 Use --report to generate processing report")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
