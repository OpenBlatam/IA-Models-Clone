"""
Data Processing Module
Ported from backend/bulk/data_processing_engine.py and adapted for Unified AI Model.
Supports multiple data formats and processing operations.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)

# Optional imports - will gracefully degrade if not available
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas/numpy not available. Some data processing features will be limited.")


class DataFormat(Enum):
    """Supported data formats."""
    CSV = "csv"
    JSON = "json"
    TEXT = "text"
    MARKDOWN = "markdown"
    # Excel and Parquet require additional dependencies
    EXCEL = "excel"
    PARQUET = "parquet"


class ProcessingOperation(Enum):
    """Data processing operations."""
    CLEAN = "clean"
    TRANSFORM = "transform"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    VALIDATE = "validate"
    NORMALIZE = "normalize"
    DEDUPLICATE = "deduplicate"


@dataclass
class ProcessingResult:
    """Result of a data processing operation."""
    success: bool
    data: Any
    operation: str
    input_size: int
    output_size: int
    processing_time_ms: float
    message: str = ""


class DataProcessor:
    """
    Data processor for Unified AI Model.
    Handles multiple formats and processing operations.
    """
    
    def __init__(self, output_dir: str = "./data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processing_history: List[Dict[str, Any]] = []
        logger.info(f"DataProcessor initialized. Output dir: {self.output_dir}")
    
    async def load_data(self, file_path: str, data_format: DataFormat) -> Any:
        """Load data from file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        if data_format == DataFormat.CSV:
            if not PANDAS_AVAILABLE:
                raise ImportError("pandas required for CSV processing")
            return pd.read_csv(file_path)
        
        elif data_format == DataFormat.JSON:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        elif data_format in (DataFormat.TEXT, DataFormat.MARKDOWN):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif data_format == DataFormat.EXCEL:
            if not PANDAS_AVAILABLE:
                raise ImportError("pandas required for Excel processing")
            return pd.read_excel(file_path)
        
        elif data_format == DataFormat.PARQUET:
            if not PANDAS_AVAILABLE:
                raise ImportError("pandas required for Parquet processing")
            return pd.read_parquet(file_path)
        
        else:
            raise ValueError(f"Unsupported format: {data_format}")
    
    async def save_data(self, data: Any, file_path: str, data_format: DataFormat) -> str:
        """Save data to file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if data_format == DataFormat.CSV:
            if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
            else:
                raise ValueError("Data must be a pandas DataFrame for CSV")
        
        elif data_format == DataFormat.JSON:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        
        elif data_format in (DataFormat.TEXT, DataFormat.MARKDOWN):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
        
        elif data_format == DataFormat.EXCEL:
            if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                data.to_excel(file_path, index=False)
            else:
                raise ValueError("Data must be a pandas DataFrame for Excel")
        
        elif data_format == DataFormat.PARQUET:
            if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                data.to_parquet(file_path, index=False)
            else:
                raise ValueError("Data must be a pandas DataFrame for Parquet")
        
        else:
            raise ValueError(f"Unsupported format: {data_format}")
        
        return str(path.absolute())
    
    async def process(
        self, 
        data: Any, 
        operation: ProcessingOperation, 
        parameters: Dict[str, Any] = None
    ) -> ProcessingResult:
        """Process data with specified operation."""
        start_time = datetime.now()
        parameters = parameters or {}
        
        input_size = len(data) if hasattr(data, '__len__') else 1
        
        try:
            if operation == ProcessingOperation.CLEAN:
                result = await self._clean(data, parameters)
            elif operation == ProcessingOperation.TRANSFORM:
                result = await self._transform(data, parameters)
            elif operation == ProcessingOperation.FILTER:
                result = await self._filter(data, parameters)
            elif operation == ProcessingOperation.AGGREGATE:
                result = await self._aggregate(data, parameters)
            elif operation == ProcessingOperation.VALIDATE:
                result = await self._validate(data, parameters)
            elif operation == ProcessingOperation.NORMALIZE:
                result = await self._normalize(data, parameters)
            elif operation == ProcessingOperation.DEDUPLICATE:
                result = await self._deduplicate(data, parameters)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            output_size = len(result) if hasattr(result, '__len__') else 1
            
            return ProcessingResult(
                success=True,
                data=result,
                operation=operation.value,
                input_size=input_size,
                output_size=output_size,
                processing_time_ms=processing_time,
                message="Processing completed successfully"
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Processing failed: {e}")
            return ProcessingResult(
                success=False,
                data=None,
                operation=operation.value,
                input_size=input_size,
                output_size=0,
                processing_time_ms=processing_time,
                message=str(e)
            )
    
    async def _clean(self, data: Any, params: Dict[str, Any]) -> Any:
        """Clean data."""
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            df = data.copy()
            
            if params.get('remove_duplicates', False):
                df = df.drop_duplicates()
            
            if params.get('handle_missing') == 'drop':
                df = df.dropna()
            elif params.get('handle_missing') == 'fill':
                df = df.fillna(params.get('fill_value', 0))
            
            if params.get('clean_text', False):
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].astype(str).str.strip()
            
            return df
        
        elif isinstance(data, str):
            text = re.sub(r'\s+', ' ', data).strip()
            if params.get('lowercase', False):
                text = text.lower()
            return text
        
        return data
    
    async def _transform(self, data: Any, params: Dict[str, Any]) -> Any:
        """Transform data."""
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            df = data.copy()
            
            transformations = params.get('transformations', {})
            for column, transform in transformations.items():
                if column in df.columns:
                    if transform.get('type') == 'scale':
                        min_val, max_val = df[column].min(), df[column].max()
                        if max_val != min_val:
                            df[column] = (df[column] - min_val) / (max_val - min_val)
            
            return df
        
        return data
    
    async def _filter(self, data: Any, params: Dict[str, Any]) -> Any:
        """Filter data."""
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            df = data.copy()
            
            for column, value in params.get('column_filters', {}).items():
                if column in df.columns:
                    if isinstance(value, list):
                        df = df[df[column].isin(value)]
                    else:
                        df = df[df[column] == value]
            
            return df
        
        return data
    
    async def _aggregate(self, data: Any, params: Dict[str, Any]) -> Any:
        """Aggregate data."""
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            group_by = params.get('group_by', [])
            if group_by:
                aggs = params.get('aggregations', {})
                if aggs:
                    return data.groupby(group_by).agg(aggs).reset_index()
                return data.groupby(group_by).sum().reset_index()
        
        return data
    
    async def _validate(self, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data and return validation report."""
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'missing_values': data.isnull().sum().to_dict(),
                'duplicates': int(data.duplicated().sum()),
                'data_types': {str(k): str(v) for k, v in data.dtypes.to_dict().items()}
            }
        
        return {'type': type(data).__name__, 'length': len(data) if hasattr(data, '__len__') else 1}
    
    async def _normalize(self, data: Any, params: Dict[str, Any]) -> Any:
        """Normalize numeric data."""
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            df = data.copy()
            method = params.get('method', 'minmax')
            
            for col in df.select_dtypes(include=[np.number]).columns:
                if method == 'minmax':
                    min_val, max_val = df[col].min(), df[col].max()
                    if max_val != min_val:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    mean_val, std_val = df[col].mean(), df[col].std()
                    if std_val != 0:
                        df[col] = (df[col] - mean_val) / std_val
            
            return df
        
        return data
    
    async def _deduplicate(self, data: Any, params: Dict[str, Any]) -> Any:
        """Remove duplicates."""
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            subset = params.get('subset', None)
            keep = params.get('keep', 'first')
            return data.drop_duplicates(subset=subset, keep=keep)
        
        return data
    
    def export_to_markdown(self, data: Any, title: str = "Data Report") -> str:
        """Export data to markdown format."""
        lines = [f"# {title}", f"Generated: {datetime.now().isoformat()}", ""]
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            lines.append(f"**Rows:** {len(data)} | **Columns:** {len(data.columns)}")
            lines.append("")
            lines.append("## Data Preview")
            lines.append(data.head(10).to_markdown(index=False))
        elif isinstance(data, dict):
            lines.append("## Data")
            lines.append("```json")
            lines.append(json.dumps(data, indent=2, default=str))
            lines.append("```")
        else:
            lines.append(str(data))
        
        return "\n".join(lines)
