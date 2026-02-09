"""
Modern Data Processing for BUL
==============================

Efficient data processing with pandas, numpy, and modern libraries.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Union, Iterator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import orjson
from ..utils.modern_logging import get_logger

logger = get_logger(__name__)

@dataclass
class DocumentMetrics:
    """Document processing metrics"""
    word_count: int
    character_count: int
    sentence_count: int
    paragraph_count: int
    reading_time_minutes: float
    complexity_score: float
    language: str
    created_at: datetime

class ModernDataProcessor:
    """Modern data processing utilities"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def analyze_document_content(self, content: str, language: str = "es") -> DocumentMetrics:
        """Analyze document content and extract metrics"""
        try:
            # Basic text analysis
            word_count = len(content.split())
            character_count = len(content)
            sentence_count = len([s for s in content.split('.') if s.strip()])
            paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
            
            # Reading time estimation (average 200 words per minute)
            reading_time_minutes = word_count / 200.0
            
            # Simple complexity score (0-1, higher = more complex)
            avg_word_length = np.mean([len(word) for word in content.split() if word])
            avg_sentence_length = word_count / max(sentence_count, 1)
            complexity_score = min(1.0, (avg_word_length / 10.0) * (avg_sentence_length / 20.0))
            
            return DocumentMetrics(
                word_count=word_count,
                character_count=character_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                reading_time_minutes=reading_time_minutes,
                complexity_score=complexity_score,
                language=language,
                created_at=datetime.now()
            )
        
        except Exception as e:
            self.logger.error(f"Error analyzing document content: {e}")
            # Return default metrics
            return DocumentMetrics(
                word_count=0,
                character_count=0,
                sentence_count=0,
                paragraph_count=0,
                reading_time_minutes=0.0,
                complexity_score=0.0,
                language=language,
                created_at=datetime.now()
            )
    
    def process_documents_batch(self, documents: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process multiple documents and return as DataFrame"""
        try:
            processed_data = []
            
            for doc in documents:
                metrics = self.analyze_document_content(
                    doc.get('content', ''),
                    doc.get('language', 'es')
                )
                
                processed_doc = {
                    'id': doc.get('id'),
                    'title': doc.get('title', ''),
                    'business_area': doc.get('business_area', ''),
                    'document_type': doc.get('document_type', ''),
                    'agent_used': doc.get('agent_used', ''),
                    'processing_time': doc.get('processing_time', 0.0),
                    'confidence_score': doc.get('confidence_score', 0.0),
                    'created_at': doc.get('created_at'),
                    **asdict(metrics)
                }
                
                processed_data.append(processed_doc)
            
            df = pd.DataFrame(processed_data)
            
            # Add derived columns
            df['efficiency_score'] = df['word_count'] / df['processing_time'].clip(lower=0.1)
            df['quality_score'] = df['confidence_score'] * (1 - df['complexity_score'])
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error processing documents batch: {e}")
            return pd.DataFrame()
    
    def generate_analytics_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate analytics report from processed documents"""
        try:
            if df.empty:
                return {}
            
            report = {
                'summary': {
                    'total_documents': len(df),
                    'total_words': df['word_count'].sum(),
                    'average_processing_time': df['processing_time'].mean(),
                    'average_confidence': df['confidence_score'].mean(),
                    'average_complexity': df['complexity_score'].mean()
                },
                'by_business_area': df.groupby('business_area').agg({
                    'id': 'count',
                    'word_count': 'mean',
                    'processing_time': 'mean',
                    'confidence_score': 'mean'
                }).to_dict('index'),
                'by_agent': df.groupby('agent_used').agg({
                    'id': 'count',
                    'processing_time': 'mean',
                    'confidence_score': 'mean',
                    'efficiency_score': 'mean'
                }).to_dict('index'),
                'by_document_type': df.groupby('document_type').agg({
                    'id': 'count',
                    'word_count': 'mean',
                    'complexity_score': 'mean'
                }).to_dict('index'),
                'performance_metrics': {
                    'fastest_agent': df.loc[df['processing_time'].idxmin(), 'agent_used'] if not df.empty else None,
                    'most_productive_agent': df.loc[df['efficiency_score'].idxmax(), 'agent_used'] if not df.empty else None,
                    'highest_quality_agent': df.loc[df['quality_score'].idxmax(), 'agent_used'] if not df.empty else None
                },
                'trends': self._calculate_trends(df)
            }
            
            return report
        
        except Exception as e:
            self.logger.error(f"Error generating analytics report: {e}")
            return {}
    
    def _calculate_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trends from document data"""
        try:
            if df.empty or 'created_at' not in df.columns:
                return {}
            
            # Convert created_at to datetime if it's not already
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['date'] = df['created_at'].dt.date
            
            # Daily trends
            daily_trends = df.groupby('date').agg({
                'id': 'count',
                'word_count': 'sum',
                'processing_time': 'mean',
                'confidence_score': 'mean'
            }).to_dict('index')
            
            # Calculate growth rates
            dates = sorted(daily_trends.keys())
            if len(dates) > 1:
                latest = daily_trends[dates[-1]]
                previous = daily_trends[dates[-2]]
                
                growth_rates = {
                    'documents_growth': ((latest['id'] - previous['id']) / max(previous['id'], 1)) * 100,
                    'words_growth': ((latest['word_count'] - previous['word_count']) / max(previous['word_count'], 1)) * 100,
                    'processing_time_change': ((latest['processing_time'] - previous['processing_time']) / max(previous['processing_time'], 0.1)) * 100,
                    'confidence_change': ((latest['confidence_score'] - previous['confidence_score']) / max(previous['confidence_score'], 0.1)) * 100
                }
            else:
                growth_rates = {}
            
            return {
                'daily_trends': daily_trends,
                'growth_rates': growth_rates
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating trends: {e}")
            return {}
    
    def optimize_json_serialization(self, data: Any) -> str:
        """Optimize JSON serialization using orjson"""
        try:
            return orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY).decode('utf-8')
        except Exception as e:
            self.logger.warning(f"orjson serialization failed, falling back to standard json: {e}")
            return json.dumps(data, default=str)
    
    def optimize_json_deserialization(self, json_str: str) -> Any:
        """Optimize JSON deserialization using orjson"""
        try:
            return orjson.loads(json_str)
        except Exception as e:
            self.logger.warning(f"orjson deserialization failed, falling back to standard json: {e}")
            return json.loads(json_str)
    
    def create_data_export(self, df: pd.DataFrame, format: str = "csv") -> bytes:
        """Create data export in various formats"""
        try:
            if format.lower() == "csv":
                return df.to_csv(index=False).encode('utf-8')
            elif format.lower() == "json":
                return self.optimize_json_serialization(df.to_dict('records')).encode('utf-8')
            elif format.lower() == "excel":
                # For Excel, we'd need openpyxl, but keeping it simple
                return df.to_csv(index=False).encode('utf-8')
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        except Exception as e:
            self.logger.error(f"Error creating data export: {e}")
            return b""
    
    def detect_anomalies(self, df: pd.DataFrame, column: str) -> List[int]:
        """Detect anomalies in data using statistical methods"""
        try:
            if column not in df.columns or df.empty:
                return []
            
            data = df[column].dropna()
            if len(data) < 3:
                return []
            
            # Use IQR method for anomaly detection
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
            
            return anomalies
        
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def calculate_correlation_matrix(self, df: pd.DataFrame, numeric_columns: List[str] = None) -> pd.DataFrame:
        """Calculate correlation matrix for numeric columns"""
        try:
            if numeric_columns is None:
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                return pd.DataFrame()
            
            correlation_matrix = df[numeric_columns].corr()
            return correlation_matrix
        
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()

class AsyncDataProcessor:
    """Async version of data processor for better performance"""
    
    def __init__(self):
        self.processor = ModernDataProcessor()
        self.logger = get_logger(__name__)
    
    async def process_documents_async(self, documents: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process documents asynchronously"""
        try:
            # Process in chunks to avoid blocking
            chunk_size = 10
            chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]
            
            results = []
            for chunk in chunks:
                # Process chunk in thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.processor.process_documents_batch, chunk
                )
                results.append(result)
            
            # Combine results
            if results:
                return pd.concat(results, ignore_index=True)
            else:
                return pd.DataFrame()
        
        except Exception as e:
            self.logger.error(f"Error in async document processing: {e}")
            return pd.DataFrame()
    
    async def generate_analytics_async(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate analytics report asynchronously"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.processor.generate_analytics_report, df
            )
        except Exception as e:
            self.logger.error(f"Error in async analytics generation: {e}")
            return {}

# Global instances
_data_processor: Optional[ModernDataProcessor] = None
_async_data_processor: Optional[AsyncDataProcessor] = None

def get_data_processor() -> ModernDataProcessor:
    """Get the global data processor instance"""
    global _data_processor
    if _data_processor is None:
        _data_processor = ModernDataProcessor()
    return _data_processor

def get_async_data_processor() -> AsyncDataProcessor:
    """Get the global async data processor instance"""
    global _async_data_processor
    if _async_data_processor is None:
        _async_data_processor = AsyncDataProcessor()
    return _async_data_processor




