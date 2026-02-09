"""
CloudWatch Service para logging y métricas
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available. Install with: pip install boto3")


class CloudWatchService:
    """
    Servicio para interactuar con CloudWatch Logs y Metrics
    """
    
    def __init__(self, region_name: Optional[str] = None):
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for CloudWatchService")
        
        self.logs_client = boto3.client('logs', region_name=region_name)
        self.metrics_client = boto3.client('cloudwatch', region_name=region_name)
    
    def put_log_event(
        self,
        log_group_name: str,
        log_stream_name: str,
        message: str,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Envía un evento de log a CloudWatch
        
        Args:
            log_group_name: Nombre del grupo de logs
            log_stream_name: Nombre del stream de logs
            message: Mensaje del log
            timestamp: Timestamp del evento (default: ahora)
            
        Returns:
            True si fue exitoso
        """
        try:
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            # Convertir a milisegundos
            timestamp_ms = int(timestamp.timestamp() * 1000)
            
            self.logs_client.put_log_events(
                logGroupName=log_group_name,
                logStreamName=log_stream_name,
                logEvents=[
                    {
                        'timestamp': timestamp_ms,
                        'message': message
                    }
                ]
            )
            return True
        except ClientError as e:
            # Si el log group/stream no existe, crearlo
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                try:
                    self._ensure_log_group_stream(log_group_name, log_stream_name)
                    return self.put_log_event(log_group_name, log_stream_name, message, timestamp)
                except Exception as create_error:
                    logger.error(f"Failed to create log group/stream: {create_error}")
            logger.error(f"CloudWatch put_log_event error: {e}")
            return False
    
    def _ensure_log_group_stream(self, log_group_name: str, log_stream_name: str):
        """Asegura que el log group y stream existan"""
        try:
            # Crear log group si no existe
            try:
                self.logs_client.create_log_group(logGroupName=log_group_name)
            except ClientError as e:
                if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                    raise
            
            # Crear log stream si no existe
            try:
                self.logs_client.create_log_stream(
                    logGroupName=log_group_name,
                    logStreamName=log_stream_name
                )
            except ClientError as e:
                if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                    raise
        except Exception as e:
            logger.error(f"Failed to ensure log group/stream: {e}")
            raise
    
    def put_metric(
        self,
        namespace: str,
        metric_name: str,
        value: float,
        unit: str = "Count",
        dimensions: Optional[List[Dict[str, str]]] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Envía una métrica a CloudWatch
        
        Args:
            namespace: Namespace de la métrica
            metric_name: Nombre de la métrica
            value: Valor de la métrica
            unit: Unidad (Count, Bytes, Seconds, etc.)
            dimensions: Dimensiones adicionales
            timestamp: Timestamp del evento
            
        Returns:
            True si fue exitoso
        """
        try:
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': timestamp
            }
            
            if dimensions:
                metric_data['Dimensions'] = dimensions
            
            self.metrics_client.put_metric_data(
                Namespace=namespace,
                MetricData=[metric_data]
            )
            return True
        except ClientError as e:
            logger.error(f"CloudWatch put_metric error: {e}")
            return False
    
    def put_metric_batch(
        self,
        namespace: str,
        metric_data: List[Dict[str, Any]]
    ) -> bool:
        """
        Envía múltiples métricas en batch (máximo 20)
        
        Args:
            namespace: Namespace de las métricas
            metric_data: Lista de diccionarios con métricas
            
        Returns:
            True si fue exitoso
        """
        try:
            # CloudWatch limita a 20 métricas por batch
            for batch in [metric_data[i:i+20] for i in range(0, len(metric_data), 20)]:
                self.metrics_client.put_metric_data(
                    Namespace=namespace,
                    MetricData=batch
                )
            return True
        except ClientError as e:
            logger.error(f"CloudWatch put_metric_batch error: {e}")
            return False















