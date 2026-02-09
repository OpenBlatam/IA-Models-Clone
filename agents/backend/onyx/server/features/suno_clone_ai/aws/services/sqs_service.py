"""
SQS Service para message queuing
"""

import logging
import json
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


class SQSService:
    """
    Servicio para interactuar con SQS
    Para procesamiento asíncrono de tareas
    """
    
    def __init__(
        self,
        queue_url: Optional[str] = None,
        queue_name: Optional[str] = None,
        region_name: Optional[str] = None
    ):
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for SQSService")
        
        self.sqs_client = boto3.client('sqs', region_name=region_name)
        
        if queue_url:
            self.queue_url = queue_url
        elif queue_name:
            # Obtener URL de la cola
            try:
                response = self.sqs_client.get_queue_url(QueueName=queue_name)
                self.queue_url = response['QueueUrl']
            except ClientError as e:
                logger.error(f"Failed to get queue URL: {e}")
                raise
        else:
            raise ValueError("Either queue_url or queue_name must be provided")
    
    def send_message(
        self,
        message_body: Dict[str, Any],
        delay_seconds: int = 0,
        message_attributes: Optional[Dict[str, Any]] = None,
        message_group_id: Optional[str] = None  # Para FIFO queues
    ) -> Dict[str, Any]:
        """
        Envía un mensaje a la cola
        
        Args:
            message_body: Cuerpo del mensaje (será serializado a JSON)
            delay_seconds: Retraso antes de que el mensaje sea visible
            message_attributes: Atributos adicionales del mensaje
            message_group_id: ID de grupo para colas FIFO
            
        Returns:
            Respuesta de SQS con MessageId
        """
        try:
            kwargs = {
                'QueueUrl': self.queue_url,
                'MessageBody': json.dumps(message_body)
            }
            
            if delay_seconds > 0:
                kwargs['DelaySeconds'] = delay_seconds
            
            if message_attributes:
                kwargs['MessageAttributes'] = message_attributes
            
            if message_group_id:
                kwargs['MessageGroupId'] = message_group_id
                # FIFO queues requieren MessageDeduplicationId
                kwargs['MessageDeduplicationId'] = f"{message_group_id}-{datetime.utcnow().isoformat()}"
            
            response = self.sqs_client.send_message(**kwargs)
            logger.info(f"Message sent to SQS: {response.get('MessageId')}")
            return response
        except ClientError as e:
            logger.error(f"SQS send_message error: {e}")
            raise
    
    def receive_messages(
        self,
        max_number: int = 1,
        wait_time_seconds: int = 0,
        visibility_timeout: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Recibe mensajes de la cola
        
        Args:
            max_number: Número máximo de mensajes a recibir (1-10)
            wait_time_seconds: Long polling (0-20)
            visibility_timeout: Tiempo de visibilidad en segundos
            
        Returns:
            Lista de mensajes
        """
        try:
            kwargs = {
                'QueueUrl': self.queue_url,
                'MaxNumberOfMessages': min(max_number, 10),
                'WaitTimeSeconds': wait_time_seconds
            }
            
            if visibility_timeout:
                kwargs['VisibilityTimeout'] = visibility_timeout
            
            response = self.sqs_client.receive_message(**kwargs)
            
            messages = []
            for msg in response.get('Messages', []):
                try:
                    body = json.loads(msg['Body'])
                    messages.append({
                        'ReceiptHandle': msg['ReceiptHandle'],
                        'MessageId': msg['MessageId'],
                        'Body': body,
                        'Attributes': msg.get('Attributes', {}),
                        'MessageAttributes': msg.get('MessageAttributes', {})
                    })
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse message body: {msg.get('Body')}")
            
            return messages
        except ClientError as e:
            logger.error(f"SQS receive_messages error: {e}")
            raise
    
    def delete_message(self, receipt_handle: str) -> bool:
        """
        Elimina un mensaje de la cola después de procesarlo
        
        Args:
            receipt_handle: Handle del mensaje recibido
            
        Returns:
            True si fue exitoso
        """
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            return True
        except ClientError as e:
            logger.error(f"SQS delete_message error: {e}")
            raise
    
    def send_batch(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Envía múltiples mensajes en batch (máximo 10)
        
        Args:
            messages: Lista de diccionarios con 'Id' y 'MessageBody'
            
        Returns:
            Respuesta de SQS
        """
        try:
            entries = []
            for i, msg in enumerate(messages[:10]):  # SQS limita a 10
                entry = {
                    'Id': msg.get('Id', str(i)),
                    'MessageBody': json.dumps(msg.get('MessageBody', msg))
                }
                
                if 'DelaySeconds' in msg:
                    entry['DelaySeconds'] = msg['DelaySeconds']
                if 'MessageAttributes' in msg:
                    entry['MessageAttributes'] = msg['MessageAttributes']
                if 'MessageGroupId' in msg:
                    entry['MessageGroupId'] = msg['MessageGroupId']
                    entry['MessageDeduplicationId'] = msg.get(
                        'MessageDeduplicationId',
                        f"{msg['MessageGroupId']}-{datetime.utcnow().isoformat()}"
                    )
                
                entries.append(entry)
            
            response = self.sqs_client.send_message_batch(
                QueueUrl=self.queue_url,
                Entries=entries
            )
            
            logger.info(f"Batch sent to SQS: {len(response.get('Successful', []))} successful")
            return response
        except ClientError as e:
            logger.error(f"SQS send_batch error: {e}")
            raise















