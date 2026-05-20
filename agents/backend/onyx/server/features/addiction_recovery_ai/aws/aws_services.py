"""
AWS Services Integration
Provides wrappers for AWS services with retry logic and error handling
"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config

from config.aws_settings import get_aws_settings
from .circuit_breaker import CircuitBreaker
from .retry_handler import retry_with_backoff

logger = logging.getLogger(__name__)
aws_settings = get_aws_settings()

# Boto3 client configuration with retries
boto_config = Config(
    retries={
        'max_attempts': aws_settings.max_retries,
        'mode': 'adaptive'
    },
    connect_timeout=5,
    read_timeout=5
)


class DynamoDBService:
    """DynamoDB service wrapper with circuit breaker and retry logic"""
    
    def __init__(self):
        self.settings = get_aws_settings()
        self.client = boto3.client(
            'dynamodb',
            region_name=self.settings.aws_region,
            endpoint_url=self.settings.dynamodb_endpoint_url,
            config=boto_config
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.settings.circuit_breaker_failure_threshold,
            timeout=self.settings.circuit_breaker_timeout
        )
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def get_item(self, key: Dict[str, Any], table_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get item from DynamoDB"""
        table = table_name or self.settings.dynamodb_table_name
        
        try:
            with self.circuit_breaker:
                response = self.client.get_item(
                    TableName=table,
                    Key=key
                )
                return response.get('Item')
        except Exception as e:
            logger.error(f"Error getting item from DynamoDB: {str(e)}")
            raise
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def put_item(self, item: Dict[str, Any], table_name: Optional[str] = None) -> Dict[str, Any]:
        """Put item in DynamoDB"""
        table = table_name or self.settings.dynamodb_table_name
        
        try:
            with self.circuit_breaker:
                # Add timestamp
                item['updated_at'] = datetime.utcnow().isoformat()
                if 'created_at' not in item:
                    item['created_at'] = item['updated_at']
                
                response = self.client.put_item(
                    TableName=table,
                    Item=self._serialize_item(item)
                )
                return response
        except Exception as e:
            logger.error(f"Error putting item to DynamoDB: {str(e)}")
            raise
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def query(self, key_condition_expression: str, expression_attribute_values: Dict[str, Any],
              table_name: Optional[str] = None, index_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query DynamoDB table"""
        table = table_name or self.settings.dynamodb_table_name
        
        try:
            with self.circuit_breaker:
                params = {
                    'TableName': table,
                    'KeyConditionExpression': key_condition_expression,
                    'ExpressionAttributeValues': expression_attribute_values
                }
                if index_name:
                    params['IndexName'] = index_name
                
                response = self.client.query(**params)
                return [self._deserialize_item(item) for item in response.get('Items', [])]
        except Exception as e:
            logger.error(f"Error querying DynamoDB: {str(e)}")
            raise
    
    def _serialize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize Python types to DynamoDB format"""
        dynamodb_item = {}
        for key, value in item.items():
            if isinstance(value, str):
                dynamodb_item[key] = {'S': value}
            elif isinstance(value, (int, float)):
                dynamodb_item[key] = {'N': str(value)}
            elif isinstance(value, bool):
                dynamodb_item[key] = {'BOOL': value}
            elif isinstance(value, dict):
                dynamodb_item[key] = {'M': self._serialize_item(value)}
            elif isinstance(value, list):
                dynamodb_item[key] = {'L': [self._serialize_value(v) for v in value]}
            elif value is None:
                dynamodb_item[key] = {'NULL': True}
        return dynamodb_item
    
    def _deserialize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize DynamoDB format to Python types"""
        python_item = {}
        for key, value in item.items():
            python_item[key] = self._deserialize_value(value)
        return python_item
    
    def _serialize_value(self, value: Any) -> Dict[str, Any]:
        """Serialize single value"""
        if isinstance(value, str):
            return {'S': value}
        elif isinstance(value, (int, float)):
            return {'N': str(value)}
        elif isinstance(value, bool):
            return {'BOOL': value}
        elif isinstance(value, dict):
            return {'M': self._serialize_item(value)}
        elif isinstance(value, list):
            return {'L': [self._serialize_value(v) for v in value]}
        elif value is None:
            return {'NULL': True}
        return {'S': str(value)}
    
    def _deserialize_value(self, value: Dict[str, Any]) -> Any:
        """Deserialize single DynamoDB value"""
        if 'S' in value:
            return value['S']
        elif 'N' in value:
            return float(value['N']) if '.' in value['N'] else int(value['N'])
        elif 'BOOL' in value:
            return value['BOOL']
        elif 'M' in value:
            return self._deserialize_item(value['M'])
        elif 'L' in value:
            return [self._deserialize_value(v) for v in value['L']]
        elif 'NULL' in value:
            return None
        return value


class S3Service:
    """S3 service wrapper with circuit breaker and retry logic"""
    
    def __init__(self):
        self.settings = get_aws_settings()
        self.client = boto3.client(
            's3',
            region_name=self.settings.aws_region,
            endpoint_url=self.settings.s3_endpoint_url,
            config=boto_config
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.settings.circuit_breaker_failure_threshold,
            timeout=self.settings.circuit_breaker_timeout
        )
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def upload_file(self, file_content: bytes, key: str, bucket_name: Optional[str] = None,
                   content_type: Optional[str] = None) -> str:
        """Upload file to S3"""
        bucket = bucket_name or self.settings.s3_bucket_name
        
        try:
            with self.circuit_breaker:
                extra_args = {}
                if content_type:
                    extra_args['ContentType'] = content_type
                
                self.client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=file_content,
                    **extra_args
                )
                return f"s3://{bucket}/{key}"
        except Exception as e:
            logger.error(f"Error uploading file to S3: {str(e)}")
            raise
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def download_file(self, key: str, bucket_name: Optional[str] = None) -> bytes:
        """Download file from S3"""
        bucket = bucket_name or self.settings.s3_bucket_name
        
        try:
            with self.circuit_breaker:
                response = self.client.get_object(Bucket=bucket, Key=key)
                return response['Body'].read()
        except Exception as e:
            logger.error(f"Error downloading file from S3: {str(e)}")
            raise
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def delete_file(self, key: str, bucket_name: Optional[str] = None) -> None:
        """Delete file from S3"""
        bucket = bucket_name or self.settings.s3_bucket_name
        
        try:
            with self.circuit_breaker:
                self.client.delete_object(Bucket=bucket, Key=key)
        except Exception as e:
            logger.error(f"Error deleting file from S3: {str(e)}")
            raise


class CloudWatchService:
    """CloudWatch service for metrics and logging"""
    
    def __init__(self):
        self.settings = get_aws_settings()
        self.client = boto3.client(
            'cloudwatch',
            region_name=self.settings.aws_region,
            config=boto_config
        )
    
    def put_metric(self, metric_name: str, value: float, unit: str = "Count",
                   dimensions: Optional[Dict[str, str]] = None) -> None:
        """Put custom metric to CloudWatch"""
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.utcnow(),
                'Namespace': self.settings.cloudwatch_metrics_namespace
            }
            
            if dimensions:
                metric_data['Dimensions'] = [
                    {'Name': k, 'Value': v} for k, v in dimensions.items()
                ]
            
            self.client.put_metric_data(
                Namespace=self.settings.cloudwatch_metrics_namespace,
                MetricData=[metric_data]
            )
        except Exception as e:
            logger.error(f"Error putting metric to CloudWatch: {str(e)}")
    
    def log_event(self, message: str, level: str = "INFO", metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log structured event to CloudWatch"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message
        }
        
        if metadata:
            log_data['metadata'] = metadata
        
        logger.log(getattr(logging, level.upper(), logging.INFO), json.dumps(log_data))


class SNSService:
    """SNS service for notifications"""
    
    def __init__(self):
        self.settings = get_aws_settings()
        self.client = boto3.client(
            'sns',
            region_name=self.settings.aws_region,
            config=boto_config
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.settings.circuit_breaker_failure_threshold,
            timeout=self.settings.circuit_breaker_timeout
        )
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def publish(self, message: str, subject: Optional[str] = None,
               topic_arn: Optional[str] = None) -> str:
        """Publish message to SNS topic"""
        topic = topic_arn or self.settings.sns_topic_arn
        
        if not topic:
            raise ValueError("SNS topic ARN not configured")
        
        try:
            with self.circuit_breaker:
                params = {
                    'TopicArn': topic,
                    'Message': message
                }
                if subject:
                    params['Subject'] = subject
                
                response = self.client.publish(**params)
                return response['MessageId']
        except Exception as e:
            logger.error(f"Error publishing to SNS: {str(e)}")
            raise


class SQSService:
    """SQS service for async message processing"""
    
    def __init__(self):
        self.settings = get_aws_settings()
        self.client = boto3.client(
            'sqs',
            region_name=self.settings.aws_region,
            config=boto_config
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.settings.circuit_breaker_failure_threshold,
            timeout=self.settings.circuit_breaker_timeout
        )
    
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def receive_messages(self, queue_url: Optional[str] = None, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Receive messages from SQS queue"""
        queue = queue_url or self.settings.sqs_queue_url
        
        if not queue:
            raise ValueError("SQS queue URL not configured")
        
        try:
            with self.circuit_breaker:
                response = self.client.receive_message(
                    QueueUrl=queue,
                    MaxNumberOfMessages=max_messages
                )
                return response.get('Messages', [])
        except Exception as e:
            logger.error(f"Error receiving messages from SQS: {str(e)}")
            raise
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def delete_message(self, queue_url: str, receipt_handle: str) -> None:
        """Delete message from SQS queue"""
        try:
            with self.circuit_breaker:
                self.client.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle
                )
        except Exception as e:
            logger.error(f"Error deleting message from SQS: {str(e)}")
            raise
    
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def send_message(self, message_body: Dict[str, Any], queue_url: Optional[str] = None,
                    delay_seconds: int = 0, message_attributes: Optional[Dict[str, Any]] = None) -> str:
        """Send message to SQS queue with optional delay and attributes"""
        queue = queue_url or self.settings.sqs_queue_url
        
        if not queue:
            raise ValueError("SQS queue URL not configured")
        
        try:
            with self.circuit_breaker:
                import json
                params = {
                    'QueueUrl': queue,
                    'MessageBody': json.dumps(message_body) if isinstance(message_body, dict) else message_body
                }
                
                if delay_seconds > 0:
                    params['DelaySeconds'] = delay_seconds
                
                if message_attributes:
                    params['MessageAttributes'] = message_attributes
                
                response = self.client.send_message(**params)
                return response['MessageId']
        except Exception as e:
            logger.error(f"Error sending message to SQS: {str(e)}")
            raise


class SecretsManagerService:
    """AWS Secrets Manager service"""
    
    def __init__(self):
        self.settings = get_aws_settings()
        self.client = boto3.client(
            'secretsmanager',
            region_name=self.settings.aws_region,
            config=boto_config
        )
        self._cache: Dict[str, Any] = {}
    
    def get_secret(self, secret_name: Optional[str] = None, use_cache: bool = True) -> Dict[str, Any]:
        """Get secret from Secrets Manager"""
        secret_name = secret_name or self.settings.secrets_manager_secret_name
        
        if not secret_name:
            raise ValueError("Secrets Manager secret name not configured")
        
        # Check cache
        if use_cache and secret_name in self._cache:
            return self._cache[secret_name]
        
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            secret = json.loads(response['SecretString'])
            
            if use_cache:
                self._cache[secret_name] = secret
            
            return secret
        except Exception as e:
            logger.error(f"Error getting secret from Secrets Manager: {str(e)}")
            raise


class ParameterStoreService:
    """AWS Systems Manager Parameter Store service"""
    
    def __init__(self):
        self.settings = get_aws_settings()
        self.client = boto3.client(
            'ssm',
            region_name=self.settings.aws_region,
            config=boto_config
        )
        self._cache: Dict[str, str] = {}
    
    def get_parameter(self, parameter_name: str, use_cache: bool = True) -> str:
        """Get parameter from Parameter Store"""
        full_name = f"{self.settings.parameter_store_prefix}/{parameter_name}"
        
        # Check cache
        if use_cache and full_name in self._cache:
            return self._cache[full_name]
        
        try:
            response = self.client.get_parameter(
                Name=full_name,
                WithDecryption=True
            )
            value = response['Parameter']['Value']
            
            if use_cache:
                self._cache[full_name] = value
            
            return value
        except Exception as e:
            logger.error(f"Error getting parameter from Parameter Store: {str(e)}")
            raise

