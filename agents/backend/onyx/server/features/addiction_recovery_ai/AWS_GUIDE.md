# AWS Guide - Addiction Recovery AI

## ✅ AWS Components

### AWS Structure

```
aws/
├── aws_services.py          # ✅ AWS services wrapper
├── lambda_handler.py        # ✅ Lambda function handler
├── event_processor.py       # ✅ Event processor
├── background_workers.py    # ✅ Background workers
├── circuit_breaker.py       # ✅ Circuit breaker
├── retry_handler.py         # ✅ Retry handler
├── prometheus_metrics.py    # ✅ Prometheus metrics
├── deploy.sh                # ✅ Deployment script
├── Dockerfile.lambda        # ✅ Lambda Dockerfile
├── sam_template.yaml        # ✅ SAM template
└── README.md                # ✅ AWS documentation
```

## 📦 AWS Components

### `aws/aws_services.py` - AWS Services
- **Status**: ✅ Active
- **Purpose**: AWS service wrappers
- **Services**:
  - DynamoDB
  - S3
  - CloudWatch
  - SNS
  - SQS
  - Parameter Store
  - Secrets Manager

**Usage:**
```python
from aws.aws_services import DynamoDBService, S3Service

dynamodb = DynamoDBService()
s3 = S3Service()

# Use services
dynamodb.put_item("table", {"key": "value"})
s3.upload_file("bucket", "key", "file.txt")
```

### `aws/lambda_handler.py` - Lambda Handler
- **Status**: ✅ Active
- **Purpose**: AWS Lambda function handler
- **Features**: FastAPI integration, event processing

### `aws/event_processor.py` - Event Processor
- **Status**: ✅ Active
- **Purpose**: Process AWS events
- **Features**: SQS, SNS, EventBridge integration

### `aws/background_workers.py` - Background Workers
- **Status**: ✅ Active
- **Purpose**: Background task processing
- **Features**: SQS integration, worker management

### `aws/circuit_breaker.py` - Circuit Breaker
- **Status**: ✅ Active
- **Purpose**: Circuit breaker pattern for AWS services
- **Features**: Failure detection, automatic recovery

### `aws/retry_handler.py` - Retry Handler
- **Status**: ✅ Active
- **Purpose**: Retry logic for AWS operations
- **Features**: Exponential backoff, jitter

### `aws/prometheus_metrics.py` - Prometheus Metrics
- **Status**: ✅ Active
- **Purpose**: Prometheus metrics for AWS
- **Features**: Metric collection, export

## 📝 Usage Examples

### AWS Services
```python
from aws.aws_services import DynamoDBService, S3Service

dynamodb = DynamoDBService()
item = dynamodb.get_item("users", {"user_id": "123"})

s3 = S3Service()
s3.upload_file("reports", "report.pdf", file_data)
```

### Lambda Handler
```python
from aws.lambda_handler import lambda_handler

# Lambda function
def handler(event, context):
    return lambda_handler(event, context)
```

### Event Processing
```python
from aws.event_processor import EventProcessor

processor = EventProcessor()
result = processor.process_sqs_event(event)
```

## 🚀 Deployment

### Using SAM
```bash
cd aws
sam build
sam deploy --guided
```

### Using deploy.sh
```bash
chmod +x aws/deploy.sh
./aws/deploy.sh
```

## 📚 Additional Resources

- See `AWS_DEPLOYMENT.md` for detailed deployment guide
- See `config/aws_settings.py` for AWS configuration
- See `api/health_advanced.py` for AWS health checks
- See `aws/README.md` for AWS-specific documentation






