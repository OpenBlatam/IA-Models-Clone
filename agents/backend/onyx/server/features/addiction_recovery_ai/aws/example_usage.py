"""
Example usage of AWS services in Addiction Recovery AI
Demonstrates how to use DynamoDB, S3, CloudWatch, etc.
"""

import asyncio
from aws.aws_services import (
    DynamoDBService,
    S3Service,
    CloudWatchService,
    SNSService,
    SQSService,
    SecretsManagerService,
    ParameterStoreService
)
from config.aws_settings import get_aws_settings


async def example_dynamodb():
    """Example: Using DynamoDB to store user data"""
    print("=== DynamoDB Example ===")
    
    dynamodb = DynamoDBService()
    
    # Store user data
    user_data = {
        "user_id": {"S": "user123"},
        "email": {"S": "user@example.com"},
        "name": {"S": "John Doe"},
        "addiction_type": {"S": "smoking"},
        "days_sober": {"N": "30"}
    }
    
    try:
        # Put item
        dynamodb.put_item(user_data)
        print("✓ User data stored in DynamoDB")
        
        # Get item
        user = dynamodb.get_item({"user_id": {"S": "user123"}})
        print(f"✓ Retrieved user: {user}")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")


async def example_s3():
    """Example: Using S3 to store files"""
    print("\n=== S3 Example ===")
    
    s3 = S3Service()
    
    # Upload a report
    report_content = b"User recovery report data..."
    report_key = "reports/user123/report_2024.pdf"
    
    try:
        s3_url = s3.upload_file(
            file_content=report_content,
            key=report_key,
            content_type="application/pdf"
        )
        print(f"✓ Report uploaded to: {s3_url}")
        
        # Download file
        downloaded = s3.download_file(report_key)
        print(f"✓ Downloaded {len(downloaded)} bytes")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")


async def example_cloudwatch():
    """Example: Sending custom metrics to CloudWatch"""
    print("\n=== CloudWatch Example ===")
    
    cloudwatch = CloudWatchService()
    
    try:
        # Send custom metric
        cloudwatch.put_metric(
            metric_name="RecoveryProgress",
            value=75.5,
            unit="Percent",
            dimensions={
                "UserID": "user123",
                "AddictionType": "smoking"
            }
        )
        print("✓ Metric sent to CloudWatch")
        
        # Log structured event
        cloudwatch.log_event(
            message="User completed recovery milestone",
            level="INFO",
            metadata={
                "user_id": "user123",
                "milestone": "30_days_sober"
            }
        )
        print("✓ Event logged to CloudWatch")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")


async def example_sns():
    """Example: Sending notifications via SNS"""
    print("\n=== SNS Example ===")
    
    sns = SNSService()
    
    try:
        message = {
            "user_id": "user123",
            "type": "milestone",
            "message": "Congratulations on 30 days sober!",
            "timestamp": "2024-01-15T10:00:00Z"
        }
        
        import json
        message_id = sns.publish(
            message=json.dumps(message),
            subject="Recovery Milestone Achieved"
        )
        print(f"✓ Notification sent: {message_id}")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")


async def example_sqs():
    """Example: Sending background tasks to SQS"""
    print("\n=== SQS Example ===")
    
    sqs = SQSService()
    
    try:
        # Send background task
        task = {
            "task_type": "generate_report",
            "user_id": "user123",
            "report_type": "monthly_summary"
        }
        
        message_id = sqs.send_message(task)
        print(f"✓ Background task queued: {message_id}")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")


async def example_secrets():
    """Example: Retrieving secrets from Secrets Manager"""
    print("\n=== Secrets Manager Example ===")
    
    secrets = SecretsManagerService()
    
    try:
        secret_data = secrets.get_secret()
        print(f"✓ Retrieved secrets (keys: {list(secret_data.keys())})")
        # Don't print actual secret values for security
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")


async def example_parameter_store():
    """Example: Getting configuration from Parameter Store"""
    print("\n=== Parameter Store Example ===")
    
    params = ParameterStoreService()
    
    try:
        # Get configuration parameter
        api_key = params.get_parameter("openai/api_key")
        print(f"✓ Retrieved parameter (length: {len(api_key)})")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")


async def main():
    """Run all examples"""
    print("AWS Services Integration Examples\n")
    print("=" * 50)
    
    # Check if running in AWS environment
    settings = get_aws_settings()
    print(f"Environment: {settings.environment}")
    print(f"Region: {settings.aws_region}")
    print(f"Is Lambda: {settings.is_lambda}")
    print("=" * 50)
    
    # Run examples
    await example_dynamodb()
    await example_s3()
    await example_cloudwatch()
    await example_sns()
    await example_sqs()
    await example_secrets()
    await example_parameter_store()
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())















