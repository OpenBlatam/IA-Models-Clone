"""
Event Processor Lambda Function
Processes events from SNS and SQS for async processing
"""

import json
import os
import logging
import boto3
from typing import Dict, Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sqs = boto3.client('sqs')
sns = boto3.client('sns')

ANALYSIS_QUEUE_URL = os.getenv('SQS_QUEUE_URL')
SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN', '')


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Process events from SNS or direct invocation
    
    Args:
        event: Lambda event (SNS or direct)
        context: Lambda context
        
    Returns:
        Processing result
    """
    try:
        # Handle SNS event
        if 'Records' in event:
            for record in event['Records']:
                if record.get('EventSource') == 'aws:sns':
                    process_sns_record(record)
                elif record.get('eventSource') == 'aws:sqs':
                    process_sqs_record(record)
        else:
            # Direct invocation
            process_direct_event(event)
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Events processed successfully'})
        }
        
    except Exception as e:
        logger.error(f"Error processing events: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def process_sns_record(record: Dict[str, Any]):
    """Process SNS record"""
    message = json.loads(record['Sns']['Message'])
    topic_arn = record['Sns']['TopicArn']
    
    logger.info(f"Processing SNS message from {topic_arn}")
    
    # Route based on event type
    event_type = message.get('event_type', 'unknown')
    
    if event_type == 'analysis_complete':
        handle_analysis_complete(message)
    elif event_type == 'analysis_failed':
        handle_analysis_failed(message)
    else:
        logger.warning(f"Unknown event type: {event_type}")


def process_sqs_record(record: Dict[str, Any]):
    """Process SQS record"""
    body = json.loads(record['body'])
    
    logger.info(f"Processing SQS message: {body.get('event_type', 'unknown')}")
    
    # Process the message
    event_type = body.get('event_type', 'unknown')
    
    if event_type == 'analyze_track':
        handle_analyze_track(body)
    else:
        logger.warning(f"Unknown event type: {event_type}")


def process_direct_event(event: Dict[str, Any]):
    """Process direct invocation event"""
    logger.info(f"Processing direct event: {event.get('event_type', 'unknown')}")
    
    # Add to SQS queue for async processing
    if ANALYSIS_QUEUE_URL:
        sqs.send_message(
            QueueUrl=ANALYSIS_QUEUE_URL,
            MessageBody=json.dumps(event)
        )


def handle_analysis_complete(message: Dict[str, Any]):
    """Handle analysis complete event"""
    track_id = message.get('track_id')
    logger.info(f"Analysis complete for track: {track_id}")
    
    # Publish to SNS for other subscribers
    if SNS_TOPIC_ARN:
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=json.dumps({
                'event_type': 'analysis_complete',
                'track_id': track_id,
                'timestamp': message.get('timestamp')
            })
        )


def handle_analysis_failed(message: Dict[str, Any]):
    """Handle analysis failed event"""
    track_id = message.get('track_id')
    error = message.get('error')
    logger.error(f"Analysis failed for track {track_id}: {error}")


def handle_analyze_track(message: Dict[str, Any]):
    """Handle analyze track request"""
    track_id = message.get('track_id')
    logger.info(f"Processing analysis request for track: {track_id}")
    
    # Here you would call the actual analysis service
    # For now, just log it
    logger.info(f"Track analysis queued: {track_id}")




