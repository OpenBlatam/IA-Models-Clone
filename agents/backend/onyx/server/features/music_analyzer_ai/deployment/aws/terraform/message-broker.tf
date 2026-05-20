# Message Broker Infrastructure (RabbitMQ/Kafka alternatives)
# Using AWS SQS + SNS for event-driven architecture

# SNS Topic for Events
resource "aws_sns_topic" "events" {
  name              = "music-analyzer-events-${var.environment}"
  display_name      = "Music Analyzer Events"
  kms_master_key_id = aws_kms_key.sns.id

  tags = {
    Name = "music-analyzer-events-${var.environment}"
  }
}

# SNS Subscriptions
resource "aws_sns_topic_subscription" "lambda_processor" {
  topic_arn = aws_sns_topic.events.arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.event_processor.arn
}

resource "aws_sns_topic_subscription" "sqs_processor" {
  topic_arn = aws_sns_topic.events.arn
  protocol  = "sqs"
  endpoint  = aws_sqs_queue.event_processor.arn
}

# SQS Queue for Event Processing
resource "aws_sqs_queue" "event_processor" {
  name                      = "music-analyzer-events-${var.environment}"
  visibility_timeout_seconds = 300
  message_retention_seconds  = 1209600
  receive_wait_time_seconds  = 20

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.event_dlq.arn
    maxReceiveCount     = 3
  })

  kms_master_key_id = aws_kms_key.sqs.id

  tags = {
    Name = "music-analyzer-events-queue-${var.environment}"
  }
}

resource "aws_sqs_queue" "event_dlq" {
  name = "music-analyzer-events-dlq-${var.environment}"

  message_retention_seconds = 1209600

  kms_master_key_id = aws_kms_key.sqs.id

  tags = {
    Name = "music-analyzer-events-dlq-${var.environment}"
  }
}

# SQS Queue Policy
resource "aws_sqs_queue_policy" "event_processor" {
  queue_url = aws_sqs_queue.event_processor.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sns.amazonaws.com"
        }
        Action   = "sqs:SendMessage"
        Resource = aws_sqs_queue.event_processor.arn
        Condition = {
          ArnEquals = {
            "aws:SourceArn" = aws_sns_topic.events.arn
          }
        }
      }
    ]
  })
}

# KMS Keys for Encryption
resource "aws_kms_key" "sns" {
  description             = "KMS key for SNS encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = {
    Name = "music-analyzer-sns-key-${var.environment}"
  }
}

resource "aws_kms_alias" "sns" {
  name          = "alias/music-analyzer-sns-${var.environment}"
  target_key_id = aws_kms_key.sns.key_id
}

resource "aws_kms_key" "sqs" {
  description             = "KMS key for SQS encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = {
    Name = "music-analyzer-sqs-key-${var.environment}"
  }
}

resource "aws_kms_alias" "sqs" {
  name          = "alias/music-analyzer-sqs-${var.environment}"
  target_key_id = aws_kms_key.sqs.key_id
}

# Lambda Function for Event Processing
resource "aws_lambda_function" "event_processor" {
  function_name = "music-analyzer-event-processor-${var.environment}"
  runtime       = "python3.11"
  handler       = "deployment.aws.event_processor.handler"
  memory_size   = 512
  timeout       = 300

  role = aws_iam_role.event_processor.arn

  environment {
    variables = {
      ENVIRONMENT = var.environment
      SQS_QUEUE_URL = aws_sqs_queue.analysis_queue.id
    }
  }

  tags = {
    Name = "music-analyzer-event-processor-${var.environment}"
  }
}

# IAM Role for Event Processor
resource "aws_iam_role" "event_processor" {
  name = "music-analyzer-event-processor-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "event_processor" {
  name = "music-analyzer-event-processor-policy-${var.environment}"
  role = aws_iam_role.event_processor.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sqs:SendMessage",
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = [
          aws_sqs_queue.analysis_queue.arn,
          aws_sqs_queue.event_processor.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = aws_sns_topic.events.arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# Lambda Permission for SNS
resource "aws_lambda_permission" "sns" {
  statement_id  = "AllowExecutionFromSNS"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.event_processor.function_name
  principal     = "sns.amazonaws.com"
  source_arn    = aws_sns_topic.events.arn
}




