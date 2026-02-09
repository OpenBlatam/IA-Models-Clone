# Advanced IAM Configuration

# IAM policy for S3 access with specific buckets
resource "aws_iam_role_policy" "app_s3_advanced" {
  count = length(var.s3_bucket_arns) > 0 ? 1 : 0
  name  = "${var.project_name}-s3-advanced-${var.environment}"
  role  = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = concat(
          var.s3_bucket_arns,
          [for arn in var.s3_bucket_arns : "${arn}/*"]
        )
      }
    ]
  })
}

# IAM policy for CloudWatch Logs
resource "aws_iam_role_policy" "app_cloudwatch_logs" {
  name = "${var.project_name}-cloudwatch-logs-${var.environment}"
  role = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = [
          "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/ec2/${var.project_name}-${var.environment}*"
        ]
      }
    ]
  })
}

# IAM policy for Secrets Manager (if using)
resource "aws_iam_role_policy" "app_secrets_manager" {
  count = var.enable_secrets_manager ? 1 : 0
  name  = "${var.project_name}-secrets-manager-${var.environment}"
  role  = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = var.secrets_manager_arns
      }
    ]
  })
}

# IAM policy for Parameter Store (if using)
resource "aws_iam_role_policy" "app_parameter_store" {
  count = var.enable_parameter_store ? 1 : 0
  name  = "${var.project_name}-parameter-store-${var.environment}"
  role  = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
          "ssm:GetParametersByPath"
        ]
        Resource = [
          "arn:aws:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:parameter/${var.project_name}/${var.environment}/*"
        ]
      }
    ]
  })
}

# IAM policy for ECR (if using container registry)
resource "aws_iam_role_policy" "app_ecr" {
  count = var.enable_ecr_access ? 1 : 0
  name  = "${var.project_name}-ecr-${var.environment}"
  role  = aws_iam_role.app.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

# Service-linked role for Auto Scaling (if needed)
resource "aws_iam_service_linked_role" "autoscaling" {
  count            = var.create_autoscaling_service_role ? 1 : 0
  aws_service_name = "autoscaling.amazonaws.com"
  description      = "Service-linked role for Auto Scaling"
}

