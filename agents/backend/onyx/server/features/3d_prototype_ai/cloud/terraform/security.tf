# Security resources
# Security Groups, IAM Roles and Policies

# Security Group for Application
resource "aws_security_group" "app" {
  name        = "${local.name_prefix}-sg"
  description = "Security group for 3D Prototype AI application"
  vpc_id      = local.vpc_id

  # HTTP
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTPS
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Application Port
  ingress {
    description     = "Application Port"
    from_port       = var.app_port
    to_port         = var.app_port
    protocol        = "tcp"
    cidr_blocks     = var.allowed_cidr_blocks
    security_groups = var.create_vpc ? [] : [aws_security_group.app.id]
  }

  # SSH
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_allowed_cidr_blocks
  }

  # Prometheus metrics (optional)
  ingress {
    description     = "Prometheus metrics"
    from_port       = 9090
    to_port         = 9090
    protocol        = "tcp"
    cidr_blocks     = var.allowed_cidr_blocks
    security_groups = []
  }

  # All outbound traffic
  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    {
      Name = "${local.name_prefix}-sg"
    },
    local.common_tags
  )

  lifecycle {
    create_before_destroy = true
  }
}

# IAM Role for EC2 Instance
resource "aws_iam_role" "ec2_role" {
  name = "${local.name_prefix}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(
    {
      Name = "${local.name_prefix}-ec2-role"
    },
    local.common_tags
  )
}

# IAM Policy for CloudWatch
resource "aws_iam_role_policy" "cloudwatch" {
  name = "${local.name_prefix}-cloudwatch-policy"
  role = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "*"
      }
    ]
  })
}

# IAM Policy for S3 (for backups)
resource "aws_iam_role_policy" "s3_backup" {
  count = var.enable_s3_backup ? 1 : 0
  name  = "${local.name_prefix}-s3-backup-policy"
  role  = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.backup[0].arn,
          "${aws_s3_bucket.backup[0].arn}/*"
        ]
      }
    ]
  })
}

# IAM Instance Profile
resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${local.name_prefix}-ec2-profile"
  role = aws_iam_role.ec2_role.name

  tags = merge(
    {
      Name = "${local.name_prefix}-ec2-profile"
    },
    local.common_tags
  )
}

