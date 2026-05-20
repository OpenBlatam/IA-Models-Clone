# Disaster Recovery Configuration

# S3 bucket for backups
resource "aws_s3_bucket" "backups" {
  count  = var.enable_backup_bucket ? 1 : 0
  bucket = "${var.project_name}-backups-${var.environment}"

  tags = merge(
    var.tags,
    {
      Name        = "${var.project_name}-backups-${var.environment}"
      Purpose     = "Disaster Recovery"
      BackupPolicy = "Daily"
    }
  )
}

resource "aws_s3_bucket_versioning" "backups" {
  count  = var.enable_backup_bucket ? 1 : 0
  bucket = aws_s3_bucket.backups[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backups" {
  count  = var.enable_backup_bucket ? 1 : 0
  bucket = aws_s3_bucket.backups[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  count  = var.enable_backup_bucket ? 1 : 0
  bucket = aws_s3_bucket.backups[0].id

  rule {
    id     = "backup-retention"
    status = "Enabled"

    expiration {
      days = var.backup_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = var.backup_retention_days
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}

# Cross-region replication for critical backups
resource "aws_s3_bucket_replication_configuration" "backups" {
  count  = var.enable_backup_replication && var.enable_backup_bucket ? 1 : 0
  role   = aws_iam_role.replication[0].arn
  bucket = aws_s3_bucket.backups[0].id

  rule {
    id     = "replicate-to-${var.backup_replication_region}"
    status = "Enabled"

    destination {
      bucket        = aws_s3_bucket.backup_replica[0].arn
      storage_class = "STANDARD"
    }
  }
}

resource "aws_s3_bucket" "backup_replica" {
  count  = var.enable_backup_replication && var.enable_backup_bucket ? 1 : 0
  bucket = "${var.project_name}-backups-${var.environment}-replica"
  provider = aws.replica

  tags = merge(
    var.tags,
    {
      Name    = "${var.project_name}-backups-${var.environment}-replica"
      Purpose = "Disaster Recovery Replica"
    }
  )
}

# IAM role for replication
resource "aws_iam_role" "replication" {
  count = var.enable_backup_replication && var.enable_backup_bucket ? 1 : 0
  name  = "${var.project_name}-s3-replication-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "replication" {
  count = var.enable_backup_replication && var.enable_backup_bucket ? 1 : 0
  role  = aws_iam_role.replication[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetReplicationConfiguration",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.backups[0].arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging"
        ]
        Resource = [
          "${aws_s3_bucket.backups[0].arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags"
        ]
        Resource = [
          "${aws_s3_bucket.backup_replica[0].arn}/*"
        ]
      }
    ]
  })
}

# EBS snapshot lifecycle policy
resource "aws_dlm_lifecycle_policy" "ebs_snapshots" {
  count              = var.enable_ebs_snapshots ? 1 : 0
  description        = "EBS snapshot lifecycle policy for ${var.project_name}"
  execution_role_arn = aws_iam_role.dlm[0].arn
  state              = "ENABLED"

  policy_details {
    resource_types = ["VOLUME"]

    schedule {
      name = "daily-snapshots"

      create_rule {
        interval      = 24
        interval_unit = "HOURS"
        times         = ["03:00"]
      }

      retain_rule {
        count = var.snapshot_retention_count
      }

      tags_to_add = {
        SnapshotCreator = "DLM"
        Environment     = var.environment
        Project         = var.project_name
      }
    }

    target_tags = {
      Snapshot = "true"
    }
  }
}

# IAM role for DLM
resource "aws_iam_role" "dlm" {
  count = var.enable_ebs_snapshots ? 1 : 0
  name  = "${var.project_name}-dlm-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "dlm.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "dlm" {
  count = var.enable_ebs_snapshots ? 1 : 0
  role  = aws_iam_role.dlm[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:CreateSnapshot",
          "ec2:CreateSnapshots",
          "ec2:DeleteSnapshot",
          "ec2:DescribeInstances",
          "ec2:DescribeVolumes",
          "ec2:DescribeSnapshots"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:CreateTags"
        ]
        Resource = "arn:aws:ec2:*::snapshot/*"
      }
    ]
  })
}

