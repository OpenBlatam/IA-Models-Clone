# Disaster Recovery and Backup Configuration

# Automated Backups
resource "aws_backup_plan" "main" {
  name = "music-analyzer-backup-plan-${var.environment}"

  rule {
    rule_name         = "daily-backup"
    target_vault_name = aws_backup_vault.main.name
    schedule          = "cron(0 2 * * ? *)"  # Daily at 2 AM

    lifecycle {
      cold_storage_after = 30
      delete_after       = 90
    }

    recovery_point_tags = {
      Name = "music-analyzer-backup-${var.environment}"
    }
  }

  rule {
    rule_name         = "weekly-backup"
    target_vault_name = aws_backup_vault.main.name
    schedule          = "cron(0 3 ? * SUN *)"  # Weekly on Sunday at 3 AM

    lifecycle {
      cold_storage_after = 90
      delete_after       = 365
    }

    recovery_point_tags = {
      Name = "music-analyzer-weekly-backup-${var.environment}"
    }
  }

  tags = {
    Name = "music-analyzer-backup-plan-${var.environment}"
  }
}

# Backup Vault
resource "aws_backup_vault" "main" {
  name        = "music-analyzer-vault-${var.environment}"
  kms_key_arn = aws_kms_key.backup.arn

  tags = {
    Name = "music-analyzer-vault-${var.environment}"
  }
}

# KMS Key for Backup
resource "aws_kms_key" "backup" {
  description             = "KMS key for backup encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = {
    Name = "music-analyzer-backup-key-${var.environment}"
  }
}

resource "aws_kms_alias" "backup" {
  name          = "alias/music-analyzer-backup-${var.environment}"
  target_key_id = aws_kms_key.backup.key_id
}

# Backup Selection - RDS
resource "aws_backup_selection" "rds" {
  iam_role_arn = aws_iam_role.backup.arn
  name         = "music-analyzer-rds-backup-${var.environment}"
  plan_id      = aws_backup_plan.main.id

  resources = [
    aws_db_instance.main.arn
  ]

  selection_tag {
    type  = "STRINGEQUALS"
    key   = "Backup"
    value = "true"
  }
}

# Backup Selection - EBS Volumes
resource "aws_backup_selection" "ebs" {
  iam_role_arn = aws_iam_role.backup.arn
  name         = "music-analyzer-ebs-backup-${var.environment}"
  plan_id      = aws_backup_plan.main.id

  resources = [
    "arn:aws:ec2:${data.aws_region.current.name}::volume/*"
  ]

  selection_tag {
    type  = "STRINGEQUALS"
    key   = "Backup"
    value = "true"
  }
}

# IAM Role for Backup
resource "aws_iam_role" "backup" {
  name = "music-analyzer-backup-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "backup.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "backup" {
  role       = aws_iam_role.backup.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForBackup"
}

# Cross-Region Replication (Disaster Recovery)
resource "aws_db_instance" "dr_replica" {
  count              = var.environment == "production" && var.enable_dr ? 1 : 0
  identifier         = "music-analyzer-db-dr-${var.environment}"
  replicate_source_db = aws_db_instance.main.identifier
  instance_class     = var.rds_instance_class

  # Deploy in different region
  provider = aws.dr_region

  publicly_accessible = false
  multi_az           = false

  tags = {
    Name = "music-analyzer-db-dr-${var.environment}"
  }
}

# Provider for DR Region
provider "aws" {
  alias  = "dr_region"
  region = var.dr_region
}

# S3 Cross-Region Replication
resource "aws_s3_bucket_replication_configuration" "alb_logs" {
  count  = var.environment == "production" ? 1 : 0
  role   = aws_iam_role.s3_replication.arn
  bucket = aws_s3_bucket.alb_logs.id

  rule {
    id     = "replicate-to-dr"
    status = "Enabled"

    destination {
      bucket        = aws_s3_bucket.alb_logs_dr[0].arn
      storage_class = "STANDARD_IA"
    }
  }
}

resource "aws_s3_bucket" "alb_logs_dr" {
  count  = var.environment == "production" ? 1 : 0
  bucket = "music-analyzer-alb-logs-dr-${var.environment}-${data.aws_caller_identity.current.account_id}"
  provider = aws.dr_region

  tags = {
    Name = "music-analyzer-alb-logs-dr-${var.environment}"
  }
}

resource "aws_iam_role" "s3_replication" {
  count = var.environment == "production" ? 1 : 0
  name  = "music-analyzer-s3-replication-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "s3.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "s3_replication" {
  count = var.environment == "production" ? 1 : 0
  name  = "music-analyzer-s3-replication-policy-${var.environment}"
  role  = aws_iam_role.s3_replication[0].id

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
          aws_s3_bucket.alb_logs.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObjectVersion",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging"
        ]
        Resource = [
          "${aws_s3_bucket.alb_logs.arn}/*"
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
          "${aws_s3_bucket.alb_logs_dr[0].arn}/*"
        ]
      }
    ]
  })
}

# Additional DR Variables
variable "enable_dr" {
  description = "Enable disaster recovery"
  type        = bool
  default     = false
}

variable "dr_region" {
  description = "Disaster recovery region"
  type        = string
  default     = "us-west-2"
}




