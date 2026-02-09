# Storage resources
# S3 buckets for backups and logs

# S3 Bucket for Backups
resource "aws_s3_bucket" "backup" {
  count  = var.enable_s3_backup ? 1 : 0
  bucket = "${local.name_prefix}-backups-${data.aws_caller_identity.current.account_id}"

  tags = merge(
    {
      Name = "${local.name_prefix}-backups"
    },
    local.common_tags
  )
}

# S3 Bucket Versioning
resource "aws_s3_bucket_versioning" "backup" {
  count  = var.enable_s3_backup ? 1 : 0
  bucket = aws_s3_bucket.backup[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Bucket Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "backup" {
  count  = var.enable_s3_backup ? 1 : 0
  bucket = aws_s3_bucket.backup[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 Bucket Lifecycle Policy
resource "aws_s3_bucket_lifecycle_configuration" "backup" {
  count  = var.enable_s3_backup ? 1 : 0
  bucket = aws_s3_bucket.backup[0].id

  rule {
    id     = "backup-retention"
    status = "Enabled"

    expiration {
      days = var.backup_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = var.backup_retention_days
    }
  }
}

# S3 Bucket Public Access Block
resource "aws_s3_bucket_public_access_block" "backup" {
  count  = var.enable_s3_backup ? 1 : 0
  bucket = aws_s3_bucket.backup[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets  = true
}

