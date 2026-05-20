# S3 Backend and State Locking Resources
# Create these resources first, then configure backend in backend.tf

resource "aws_s3_bucket" "terraform_state" {
  bucket = "${var.project_name}-terraform-state-${var.environment}"

  tags = merge(
    var.tags,
    {
      Name        = "${var.project_name}-terraform-state-${var.environment}"
      Description = "Terraform state bucket"
    }
  )
}

resource "aws_s3_bucket_versioning" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    id     = "delete-old-versions"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

# DynamoDB table for state locking
resource "aws_dynamodb_table" "terraform_state_lock" {
  name           = "${var.project_name}-terraform-state-lock-${var.environment}"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  tags = merge(
    var.tags,
    {
      Name        = "${var.project_name}-terraform-state-lock-${var.environment}"
      Description = "Terraform state lock table"
    }
  )
}

# Optional: KMS key for encryption
resource "aws_kms_key" "terraform_state" {
  count = var.enable_kms_encryption ? 1 : 0

  description             = "KMS key for Terraform state encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = merge(
    var.tags,
    {
      Name = "${var.project_name}-terraform-state-key-${var.environment}"
    }
  )
}

resource "aws_kms_alias" "terraform_state" {
  count = var.enable_kms_encryption ? 1 : 0

  name          = "alias/${var.project_name}-terraform-state-${var.environment}"
  target_key_id = aws_kms_key.terraform_state[0].key_id
}

