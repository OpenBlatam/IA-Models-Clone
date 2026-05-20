# Terraform Backend Configuration
# Configure S3 backend for state management and DynamoDB for state locking

terraform {
  backend "s3" {
    # These values should be provided via backend configuration file
    # or environment variables
    # bucket         = "ai-project-generator-terraform-state"
    # key            = "terraform.tfstate"
    # region         = "us-east-1"
    # dynamodb_table = "terraform-state-lock"
    # encrypt        = true
    # kms_key_id     = "alias/terraform-state-key" # Optional: KMS encryption
  }
}

# Alternative: Local backend for development
# Uncomment for local development
# terraform {
#   backend "local" {
#     path = "terraform.tfstate"
#   }
# }

