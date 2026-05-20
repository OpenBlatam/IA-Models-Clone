# Alternative Terraform configuration (optional)
# This is an alternative to CDK if you prefer Terraform

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    # Configure your S3 backend for state storage
    # bucket = "your-terraform-state-bucket"
    # key    = "manuales-hogar-ai/terraform.tfstate"
    # region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "app_name" {
  description = "Application name"
  type        = string
  default     = "manuales-hogar-ai"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "dev"
}

# Note: This is a placeholder. For full Terraform implementation,
# you would need to define all resources similar to the CDK stack.
# Consider using CDK for this project as it's already implemented.




