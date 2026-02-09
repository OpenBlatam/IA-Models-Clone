# Provider Configuration for Multi-Region Support

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Primary provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = merge(
      {
        Project     = var.project_name
        Environment = var.environment
        ManagedBy   = "Terraform"
        Application = "AI Project Generator"
      },
      var.tags
    )
  }
}

# Replica provider for cross-region replication
provider "aws" {
  alias  = "replica"
  region = var.backup_replication_region

  default_tags {
    tags = merge(
      {
        Project     = var.project_name
        Environment = var.environment
        ManagedBy   = "Terraform"
        Application = "AI Project Generator"
        Purpose     = "Disaster Recovery"
      },
      var.tags
    )
  }
}

