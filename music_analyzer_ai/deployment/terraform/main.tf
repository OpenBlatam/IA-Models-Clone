# Terraform configuration for Music Analyzer AI
# Multi-cloud infrastructure as code

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
  }
  
  backend "s3" {
    bucket = "music-analyzer-ai-terraform-state"
    key    = "terraform.tfstate"
    region = "us-east-1"
  }
}

# Variables
variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "azure_location" {
  description = "Azure location"
  type        = string
  default     = "East US"
}

variable "enable_aws" {
  description = "Enable AWS resources"
  type        = bool
  default     = true
}

variable "enable_azure" {
  description = "Enable Azure resources"
  type        = bool
  default     = false
}

variable "enable_kubernetes" {
  description = "Enable Kubernetes resources"
  type        = bool
  default     = true
}

# Providers
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = "music-analyzer-ai"
      ManagedBy   = "terraform"
    }
  }
}

provider "azurerm" {
  features {}
  
  skip_provider_registration = false
}

# AWS Resources
module "aws_infrastructure" {
  source = "./modules/aws"
  count  = var.enable_aws ? 1 : 0
  
  environment = var.environment
  region      = var.aws_region
}

# Azure Resources
module "azure_infrastructure" {
  source = "./modules/azure"
  count  = var.enable_azure ? 1 : 0
  
  environment = var.environment
  location    = var.azure_location
}

# Kubernetes Resources
module "kubernetes_infrastructure" {
  source = "./modules/kubernetes"
  count  = var.enable_kubernetes ? 1 : 0
  
  environment = var.environment
}

# Outputs
output "aws_endpoints" {
  value     = var.enable_aws ? module.aws_infrastructure[0].endpoints : null
  sensitive = false
}

output "azure_endpoints" {
  value     = var.enable_azure ? module.azure_infrastructure[0].endpoints : null
  sensitive = false
}

output "kubernetes_endpoints" {
  value     = var.enable_kubernetes ? module.kubernetes_infrastructure[0].endpoints : null
  sensitive = false
}




