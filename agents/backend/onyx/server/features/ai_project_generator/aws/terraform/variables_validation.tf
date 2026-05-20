# Variable Validation - Enhanced input validation
# Note: Variable definitions are in variables.tf
# This file contains validation blocks that should be added to variables.tf
# Keeping as reference for validation patterns

# Example validation patterns (add to variables in variables.tf):

# AWS Region Validation
# validation {
#   condition = contains([
#     "us-east-1", "us-east-2", "us-west-1", "us-west-2",
#     "eu-west-1", "eu-west-2", "eu-central-1", "ap-southeast-1",
#     "ap-southeast-2", "ap-northeast-1", "sa-east-1"
#   ], var.aws_region)
#   error_message = "AWS region must be a valid region."
# }

# Environment Validation
# validation {
#   condition     = contains(["dev", "staging", "production"], var.environment)
#   error_message = "Environment must be one of: dev, staging, production."
# }

# Instance Type Validation
# validation {
#   condition = can(regex("^[a-z][0-9]+\\.[a-z]+$", var.instance_type))
#   error_message = "Instance type must be a valid AWS instance type (e.g., t3.medium, m5.large)."
# }

# Min/Max Size Validation
# validation {
#   condition     = var.min_size >= 0 && var.min_size <= 10
#   error_message = "Min size must be between 0 and 10."
# }

# validation {
#   condition     = var.max_size >= 1 && var.max_size <= 20
#   error_message = "Max size must be between 1 and 20."
# }

# Desired Capacity Validation
# validation {
#   condition     = var.desired_capacity >= var.min_size && var.desired_capacity <= var.max_size
#   error_message = "Desired capacity must be between min_size and max_size."
# }

# Volume Size Validation
# validation {
#   condition     = var.root_volume_size >= 20 && var.root_volume_size <= 1000
#   error_message = "Root volume size must be between 20 and 1000 GB."
# }

# Port Validation
# validation {
#   condition     = var.app_port >= 1024 && var.app_port <= 65535
#   error_message = "Application port must be between 1024 and 65535."
# }

# CIDR Validation
# validation {
#   condition     = can(cidrhost(var.vpc_cidr, 0))
#   error_message = "VPC CIDR must be a valid CIDR block."
# }

# Email Validation
# validation {
#   condition     = var.alert_email == "" || can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", var.alert_email))
#   error_message = "Alert email must be a valid email address or empty."
# }

