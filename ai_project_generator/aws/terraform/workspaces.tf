# Terraform Workspaces Configuration
# Use workspaces for managing multiple environments

# Workspace-specific locals
locals {
  # Workspace-specific tags
  workspace_tags = {
    Workspace = terraform.workspace
  }

  # Environment-specific configurations based on workspace
  workspace_config = {
    dev = {
      instance_type     = "t3.small"
      min_size         = 1
      max_size         = 2
      desired_capacity = 1
      enable_monitoring = false
      enable_spot_instances = true
    }
    staging = {
      instance_type     = "t3.medium"
      min_size         = 1
      max_size         = 3
      desired_capacity = 2
      enable_monitoring = true
      enable_spot_instances = false
    }
    production = {
      instance_type     = "t3.large"
      min_size         = 2
      max_size         = 5
      desired_capacity = 3
      enable_monitoring = true
      enable_spot_instances = false
    }
  }

  # Get current workspace config or use defaults
  current_config = lookup(
    local.workspace_config,
    terraform.workspace,
    local.workspace_config.production
  )
}

# Override variables based on workspace (optional)
# Uncomment to use workspace-based overrides
# variable "instance_type" {
#   description = "EC2 instance type"
#   type        = string
#   default     = local.current_config.instance_type
# }

# Workspace-specific outputs
output "workspace_name" {
  description = "Current Terraform workspace"
  value       = terraform.workspace
}

output "workspace_config" {
  description = "Workspace-specific configuration"
  value       = local.current_config
  sensitive   = false
}

