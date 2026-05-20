# Local values for computed and reusable values
# Following DevOps best practices for maintainability

locals {
  # Common tags applied to all resources
  common_tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
      CreatedAt   = timestamp()
    },
    var.tags
  )

  # Resource naming prefix
  name_prefix = "${var.project_name}-${var.environment}"

  # VPC configuration
  vpc_id = var.create_vpc ? aws_vpc.main[0].id : var.vpc_id

  # Subnet configuration
  subnet_id = var.create_vpc ? aws_subnet.main[0].id : var.subnet_id

  # Availability zones
  availability_zones = data.aws_availability_zones.available.names

  # AMI ID
  ami_id = var.ami_id != "" ? var.ami_id : data.aws_ami.ubuntu.id

  # User data template path
  user_data_template = "${path.module}/../user_data/init.sh"

  # Application URLs
  app_url_http  = "http://${var.allocate_elastic_ip ? aws_eip.app[0].public_ip : aws_instance.app.public_ip}:${var.app_port}"
  app_url_https = var.domain_name != "" ? "https://${var.domain_name}" : null
}

