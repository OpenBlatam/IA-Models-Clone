# Data sources for Terraform configuration
# Following DevOps best practices for dynamic resource discovery

data "aws_availability_zones" "available" {
  state = "available"
  
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/h2-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "root-device-type"
    values = ["ebs"]
  }
}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

# Get default VPC if not creating new one
data "aws_vpc" "default" {
  count   = var.create_vpc ? 0 : 1
  default = var.vpc_id == "" ? true : false
  id      = var.vpc_id == "" ? null : var.vpc_id
}

# Get default subnets if not creating new VPC
data "aws_subnets" "default" {
  count = var.create_vpc ? 0 : 1
  filter {
    name   = "vpc-id"
    values = [var.vpc_id == "" ? data.aws_vpc.default[0].id : var.vpc_id]
  }
}

