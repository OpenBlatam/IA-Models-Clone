# Networking resources
# VPC, Subnets, Internet Gateway, Route Tables

# VPC
resource "aws_vpc" "main" {
  count                = var.create_vpc ? 1 : 0
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(
    {
      Name = "${local.name_prefix}-vpc"
    },
    local.common_tags
  )
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  count  = var.create_vpc ? 1 : 0
  vpc_id = aws_vpc.main[0].id

  tags = merge(
    {
      Name = "${local.name_prefix}-igw"
    },
    local.common_tags
  )
}

# Public Subnet
resource "aws_subnet" "main" {
  count                   = var.create_vpc ? 1 : 0
  vpc_id                  = aws_vpc.main[0].id
  cidr_block              = var.subnet_cidr
  availability_zone       = local.availability_zones[0]
  map_public_ip_on_launch = true

  tags = merge(
    {
      Name = "${local.name_prefix}-subnet-public"
      Type = "public"
    },
    local.common_tags
  )
}

# Route Table
resource "aws_route_table" "main" {
  count  = var.create_vpc ? 1 : 0
  vpc_id = aws_vpc.main[0].id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main[0].id
  }

  tags = merge(
    {
      Name = "${local.name_prefix}-rt-public"
    },
    local.common_tags
  )
}

# Route Table Association
resource "aws_route_table_association" "main" {
  count          = var.create_vpc ? 1 : 0
  subnet_id      = aws_subnet.main[0].id
  route_table_id = aws_route_table.main[0].id
}

