# =====================================================================
# Terraform Module: product_descriptions Microservice EC2 Deployment
# =====================================================================
# - Deploys a single EC2 instance for this microservice
# - Uses Docker and pulls code from a Git repo
# - All configuration is parameterized for reuse and CI/CD
# =====================================================================

locals {
  service_name = "product-descriptions"
  service_path = "agents/backend/onyx/server/features/product_descriptions"
}

module "product_descriptions_ec2" {
  source            = "../../../../../../../../terraform/modules/ec2_docker"
  ami_id            = var.ami_id
  instance_type     = var.instance_type
  subnet_id         = var.subnet_id
  key_name          = var.key_name
  security_group_id = var.security_group_id
  tags              = merge(var.tags, { "Service" = local.service_name })
  name              = local.service_name
  repo_url          = var.repo_url
  branch            = var.branch
  compose_file      = "docker-compose.yml"
  feature_path      = local.service_path
} 