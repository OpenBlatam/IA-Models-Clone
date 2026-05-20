variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "3d-prototype-ai"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.large"
}

variable "key_name" {
  description = "AWS key pair name for SSH access"
  type        = string
}

variable "ami_id" {
  description = "Custom AMI ID (leave empty to use latest Ubuntu)"
  type        = string
  default     = ""
}

variable "app_port" {
  description = "Application port"
  type        = number
  default     = 8030
}

variable "app_host" {
  description = "Application host"
  type        = string
  default     = "0.0.0.0"
}

variable "volume_size" {
  description = "Root volume size in GB"
  type        = number
  default     = 30
}

variable "create_vpc" {
  description = "Create new VPC or use existing"
  type        = bool
  default     = false
}

variable "vpc_id" {
  description = "Existing VPC ID (required if create_vpc is false)"
  type        = string
  default     = ""
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "subnet_id" {
  description = "Existing subnet ID (required if create_vpc is false)"
  type        = string
  default     = ""
}

variable "subnet_cidr" {
  description = "CIDR block for subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access application port"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "ssh_allowed_cidr_blocks" {
  description = "CIDR blocks allowed to SSH"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "allocate_elastic_ip" {
  description = "Allocate Elastic IP for instance"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

variable "domain_name" {
  description = "Domain name for application (optional)"
  type        = string
  default     = ""
}

variable "enable_s3_backup" {
  description = "Enable S3 bucket for backups"
  type        = bool
  default     = false
}

variable "backup_retention_days" {
  description = "Number of days to retain backups in S3"
  type        = number
  default     = 30
}

variable "enable_cloudwatch_alarms" {
  description = "Enable CloudWatch alarms"
  type        = bool
  default     = true
}

variable "enable_memory_alarm" {
  description = "Enable memory utilization alarm (requires CloudWatch agent)"
  type        = bool
  default     = false
}

variable "cpu_threshold" {
  description = "CPU utilization threshold for alarm (percentage)"
  type        = number
  default     = 80
}

variable "memory_threshold" {
  description = "Memory utilization threshold for alarm (percentage)"
  type        = number
  default     = 85
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for alarm notifications"
  type        = string
  default     = ""
}

variable "enable_detailed_monitoring" {
  description = "Enable detailed CloudWatch monitoring"
  type        = bool
  default     = false
}

variable "enable_termination_protection" {
  description = "Enable termination protection for EC2 instance"
  type        = bool
  default     = false
}

variable "volume_iops" {
  description = "IOPS for root volume (only for gp3)"
  type        = number
  default     = 0
}

variable "volume_throughput" {
  description = "Throughput for root volume in MB/s (only for gp3)"
  type        = number
  default     = 0
}

variable "additional_volumes" {
  description = "Additional EBS volumes to attach"
  type = list(object({
    device_name = string
    volume_type = string
    volume_size = number
    encrypted   = bool
    iops        = number
    throughput  = number
  }))
  default = []
}

