variable "ami_id" {
  description = "AMI ID for the EC2 instance."
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type."
  type        = string
  default     = "t3.micro"
}

variable "subnet_id" {
  description = "Subnet ID for the EC2 instance."
  type        = string
}

variable "key_name" {
  description = "SSH key name for EC2 access."
  type        = string
}

variable "security_group_id" {
  description = "Security group ID for the EC2 instance."
  type        = string
}

variable "tags" {
  description = "Tags to apply to the EC2 instance."
  type        = map(string)
  default     = {}
}

variable "repo_url" {
  description = "Git repository URL for the microservice code."
  type        = string
}

variable "branch" {
  description = "Git branch to deploy."
  type        = string
  default     = "main"
} 