output "public_ip" {
  description = "Public IP address of the product_descriptions EC2 instance."
  value       = module.product_descriptions_ec2.public_ip
} 