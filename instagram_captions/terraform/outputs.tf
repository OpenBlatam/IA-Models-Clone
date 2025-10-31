output "public_ip" {
  description = "Public IP address of the instagram_captions EC2 instance."
  value       = module.instagram_captions_ec2.public_ip
} 