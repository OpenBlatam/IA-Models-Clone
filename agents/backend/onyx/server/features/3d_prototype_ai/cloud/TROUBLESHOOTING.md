# Troubleshooting Guide

Common issues and solutions for AWS EC2 deployment.

## Instance Not Accessible

### Issue: Cannot SSH into instance

**Solutions:**
1. Check security group rules - ensure port 22 is open
2. Verify key pair name matches
3. Check instance state: `aws ec2 describe-instances --instance-ids <id>`
4. Verify key file permissions: `chmod 400 ~/.ssh/your-key.pem`

### Issue: Application not responding

**Solutions:**
1. Check if instance is running: `aws ec2 describe-instance-status --instance-ids <id>`
2. Verify security group allows port 8030
3. Check application logs:
   ```bash
   ssh -i ~/.ssh/key.pem ubuntu@<ip> 'docker-compose logs'
   ```
4. Check Nginx status:
   ```bash
   ssh -i ~/.ssh/key.pem ubuntu@<ip> 'sudo systemctl status nginx'
   ```

## Deployment Issues

### Issue: Terraform apply fails

**Solutions:**
1. Check AWS credentials: `aws sts get-caller-identity`
2. Verify terraform.tfvars is configured correctly
3. Check for resource limits in AWS account
4. Review Terraform logs for specific errors

### Issue: Ansible playbook fails

**Solutions:**
1. Verify inventory file is correct
2. Check SSH access to instance
3. Ensure Python 3 is installed on target
4. Run with verbose output: `ansible-playbook -vvv playbooks/deploy.yml`

### Issue: Docker Compose fails

**Solutions:**
1. Check Docker is running: `sudo systemctl status docker`
2. Verify docker-compose.yml syntax
3. Check disk space: `df -h`
4. Review Docker logs: `docker-compose logs`

## Application Issues

### Issue: Health check fails

**Solutions:**
1. Check application is running:
   ```bash
   curl http://localhost:8030/health
   ```
2. Review application logs
3. Check port conflicts: `sudo netstat -tulpn | grep 8030`
4. Verify environment variables

### Issue: High memory usage

**Solutions:**
1. Upgrade instance type
2. Optimize Docker containers
3. Check for memory leaks
4. Review application configuration

## Network Issues

### Issue: Cannot access from internet

**Solutions:**
1. Verify Elastic IP is attached (if using)
2. Check route tables in VPC
3. Verify internet gateway is attached
4. Check NACL rules

### Issue: SSL/TLS errors

**Solutions:**
1. Configure SSL certificate in Nginx
2. Use Let's Encrypt for free certificates
3. Update security group for port 443
4. Verify certificate validity

## Performance Issues

### Issue: Slow response times

**Solutions:**
1. Check instance metrics in CloudWatch
2. Review application logs for errors
3. Optimize database queries (if applicable)
4. Consider using CloudFront CDN

### Issue: High CPU usage

**Solutions:**
1. Upgrade instance type
2. Optimize application code
3. Check for infinite loops
4. Review background tasks

## Logs and Debugging

### View Application Logs

```bash
# Docker Compose
ssh -i ~/.ssh/key.pem ubuntu@<ip> 'docker-compose logs -f'

# Systemd
ssh -i ~/.ssh/key.pem ubuntu@<ip> 'sudo journalctl -u 3d-prototype-ai -f'
```

### View User Data Logs

```bash
ssh -i ~/.ssh/key.pem ubuntu@<ip> 'cat /var/log/user-data.log'
```

### View Nginx Logs

```bash
ssh -i ~/.ssh/key.pem ubuntu@<ip> 'sudo tail -f /var/log/nginx/access.log'
ssh -i ~/.ssh/key.pem ubuntu@<ip> 'sudo tail -f /var/log/nginx/error.log'
```

## Common Commands

### Restart Application

```bash
ssh -i ~/.ssh/key.pem ubuntu@<ip> 'cd /opt/3d-prototype-ai && docker-compose restart'
```

### Update Application

```bash
ssh -i ~/.ssh/key.pem ubuntu@<ip> 'cd /opt/3d-prototype-ai && git pull && docker-compose up -d --build'
```

### Check Disk Space

```bash
ssh -i ~/.ssh/key.pem ubuntu@<ip> 'df -h'
```

### Check Memory Usage

```bash
ssh -i ~/.ssh/key.pem ubuntu@<ip> 'free -h'
```

## Getting Help

If you encounter issues not covered here:

1. Check AWS CloudWatch logs
2. Review application documentation
3. Check GitHub issues
4. Contact support team

