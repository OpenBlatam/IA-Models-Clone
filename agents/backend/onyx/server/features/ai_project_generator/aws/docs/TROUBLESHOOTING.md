# Troubleshooting Guide - AI Project Generator

Common issues and solutions when deploying the AI Project Generator to AWS EC2.

## Table of Contents

1. [Infrastructure Issues](#infrastructure-issues)
2. [Application Deployment Issues](#application-deployment-issues)
3. [Connectivity Issues](#connectivity-issues)
4. [Performance Issues](#performance-issues)
5. [ML/AI Specific Issues](#mlai-specific-issues)

## Infrastructure Issues

### Terraform State Lock

**Problem**: Terraform state is locked.

**Solution**:
```bash
# Check for lock
aws dynamodb list-tables | grep terraform

# Force unlock (use with caution)
terraform force-unlock <lock-id>
```

### EC2 Instances Not Launching

**Problem**: Instances fail to launch in Auto Scaling Group.

**Solution**:
1. Check CloudWatch logs for user data script:
   ```bash
   aws logs tail /aws/ec2/user-data --follow
   ```

2. Check launch template:
   ```bash
   aws ec2 describe-launch-template-versions \
     --launch-template-name ai-project-generator-production-*
   ```

3. Verify instance type supports your region:
   ```bash
   aws ec2 describe-instance-type-offerings \
     --location-type availability-zone \
     --filters Name=instance-type,Values=t3.medium
   ```

## Application Deployment Issues

### Ansible Connection Failures

**Problem**: Can't connect to EC2 instances via Ansible.

**Solution**:
1. Verify SSH key permissions:
   ```bash
   chmod 400 ~/.ssh/ai-project-generator-key.pem
   ```

2. Test SSH connection manually:
   ```bash
   ssh -i ~/.ssh/ai-project-generator-key.pem ubuntu@<instance-ip>
   ```

3. Check security group allows SSH (port 22)

4. Verify EC2 inventory:
   ```bash
   ansible-inventory -i inventory/ec2.ini --list
   ```

### Docker Installation Failures

**Problem**: Docker fails to install on instances.

**Solution**:
1. Check user data script logs:
   ```bash
   ssh ubuntu@<instance-ip> "sudo cat /var/log/user-data.log"
   ```

2. Manually install Docker:
   ```bash
   ansible all -i inventory/ec2.ini -m shell \
     -a "curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh" \
     --become
   ```

### Application Not Starting

**Problem**: Application containers fail to start.

**Solution**:
1. Check Docker logs:
   ```bash
   ssh ubuntu@<instance-ip>
   docker-compose -f /opt/ai-project-generator/docker-compose.yml logs
   ```

2. Verify environment variables:
   ```bash
   cat /opt/ai-project-generator/.env
   ```

3. Check Redis connection:
   ```bash
   redis-cli ping
   ```

4. Check disk space (ML models can be large):
   ```bash
   df -h
   ```

5. Check Python dependencies:
   ```bash
   /opt/venv/bin/pip list | grep -E "(torch|transformers|fastapi)"
   ```

### Port 8020 Already in Use

**Problem**: Port 8020 is already in use.

**Solution**:
```bash
# Find process using port 8020
sudo lsof -i :8020

# Kill the process
sudo kill -9 <PID>

# Or change port in .env file
sed -i 's/APP_PORT=8020/APP_PORT=8021/' /opt/ai-project-generator/.env
```

## Connectivity Issues

### Can't Access Application via Load Balancer

**Problem**: Load balancer returns 502 or connection timeout.

**Solution**:
1. Check target group health:
   ```bash
   aws elbv2 describe-target-health \
     --target-group-arn <target-group-arn>
   ```

2. Verify security group allows traffic from ALB to instances on port 8020

3. Check application is listening on correct port:
   ```bash
   ssh ubuntu@<instance-ip> "sudo netstat -tlnp | grep 8020"
   ```

4. Test health endpoint directly on instance:
   ```bash
   curl http://localhost:8020/health
   ```

### Redis Connection Issues

**Problem**: Application can't connect to Redis.

**Solution**:
1. Check Redis is running:
   ```bash
   sudo systemctl status redis-server
   redis-cli ping
   ```

2. Verify Redis URL in .env:
   ```bash
   grep REDIS_URL /opt/ai-project-generator/.env
   ```

3. Check Redis bind address:
   ```bash
   grep "^bind" /etc/redis/redis.conf
   ```

4. Test Redis connection:
   ```bash
   redis-cli -h localhost -p 6379 ping
   ```

## Performance Issues

### High CPU Usage

**Problem**: Instances show high CPU utilization.

**Solution**:
1. Check CloudWatch metrics:
   ```bash
   aws cloudwatch get-metric-statistics \
     --namespace AWS/EC2 \
     --metric-name CPUUtilization \
     --dimensions Name=InstanceId,Value=<instance-id> \
     --start-time <start-time> \
     --end-time <end-time> \
     --period 300 \
     --statistics Average
   ```

2. Identify resource-intensive processes:
   ```bash
   ssh ubuntu@<instance-ip> "top -b -n 1 | head -20"
   ```

3. Consider upgrading instance type (especially for ML workloads)

### High Memory Usage

**Problem**: Instances running out of memory (common with ML models).

**Solution**:
1. Check memory usage:
   ```bash
   ssh ubuntu@<instance-ip> "free -h"
   ```

2. Review Docker container memory limits

3. Consider:
   - Upgrading to larger instance (t3.large, t3.xlarge)
   - Using ElastiCache Redis instead of local Redis
   - Optimizing ML model loading

### Slow Application Response

**Problem**: Application responds slowly, especially for ML operations.

**Solution**:
1. Check application logs for errors

2. Verify Redis connection (if using cache)

3. Review Nginx timeout settings:
   ```bash
   grep -E "(proxy_read_timeout|proxy_connect_timeout)" \
     /etc/nginx/sites-available/ai-project-generator
   ```

4. Consider:
   - Increasing instance size
   - Using GPU instances (g4dn.xlarge) for ML workloads
   - Implementing request queuing

## ML/AI Specific Issues

### PyTorch Installation Issues

**Problem**: PyTorch fails to install or import.

**Solution**:
1. Check Python version (requires 3.11):
   ```bash
   python3.11 --version
   ```

2. Install PyTorch manually:
   ```bash
   /opt/venv/bin/pip install torch torchvision torchaudio
   ```

3. Verify CUDA support (if using GPU):
   ```bash
   /opt/venv/bin/python -c "import torch; print(torch.cuda.is_available())"
   ```

### Model Loading Failures

**Problem**: ML models fail to load.

**Solution**:
1. Check disk space (models can be large):
   ```bash
   df -h
   ```

2. Verify model files exist:
   ```bash
   find /opt/ai-project-generator -name "*.pth" -o -name "*.pt"
   ```

3. Check memory availability:
   ```bash
   free -h
   ```

4. Review application logs for specific error messages

### Out of Memory Errors

**Problem**: Application crashes with OOM errors.

**Solution**:
1. Increase instance memory (upgrade instance type)

2. Reduce batch size in ML operations

3. Use model quantization:
   ```python
   # In application code
   import torch
   model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
   ```

4. Implement model caching/unloading strategies

## Getting Help

### Useful Commands

```bash
# View all instances
aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=ai-project-generator" \
  --query 'Reservations[*].Instances[*].[InstanceId,State.Name,PublicIpAddress,PrivateIpAddress]' \
  --output table

# View security groups
aws ec2 describe-security-groups \
  --filters "Name=tag:Project,Values=ai-project-generator"

# View load balancer
aws elbv2 describe-load-balancers \
  --query 'LoadBalancers[?contains(LoadBalancerName, `ai-project-generator`)]'

# View CloudWatch logs
aws logs tail /aws/ec2/user-data --follow
```

### Log Locations

- User data script: `/var/log/user-data.log`
- Application logs: `/opt/ai-project-generator/logs/` or Docker logs
- Nginx logs: `/var/log/nginx/`
- System logs: `journalctl -u ai-project-generator`
- Redis logs: `/var/log/redis/redis-server.log`

### Support Resources

- AWS Documentation: https://docs.aws.amazon.com/
- Terraform Documentation: https://www.terraform.io/docs
- Ansible Documentation: https://docs.ansible.com/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- PyTorch Documentation: https://pytorch.org/docs/

