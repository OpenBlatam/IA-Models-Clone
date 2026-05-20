# Quick Reference Guide - Automation Scripts

Quick reference for all automation scripts.

## 🚀 Quick Commands

### Backup
```bash
# Basic backup
./automated_backup.sh

# With S3
export S3_BUCKET=my-bucket
./automated_backup.sh
```

### Monitoring
```bash
# Basic monitoring
./automated_monitoring.sh

# With alerts
export SNS_TOPIC_ARN=arn:aws:sns:...
./automated_monitoring.sh
```

### Updates
```bash
# Update application
export GIT_REPO=https://github.com/user/repo.git
./automated_update.sh
```

### Scaling
```bash
# Auto-scale
export AUTO_SCALING_GROUP=my-asg
./automated_scaling.sh
```

### Disaster Recovery
```bash
# Restore from backup
export RECOVERY_MODE=restore
./disaster_recovery.sh
```

### Log Analysis
```bash
# Analyze logs
./log_analyzer.sh
```

### Security Audit
```bash
# Security check
./security_audit.sh
```

## 📅 Recommended Schedule

### Daily
- **Monitoring**: Every 5 minutes
- **Log Analysis**: Once daily

### Weekly
- **Backup**: Sunday 2 AM
- **Update**: Sunday 3 AM
- **Security Audit**: Sunday 4 AM

### Monthly
- **Disaster Recovery Test**: First Sunday
- **Full Security Review**: Last Sunday

## 🔧 Common Configurations

### Environment Setup
```bash
# AWS
export AWS_REGION=us-east-1
export SNS_TOPIC_ARN=arn:aws:sns:...
export S3_BUCKET=my-backup-bucket

# Application
export APP_DIR=/opt/ai-project-generator
export APP_PORT=8020

# CloudWatch
export CLOUDWATCH_NAMESPACE=AIProjectGenerator
export ENABLE_CLOUDWATCH=true
```

### Thresholds
```bash
# Monitoring
export CPU_THRESHOLD=80
export MEMORY_THRESHOLD=85
export DISK_THRESHOLD=90

# Scaling
export CPU_SCALE_UP_THRESHOLD=75
export CPU_SCALE_DOWN_THRESHOLD=25
export MIN_INSTANCES=1
export MAX_INSTANCES=10
```

## 📊 CloudWatch Metrics

All scripts send metrics to CloudWatch:
- `AIProjectGenerator/Backups` - Backup metrics
- `AIProjectGenerator/Monitoring` - Monitoring metrics
- `AIProjectGenerator/Scaling` - Scaling metrics
- `AIProjectGenerator/Security` - Security metrics
- `AIProjectGenerator/Logs` - Log analysis metrics

## 🔔 SNS Alerts

Configure alerts for:
- Backup failures
- High resource usage
- Application health issues
- Security vulnerabilities
- Scaling actions
- Disaster recovery events

## 🛠️ Troubleshooting

### Script won't run
```bash
chmod +x script.sh
./script.sh
```

### Check logs
```bash
tail -f /var/log/backup.log
tail -f /var/log/monitoring.log
```

### Test AWS connection
```bash
aws sts get-caller-identity
```

### Verify CloudWatch
```bash
aws cloudwatch list-metrics --namespace AIProjectGenerator
```

## 📚 Documentation

- [README_AUTOMATION.md](README_AUTOMATION.md) - Full documentation
- [ALL_SCRIPTS.md](ALL_SCRIPTS.md) - Complete script reference
- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Improvement details
- [CHANGELOG.md](CHANGELOG.md) - Version history

