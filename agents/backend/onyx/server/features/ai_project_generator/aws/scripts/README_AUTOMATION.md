# Automated Scripts Documentation

This directory contains automated scripts for managing the AI Project Generator deployment on AWS EC2.

## 🚀 Recent Improvements

**Version 2.0** includes significant enhancements:
- ✅ Common functions library for code reuse
- ✅ CloudWatch metrics integration
- ✅ Enhanced error handling and retry logic
- ✅ Backup integrity verification
- ✅ Improved monitoring with detailed metrics
- ✅ Better logging and reporting
- ✅ SNS alerting integration

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed changelog.

## 📦 Complete Script Suite

### Core Automation (7 scripts)
1. **automated_backup.sh** - Backup automation with integrity checks
2. **automated_monitoring.sh** - System and application monitoring
3. **automated_update.sh** - Zero-downtime application updates
4. **automated_scaling.sh** ⭐ NEW - Automatic scaling based on metrics
5. **disaster_recovery.sh** ⭐ NEW - Automated disaster recovery
6. **log_analyzer.sh** ⭐ NEW - Log analysis and pattern detection
7. **security_audit.sh** ⭐ NEW - Security auditing and hardening

### Utility Scripts
8. **setup_cron_jobs.sh** - Cron job configuration
9. **deploy.sh** - Main deployment script
10. **health_check.sh** - Health checking
11. **common_functions.sh** - Shared functions library

See [ALL_SCRIPTS.md](ALL_SCRIPTS.md) for complete documentation.

## Available Scripts

### 0. `common_functions.sh` (NEW)
Shared library of common functions used by all automation scripts.

**Features:**
- Logging functions (info, warn, error, debug, success)
- AWS integration (SNS, CloudWatch, S3)
- System monitoring utilities
- Network utilities
- Docker management functions
- Validation functions
- Retry mechanisms

**Usage:**
```bash
# Source in your script
source "${SCRIPT_DIR}/common_functions.sh"

# Use functions
log_info "Message"
check_command "docker"
send_cloudwatch_metric "Namespace" "Metric" 100 "Count"
```

### 1. `automated_backup.sh` (IMPROVED)
Automated backup script that creates backups of application data, logs, and configurations.

**Features:**
- Backs up application files (excluding large/unnecessary files)
- Backs up environment configuration
- Backs up database (PostgreSQL) and Redis data
- Backs up logs and system configurations
- Optional S3 upload with verification
- Automatic cleanup of old backups
- **NEW**: Backup integrity verification
- **NEW**: CloudWatch metrics integration
- **NEW**: Backup size and duration tracking
- **NEW**: Disk space validation
- **NEW**: Retry logic for S3 uploads

**Usage:**
```bash
# Basic usage
./automated_backup.sh

# With S3 backup
export S3_BUCKET=my-backup-bucket
export RETENTION_DAYS=30
./automated_backup.sh

# Schedule with cron
0 2 * * * /path/to/automated_backup.sh
```

**Environment Variables:**
- `BACKUP_DIR`: Backup directory (default: `/opt/backups/ai-project-generator`)
- `S3_BUCKET`: S3 bucket for backups (optional)
- `S3_PREFIX`: S3 prefix for backups (default: `backups/`)
- `RETENTION_DAYS`: Days to keep backups (default: 7)
- `APP_DIR`: Application directory (default: `/opt/ai-project-generator`)
- `CLOUDWATCH_NAMESPACE`: CloudWatch namespace (default: `AIProjectGenerator/Backups`)
- `ENABLE_COMPRESSION`: Enable compression (default: `true`)
- `VERIFY_INTEGRITY`: Verify backup integrity (default: `true`)
- `MIN_DISK_SPACE_GB`: Minimum disk space required (default: 5)
- `SNS_TOPIC_ARN`: SNS topic for notifications (optional)

### 2. `automated_monitoring.sh` / `automated_monitoring_improved.sh`
Automated monitoring script that checks application health and system resources.

**Features:**
- Application health checks
- CPU, memory, and disk usage monitoring
- Docker container status and health checks
- Redis and Nginx status checks
- Response time monitoring
- Alert notifications (SNS/Email)
- Metrics collection
- **NEW**: CloudWatch metrics for all monitored values
- **NEW**: Application status endpoint checking
- **NEW**: Comprehensive monitoring reports
- **NEW**: Success rate calculation
- **NEW**: Alert count tracking

**Usage:**
```bash
# Basic usage
./automated_monitoring.sh

# With alerts
export SNS_TOPIC_ARN=arn:aws:sns:region:account:topic
export ALERT_EMAIL=admin@example.com
./automated_monitoring.sh

# Schedule with cron (every 5 minutes)
*/5 * * * * /path/to/automated_monitoring.sh
```

**Environment Variables:**
- `APP_PORT`: Application port (default: 8020)
- `CPU_THRESHOLD`: CPU usage threshold % (default: 80)
- `MEMORY_THRESHOLD`: Memory usage threshold % (default: 85)
- `DISK_THRESHOLD`: Disk usage threshold % (default: 90)
- `RESPONSE_TIME_THRESHOLD`: Response time threshold in ms (default: 5000)
- `ERROR_RATE_THRESHOLD`: Error rate threshold % (default: 5)
- `SNS_TOPIC_ARN`: AWS SNS topic for alerts
- `ALERT_EMAIL`: Email address for alerts
- `CLOUDWATCH_NAMESPACE`: CloudWatch namespace (default: `AIProjectGenerator/Monitoring`)
- `ENABLE_CLOUDWATCH`: Enable CloudWatch metrics (default: `true`)

### 3. `automated_update.sh`
Automated update script that updates the application with zero-downtime deployment.

**Features:**
- Automatic backup before update
- Git-based code updates
- Ansible-based deployment
- Health check verification
- Automatic rollback on failure
- Version tracking

**Usage:**
```bash
# Basic usage
export GIT_REPO=https://github.com/user/repo.git
export GIT_BRANCH=main
./automated_update.sh

# With custom settings
export BACKUP_BEFORE_UPDATE=true
export HEALTH_CHECK_RETRIES=15
./automated_update.sh

# Schedule with cron (weekly updates)
0 3 * * 0 /path/to/automated_update.sh
```

**Environment Variables:**
- `GIT_REPO`: Git repository URL
- `GIT_BRANCH`: Git branch to deploy (default: main)
- `BACKUP_BEFORE_UPDATE`: Create backup before update (default: true)
- `HEALTH_CHECK_RETRIES`: Number of health check retries (default: 10)
- `HEALTH_CHECK_DELAY`: Delay between health checks in seconds (default: 5)

### 4. `setup_cron_jobs.sh`
Sets up automated cron jobs for all automation scripts.

**Features:**
- Configures backup cron job
- Configures monitoring cron job
- Configures update cron job
- Sets up log rotation
- Prevents duplicate cron entries

**Usage:**
```bash
# Run as root or with sudo
sudo ./setup_cron_jobs.sh

# With custom schedules
export BACKUP_SCHEDULE="0 2 * * *"
export MONITORING_SCHEDULE="*/5 * * * *"
export UPDATE_SCHEDULE="0 3 * * 0"
sudo ./setup_cron_jobs.sh
```

**Environment Variables:**
- `BACKUP_SCHEDULE`: Cron schedule for backups (default: `0 2 * * *` - 2 AM daily)
- `MONITORING_SCHEDULE`: Cron schedule for monitoring (default: `*/5 * * * *` - every 5 minutes)
- `UPDATE_SCHEDULE`: Cron schedule for updates (default: `0 3 * * 0` - 3 AM Sundays)
- `CRON_USER`: User to run cron jobs (default: ubuntu)

## Quick Setup

### 1. Make Scripts Executable
```bash
chmod +x automated_backup.sh
chmod +x automated_monitoring.sh
chmod +x automated_update.sh
chmod +x setup_cron_jobs.sh
```

### 2. Configure Environment Variables
Create a `.env` file or export variables:
```bash
# Backup configuration
export S3_BUCKET=my-backup-bucket
export RETENTION_DAYS=30

# Monitoring configuration
export SNS_TOPIC_ARN=arn:aws:sns:us-east-1:123456789:alerts
export ALERT_EMAIL=admin@example.com

# Update configuration
export GIT_REPO=https://github.com/user/ai-project-generator.git
export GIT_BRANCH=main
```

### 3. Setup Cron Jobs
```bash
sudo ./setup_cron_jobs.sh
```

## Cron Schedule Examples

### Backup Schedules
```bash
# Daily at 2 AM
0 2 * * *

# Every 6 hours
0 */6 * * *

# Daily at midnight
0 0 * * *
```

### Monitoring Schedules
```bash
# Every 5 minutes
*/5 * * * *

# Every minute
* * * * *

# Every 15 minutes
*/15 * * * *
```

### Update Schedules
```bash
# Weekly on Sundays at 3 AM
0 3 * * 0

# Daily at 4 AM
0 4 * * *

# Monthly on the 1st at 2 AM
0 2 1 * *
```

## Log Files

All scripts write to log files:
- Backup: `/var/log/backup.log` and `/var/log/backup-cron.log`
- Monitoring: `/var/log/monitoring.log` and `/var/log/monitoring-cron.log`
- Updates: `/var/log/automated-update.log` and `/var/log/update-cron.log`

## Troubleshooting

### Scripts Not Executing
1. Check file permissions: `chmod +x script.sh`
2. Check cron service: `sudo systemctl status cron`
3. Check cron logs: `grep CRON /var/log/syslog`

### Scripts Failing
1. Check script logs in `/var/log/`
2. Run scripts manually to see errors
3. Verify environment variables are set
4. Check AWS credentials (for S3/SNS)

### High Resource Usage
1. Adjust monitoring frequency
2. Optimize backup schedules
3. Review log rotation settings

## Best Practices

1. **Test Scripts Manually First**: Run each script manually before setting up cron jobs
2. **Monitor Logs**: Regularly check log files for errors
3. **Set Appropriate Thresholds**: Adjust monitoring thresholds based on your workload
4. **Backup Before Updates**: Always enable `BACKUP_BEFORE_UPDATE=true`
5. **Use S3 for Backups**: Store backups in S3 for disaster recovery
6. **Set Up Alerts**: Configure SNS or email alerts for critical issues
7. **Review Schedules**: Adjust cron schedules based on your needs

## Integration with Ansible

These scripts can be integrated into Ansible playbooks:

```yaml
- name: Setup automated scripts
  copy:
    src: "{{ item }}"
    dest: "/opt/scripts/{{ item }}"
    mode: '0755'
  loop:
    - automated_backup.sh
    - automated_monitoring.sh
    - automated_update.sh

- name: Setup cron jobs
  shell: /opt/scripts/setup_cron_jobs.sh
  become: yes
```

## Security Considerations

1. **File Permissions**: Ensure scripts have appropriate permissions (755)
2. **Sensitive Data**: Use environment variables or AWS Secrets Manager for secrets
3. **S3 Bucket Policies**: Restrict S3 bucket access appropriately
4. **Cron User**: Run cron jobs as non-root user when possible
5. **Log Rotation**: Prevent log files from growing indefinitely

