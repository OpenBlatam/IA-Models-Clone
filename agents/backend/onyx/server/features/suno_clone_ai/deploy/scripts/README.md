# Deployment Scripts

Collection of utility scripts for managing Suno Clone AI deployment on EC2.

## 📋 Available Scripts

### Core Deployment

- **`deploy.sh`** - Main deployment script
  ```bash
  ./deploy.sh
  ```

### Backup & Restore

- **`backup.sh`** - Create backups
  ```bash
  ./backup.sh
  # With S3 upload (set S3_BACKUP_BUCKET env var)
  S3_BACKUP_BUCKET=my-bucket ./backup.sh
  ```

- **`restore.sh`** - Restore from backup
  ```bash
  # Restore all
  ./restore.sh suno-clone-ai-backup-20240101_120000.tar.gz
  
  # Restore specific components
  ./restore.sh backup.tar.gz --database
  ./restore.sh backup.tar.gz --audio
  ./restore.sh backup.tar.gz --config
  ```

### Rollback

- **`rollback.sh`** - Rollback to previous version
  ```bash
  # List available versions
  ./rollback.sh --list
  
  # Rollback to previous version
  ./rollback.sh
  
  # Rollback to specific version
  ./rollback.sh 20240101_120000
  ```

### Monitoring

- **`monitor.sh`** - Health monitoring and alerts
  ```bash
  # Full report
  ./monitor.sh report
  
  # Check specific components
  ./monitor.sh health
  ./monitor.sh metrics
  ./monitor.sh disk
  ./monitor.sh logs
  ```

### Updates

- **`update.sh`** - Update application
  ```bash
  ./update.sh
  ```

### SSL/TLS

- **`setup_ssl.sh`** - Setup SSL certificates
  ```bash
  # Let's Encrypt (production)
  sudo ./setup_ssl.sh example.com admin@example.com
  
  # Self-signed (testing)
  sudo ./setup_ssl.sh --self-signed
  ```

### Maintenance

- **`maintenance.sh`** - Routine maintenance
  ```bash
  # Run all maintenance tasks
  ./maintenance.sh all
  
  # Specific tasks
  ./maintenance.sh docker
  ./maintenance.sh logs
  ./maintenance.sh database
  ./maintenance.sh disk
  ```

## 🔧 Configuration

### Environment Variables

Set these in your `.env` file or export them:

```bash
# Backup
BACKUP_DIR=/backups/suno-clone-ai
RETENTION_DAYS=7
S3_BACKUP_BUCKET=my-backup-bucket

# Monitoring
ALERT_EMAIL=admin@example.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

## 📅 Cron Jobs

Add to crontab for automated tasks:

```bash
# Daily backup at 2 AM
0 2 * * * /home/ubuntu/suno-clone-ai/deploy/scripts/backup.sh

# Health check every 5 minutes
*/5 * * * * /home/ubuntu/suno-clone-ai/deploy/scripts/monitor.sh health

# Weekly maintenance on Sundays at 3 AM
0 3 * * 0 /home/ubuntu/suno-clone-ai/deploy/scripts/maintenance.sh all
```

## 🔒 Security

- All scripts use `set -euo pipefail` for error handling
- Sensitive operations require confirmation
- Scripts validate inputs before execution
- Use `sudo` only when necessary

## 📚 Usage Examples

### Daily Operations

```bash
# Morning health check
./monitor.sh report

# Create backup before update
./backup.sh

# Update application
./update.sh

# Verify deployment
./monitor.sh health
```

### Emergency Procedures

```bash
# Container not responding
docker restart suno-clone-ai
./monitor.sh health

# Rollback after failed update
./rollback.sh

# Restore from backup
./restore.sh latest-backup.tar.gz
```

### Maintenance Window

```bash
# Run full maintenance
./maintenance.sh all

# Check disk space
./maintenance.sh disk

# Clean up old resources
./maintenance.sh docker
```

## 🆘 Troubleshooting

### Script Permission Denied

```bash
chmod +x deploy/scripts/*.sh
```

### Docker Not Found

```bash
# Check Docker installation
docker --version

# Start Docker service
sudo systemctl start docker
```

### Backup Fails

```bash
# Check disk space
df -h

# Check backup directory permissions
ls -la /backups/suno-clone-ai
```

## 📝 Notes

- All scripts log to stdout/stderr
- Use `--help` flag for usage information (where supported)
- Scripts are idempotent (safe to run multiple times)
- Check logs in `/var/log/` for detailed information




