# Complete Scripts Documentation

Complete reference for all automation scripts in the AI Project Generator AWS deployment.

## 📋 Script Index

### Core Automation Scripts
1. [automated_backup.sh](#automated_backupsh) - Backup automation
2. [automated_monitoring.sh](#automated_monitoringsh) - System monitoring
3. [automated_update.sh](#automated_updatesh) - Application updates
4. [automated_scaling.sh](#automated_scalingsh) - Auto-scaling
5. [disaster_recovery.sh](#disaster_recoverysh) - Disaster recovery
6. [log_analyzer.sh](#log_analyzersh) - Log analysis
7. [security_audit.sh](#security_auditsh) - Security auditing

### Deployment Strategy Scripts
8. [canary_deploy.sh](#canary_deploysh) - Canary deployment
9. [blue_green_deploy.sh](#blue_green_deploysh) - Blue-green deployment
10. [feature_flags.sh](#feature_flagssh) - Feature flags management
11. [performance_test.sh](#performance_testsh) - Performance testing
12. [database_migration.sh](#database_migrationsh) - Database migrations

### Enterprise Scripts
13. [multi_region_deploy.sh](#multi_region_deploysh) - Multi-region deployment
14. [automated_dr.sh](#automated_drsh) - Automated disaster recovery
15. [advanced_logging.sh](#advanced_loggingsh) - Advanced logging & tracing
16. [cost_optimizer.sh](#cost_optimizersh) - Cost optimization

### Observability & Security Scripts
17. [prometheus_setup.sh](#prometheus_setupsh) - Prometheus & Grafana setup
18. [security_hardening.sh](#security_hardeningsh) - Security hardening
19. [api_gateway_setup.sh](#api_gateway_setupsh) - API Gateway configuration
20. [chaos_engineering.sh](#chaos_engineeringsh) - Chaos engineering tests

### Utility Scripts
17. [setup_cron_jobs.sh](#setup_cron_jobssh) - Cron job configuration
18. [deploy.sh](#deploysh) - Main deployment script
19. [health_check.sh](#health_checksh) - Health checking
20. [common_functions.sh](#common_functionssh) - Shared functions library
21. [rollback.sh](#rollbacksh) - Application rollback

## 📚 Detailed Documentation

### automated_backup.sh

**Purpose**: Automated backup of application data, logs, and configurations.

**Features**:
- Application files backup
- Database backup (PostgreSQL)
- Redis data backup
- Configuration backup
- S3 upload with verification
- Integrity verification
- CloudWatch metrics
- Automatic cleanup

**Usage**:
```bash
./automated_backup.sh
```

**Environment Variables**:
- `BACKUP_DIR`: Backup directory
- `S3_BUCKET`: S3 bucket for backups
- `RETENTION_DAYS`: Days to keep backups
- `VERIFY_INTEGRITY`: Enable integrity checks
- `CLOUDWATCH_NAMESPACE`: CloudWatch namespace

### automated_monitoring.sh

**Purpose**: Monitor application health and system resources.

**Features**:
- Application health checks
- CPU/Memory/Disk monitoring
- Docker container monitoring
- Service status checks
- CloudWatch metrics
- SNS alerts
- Comprehensive reporting

**Usage**:
```bash
./automated_monitoring.sh
```

**Environment Variables**:
- `APP_PORT`: Application port
- `CPU_THRESHOLD`: CPU threshold %
- `MEMORY_THRESHOLD`: Memory threshold %
- `SNS_TOPIC_ARN`: SNS topic for alerts
- `ENABLE_CLOUDWATCH`: Enable CloudWatch

### automated_update.sh

**Purpose**: Automated application updates with zero-downtime.

**Features**:
- Automatic backup before update
- Git-based code updates
- Ansible deployment
- Health check verification
- Automatic rollback on failure
- Version tracking

**Usage**:
```bash
export GIT_REPO=https://github.com/user/repo.git
./automated_update.sh
```

**Environment Variables**:
- `GIT_REPO`: Git repository URL
- `GIT_BRANCH`: Branch to deploy
- `BACKUP_BEFORE_UPDATE`: Create backup first
- `HEALTH_CHECK_RETRIES`: Retry count

### automated_scaling.sh (NEW)

**Purpose**: Automatic scaling based on metrics.

**Features**:
- CPU-based scaling
- Memory-based scaling
- Request rate scaling
- Cooldown periods
- CloudWatch integration
- SNS notifications

**Usage**:
```bash
export AUTO_SCALING_GROUP=ai-project-generator-asg
./automated_scaling.sh
```

**Environment Variables**:
- `AUTO_SCALING_GROUP`: Auto Scaling Group name
- `CPU_SCALE_UP_THRESHOLD`: CPU threshold for scale up
- `MIN_INSTANCES`: Minimum instances
- `MAX_INSTANCES`: Maximum instances

### disaster_recovery.sh (NEW)

**Purpose**: Automated disaster recovery procedures.

**Features**:
- Disaster detection
- Backup restoration
- S3 backup restoration
- Application rebuild
- Failover support
- Recovery verification

**Usage**:
```bash
export RECOVERY_MODE=restore
./disaster_recovery.sh
```

**Environment Variables**:
- `RECOVERY_MODE`: restore, rebuild, or failover
- `RESTORE_FROM_S3`: Use S3 backups
- `S3_BUCKET`: S3 bucket for backups

### log_analyzer.sh (NEW)

**Purpose**: Analyze logs for errors and patterns.

**Features**:
- Error detection
- Warning detection
- Pattern analysis
- Anomaly detection
- Report generation
- CloudWatch metrics

**Usage**:
```bash
./log_analyzer.sh
```

**Environment Variables**:
- `LOG_DIR`: Log directory
- `ANALYSIS_PERIOD`: Hours to analyze
- `SEND_REPORT`: Send report via email

### security_audit.sh (NEW)

**Purpose**: Security auditing and hardening checks.

**Features**:
- SSH security checks
- Firewall verification
- File permissions audit
- Secrets detection
- System update checks
- Docker security checks
- Report generation

**Usage**:
```bash
./security_audit.sh
```

**Environment Variables**:
- `REPORT_DIR`: Report output directory
- `CHECK_SSH`: Enable SSH checks
- `CHECK_FIREWALL`: Enable firewall checks

### setup_cron_jobs.sh

**Purpose**: Configure automated cron jobs.

**Features**:
- Backup cron setup
- Monitoring cron setup
- Update cron setup
- Log rotation configuration

**Usage**:
```bash
sudo ./setup_cron_jobs.sh
```

### canary_deploy.sh

**Purpose**: Gradual rollout deployment with traffic splitting.

**Features**:
- Deploy new version alongside existing
- Route percentage of traffic to new version
- Monitor metrics (error rate, latency)
- Promote to 100% or rollback based on results
- Zero-downtime deployment

**Usage**:
```bash
# Deploy canary at 10% traffic, monitor for 5 minutes
sudo ./canary_deploy.sh 10 300 deploy

# Promote canary to 100%
sudo ./canary_deploy.sh 10 300 promote

# Rollback canary
sudo ./canary_deploy.sh 10 300 rollback
```

**Parameters**:
- Percentage: Traffic percentage (1-100)
- Monitoring Duration: Seconds to monitor
- Action: deploy, promote, rollback, monitor

### blue_green_deploy.sh

**Purpose**: Zero-downtime deployment with instant rollback.

**Features**:
- Maintain two identical environments
- Instant traffic switching
- Full environment testing
- Quick rollback capability

**Usage**:
```bash
# Deploy to inactive environment and switch
sudo ./blue_green_deploy.sh deploy

# Check current deployment
sudo ./blue_green_deploy.sh status

# Rollback to previous environment
sudo ./blue_green_deploy.sh rollback
```

### feature_flags.sh

**Purpose**: Feature flags management for A/B testing and gradual rollouts.

**Features**:
- Enable/disable features without deployment
- Gradual feature rollout (percentage-based)
- A/B testing support
- Instant feature toggling

**Usage**:
```bash
# Enable feature at 100%
sudo ./feature_flags.sh enable new_feature 100

# Enable feature at 25%
sudo ./feature_flags.sh enable new_feature 25

# Disable feature
sudo ./feature_flags.sh disable new_feature

# List all features
sudo ./feature_flags.sh list

# Check feature status
sudo ./feature_flags.sh status new_feature
```

### performance_test.sh

**Purpose**: Performance testing before deployment.

**Features**:
- Load testing with Apache Bench or wrk
- Latency analysis
- Throughput measurement
- Error rate detection
- Threshold validation

**Usage**:
```bash
# Basic performance test
sudo ./performance_test.sh http://localhost:8020 10 60 100

# Parameters:
# - URL: Target URL
# - Concurrent users: 10
# - Duration: 60 seconds
# - Requests per second: 100
```

**Metrics Checked**:
- Average Latency: Must be < 500ms
- Requests per Second: Must be > 50 req/s
- Failed Requests: Must be 0

### database_migration.sh

**Purpose**: Safe database migrations with rollback support.

**Features**:
- Automatic backups before migration
- Support for PostgreSQL, MySQL, SQLite
- Migration verification
- Automatic rollback on failure
- Alembic integration

**Usage**:
```bash
# Run all pending migrations
sudo ./database_migration.sh migrate

# Run specific migration
sudo ./database_migration.sh migrate migration_001.sql

# Create backup
sudo ./database_migration.sh backup

# Rollback to latest backup
sudo ./database_migration.sh rollback

# Verify migration
sudo ./database_migration.sh verify
```

### multi_region_deploy.sh

**Purpose**: Deploy application to multiple AWS regions.

**Features**:
- Parallel or sequential deployment
- Health checks per region
- Selective rollback
- Global availability

**Usage**:
```bash
# Deploy to multiple regions in parallel
export REGIONS="us-east-1,us-west-2,eu-west-1"
export DEPLOYMENT_STRATEGY="parallel"
sudo ./multi_region_deploy.sh deploy

# Health check all regions
sudo ./multi_region_deploy.sh health-check

# Rollback specific region
sudo ./multi_region_deploy.sh rollback us-west-2
```

**Environment Variables**:
- `REGIONS`: Comma-separated list of regions
- `DEPLOYMENT_STRATEGY`: parallel or sequential

### automated_dr.sh

**Purpose**: Automated disaster recovery with failover.

**Features**:
- Continuous health monitoring
- Automatic failover to DR region
- SNS notifications
- Failback support
- DR testing

**Usage**:
```bash
# Continuous monitoring mode
sudo ./automated_dr.sh monitor

# Manual failover to DR region
sudo ./automated_dr.sh failover

# Failback to primary region
sudo ./automated_dr.sh failback

# Test DR procedure
sudo ./automated_dr.sh test
```

**Environment Variables**:
- `PRIMARY_REGION`: Primary AWS region
- `DR_REGION`: DR AWS region
- `FAILOVER_THRESHOLD`: Failed health checks before failover
- `SNS_TOPIC_ARN`: SNS topic for notifications

### advanced_logging.sh

**Purpose**: Advanced logging and distributed tracing.

**Features**:
- Structured JSON logging
- CloudWatch integration
- Distributed tracing (OpenTelemetry)
- Log rotation
- Log analysis tools
- S3 export

**Usage**:
```bash
# Setup advanced logging
sudo ./advanced_logging.sh setup

# Analyze logs
sudo ./advanced_logging.sh analyze /var/log/ai-project-generator/app.log

# Export logs to S3
sudo ./advanced_logging.sh export ai-project-generator-logs
```

**Environment Variables**:
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `ENABLE_TRACING`: true or false
- `CLOUDWATCH_ENABLED`: true or false

### cost_optimizer.sh

**Purpose**: Monitor and optimize AWS costs.

**Features**:
- Daily cost tracking
- Cost by service breakdown
- Unused resource identification
- Instance optimization recommendations
- Cost alerts
- Automated cost reports

**Usage**:
```bash
# Generate cost report
sudo ./cost_optimizer.sh report

# Find optimization opportunities
sudo ./cost_optimizer.sh optimize

# Monitor daily costs
sudo ./cost_optimizer.sh monitor
```

**Environment Variables**:
- `COST_THRESHOLD`: Daily cost threshold in USD
- `ALERT_EMAIL`: Email for cost reports
- `SNS_TOPIC_ARN`: SNS topic for alerts

### prometheus_setup.sh

**Purpose**: Setup Prometheus and Grafana for advanced observability.

**Features**:
- Prometheus installation and configuration
- Grafana installation and setup
- Node Exporter for system metrics
- Pre-configured scrape targets
- Systemd services
- Dashboard import instructions

**Usage**:
```bash
# Install Prometheus and Grafana
sudo ./prometheus_setup.sh install

# Start services
sudo ./prometheus_setup.sh start

# Stop services
sudo ./prometheus_setup.sh stop

# Check status
sudo ./prometheus_setup.sh status
```

**Access**:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### security_hardening.sh

**Purpose**: Comprehensive security hardening.

**Features**:
- Firewall (UFW) configuration
- Fail2ban intrusion prevention
- Automatic security updates
- SSH hardening
- Audit logging (auditd)
- AppArmor security profiles
- Security scanning tools (rkhunter, chkrootkit, ClamAV)

**Usage**:
```bash
# Full security hardening
sudo ./security_hardening.sh all

# Individual components
sudo ./security_hardening.sh firewall
sudo ./security_hardening.sh fail2ban
sudo ./security_hardening.sh ssh

# Run security scan
sudo ./security_hardening.sh scan
```

### api_gateway_setup.sh

**Purpose**: API Gateway with rate limiting and versioning.

**Features**:
- Nginx-based API Gateway
- Rate limiting (per IP)
- API versioning support
- Security headers
- Connection limiting
- Health check endpoint
- Metrics endpoint
- Custom error pages

**Usage**:
```bash
# Setup API Gateway
sudo ./api_gateway_setup.sh setup

# Reload configuration
sudo ./api_gateway_setup.sh reload

# Monitor API metrics
sudo ./api_gateway_setup.sh monitor
```

**Rate Limits**:
- API endpoints: 10 req/s (burst: 20)
- Auth endpoints: 5 req/s (burst: 5)
- Connection limit: 10 per IP

### chaos_engineering.sh

**Purpose**: Chaos engineering for resilience testing.

**Features**:
- CPU stress testing
- Memory pressure injection
- Network chaos (latency, packet loss)
- Disk I/O stress
- Service failure simulation
- Container failure simulation
- Automatic monitoring during chaos
- Full chaos test suite

**Usage**:
```bash
# CPU chaos (60s, 50% load)
sudo ./chaos_engineering.sh cpu 60 50

# Memory chaos (60s, 512MB)
sudo ./chaos_engineering.sh memory 60 512

# Network chaos (60s, 100ms latency, 5% loss)
sudo ./chaos_engineering.sh network 60 100 5

# Disk I/O chaos
sudo ./chaos_engineering.sh disk 60

# Service chaos
sudo ./chaos_engineering.sh service ai-project-generator 30

# Container chaos
sudo ./chaos_engineering.sh container ai-project-generator 30

# Run full chaos test suite
sudo ./chaos_engineering.sh suite
```

### rollback.sh

**Purpose**: Application rollback to previous version.

**Features**:
- Automatic version detection
- Backup restoration
- Health check verification
- Database rollback support

**Usage**:
```bash
# Rollback to previous version
sudo ./rollback.sh

# Rollback to specific version
sudo ./rollback.sh v1.2.3
```

### common_functions.sh

**Purpose**: Shared functions library.

**Features**:
- Logging functions
- AWS integration
- System utilities
- Network utilities
- Validation functions
- Retry mechanisms

**Usage**:
```bash
source common_functions.sh
```

## 🔄 Workflow Examples

### Daily Operations
```bash
# Morning: Check system health
./automated_monitoring.sh

# Afternoon: Analyze logs
./log_analyzer.sh

# Evening: Security audit
./security_audit.sh
```

### Weekly Operations
```bash
# Sunday: Full backup
./automated_backup.sh

# Sunday: System update
export GIT_BRANCH=main
./automated_update.sh
```

### On-Demand Operations
```bash
# Scale up for high load
export AUTO_SCALING_GROUP=my-asg
./automated_scaling.sh

# Disaster recovery
export RECOVERY_MODE=restore
./disaster_recovery.sh

# Canary deployment
./canary_deploy.sh 10 300 deploy

# Multi-region deployment
export REGIONS="us-east-1,us-west-2"
./multi_region_deploy.sh deploy

# Cost optimization
./cost_optimizer.sh optimize
```

## 📊 Monitoring Dashboard

All scripts send metrics to CloudWatch. Create dashboards for:
- Backup success/failure rates
- System resource usage
- Application health
- Scaling actions
- Security issues
- Log analysis results

## 🔔 Alerting

Configure SNS topics for:
- Backup failures
- High resource usage
- Application health issues
- Security vulnerabilities
- Scaling actions
- Disaster recovery events

## 📈 Best Practices

1. **Schedule Regular Backups**: Daily backups with S3 storage
2. **Monitor Continuously**: 5-minute monitoring intervals
3. **Update Weekly**: Automated updates on low-traffic periods
4. **Audit Monthly**: Security audits and log analysis
5. **Test Recovery**: Quarterly disaster recovery drills
6. **Review Metrics**: Weekly review of CloudWatch dashboards

## 🛠️ Troubleshooting

### Scripts Not Running
- Check file permissions: `chmod +x script.sh`
- Check cron service: `systemctl status cron`
- Check logs: `/var/log/*.log`

### AWS Integration Issues
- Verify credentials: `aws sts get-caller-identity`
- Check IAM permissions
- Verify SNS topic ARN
- Check CloudWatch namespace

### Performance Issues
- Review script execution time
- Check system resources
- Optimize script logic
- Use parallel execution where possible

## 📝 Maintenance

### Regular Tasks
- Review and update thresholds
- Update script versions
- Test disaster recovery
- Review security audit results
- Optimize backup retention

### Version Control
- Keep scripts in version control
- Tag releases
- Document changes
- Test before deployment

