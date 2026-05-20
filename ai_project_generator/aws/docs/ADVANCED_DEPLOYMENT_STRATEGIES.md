# Advanced Deployment Strategies

Complete guide for advanced deployment strategies and features.

## 📋 Table of Contents

1. [Canary Deployments](#canary-deployments)
2. [Blue-Green Deployments](#blue-green-deployments)
3. [Feature Flags](#feature-flags)
4. [Performance Testing](#performance-testing)
5. [Database Migrations](#database-migrations)

## 🎯 Canary Deployments

### Overview

Canary deployment gradually rolls out new version to a percentage of traffic, allowing you to monitor and validate before full rollout.

### Usage

```bash
# Deploy canary at 10% traffic, monitor for 5 minutes
sudo /opt/ai-project-generator/scripts/canary_deploy.sh 10 300 deploy

# Promote canary to 100%
sudo /opt/ai-project-generator/scripts/canary_deploy.sh 10 300 promote

# Rollback canary
sudo /opt/ai-project-generator/scripts/canary_deploy.sh 10 300 rollback
```

### Parameters

- **Percentage**: Traffic percentage (1-100)
- **Monitoring Duration**: Seconds to monitor (default: 300)
- **Action**: deploy, promote, rollback, monitor

### Workflow

1. Deploy new version alongside existing
2. Route X% traffic to new version
3. Monitor metrics (error rate, latency)
4. Promote to 100% or rollback based on results

### Benefits

- ✅ Gradual rollout reduces risk
- ✅ Real-time monitoring
- ✅ Quick rollback if issues detected
- ✅ Zero-downtime deployment

## 🔵 Blue-Green Deployments

### Overview

Blue-green deployment maintains two identical production environments. Traffic switches instantly between them.

### Usage

```bash
# Deploy to inactive environment and switch
sudo /opt/ai-project-generator/scripts/blue_green_deploy.sh deploy

# Check current deployment
sudo /opt/ai-project-generator/scripts/blue_green_deploy.sh status

# Rollback to previous environment
sudo /opt/ai-project-generator/scripts/blue_green_deploy.sh rollback
```

### Workflow

1. Deploy new version to inactive environment (green)
2. Run health checks
3. Switch traffic from active (blue) to new (green)
4. Monitor new environment
5. Keep old environment for quick rollback

### Benefits

- ✅ Instant rollback
- ✅ Zero-downtime
- ✅ Full environment testing
- ✅ Reduced deployment risk

## 🚩 Feature Flags

### Overview

Feature flags allow gradual feature rollouts and A/B testing without code deployments.

### Usage

```bash
# Enable feature at 100%
sudo /opt/ai-project-generator/scripts/feature_flags.sh enable new_feature 100

# Enable feature at 25%
sudo /opt/ai-project-generator/scripts/feature_flags.sh enable new_feature 25

# Disable feature
sudo /opt/ai-project-generator/scripts/feature_flags.sh disable new_feature

# List all features
sudo /opt/ai-project-generator/scripts/feature_flags.sh list

# Check feature status
sudo /opt/ai-project-generator/scripts/feature_flags.sh status new_feature
```

### Feature Flag File

Location: `/opt/ai-project-generator/config/feature_flags.json`

```json
{
  "features": {
    "new_feature": {
      "enabled": true,
      "percentage": 25,
      "updated_at": "2024-01-01T00:00:00Z"
    }
  },
  "version": "1.0",
  "last_updated": "2024-01-01T00:00:00Z"
}
```

### Benefits

- ✅ Gradual feature rollout
- ✅ A/B testing support
- ✅ Instant enable/disable
- ✅ No deployment needed

## ⚡ Performance Testing

### Overview

Performance testing validates application performance before deployment.

### Usage

```bash
# Basic performance test
sudo /opt/ai-project-generator/scripts/performance_test.sh http://localhost:8020 10 60 100

# Parameters:
# - URL: Target URL
# - Concurrent users: 10
# - Duration: 60 seconds
# - Requests per second: 100
```

### Metrics Checked

- **Average Latency**: Must be < 500ms
- **Requests per Second**: Must be > 50 req/s
- **Failed Requests**: Must be 0

### Tools Supported

- **Apache Bench (ab)**: Basic load testing
- **wrk**: Advanced load testing with latency stats

### Installation

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Install wrk
sudo apt-get install wrk
```

### Benefits

- ✅ Catch performance issues early
- ✅ Validate under load
- ✅ Set performance baselines
- ✅ Prevent production issues

## 🗄️ Database Migrations

### Overview

Safe database migrations with automatic backups and rollback support.

### Usage

```bash
# Run all pending migrations
sudo /opt/ai-project-generator/scripts/database_migration.sh migrate

# Run specific migration
sudo /opt/ai-project-generator/scripts/database_migration.sh migrate migration_001.sql

# Create backup
sudo /opt/ai-project-generator/scripts/database_migration.sh backup

# Rollback to latest backup
sudo /opt/ai-project-generator/scripts/database_migration.sh rollback

# Rollback to specific backup
sudo /opt/ai-project-generator/scripts/database_migration.sh rollback /opt/backups/database/db_backup_20240101_120000.sql

# Verify migration
sudo /opt/ai-project-generator/scripts/database_migration.sh verify
```

### Supported Databases

- **PostgreSQL**: Full support
- **MySQL**: Full support
- **SQLite**: Full support

### Migration Directory

Location: `/opt/ai-project-generator/migrations/`

Format: `YYYYMMDD_HHMMSS_description.sql`

### Workflow

1. Create database backup
2. Run migration
3. Verify migration
4. Rollback if verification fails

### Benefits

- ✅ Automatic backups
- ✅ Safe rollback
- ✅ Migration verification
- ✅ Support for multiple databases

## 🔄 Integration with CI/CD

### GitHub Actions

The advanced workflow (`.github/workflows/deploy-ec2-advanced.yml`) supports:

- **Deployment Strategy Selection**: standard, canary, blue-green
- **Performance Testing**: Before deployment
- **Database Migrations**: Automatic migration execution

### Example Workflow

```yaml
# Manual trigger with canary deployment
workflow_dispatch:
  inputs:
    deployment_strategy: canary
    canary_percentage: 10
    run_performance_tests: true
    run_migrations: true
```

## 📊 Comparison Matrix

| Strategy | Downtime | Rollback Speed | Risk Level | Use Case |
|----------|----------|----------------|------------|----------|
| **Standard** | Minimal | Medium | Medium | Regular deployments |
| **Canary** | None | Fast | Low | New features, major updates |
| **Blue-Green** | None | Instant | Low | Critical updates |
| **Feature Flags** | None | Instant | Very Low | Feature toggles |

## 🎓 Best Practices

### Canary Deployments

1. Start with 5-10% traffic
2. Monitor for at least 5 minutes
3. Gradually increase if metrics are good
4. Have rollback plan ready

### Blue-Green Deployments

1. Keep old environment for 24-48 hours
2. Monitor both environments
3. Test thoroughly before switching
4. Document switch procedures

### Feature Flags

1. Use for risky features
2. Start with low percentage
3. Monitor metrics closely
4. Have disable plan ready

### Performance Testing

1. Test before every deployment
2. Set realistic thresholds
3. Test under production-like load
4. Monitor trends over time

### Database Migrations

1. Always backup before migration
2. Test migrations in staging first
3. Run during low-traffic periods
4. Have rollback plan ready

## 🔗 Related Documentation

- [Enhanced CI/CD](ENHANCED_CI_CD.md)
- [CI/CD Setup](CI_CD_SETUP.md)
- [Troubleshooting](TROUBLESHOOTING.md)

---

**Version**: 1.0  
**Status**: Production Ready ✅  
**Last Updated**: 2024

